"""Utilities for computing per-frame multi-pitch (piano roll) signals.

Multipitch preserves register (C4 vs C5 are different bins), keeps
note-event structure as explicit bit flips, and captures drum sounds, since slakh drum stems use GM channel 10 where the note
number identifies the drum sound (kick=36, snare=38, hi-hat=42, etc.).

The full 128 dims cover the entire MIDI range uniformly across stem
types, with ``dec_inst_tokens`` providing the context to interpret the
active bins as drum vs pitched.

Output is ``[T, 128]`` ``uint8`` (binary {0, 1}) at 50 Hz frame rate to
match ``beat_cond.pt``. uint8 saves ~4x storage
vs float32 since the values are binary.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Tuple

import numpy as np

FRAME_RATE_HZ = 50
NUM_PITCHES = 128


@functools.lru_cache(maxsize=4096)
def _parse_midi_notes(
    midi_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a single-stem slakh MIDI file into note arrays.

    Returns (start_sec, end_sec, pitch, velocity). Empty arrays if no notes.
    Velocity is the MIDI velocity (1-127) of the originating ``note_on``.

    Cached per-process since every window of a track loads the same MIDI
    file, so each file is parsed once regardless of how many signals are
    derived.
    """
    import mido  # imported here so workers without mido don't error at import-time

    if not os.path.exists(midi_path):
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
        )

    try:
        mid = mido.MidiFile(midi_path)
    except (OSError, EOFError, ValueError):
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
        )

    # Iterating a mido.MidiFile yields message times in seconds with tempo
    # threaded through the merged-track iterator. Slakh stems are one stem
    # per file with tempo at the start, so this is safe.
    starts_list: list[float] = []
    ends_list: list[float] = []
    pitches: list[int] = []
    velocities: list[int] = []

    abs_t = 0.0
    # open_notes: pitch -> (start_sec, velocity)
    open_notes: dict[int, tuple[float, int]] = {}
    for msg in mid:
        abs_t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            # If the same pitch is already open, close it first (overlap).
            if msg.note in open_notes:
                start_t, vel = open_notes.pop(msg.note)
                starts_list.append(start_t)
                ends_list.append(abs_t)
                pitches.append(msg.note)
                velocities.append(vel)
            open_notes[msg.note] = (abs_t, msg.velocity)
        elif (msg.type == "note_off") or (
            msg.type == "note_on" and msg.velocity == 0
        ):
            if msg.note in open_notes:
                start_t, vel = open_notes.pop(msg.note)
                starts_list.append(start_t)
                ends_list.append(abs_t)
                pitches.append(msg.note)
                velocities.append(vel)
    # Close any still-open notes at end of file.
    for pitch, (start_t, vel) in open_notes.items():
        starts_list.append(start_t)
        ends_list.append(abs_t)
        pitches.append(pitch)
        velocities.append(vel)

    return (
        np.asarray(starts_list, dtype=np.float64),
        np.asarray(ends_list, dtype=np.float64),
        np.asarray(pitches, dtype=np.int64),
        np.asarray(velocities, dtype=np.int64),
    )


def _rasterize_notes(
    starts: np.ndarray,
    ends: np.ndarray,
    pitches: np.ndarray,
    velocities: np.ndarray,
    start_frame: int,
    num_frames: int,
    frame_rate_hz: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Internal: rasterize note-event arrays to per-frame piano roll +
    velocity. Returns (multipitch [T, 128] uint8, velocity [T, 128] uint8,
    has_notes bool).

    Velocity convention: per (frame, pitch), max velocity over overlapping
    notes (loudest active hit wins). Out-of-range pitches are dropped.
    """
    multipitch = np.zeros((num_frames, NUM_PITCHES), dtype=np.uint8)
    velocity = np.zeros((num_frames, NUM_PITCHES), dtype=np.uint8)
    if starts.size == 0:
        return multipitch, velocity, False

    win_start_sec = start_frame / float(frame_rate_hz)
    win_end_sec = (start_frame + num_frames) / float(frame_rate_hz)

    overlap_mask = (ends > win_start_sec) & (starts < win_end_sec)
    if not overlap_mask.any():
        return multipitch, velocity, False

    s = starts[overlap_mask]
    e = ends[overlap_mask]
    p = pitches[overlap_mask]
    v = velocities[overlap_mask]

    valid = (p >= 0) & (p < NUM_PITCHES)
    if not valid.any():
        return multipitch, velocity, False
    s = s[valid]
    e = e[valid]
    p = p[valid]
    v = np.clip(v[valid], 0, 127).astype(np.uint8)

    f_lo = np.clip(
        np.floor((s - win_start_sec) * frame_rate_hz).astype(np.int64),
        0,
        num_frames,
    )
    f_hi = np.clip(
        np.ceil((e - win_start_sec) * frame_rate_hz).astype(np.int64),
        0,
        num_frames,
    )

    for i in range(len(p)):
        if f_hi[i] > f_lo[i]:
            multipitch[f_lo[i]:f_hi[i], p[i]] = 1
            # Max velocity wins for overlapping notes at the same pitch.
            cur_vel = velocity[f_lo[i]:f_hi[i], p[i]]
            np.maximum(cur_vel, v[i], out=cur_vel)
            velocity[f_lo[i]:f_hi[i], p[i]] = cur_vel

    return multipitch, velocity, True


def compute_multipitch_from_midi(
    midi_path: str,
    start_frame: int,
    num_frames: int = 1000,
    frame_rate_hz: int = FRAME_RATE_HZ,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Compute per-frame 128-class binary piano roll + velocity from a
    single slakh stem MIDI. Returns (multipitch [T, 128] uint8 in {0, 1},
    velocity [T, 128] uint8 in 0-127, has_notes bool).

    ``has_notes`` is False when the MIDI is missing, unparseable, or
    contains no notes overlapping the requested window.

    Drum stems are not filtered. Their MIDI note numbers identify drum
    sounds (GM channel 10, 36=kick, 38=snare, etc.), so the 128-dim output
    encodes which drum sounds are active and the caller uses the
    instrument-token context to interpret the bins.
    """
    starts, ends, pitches, velocities = _parse_midi_notes(midi_path)
    return _rasterize_notes(
        starts, ends, pitches, velocities,
        start_frame=start_frame,
        num_frames=num_frames,
        frame_rate_hz=frame_rate_hz,
    )


def compute_multipitch_from_midis(
    midi_paths: list,
    start_frame: int,
    num_frames: int = 1000,
    frame_rate_hz: int = FRAME_RATE_HZ,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Compute per-frame multipitch + velocity over the *union* of multiple
    stem MIDIs. Used for the input mix's polyphonic content.

    For each frame and pitch:
      multipitch[t, p] = 1 if any input stem has that note active
      velocity[t, p]   = max velocity over input stems

    Drum stems are included. Their channel-10 note numbers populate the
    typical drum-sound bin range (35-81), and the caller filters drums
    via the dataset metadata if desired.

    Returns (multipitch, velocity, has_any) where has_any is True if at
    least one stem contributed any note overlapping the window.
    """
    multipitch = np.zeros((num_frames, NUM_PITCHES), dtype=np.uint8)
    velocity = np.zeros((num_frames, NUM_PITCHES), dtype=np.uint8)
    has_any = False
    for midi_path in midi_paths:
        if not midi_path or not os.path.exists(midi_path):
            continue
        mp, vel, has = compute_multipitch_from_midi(
            midi_path, start_frame, num_frames, frame_rate_hz
        )
        if not has:
            continue
        has_any = True
        # Union: OR multipitch, max velocity.
        multipitch |= mp
        np.maximum(velocity, vel, out=velocity)
    return multipitch, velocity, has_any


def target_audio_path_to_midi_path(target_audio_path: str) -> str:
    """Map a slakh target/input-stem flac path to its sibling MIDI path.

    `<...>/Track01819/stems/S07.flac` -> `<...>/Track01819/MIDI/S07.mid`
    """
    p = Path(target_audio_path)
    return str(p.parent.parent / "MIDI" / (p.stem + ".mid"))
