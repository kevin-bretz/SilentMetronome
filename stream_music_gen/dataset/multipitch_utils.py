"""Utilities for computing per-frame multi-pitch (piano roll) signals.

Companion to ``chroma_utils.py``. Multipitch carries strictly more
information than chroma: it preserves both pitch class AND register, AND
preserves note-event structure (transitions are explicit bit flips).

Compared to chroma:
  * 128-dim binary instead of 12-dim float — note number resolution
  * Captures register (C4 vs C5 are different bins)
  * Captures drum sounds (slakh drum stems use MIDI channel 10 where the
    note number IS the drum-sound identity per the General MIDI standard
    -- kick=36, snare=38, hi-hat=42, crash=49, etc.)

The full 128-dim covers the entire MIDI range (0-127). Drum sounds live
in the 35-81 range; pitched instrument fundamentals span ~21-108. Using
the full 128 dim is uniform across all stem types, with the
``dec_inst_tokens`` carrying the context needed to interpret the active
bins (drum vs pitched).

Output is ``[T, 128]`` ``uint8`` (binary {0, 1}) at 50 Hz frame rate to
match ``beat_cond.pt`` / ``target_chroma.pt``. uint8 saves ~4x storage
vs float32 since the values are binary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

from stream_music_gen.dataset.chroma_utils import (
    FRAME_RATE_HZ,
    _parse_midi_notes,
)

NUM_PITCHES = 128


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

    NOTE on drums: This function does NOT filter drum stems. For drum
    stems, the MIDI note number is the drum-sound identity (GM channel 10
    convention; 36=kick, 38=snare, 42=closed-hat, etc.). The 128-dim
    binary output naturally encodes which drum sounds are active at each
    frame. The caller / model uses the instrument-token context to
    interpret the bins appropriately.
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

    Drum stems are included — their channel-10 note numbers will populate
    the typical drum-sound bin range (35-81). Caller filters drums via
    the dataset metadata if desired.

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
    Mirror of ``chroma_utils.target_audio_path_to_midi_path``; duplicated
    here to keep multipitch_utils self-contained for callers.
    """
    p = Path(target_audio_path)
    return str(p.parent.parent / "MIDI" / (p.stem + ".mid"))
