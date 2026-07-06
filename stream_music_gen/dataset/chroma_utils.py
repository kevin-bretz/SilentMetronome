"""Utilities for computing per-frame chroma signals.

`compute_chroma_from_midi` gives a clean deterministic aux-loss target from
the symbolic source (pitched stems only, drums flagged via ``has_chroma``).
`compute_chroma_from_audio` gives a DiT conditioning signal extracted from
audio, the same code path applied at inference. Both produce ``[T, 12]``
float32 tensors at 50 Hz, matching the ``beat_cond.pt`` shape.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

FRAME_RATE_HZ = 50
NUM_PITCH_CLASSES = 12


@functools.lru_cache(maxsize=4096)
def _parse_midi_notes(
    midi_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a single-stem slakh MIDI file into note arrays.

    Returns (start_sec, end_sec, pitch, velocity). Empty arrays if no notes.
    Velocity is the MIDI velocity (1-127) of the originating ``note_on``.

    Cached per-process since every window of a track loads the same MIDI
    file. Shared by the chroma and multipitch/velocity extractors, so each
    file is parsed once regardless of how many signals are derived.
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


def compute_chroma_from_midi(
    midi_path: str,
    start_frame: int,
    num_frames: int = 1000,
    frame_rate_hz: int = FRAME_RATE_HZ,
) -> tuple[np.ndarray, bool]:
    """Compute per-frame 12-class chroma from a slakh stem MIDI file.

    Returns (chroma [T, 12] float32, has_chroma bool). ``has_chroma`` is False
    when the MIDI is missing, unparseable, or contains no notes (e.g., empty
    stem). Drum-stem detection is handled by the caller via the track
    metadata.yaml, this function does not inspect channel 10.
    """
    starts, ends, pitches = _parse_midi_notes(midi_path)
    chroma = np.zeros((num_frames, NUM_PITCH_CLASSES), dtype=np.float32)
    if starts.size == 0:
        return chroma, False

    win_start_sec = start_frame / float(frame_rate_hz)
    win_end_sec = (start_frame + num_frames) / float(frame_rate_hz)

    overlap_mask = (ends > win_start_sec) & (starts < win_end_sec)
    if not overlap_mask.any():
        # Empty stretch. Return zeros but flag has_chroma False so the
        # dataloader / aux loss can mask this window out.
        return chroma, False

    s = starts[overlap_mask]
    e = ends[overlap_mask]
    p = pitches[overlap_mask] % NUM_PITCH_CLASSES

    # Convert per-note seconds to per-frame inclusive ranges, clipped to window.
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
            chroma[f_lo[i]:f_hi[i], p[i]] += 1.0

    # Max-normalize each frame so the dominant pitch class is 1.0, keeping
    # the signal bounded in [0, 1].
    row_max = chroma.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    chroma = chroma / row_max

    return chroma.astype(np.float32), True


def compute_chroma_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    num_frames: int = 1000,
    frame_rate_hz: int = FRAME_RATE_HZ,
    hpss: bool = False,
) -> np.ndarray:
    """Compute per-frame 12-class chroma from a mono audio array.

    ``audio`` is shape ``[N]`` or ``[1, N]``, float32 in [-1, 1].
    Returns ``[num_frames, 12]`` float32 in [0, 1] (per-frame max-normalized).

    HPSS is opt-in because full librosa.effects.hpss costs ~80s per 20s
    window, while STFT-domain HPSS (margin=1) is ~1.2s. Default off. CQT
    alone tolerates moderate drum smear because broadband transients spread
    roughly uniformly across pitch classes and are washed out by per-frame
    max-normalization.
    """
    import librosa

    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = np.ascontiguousarray(audio.astype(np.float32))

    hop_length = sample_rate // frame_rate_hz  # e.g. 32000 / 50 = 640

    if hpss:
        # STFT-domain HPSS is ~10x faster than librosa.effects.hpss because
        # it skips the inverse STFT for the unused percussive component.
        S = librosa.stft(audio, n_fft=2048, hop_length=hop_length)
        H, _ = librosa.decompose.hpss(S, margin=1.0)
        y_h = librosa.istft(H, hop_length=hop_length, length=len(audio))
    else:
        y_h = audio

    # Constant-Q chroma is harmonically aware and tolerates bass dominance
    # better than chroma_stft. n_chroma=12 is standard.
    chroma = librosa.feature.chroma_cqt(
        y=y_h, sr=sample_rate, hop_length=hop_length, n_chroma=NUM_PITCH_CLASSES
    )
    # chroma is [12, T_lib]. T_lib may differ slightly from num_frames due
    # to padding, so align by truncating / zero-padding on the right.
    chroma = chroma.T  # [T_lib, 12]
    if chroma.shape[0] >= num_frames:
        chroma = chroma[:num_frames, :]
    else:
        pad = np.zeros((num_frames - chroma.shape[0], NUM_PITCH_CLASSES), np.float32)
        chroma = np.concatenate([chroma, pad], axis=0)

    # Per-frame max-normalize for consistency with MIDI-derived chroma scale.
    row_max = chroma.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    chroma = chroma / row_max

    return chroma.astype(np.float32)


def target_audio_path_to_midi_path(target_audio_path: str) -> str:
    """Map a slakh target-stem flac path to its sibling MIDI path.

    `<...>/Track01819/stems/S07.flac` -> `<...>/Track01819/MIDI/S07.mid`
    """
    p = Path(target_audio_path)
    return str(p.parent.parent / "MIDI" / (p.stem + ".mid"))


@functools.lru_cache(maxsize=2048)
def load_track_metadata_yaml(yaml_path: str) -> dict:
    """Cached YAML loader. Returns empty dict if missing."""
    import yaml

    if not os.path.exists(yaml_path):
        return {}
    try:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def is_drum_stem(slakh_track_dir: str, stem_id: str) -> bool:
    """Read metadata.yaml in the slakh track dir and check stem_id (e.g. 'S07')."""
    yaml_path = str(Path(slakh_track_dir) / "metadata.yaml")
    meta = load_track_metadata_yaml(yaml_path)
    return bool(meta.get("stems", {}).get(stem_id, {}).get("is_drum", False))
