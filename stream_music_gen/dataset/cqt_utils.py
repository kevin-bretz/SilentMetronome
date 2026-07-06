"""Utilities for computing per-frame Constant-Q Transform (CQT) signals.

Companion to ``chroma_utils.py``. CQT is a log-frequency spectrogram
that places ``bins_per_octave`` (default 12) bins per octave, giving a
uniform pitch-class resolution that aligns with musical notes and
preserves register (chroma is octave-folded; CQT is not).

Bandwidth is 7 octaves x 12 bins = 84 bins, from C1 (~32 Hz) to C8
(~4186 Hz). This matches MERT's "music teacher" config and stays within
the bandwidth that 4-layer DAC reconstructs faithfully, so the aux loss
is not asking for information the codec drops.

Output is ``[T, 84]`` ``float16`` at 50 Hz frame rate, saved as float16
to halve storage (CQT magnitudes don't require full float32 precision).
"""

from __future__ import annotations

import numpy as np

FRAME_RATE_HZ = 50
NUM_CQT_BINS = 84
BINS_PER_OCTAVE = 12
# C1 = MIDI note 24 = 32.703 Hz. 84 bins above reach C8 (~4186 Hz).
CQT_FMIN_NOTE = "C1"


def compute_cqt_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    num_frames: int = 1000,
    frame_rate_hz: int = FRAME_RATE_HZ,
    n_bins: int = NUM_CQT_BINS,
    bins_per_octave: int = BINS_PER_OCTAVE,
) -> np.ndarray:
    """Compute per-frame log-CQT magnitude from a mono audio array.

    ``audio`` is shape ``[N]`` or ``[1, N]``, float32 in [-1, 1].
    Returns ``[num_frames, n_bins]`` float32 with log-magnitude values
    (caller may cast to float16 for storage). Per-frame max-normalized
    so values are in [-inf, 0] log scale, with frame-max == 0.

    Frame rate matches ``beat_cond`` / chroma; hop_length is derived from
    ``sample_rate / frame_rate_hz`` (e.g., 32000/50 = 640).
    """
    import librosa

    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = np.ascontiguousarray(audio.astype(np.float32))

    hop_length = sample_rate // frame_rate_hz

    fmin = librosa.note_to_hz(CQT_FMIN_NOTE)
    cqt = librosa.cqt(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
    )
    # cqt is complex [n_bins, T_lib]; take magnitude.
    mag = np.abs(cqt).T  # [T_lib, n_bins]

    # Truncate / zero-pad to exactly num_frames.
    if mag.shape[0] >= num_frames:
        mag = mag[:num_frames, :]
    else:
        pad = np.zeros((num_frames - mag.shape[0], n_bins), np.float32)
        mag = np.concatenate([mag, pad.astype(mag.dtype)], axis=0)

    # Log-magnitude with a floor to avoid log(0). Per-frame max-normalize
    # so the frame max is 0 dB and quieter bins are negative, bounding the
    # range independent of loudness (same convention as the chroma path).
    mag = np.maximum(mag, 1e-7).astype(np.float32)
    log_mag = np.log10(mag)
    row_max = log_mag.max(axis=1, keepdims=True)
    log_mag = log_mag - row_max  # range [-inf, 0]; floor it

    # Floor at -8 (160 dB dynamic range, well below perceptual relevance)
    # so the float16 representation is well-bounded.
    log_mag = np.maximum(log_mag, -8.0)

    return log_mag.astype(np.float32)
