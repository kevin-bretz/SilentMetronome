"""Utilities for loading per-track MIDI beat grids and computing per-frame
beat/bar phase signals for a window.

Beat-grid JSON schema (from `extract_beat_grid.py` on branch `beat_aligned`):
    beat_frames:         list[int]   frame indices at `frame_rate_hz`, length N+1
    beat_lengths:        list[int]   length N = diff of consecutive beat_frames
    tempo_bpm:           list[float] length N+1 (one per beat boundary)
    bar_positions:       list[int]   length N+1 (0 = downbeat, wraps by numerator)
    time_sig_numerators: list[int]   length N+1
    time_signature:      [num, den]  global (fallback; beats may differ if song changes)
    ticks_per_beat:      int         MIDI tick resolution (not used here)

All grids on disk use frame_rate_hz = 50.
"""

import functools
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

FRAME_RATE_HZ = 50
_TRACK_RE = re.compile(r"Track\d+")

_BEAT_GRID_BASE_DIR_DEFAULT = Path("stream_music_gen_data/beat_grids")


def derive_track_name(path_str: str) -> Optional[str]:
    """Extract 'Track00305' from any path that contains it. None if not found."""
    m = _TRACK_RE.search(path_str)
    return m.group(0) if m else None


@functools.lru_cache(maxsize=8192)
def load_beat_grid(
    dataset: str,
    split: str,
    track_name: str,
    base_dir: Optional[str] = None,
) -> Optional[dict]:
    """Load a beat grid JSON. Returns None if missing or invalid.

    Cached per-process: typical usage hits the same track many times (multiple
    windows per track in a batch).
    """
    root = Path(base_dir) if base_dir is not None else _BEAT_GRID_BASE_DIR_DEFAULT
    path = root / dataset / split / f"{track_name}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            grid = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    required = {
        "beat_frames",
        "beat_lengths",
        "tempo_bpm",
        "bar_positions",
        "time_sig_numerators",
    }
    if not required.issubset(grid.keys()):
        return None

    grid["beat_frames"] = np.asarray(grid["beat_frames"], dtype=np.int64)
    grid["beat_lengths"] = np.asarray(grid["beat_lengths"], dtype=np.int64)
    grid["tempo_bpm"] = np.asarray(grid["tempo_bpm"], dtype=np.float32)
    grid["bar_positions"] = np.asarray(grid["bar_positions"], dtype=np.int64)
    grid["time_sig_numerators"] = np.asarray(
        grid["time_sig_numerators"], dtype=np.int64
    )
    if len(grid["beat_frames"]) < 2 or len(grid["beat_lengths"]) < 1:
        return None
    return grid


def compute_beat_phase_frames(
    grid: dict,
    start_frame: int,
    num_frames: int = 500,
    time_sig_clamp: tuple = (2, 16),
) -> dict:
    """Compute per-frame beat/bar phase and window-level summary stats.

    Args:
        grid: Dict returned by `load_beat_grid`.
        start_frame: Window start in frames at FRAME_RATE_HZ (within the track).
        num_frames: Number of frames in the window (e.g., 500 for 10s @ 50Hz).
        time_sig_clamp: (min, max) inclusive for time-signature numerator vocab.

    Returns:
        Dict with keys:
            beat_cond:                np.float32 [num_frames, 4]
                columns = [sin(2pi*phi_beat), cos(2pi*phi_beat),
                           sin(2pi*phi_bar),  cos(2pi*phi_bar)]
            bpm_mean:                 float      mean over beats in window
            bpm_log:                  float      log(bpm_mean / 120)
            time_sig_num:             int        numerator at window start (clamped)
            time_sig_den:             int        global numerator/denominator[1]
            num_beats_in_window:      int
            num_downbeats_in_window:  int
            first_beat_frame_in_window: int       (relative to window start; -1 if none)
            time_sig_change_in_window:  bool
            tempo_change_in_window:     bool
            beat_frames_in_window:    list[int]  relative to window start
            bar_positions_in_window:  list[int]  (per beat in window)
    """
    beat_frames = grid["beat_frames"]        # [N+1]
    beat_lengths = grid["beat_lengths"]      # [N]
    tempo_bpm = grid["tempo_bpm"]            # [N+1]
    bar_positions = grid["bar_positions"]    # [N+1]
    time_sig_nums = grid["time_sig_numerators"]  # [N+1]

    N = len(beat_lengths)
    end_frame = start_frame + num_frames

    frame_idxs = np.arange(start_frame, end_frame, dtype=np.int64)

    # For each frame, find beat index k such that beat_frames[k] <= frame < beat_frames[k+1]
    # np.searchsorted with side='right' gives insertion index; subtract 1 for beat_idx.
    beat_idx = np.searchsorted(beat_frames, frame_idxs, side="right") - 1
    # Clamp: frames before first beat -> 0 (extrapolate backward); after last -> N-1 (extrapolate forward)
    beat_idx_clamped = np.clip(beat_idx, 0, N - 1)

    # Per-frame: use beat start + length from the clamped beat index
    beat_start = beat_frames[beat_idx_clamped]        # [T]
    beat_len = beat_lengths[beat_idx_clamped].astype(np.float32)
    beat_len = np.maximum(beat_len, 1.0)              # guard against zero

    # Handle extrapolation explicitly:
    #   - before first beat (beat_idx < 0): use (frame - 0) / beat_lengths[0] mod 1
    #     but we also want phase < 0 to represent anticipation. Simpler: treat as if
    #     frame_idx projects back: phase = (frame - beat_frames[0]) / beat_lengths[0] (negative)
    before_mask = beat_idx < 0
    after_mask = beat_idx >= N

    # Phase in [0, 1) via modular arithmetic (before_mask -> negative offset; mod handles wrap)
    raw_phase = (frame_idxs - beat_start).astype(np.float32) / beat_len
    # For after_mask, extrapolate past beat_frames[-1] using beat_lengths[-1]
    if after_mask.any():
        last_beat_start = beat_frames[-1]
        last_beat_len = float(max(beat_lengths[-1], 1))
        raw_phase[after_mask] = (
            (frame_idxs[after_mask] - last_beat_start).astype(np.float32)
            / last_beat_len
        )
    # Wrap into [0, 1)
    phi_beat = np.mod(raw_phase, 1.0).astype(np.float32)

    # Bar position per beat (for per-frame bar-phase)
    bar_pos_per_frame = bar_positions[beat_idx_clamped].astype(np.float32)  # [T]
    ts_num_per_frame = time_sig_nums[beat_idx_clamped].astype(np.float32)   # [T]
    ts_num_per_frame = np.maximum(ts_num_per_frame, 1.0)
    # For extrapolated (before/after) frames, bar_pos may be stale; acceptable.
    phi_bar = np.mod((bar_pos_per_frame + phi_beat) / ts_num_per_frame, 1.0).astype(
        np.float32
    )

    two_pi = np.float32(2.0 * np.pi)
    beat_cond = np.stack(
        [
            np.sin(two_pi * phi_beat),
            np.cos(two_pi * phi_beat),
            np.sin(two_pi * phi_bar),
            np.cos(two_pi * phi_bar),
        ],
        axis=-1,
    ).astype(np.float32)  # [T, 4]

    # ---- Window-level summary stats ----
    # Beats whose start frame falls within [start_frame, end_frame)
    in_window = (beat_frames[:-1] >= start_frame) & (beat_frames[:-1] < end_frame)
    beats_in_window = np.flatnonzero(in_window)

    if beats_in_window.size > 0:
        bpm_slice = tempo_bpm[beats_in_window]
        bpm_mean = float(np.mean(bpm_slice))
        num_beats_in_window = int(beats_in_window.size)
        downbeats_in_window = int(
            np.sum(bar_positions[beats_in_window] == 0)
        )
        first_beat_frame_in_window = int(
            beat_frames[beats_in_window[0]] - start_frame
        )
        beat_frames_in_window = (
            beat_frames[beats_in_window] - start_frame
        ).tolist()
        bar_positions_in_window = bar_positions[beats_in_window].tolist()
        tempo_change_in_window = bool(bpm_slice.max() - bpm_slice.min() > 0.5)
        ts_nums_slice = time_sig_nums[beats_in_window]
        time_sig_change_in_window = bool(
            ts_nums_slice.max() != ts_nums_slice.min()
        )
    else:
        # Window falls between beats (very short track?), fall back to global.
        bpm_mean = float(np.mean(tempo_bpm))
        num_beats_in_window = 0
        downbeats_in_window = 0
        first_beat_frame_in_window = -1
        beat_frames_in_window = []
        bar_positions_in_window = []
        tempo_change_in_window = False
        time_sig_change_in_window = False

    # Time sig numerator at window start (clamp to vocab)
    start_beat_idx = int(np.clip(beat_idx[0] if beat_idx.size else 0, 0, N - 1))
    ts_num_start = int(time_sig_nums[start_beat_idx])
    ts_num_start = int(np.clip(ts_num_start, time_sig_clamp[0], time_sig_clamp[1]))

    ts_den = int(grid.get("time_signature", [4, 4])[1])

    bpm_log = float(np.log(max(bpm_mean, 1e-3) / 120.0))

    return {
        "beat_cond": beat_cond,
        "bpm_mean": bpm_mean,
        "bpm_log": bpm_log,
        "time_sig_num": ts_num_start,
        "time_sig_den": ts_den,
        "num_beats_in_window": num_beats_in_window,
        "num_downbeats_in_window": downbeats_in_window,
        "first_beat_frame_in_window": first_beat_frame_in_window,
        "time_sig_change_in_window": time_sig_change_in_window,
        "tempo_change_in_window": tempo_change_in_window,
        "beat_frames_in_window": beat_frames_in_window,
        "bar_positions_in_window": bar_positions_in_window,
    }
