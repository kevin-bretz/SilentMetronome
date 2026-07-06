"""Extract per-window input-mix CQT from audio.

Companion to ``extract_input_chroma.py``. For each window dir, reads the
window slice of each input stem FLAC, sums to the input mix, computes a
log-magnitude CQT (84 bins, 7 octaves from C1) and writes ``input_cqt.pt``
(float16 [T, 84]). Unlike ``input_multipitch`` this is computable from any
audio, so it can serve as a real conditioning input at inference time.
Idempotent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stream_music_gen.dataset.cqt_utils import (
    compute_cqt_from_audio,
    FRAME_RATE_HZ,
    NUM_CQT_BINS,
)


def _read_slice(
    audio_path: str, start_sample: int, num_samples: int
) -> tuple[np.ndarray, int]:
    try:
        info = sf.info(audio_path)
    except (OSError, RuntimeError):
        return np.zeros(num_samples, dtype=np.float32), 0
    sr = info.samplerate
    total = info.frames
    if start_sample >= total:
        return np.zeros(num_samples, dtype=np.float32), sr
    avail = min(num_samples, total - start_sample)
    try:
        audio, _ = sf.read(
            audio_path, start=start_sample, frames=avail, always_2d=False
        )
    except (OSError, RuntimeError):
        return np.zeros(num_samples, dtype=np.float32), sr
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if avail < num_samples:
        audio = np.concatenate(
            [audio, np.zeros(num_samples - avail, dtype=np.float32)]
        )
    return audio, sr


def _process_window(
    window_dir: Path, project_dir: Path, overwrite: bool
) -> str:
    out_path = window_dir / "input_cqt.pt"
    if out_path.exists() and not overwrite:
        return "skipped"

    meta_path = window_dir / "metadata.json"
    if not meta_path.exists():
        return "error"
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "error"

    input_audio_paths = meta.get("input_audio_path", [])
    if not input_audio_paths:
        return "missing_audio"

    start_frame = int(meta.get("start_frame", 0))
    duration_frames = int(meta.get("duration_frames", 1000))
    frame_rate_hz = int(meta.get("frame_rate_hz", FRAME_RATE_HZ))

    abs_paths = [
        p if os.path.isabs(p) else str(project_dir / p)
        for p in input_audio_paths
    ]

    sr_probe = 0
    for p in abs_paths:
        try:
            sr_probe = int(sf.info(p).samplerate)
            break
        except (OSError, RuntimeError):
            continue
    if sr_probe <= 0:
        return "missing_audio"

    samples_per_frame = sr_probe // frame_rate_hz
    start_sample = start_frame * samples_per_frame
    num_samples = duration_frames * samples_per_frame

    mix = np.zeros(num_samples, dtype=np.float32)
    n_loaded = 0
    for p in abs_paths:
        slc, sr = _read_slice(p, start_sample, num_samples)
        if sr != sr_probe or sr == 0:
            continue
        mix += slc
        n_loaded += 1
    if n_loaded == 0:
        return "missing_audio"

    cqt = compute_cqt_from_audio(
        mix,
        sample_rate=sr_probe,
        num_frames=duration_frames,
        frame_rate_hz=frame_rate_hz,
    )
    cqt_f16 = cqt.astype(np.float16)
    torch.save(torch.from_numpy(cqt_f16), out_path)
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="stream_music_gen_data/precompute_audio_mixdown_20s_beat",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="slakh2100",
        choices=["slakh2100", "moisesdb", "musdb", "cocochorales"],
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"]
    )
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_examples", type=int, default=-1)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_dir / data_root

    counts = {"ok": 0, "skipped": 0, "missing_audio": 0, "error": 0}
    t0 = time.time()
    total_processed = 0

    for split in args.splits:
        split_dir = data_root / args.dataset / split
        if not split_dir.exists():
            print(f"[skip] split missing: {split_dir}")
            continue
        windows = sorted(p for p in split_dir.iterdir() if p.is_dir())
        windows = windows[args.shard_idx :: args.num_shards]
        if args.max_examples > 0:
            windows = windows[: args.max_examples]
        print(
            f"[{split}] shard {args.shard_idx}/{args.num_shards}, "
            f"-> {len(windows)} windows ({NUM_CQT_BINS}-bin CQT)"
        )
        for w in tqdm(windows, desc=f"{split}", mininterval=10.0):
            result = _process_window(w, project_dir, args.overwrite)
            counts[result] += 1
            total_processed += 1

    elapsed = time.time() - t0
    print(f"[done] {total_processed} windows in {elapsed:.0f}s")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
