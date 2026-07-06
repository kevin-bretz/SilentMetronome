"""Extract per-window target-stem CQT from audio.

Companion to ``extract_target_chroma.py`` and ``extract_input_chroma.py``.
For each window dir, reads the window slice of the target FLAC, computes a
log-magnitude CQT (84 bins, 7 octaves from C1) and writes ``target_cqt.pt``
(float16 [T, 84]). The C1 to C8 range stays within what 4-layer DAC
reconstructs faithfully, so the aux loss only asks for content the model
can actually generate. Idempotent.
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
    """Read a mono slice from a flac. Pads with zeros if file is shorter."""
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
    out_path = window_dir / "target_cqt.pt"
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

    target_audio_path = meta.get("target_audio_path", "")
    if not target_audio_path:
        return "missing_audio"
    start_frame = int(meta.get("start_frame", 0))
    duration_frames = int(meta.get("duration_frames", 1000))
    frame_rate_hz = int(meta.get("frame_rate_hz", FRAME_RATE_HZ))

    if not os.path.isabs(target_audio_path):
        target_audio_path = str(project_dir / target_audio_path)

    try:
        sr_probe = int(sf.info(target_audio_path).samplerate)
    except (OSError, RuntimeError):
        return "missing_audio"

    samples_per_frame = sr_probe // frame_rate_hz
    start_sample = start_frame * samples_per_frame
    num_samples = duration_frames * samples_per_frame

    audio, sr = _read_slice(target_audio_path, start_sample, num_samples)
    if sr != sr_probe or sr == 0:
        return "missing_audio"

    cqt = compute_cqt_from_audio(
        audio,
        sample_rate=sr_probe,
        num_frames=duration_frames,
        frame_rate_hz=frame_rate_hz,
    )
    # float16 halves storage, training casts back as needed.
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
        if args.max_examples > 0:
            windows = windows[: args.max_examples]
        print(
            f"[{split}] "
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
