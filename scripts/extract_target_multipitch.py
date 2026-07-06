"""Extract per-window TARGET-stem multipitch + velocity from MIDI for the
existing precompute_audio_mixdown_20s_beat dataset.

Mirrors ``extract_target_chroma.py`` in structure. For each window dir:

1. Reads ``metadata.json`` for ``target_audio_path`` and ``start_frame``.
2. Maps audio path → sibling MIDI path.
3. Computes 128-class binary piano roll + velocity per frame at 50 Hz.
4. Writes:
     - ``target_multipitch.pt`` (uint8 [T, 128] in {0, 1})
     - ``target_velocity.pt`` (uint8 [T, 128] in 0-127)
     - ``target_has_multipitch.pt`` (scalar bool)

Drums are NOT filtered. Drum stems use GM channel 10 where note number
encodes the drum sound; the 128-dim representation captures this. The
model uses ``dec_inst_tokens`` to disambiguate "note number 36 = C2"
(piano) vs "note number 36 = kick" (drums).

Idempotent: skips windows that already have all three output files unless
``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stream_music_gen.dataset.multipitch_utils import (
    compute_multipitch_from_midi,
    target_audio_path_to_midi_path,
    NUM_PITCHES,
)


def _process_window(window_dir: Path, project_dir: Path, overwrite: bool) -> str:
    mp_path = window_dir / "target_multipitch.pt"
    vel_path = window_dir / "target_velocity.pt"
    flag_path = window_dir / "target_has_multipitch.pt"
    if (
        mp_path.exists()
        and vel_path.exists()
        and flag_path.exists()
        and not overwrite
    ):
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
    start_frame = int(meta.get("start_frame", 0))
    duration_frames = int(meta.get("duration_frames", 1000))

    midi_path = target_audio_path_to_midi_path(target_audio_path)
    if not os.path.isabs(midi_path):
        midi_path = str(project_dir / midi_path)

    if not os.path.exists(midi_path):
        multipitch = np.zeros((duration_frames, NUM_PITCHES), dtype=np.uint8)
        velocity = np.zeros((duration_frames, NUM_PITCHES), dtype=np.uint8)
        has = False
    else:
        multipitch, velocity, has = compute_multipitch_from_midi(
            midi_path, start_frame, num_frames=duration_frames
        )

    torch.save(torch.from_numpy(multipitch), mp_path)
    torch.save(torch.from_numpy(velocity), vel_path)
    torch.save(torch.tensor(bool(has)), flag_path)

    return "ok" if has else "missing_midi"


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

    counts = {"ok": 0, "skipped": 0, "missing_midi": 0, "error": 0}
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
            f"-> {len(windows)} windows"
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
