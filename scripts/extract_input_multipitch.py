"""Extract per-window INPUT-mix multipitch + velocity from MIDI for use as
a TRAINING-TIME conditioning input.

Companion to ``extract_target_multipitch.py``. The input mix is the union
of all input stems; this script merges each stem's MIDI into a single
polyphonic piano roll for the window:

  multipitch[t, p] = 1 iff any input stem has that note active
  velocity[t, p]   = max velocity over input stems

Outputs in each window dir:
  - ``input_multipitch.pt`` (uint8 [T, 128] in {0, 1})
  - ``input_velocity.pt`` (uint8 [T, 128] in 0-127)
  - ``input_has_multipitch.pt`` (scalar bool)

Intended use: feed these tensors to the model as conditioning input
during training, providing exact symbolic context for what the input
mix is playing. This is a slakh-only signal (real audio at inference
doesn't have ground-truth MIDI); training configs that use this cond
must accept they are slakh-specialized unless paired with audio-domain
cond fallback.

For inference-time-recoverable cond input, ``input_chroma`` and
``input_cqt`` (extracted from audio) are the parallel options.

Idempotent. SLURM array via shard_idx/num_shards.
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
    compute_multipitch_from_midis,
    target_audio_path_to_midi_path,
    NUM_PITCHES,
)


def _process_window(window_dir: Path, project_dir: Path, overwrite: bool) -> str:
    mp_path = window_dir / "input_multipitch.pt"
    vel_path = window_dir / "input_velocity.pt"
    flag_path = window_dir / "input_has_multipitch.pt"
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

    input_audio_paths = meta.get("input_audio_path", [])
    if not input_audio_paths:
        return "missing_audio"

    start_frame = int(meta.get("start_frame", 0))
    duration_frames = int(meta.get("duration_frames", 1000))

    midi_paths = []
    for ap in input_audio_paths:
        mp = target_audio_path_to_midi_path(ap)
        if not os.path.isabs(mp):
            mp = str(project_dir / mp)
        midi_paths.append(mp)

    multipitch, velocity, has_any = compute_multipitch_from_midis(
        midi_paths, start_frame, num_frames=duration_frames
    )

    torch.save(torch.from_numpy(multipitch), mp_path)
    torch.save(torch.from_numpy(velocity), vel_path)
    torch.save(torch.tensor(bool(has_any)), flag_path)

    return "ok" if has_any else "missing_midi"


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

    counts = {"ok": 0, "skipped": 0, "missing_audio": 0, "missing_midi": 0, "error": 0}
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
            f"[{split}] shard={args.shard_idx}/{args.num_shards} "
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
