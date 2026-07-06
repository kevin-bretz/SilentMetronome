"""Extract per-window target-stem chroma from MIDI.

For each window dir, maps the target audio path to its sibling MIDI file,
flags drum stems (no chroma) via the track's ``metadata.yaml``, computes a
12-class active-pitch-class vector per frame at 50 Hz and writes
``target_chroma.pt`` ([T, 12] float32) plus ``has_chroma.pt`` (scalar
bool). Skips windows that already have the output unless ``--overwrite``
is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Allow running from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stream_music_gen.dataset.chroma_utils import (
    compute_chroma_from_midi,
    is_drum_stem,
    target_audio_path_to_midi_path,
    NUM_PITCH_CLASSES,
)

_STEM_RE = re.compile(r"(S\d+)\.flac$")


def _stem_id_from_audio_path(audio_path: str) -> str:
    m = _STEM_RE.search(audio_path)
    return m.group(1) if m else ""


def _process_window(window_dir: Path, project_dir: Path, overwrite: bool) -> str:
    """Returns one of: 'ok', 'skipped', 'no_chroma', 'missing_midi', 'error'."""
    out_path = window_dir / "target_chroma.pt"
    flag_path = window_dir / "target_has_chroma.pt"
    if out_path.exists() and flag_path.exists() and not overwrite:
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
    track_dir = "/".join(target_audio_path.split("/")[:-2])
    if not os.path.isabs(track_dir):
        track_dir = str(project_dir / track_dir)
    stem_id = _stem_id_from_audio_path(target_audio_path)

    is_drum = is_drum_stem(track_dir, stem_id)
    if is_drum:
        chroma = np.zeros((duration_frames, NUM_PITCH_CLASSES), dtype=np.float32)
        has = False
    else:
        if not os.path.exists(midi_path):
            chroma = np.zeros((duration_frames, NUM_PITCH_CLASSES), dtype=np.float32)
            has = False
        else:
            chroma, has = compute_chroma_from_midi(
                midi_path, start_frame, num_frames=duration_frames
            )

    torch.save(torch.from_numpy(chroma), out_path)
    torch.save(torch.tensor(bool(has)), flag_path)

    if is_drum:
        return "no_chroma"
    if not has:
        return "missing_midi"
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
    parser.add_argument("--max_examples", type=int, default=-1, help="-1 = all")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_dir / data_root

    counts = {"ok": 0, "skipped": 0, "no_chroma": 0, "missing_midi": 0, "error": 0}
    t0 = time.time()
    total_processed = 0

    for split in args.splits:
        split_dir = data_root / args.dataset / split
        if not split_dir.exists():
            print(f"[skip] split missing: {split_dir}")
            continue
        windows = sorted(p for p in split_dir.iterdir() if p.is_dir())
        # Shard.
        windows = windows[args.shard_idx :: args.num_shards]
        if args.max_examples > 0:
            windows = windows[: args.max_examples]
        print(
            f"[{split}] shard {args.shard_idx}/{args.num_shards}, "
            f"-> {len(windows)} windows"
        )
        for w in tqdm(windows, desc=f"{split}", mininterval=5.0):
            result = _process_window(w, project_dir, args.overwrite)
            counts[result] += 1
            total_processed += 1

    elapsed = time.time() - t0
    print(f"[done] {total_processed} windows in {elapsed:.0f}s")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
