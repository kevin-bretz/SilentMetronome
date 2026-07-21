"""Repair the half-silent 20 s windows of the `_beat` precompute test split.

Root cause: get_audio_dataloader hardcoded SelectStems(duration=10) while
PadTokens padded audio to the requested duration (20 s), so every window's
audio (and hence its DAC codes) is 10 s real + 10 s digital silence. This is
upstream-original SMG code; nothing before the LiveBand 20 s protocol ever
consumed frames past ~11 s.

This script rebuilds input/target audio + codes for the first N test windows
from the full-track FLACs referenced in each window's metadata, reproducing
the pipeline semantics exactly (crop at source rate from start_time_sec,
mean-mix input stems, resample to 32 kHz mono, zero-pad to 20 s, DAC
compress), and writes them into a NEW root:

    <fix_root>/slakh2100/test/<window>/   (repaired copies)
    <fix_root>/slakh2100/{train,valid}    (symlinks to the original)

The original precompute is never modified. Window identity (track, offset,
stem split) is preserved, so the first 10 s remain the same musical material
used by all existing 10 s evaluations.
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from audiotools import AudioSignal
from dac import DAC
from tqdm import tqdm

from stream_music_gen.constants import (
    DAC_SAMPLE_RATE,
    DAC_PRETRAINED_MODEL_PATH,
)

import json

FRAME_RATE_HZ = 50
REPAIRED_FILES = (
    "input_codes.pt",
    "target_codes.pt",
    "input_audio.pt",
    "target_audio.pt",
)


def load_crop(path, start_sec, duration_sec):
    audio, sr = torchaudio.load(path)  # [C, T] at source rate
    start_sample = int(start_sec * sr)
    length_sample = int(duration_sec * sr)
    return audio[:, start_sample : start_sample + length_sample], sr


def post_process(audio, sr):
    if sr != DAC_SAMPLE_RATE:
        audio = T.Resample(orig_freq=sr, new_freq=DAC_SAMPLE_RATE)(audio)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0)
    else:
        audio = audio.squeeze(0)
    return audio


def build_window_audio(meta, duration_sec):
    start_sec = float(meta["start_frame"]) / FRAME_RATE_HZ

    input_crops = []
    sr = None
    for p in meta["input_audio_path"]:
        a, sr = load_crop(p, start_sec, duration_sec)
        input_crops.append(a)
    target_crop, sr_t = load_crop(
        meta["target_audio_path"], start_sec, duration_sec
    )
    assert sr_t == sr

    # match SelectStems: trim all stems + target to the common min length
    min_len = min(
        [a.size(-1) for a in input_crops] + [target_crop.size(-1)]
    )
    input_crops = [a[:, :min_len] for a in input_crops]
    target_crop = target_crop[:, :min_len]

    input_mix = torch.stack(input_crops, dim=0).mean(0)  # [C, T]
    input_mix = post_process(input_mix, sr)  # [T] @ 32 kHz
    target = post_process(target_crop, sr)

    # match PadTokens: zero-pad to the full duration
    full = int(duration_sec * DAC_SAMPLE_RATE)
    if input_mix.size(-1) < full:
        input_mix = torch.nn.functional.pad(
            input_mix, (0, full - input_mix.size(-1))
        )
    if target.size(-1) < full:
        target = torch.nn.functional.pad(target, (0, full - target.size(-1)))
    return input_mix, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src_root",
        default="stream_music_gen_data/precompute_audio_mixdown_20s_beat",
    )
    parser.add_argument(
        "--fix_root",
        default="stream_music_gen_data/precompute_audio_mixdown_20s_beat_fix",
    )
    parser.add_argument("--dataset", default="slakh2100")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_windows", type=int, default=1100)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    src_split = Path(args.src_root) / args.dataset / args.split
    fix_split = Path(args.fix_root) / args.dataset / args.split
    os.makedirs(fix_split, exist_ok=True)

    # train/valid stay as symlinks to the untouched original
    for other in ("train", "valid"):
        link = Path(args.fix_root) / args.dataset / other
        target = (Path(args.src_root) / args.dataset / other).resolve()
        if not link.exists() and target.exists():
            os.symlink(target, link)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAC.load(DAC_PRETRAINED_MODEL_PATH).to(device)
    assert model.causal_decoder and model.causal_encoder
    model.eval()

    n_short = 0
    for idx in tqdm(range(args.num_windows), desc="repairing windows"):
        src_dir = src_split / f"{idx:07d}"
        dst_dir = fix_split / f"{idx:07d}"
        if not src_dir.is_dir():
            print(f"[warn] missing source window {src_dir}, stopping")
            break
        os.makedirs(dst_dir, exist_ok=True)

        with open(src_dir / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

        input_mix, target = build_window_audio(meta, args.duration)

        # a genuinely short window (track end) keeps its trailing silence
        second_half = input_mix[input_mix.size(-1) // 2 :]
        if float(second_half.abs().mean()) < 1e-6:
            n_short += 1

        for name, wav in (("input", input_mix), ("target", target)):
            signal = AudioSignal(wav, sample_rate=DAC_SAMPLE_RATE).to(device)
            win_duration = signal.shape[-1] / DAC_SAMPLE_RATE + 1
            with torch.no_grad():
                codes = (
                    model.compress(signal, win_duration=win_duration)
                    .codes.squeeze()
                    .cpu()
                )
            torch.save(codes, dst_dir / f"{name}_codes.pt")
            torch.save(signal.cpu(), dst_dir / f"{name}_audio.pt")

        # everything else (metadata, beat_cond, aux features) is copied as-is
        for f_src in src_dir.iterdir():
            if f_src.name in REPAIRED_FILES:
                continue
            f_dst = dst_dir / f_src.name
            if not f_dst.exists():
                shutil.copy2(f_src, f_dst)

    print(f"[repair] done; windows with silent second half (track end): "
          f"{n_short}")

    # verification: probe a few repaired windows for non-silent second halves
    import random

    random.seed(0)
    probe = random.sample(range(min(args.num_windows, 1000)), 12)
    n_bad = 0
    for idx in probe:
        sig = torch.load(
            fix_split / f"{idx:07d}" / "input_audio.pt", map_location="cpu"
        )
        wav = sig.audio_data.flatten()
        tail = wav[int(12 * DAC_SAMPLE_RATE) : int(18 * DAC_SAMPLE_RATE)]
        if float(tail.abs().mean()) < 1e-6:
            n_bad += 1
            print(f"[verify] window {idx:07d}: second half still silent")
    print(f"[verify] {len(probe) - n_bad}/{len(probe)} probed windows have "
          f"non-silent second halves")
    if n_bad > len(probe) // 2:
        raise SystemExit("[verify] FAILED: repaired windows look silent")
    print("[verify] PASSED")


if __name__ == "__main__":
    main()
