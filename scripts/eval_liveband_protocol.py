"""LiveBand-protocol evaluation (arXiv:2606.03803) for online prefix decoders.

Protocol, as described in the LiveBand paper (Pasini et al., 2026), Sec. 4:
generate a 20 s accompaniment per Slakh2100 test window and evaluate each
output on two non-overlapping 10 s segments, yielding a short-horizon
quality estimate (0-10 s) and a drift measure
    delta_s = s(10-20 s) - s(0-10 s)
per metric. Metrics follow their Table 1: FAD in the VGGish and LAION-CLAP
embedding spaces, beat-alignment F1 via BeatThis + madmom, and COCOLA in
full / harmonic / percussive modes. This script reuses THIS repo's metric
helpers verbatim (they are the same BeatThis+madmom / COCOLA / VGGish-FAD
stack), so segment scores are protocol-identical to the headline eval, just
computed on 10 s slices of 20 s generations.

Underspecified in the paper and resolved here (documented in the output
JSON): (1) the FAD background set — LiveBand's ground-truth row is nonzero,
so theirs is not the paired per-segment ground truth; default here is
"paired" (background = ground-truth stems of the SAME segment half,
matching this repo's headline FAD), "--fad_background pooled" uses the
ground-truth stems of BOTH halves as one fixed background, which keeps the
reference identical across the two drift terms. (2) Loudness: each 20 s
file is loudness-normalized once (as in the generation pipeline) and then
sliced, so within-sample level drift is preserved.

Generation beyond the 10 s training window uses
OnlinePrefixDecoderTransformerMultiOut.generate_sliding (the learned
absolute positions cap the context at ~10 s; the sliding window re-bases
the context past that, see its docstring).
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from lightning import seed_everything

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import stream_music_gen.eval.cocola_eval as cocola_eval
import stream_music_gen.eval.eval_utils as eval_utils
import stream_music_gen.eval.fad_eval as fad_eval
from stream_music_gen.constants import DAC_SAMPLE_RATE, EVAL_SAMPLE_RATE
from stream_music_gen.dataset.token_dataset import (
    get_precomputed_token_dataloader,
)
from stream_music_gen.eval.beat_alignment_eval import beat_alignment_score
from stream_music_gen.lit_module.online_prefix_dec import (
    LitOnlinePrefixDecoderMultiOut,
)
from stream_music_gen.utils.audio_utils import (
    loudness_normalize_audio,
    mix_with_generated_stem,
)
from stream_music_gen.utils.inference_utils import load_lit_model

F_MEASURE_KEY = "madmom_fmeasure"
COCOLA_MODES = ["both", "harmonic", "percussive"]
SEGMENT_WAVS = ["input_audio.wav", "ground_truth/pred.wav", "pred/pred.wav"]


def mean_std(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "n": int(len(values)),
    }


def pad_future_frames(kw, fv, n_frames):
    """Right-pad per-frame streams by ``fv`` frames via edge-repeat.

    The ``_beat`` precompute stores exactly 20 s (1000 frames) per window,
    so an fv>0 checkpoint generating the full 20 s has no stored future
    input for its final chunk. Edge-repeating the last frame (the
    convention of ``pad_beat_cond_for_delay``) keeps padded frames
    on-manifold; only the final ``fv`` frames (1 s) of conditioning are
    synthetic.
    """
    out = {}
    for key, t in kw.items():
        if torch.is_tensor(t) and t.dim() >= 2 and t.shape[1] == n_frames:
            pad = t[:, -1:].repeat(1, fv, *([1] * (t.dim() - 2)))
            t = torch.cat([t, pad], dim=1)
        out[key] = t
    return out


def generate_batch_sliding(
    batch,
    model,
    tokenizer,
    gen_frames,
    gen_seconds,
    device,
    window_frames,
    hop_frames,
    temperature,
    top_k,
):
    """20 s counterpart of gen_pred_prefix_dec_online.generate_prediction.

    Identical plumbing, with model.generate_sliding instead of
    model.generate and all crops driven by gen_seconds instead of the
    checkpoint's max_duration.
    """
    input_token_stems = batch["input_token_stems"]
    input_emb = batch["input_emb"].to(device)
    targets = batch["target_token"].to(device)
    dec_inst_tokens = torch.tensor(batch["target_inst_token"], device=device)

    optional = {}
    for key in (
        "beat_cond",
        "bpm_log",
        "time_sig_num",
        "time_sig_den",
        "time_sig_change",
        "tempo_change",
        "local_bpm_log",
    ):
        value = batch.get(key)
        optional[key] = value.to(device) if value is not None else None

    fv = model.future_visibility
    if fv > 0:
        merged = pad_future_frames(
            {"input_emb": input_emb, **optional}, fv, input_emb.shape[1]
        )
        input_emb = merged.pop("input_emb")
        optional = merged

    decoder_preds = model.generate_sliding(
        seq_len=gen_frames,
        window_frames=window_frames,
        hop_frames=hop_frames,
        input_emb=input_emb,
        inst_tokens=dec_inst_tokens,
        beat_cond=optional["beat_cond"],
        bpm_log=optional["bpm_log"],
        time_sig_num=optional["time_sig_num"],
        time_sig_den=optional["time_sig_den"],
        time_sig_change=optional["time_sig_change"],
        tempo_change=optional["tempo_change"],
        local_bpm_log=optional["local_bpm_log"],
        cache_kv=True,
        filter_logits_fn=["top_k_multi_out"],
        filter_kwargs=[{"k": top_k}],
        temperature=temperature,
        display_pbar=False,
    )

    max_samples = gen_seconds * tokenizer.sample_rate
    input_audio_all, target_audio_all, pred_audio_all = [], [], []
    num_input_stems_all = []
    for i in range(targets.size(0)):
        num_input_stems_all.append(len(batch["input_inst_id"][i]))

        target_tokens = tokenizer.post_process_tokens(targets[i])
        target_audio = (
            tokenizer.tokens_to_audio(target_tokens.unsqueeze(0))
            .cpu()
            .squeeze()[:max_samples]
        )
        target_audio_all.append(target_audio)

        input_audio = (
            tokenizer.codec_tokens_to_audio(
                input_token_stems[i][0].unsqueeze(0)
            )
            .cpu()
            .squeeze()[:max_samples]
        )
        input_audio_all.append(input_audio)

        pred_tokens = tokenizer.post_process_tokens(decoder_preds[i])
        pred_audio = (
            tokenizer.tokens_to_audio(pred_tokens.unsqueeze(0))
            .cpu()
            .squeeze()
        )
        if pred_audio.shape[-1] < target_audio.shape[-1]:
            pred_audio = torch.nn.functional.pad(
                pred_audio,
                (0, target_audio.shape[-1] - pred_audio.shape[-1]),
            )
        pred_audio_all.append(pred_audio[:max_samples])

    return (
        input_audio_all,
        target_audio_all,
        pred_audio_all,
        num_input_stems_all,
    )


def run_generation(args, root_folder):
    seed_everything(args.seed)

    model, tokenizer, _ = load_lit_model(
        args.model_path,
        lit_module_cls=LitOnlinePrefixDecoderMultiOut,
        batch_size=args.batch_size,
        override_args={
            "data_base_dir": args.data_base_dir,
            "rms_base_dir": args.rms_base_dir,
            "audio_base_dir": args.audio_base_dir,
        }
        if args.data_base_dir
        else {
            "rms_base_dir": args.rms_base_dir,
            "audio_base_dir": args.audio_base_dir,
        },
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    tokenizer.to(device)

    frame_rate = tokenizer.frame_rate
    gen_frames = args.gen_seconds * frame_rate
    # The precompute stores exactly gen_seconds (20 s / 1000 frames); an
    # fv>0 ckpt's future input past the window end is edge-padded in
    # generate_batch_sliding instead of loaded (loading gen_seconds + fv
    # hard-fails the dataset).
    duration = args.gen_seconds

    dataloader = get_precomputed_token_dataloader(
        batch_size=args.batch_size,
        dataset_names=model.config["dataset_names"],
        data_base_dir=model.config["data_base_dir"],
        rms_base_dir=model.config["rms_base_dir"],
        weights=None,
        duration=duration,
        num_rvq_layers=tokenizer.num_rvq_layers,
        split=args.split,
        num_workers=8,
        audio_base_dir=args.audio_base_dir,
        load_audio="true",
        pattern="multilayer",
        shuffle=False,
    )

    sample_rate = tokenizer.sample_rate
    max_samples = args.gen_seconds * sample_rate
    file_id = 0
    from tqdm import tqdm

    for batch in tqdm(dataloader, desc="LiveBand-protocol generation"):
        (
            input_audio_all,
            target_audio_all,
            pred_audio_all,
            num_input_stems_all,
        ) = generate_batch_sliding(
            batch,
            model,
            tokenizer,
            gen_frames,
            args.gen_seconds,
            device,
            args.window_frames,
            args.hop_frames,
            args.temperature,
            args.top_k,
        )

        for idx in range(len(input_audio_all)):
            input_audio = batch["input_audio"][idx].cpu().numpy()[:max_samples]
            target_audio = (
                batch["target_audio"][idx].cpu().numpy()[:max_samples]
            )
            pred_audio = pred_audio_all[idx].cpu().numpy()

            input_audio = loudness_normalize_audio(input_audio, sample_rate)
            target_audio = loudness_normalize_audio(target_audio, sample_rate)
            pred_audio = loudness_normalize_audio(pred_audio, sample_rate)

            metadata = {
                "input_inst_id": batch["input_inst_id"][idx],
                "target_inst_id": batch["target_inst_id"][idx],
                "input_token_path": batch["input_token_path"][idx],
                "target_token_path": batch["target_token_path"][idx],
                "input_audio_path": batch["input_audio_path"][idx],
                "target_audio_path": batch["target_audio_path"][idx],
                "num_input_stems": num_input_stems_all[idx],
            }

            sample_dir = root_folder / f"{file_id:05d}"
            os.makedirs(sample_dir / "ground_truth", exist_ok=True)
            os.makedirs(sample_dir / "pred", exist_ok=True)
            sf.write(
                sample_dir / "input_audio.wav", input_audio, sample_rate
            )
            sf.write(
                sample_dir / "ground_truth" / "pred.wav",
                target_audio,
                sample_rate,
            )
            sf.write(sample_dir / "pred" / "pred.wav", pred_audio, sample_rate)

            mix_pred = mix_with_generated_stem(
                input_audio,
                pred_audio,
                num_input_stems_all[idx],
                sample_rate,
            )
            sf.write(sample_dir / "pred" / "mix.wav", mix_pred, sample_rate)

            with open(sample_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            file_id += 1

        if args.num_samples > 0 and file_id >= args.num_samples:
            break

    print(f"[liveband] generated {file_id} samples under {root_folder}")


def build_segment_trees(root_folder, seg_roots, segment_seconds):
    sample_dirs = sorted(
        d
        for d in os.listdir(root_folder)
        if d.isdigit() and os.path.isdir(os.path.join(root_folder, d))
    )
    if not sample_dirs:
        raise SystemExit(f"no per-sample dirs under {root_folder}")

    for d in sample_dirs:
        src = Path(root_folder) / d
        for seg_idx, seg_root in enumerate(seg_roots):
            dst = Path(seg_root) / d
            os.makedirs(dst / "ground_truth", exist_ok=True)
            os.makedirs(dst / "pred", exist_ok=True)
            for rel in SEGMENT_WAVS:
                audio, sr = sf.read(src / rel)
                lo = seg_idx * segment_seconds * sr
                hi = (seg_idx + 1) * segment_seconds * sr
                segment = audio[lo:hi]
                if len(segment) < hi - lo:
                    raise SystemExit(
                        f"{src / rel}: only {len(audio)} samples, cannot cut "
                        f"segment {seg_idx} ({lo}:{hi}); was generation run "
                        f"with gen_seconds >= {(seg_idx + 1) * segment_seconds}?"
                    )
                sf.write(dst / rel, segment, sr)
    print(
        f"[liveband] segment trees built: "
        f"{', '.join(str(s) for s in seg_roots)} ({len(sample_dirs)} samples)"
    )


def evaluate_segment(seg_root, args, seg_roots):
    result = {}

    gt_beat, pred_beat = beat_alignment_score(
        str(seg_root),
        context_path="input_audio.wav",
        gt_path="ground_truth/pred.wav",
        pred_path="pred/pred.wav",
    )
    result["beat_alignment"] = {
        "gt_f_measure": mean_std(gt_beat[F_MEASURE_KEY]),
        "pred_f_measure": mean_std(pred_beat[F_MEASURE_KEY]),
    }

    gt_cocola, pred_cocola = cocola_eval.cocola_score(
        str(seg_root),
        context_path=f"input_audio_{EVAL_SAMPLE_RATE}.wav",
        gt_path=f"ground_truth/pred_{EVAL_SAMPLE_RATE}.wav",
        pred_path=f"pred/pred_{EVAL_SAMPLE_RATE}.wav",
        embedding_modes=COCOLA_MODES,
    )
    result["cocola"] = {
        mode: {
            "gt_scores": mean_std(gt_cocola[mode]),
            "pred_scores": mean_std(pred_cocola[mode]),
        }
        for mode in COCOLA_MODES
    }

    background_root_dirs = (
        [str(s) for s in seg_roots]
        if args.fad_background == "pooled"
        else None
    )
    result["fad"] = {}
    if not args.skip_fad_vggish:
        result["fad"]["vggish"] = {
            "pred": float(
                fad_eval.calculate_fad(
                    str(seg_root),
                    gt_path=f"ground_truth/pred_{EVAL_SAMPLE_RATE}.wav",
                    gen_path=f"pred/pred_{EVAL_SAMPLE_RATE}.wav",
                    metric="vggish",
                    verbose=False,
                    background_root_dirs=background_root_dirs,
                )
            )
        }
        if args.fad_background == "pooled":
            # LiveBand-style ground-truth row: only meaningful with a fixed
            # background (paired GT-vs-GT is 0 by construction).
            result["fad"]["vggish"]["gt"] = float(
                fad_eval.calculate_fad(
                    str(seg_root),
                    gt_path=f"ground_truth/pred_{EVAL_SAMPLE_RATE}.wav",
                    gen_path=f"ground_truth/pred_{EVAL_SAMPLE_RATE}.wav",
                    metric="vggish",
                    verbose=False,
                    background_root_dirs=background_root_dirs,
                )
            )
    if not args.skip_fad_clap:
        result["fad"]["clap"] = {
            "pred": float(
                fad_eval.calculate_fad(
                    str(seg_root),
                    gt_path="ground_truth/pred.wav",
                    gen_path="pred/pred.wav",
                    metric="clap",
                    verbose=False,
                    background_root_dirs=background_root_dirs,
                    clap_submodel_name=args.clap_submodel,
                )
            )
        }
        if args.fad_background == "pooled":
            result["fad"]["clap"]["gt"] = float(
                fad_eval.calculate_fad(
                    str(seg_root),
                    gt_path="ground_truth/pred.wav",
                    gen_path="ground_truth/pred.wav",
                    metric="clap",
                    verbose=False,
                    background_root_dirs=background_root_dirs,
                    clap_submodel_name=args.clap_submodel,
                )
            )
    return result


def compute_drift(seg_results):
    first, second = seg_results
    drift = {
        "beat_alignment": {
            "pred_f_measure": second["beat_alignment"]["pred_f_measure"][
                "mean"
            ]
            - first["beat_alignment"]["pred_f_measure"]["mean"],
            "gt_f_measure": second["beat_alignment"]["gt_f_measure"]["mean"]
            - first["beat_alignment"]["gt_f_measure"]["mean"],
        },
        "cocola": {
            mode: second["cocola"][mode]["pred_scores"]["mean"]
            - first["cocola"][mode]["pred_scores"]["mean"]
            for mode in COCOLA_MODES
        },
        "fad": {},
    }
    for metric in first.get("fad", {}):
        drift["fad"][metric] = (
            second["fad"][metric]["pred"] - first["fad"][metric]["pred"]
        )
    return drift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--results_tag", required=True)
    parser.add_argument("--results_save_dir", default="logs/eval_results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen_seconds", type=int, default=20)
    parser.add_argument("--segment_seconds", type=int, default=10)
    parser.add_argument("--window_frames", type=int, default=500)
    parser.add_argument(
        "--hop_frames",
        type=int,
        default=50,
        help="sliding-window hop; 50 (= chunk) keeps 9-10 s of history "
        "for every generated frame",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument(
        "--data_base_dir",
        default=None,
        help="override the ckpt args.yml data_base_dir",
    )
    parser.add_argument("--audio_base_dir", default="stream_music_gen_data/")
    parser.add_argument("--rms_base_dir", default="stream_music_gen_data/rms_50hz")
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        default=False,
        help="reuse an existing 20 s tree at the results_tag location",
    )
    parser.add_argument(
        "--skip_segmentation", action="store_true", default=False
    )
    parser.add_argument("--skip_resampling", action="store_true", default=False)
    parser.add_argument("--skip_fad_vggish", action="store_true", default=False)
    parser.add_argument("--skip_fad_clap", action="store_true", default=False)
    parser.add_argument(
        "--fad_background",
        choices=["paired", "pooled"],
        default="paired",
        help="paired: background = ground-truth stems of the same segment "
        "half (repo headline convention). pooled: one fixed background of "
        "both halves' ground truth (reference identical across the two "
        "drift terms, and gives a nonzero LiveBand-style GT row)",
    )
    parser.add_argument("--clap_submodel", default="630k-audioset")
    args = parser.parse_args()

    assert args.gen_seconds == 2 * args.segment_seconds, (
        "LiveBand protocol: two non-overlapping segments covering the "
        "whole generation"
    )

    ckpt_dir = os.path.dirname(os.path.abspath(args.model_path))
    root_folder = Path(ckpt_dir) / args.results_tag
    seg_roots = [
        Path(str(root_folder) + f"_seg_{i * args.segment_seconds}_"
             f"{(i + 1) * args.segment_seconds}")
        for i in range(2)
    ]

    if not args.skip_generation:
        run_generation(args, root_folder)
    else:
        print(f"[liveband] reusing existing tree {root_folder}")

    if not args.skip_segmentation:
        build_segment_trees(root_folder, seg_roots, args.segment_seconds)

    if not args.skip_resampling:
        files = [
            f
            for seg_root in seg_roots
            for f in glob.glob(
                os.path.join(seg_root, "**", "*.wav"), recursive=True
            )
            if not f.endswith(f"_{EVAL_SAMPLE_RATE}.wav")
        ]
        eval_utils.load_resample_save(files, DAC_SAMPLE_RATE, EVAL_SAMPLE_RATE)

    seg_results = []
    for seg_idx, seg_root in enumerate(seg_roots):
        print(f"[liveband] scoring segment {seg_idx} ({seg_root})")
        seg_results.append(evaluate_segment(seg_root, args, seg_roots))

    results = {
        "protocol": {
            "reference": "LiveBand, arXiv:2606.03803 (20 s generations, two "
            "non-overlapping 10 s segments, drift = second - first)",
            "model_path": os.path.abspath(args.model_path),
            "split": args.split,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "gen_seconds": args.gen_seconds,
            "segment_seconds": args.segment_seconds,
            "sliding_window_frames": args.window_frames,
            "sliding_hop_frames": args.hop_frames,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "fad_background": args.fad_background,
            "clap_submodel": args.clap_submodel,
            "loudness_normalization": "once per 20 s file, before slicing",
            "fv_future_padding": "edge-repeat last frame; for fv>0 ckpts "
            "the final fv frames (1 s) of conditioning are synthetic "
            "(precompute stores exactly 20 s)",
        },
        "segments": {
            f"{i * args.segment_seconds}_{(i + 1) * args.segment_seconds}s": r
            for i, r in enumerate(seg_results)
        },
        "drift": compute_drift(seg_results),
    }

    os.makedirs(args.results_save_dir, exist_ok=True)
    results_file = os.path.join(
        args.results_save_dir, args.results_tag + ".json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[liveband] results saved to: {results_file}")

    b0 = seg_results[0]["beat_alignment"]["pred_f_measure"]["mean"]
    b1 = seg_results[1]["beat_alignment"]["pred_f_measure"]["mean"]
    print(
        f"[liveband] Beat-F 0-10s {b0:.4f} | 10-20s {b1:.4f} | "
        f"drift {b1 - b0:+.4f}"
    )


if __name__ == "__main__":
    main()
