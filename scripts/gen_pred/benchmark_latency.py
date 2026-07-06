"""Streaming-inference latency benchmark for the online prefix-decoder.

Measures per-chunk wall-clock time on a real validation batch (batch_size=1)
by monkey-patching `generate_chunk` to record each chunk's latency inside a
warm KV-cache session. Separates cold-cache chunk-1 from warm-cache chunks
2..N. Also times the DAC decode step.

Hard requirement: must run on an A100 (verified at startup; aborts otherwise),
so that numbers are comparable to the ones we report.

Example:

    python scripts/gen_pred/benchmark_latency.py \
        --model_path models/<EXP_NAME>/step=200000.ckpt
"""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List

import numpy as np
import torch

from stream_music_gen.lit_module.online_prefix_dec import (
    LitOnlinePrefixDecoderMultiOut,
)
from stream_music_gen.utils.inference_utils import load_lit_model


FRAME_RATE_HZ = 50  # DAC tokens are at 50 Hz; 1 frame = 20 ms


def verify_a100():
    """Hard-fail unless the visible GPU is an A100."""
    assert torch.cuda.is_available(), "CUDA not available."
    name = torch.cuda.get_device_name(0)
    if "A100" not in name:
        raise RuntimeError(
            f"This benchmark requires an A100. Got: {name!r}"
        )
    print(f"[gpu-check] visible device: {name}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True
        )
        print(f"[gpu-check] nvidia-smi: {out.strip()}")
    except Exception as e:
        print(f"[gpu-check] nvidia-smi failed (non-fatal): {e}")


def stats(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _build_gen_kwargs(batch, device):
    keys = [
        "input_emb",
        "beat_cond",
        "bpm_log",
        "time_sig_num",
        "time_sig_den",
        "time_sig_change",
        "tempo_change",
        "local_bpm_log",
    ]
    out = {}
    for k in keys:
        v = batch.get(k)
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def time_session(model, gen_kwargs, inst_tokens, n_chunks,
                 dit_precompute=False):
    """Run one full streaming session (n_chunks chunks back-to-back, warm
    KV cache between chunks) and return per-chunk wall-clock seconds.

    Implementation: monkey-patch ``model.generate_chunk`` with a wrapper that
    syncs CUDA, perf_counters the call, and appends the elapsed time to a
    list. This measures exactly the per-chunk forward+sampling latency that
    a real-time deployment would see between consecutive chunks.
    """
    times: List[float] = []
    original = model.generate_chunk

    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        return result

    model.generate_chunk = wrapped
    try:
        _ = model.generate(
            seq_len=n_chunks * model.chunk_length,
            inst_tokens=inst_tokens,
            cache_kv=True,
            display_pbar=False,
            dit_modulation_precompute=dit_precompute,
            **gen_kwargs,
        )
    finally:
        model.generate_chunk = original
    return times


def time_decode(tokenizer, tokens_per_chunk, n_runs):
    """Time the DAC decode for one chunk's tokens, n_runs times."""
    times: List[float] = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = tokenizer.tokens_to_audio(tokens_per_chunk)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_chunks_per_session", type=int, default=10,
                        help="Chunks generated back-to-back in one warm "
                             "session. 10 = 10 s of audio.")
    parser.add_argument("--n_sessions", type=int, default=30,
                        help="Number of independent warm sessions. Each "
                             "produces n_chunks_per_session per-chunk times.")
    parser.add_argument("--n_warmup_sessions", type=int, default=3,
                        help="Discarded warm-up sessions to stabilize JIT/"
                             "CUDA cache.")
    parser.add_argument("--n_decode_runs", type=int, default=100)
    parser.add_argument("--dit_precompute", action="store_true",
                        help="Enable chunk-ahead DiT modulation precompute "
                             "(dit_modulation_precompute=True in generate).")
    parser.add_argument("--data_base_dir", default=None,
                        help="Override dataloader data_base_dir (needed when "
                             "the ckpt's recorded path no longer resolves).")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    verify_a100()

    # Stable, fast inference
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    device = "cuda"
    override_args = {}
    if args.data_base_dir is not None:
        override_args["data_base_dir"] = args.data_base_dir
    model, tokenizer, (_, val_dataloader) = load_lit_model(
        args.model_path,
        lit_module_cls=LitOnlinePrefixDecoderMultiOut,
        batch_size=1,
        compile=False,
        override_args=override_args if override_args else None,
    )
    model.to(device).eval()
    tokenizer.to(device)

    chunk_length = model.chunk_length
    audio_secs_per_chunk = chunk_length / FRAME_RATE_HZ

    batch = next(iter(val_dataloader))
    gen_kwargs = _build_gen_kwargs(batch, device)
    inst_tokens = torch.tensor(batch["target_inst_token"], device=device)

    print(f"[setup] chunk_length: {chunk_length}  ({audio_secs_per_chunk:.3f}s audio)")
    print(f"[setup] input_emb shape: {batch['input_emb'].shape}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] model parameters: {n_params/1e6:.2f}M")
    print(f"[setup] dit_precompute: {args.dit_precompute}")

    # Warm up CUDA / cache / cuDNN benchmark
    print(f"[warmup] {args.n_warmup_sessions} warm-up sessions x "
          f"{args.n_chunks_per_session} chunks each...")
    for _ in range(args.n_warmup_sessions):
        _ = time_session(model, gen_kwargs, inst_tokens,
                         args.n_chunks_per_session,
                         dit_precompute=args.dit_precompute)
    torch.cuda.synchronize()

    # Timed sessions
    print(f"[time] {args.n_sessions} sessions x "
          f"{args.n_chunks_per_session} chunks each...")
    cold_chunk_times: List[float] = []   # chunk index 0 in each session
    warm_chunk_times: List[float] = []   # chunk index 1..end in each session
    all_chunk_times: List[List[float]] = []  # full per-session arrays

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for s in range(args.n_sessions):
            t = time_session(model, gen_kwargs, inst_tokens,
                             args.n_chunks_per_session,
                             dit_precompute=args.dit_precompute)
            assert len(t) == args.n_chunks_per_session, (
                f"Expected {args.n_chunks_per_session} chunks, got {len(t)}"
            )
            cold_chunk_times.append(t[0])
            warm_chunk_times.extend(t[1:])
            all_chunk_times.append(t)
            if (s + 1) % 5 == 0:
                print(f"  session {s+1}/{args.n_sessions} done "
                      f"(per-chunk mean {1000*mean(t):.1f} ms)")
    finally:
        if gc_was_enabled:
            gc.enable()

    # Decode timing
    print(f"[decode] generating one chunk for DAC decode timing...")
    out = model.generate(
        seq_len=chunk_length,
        inst_tokens=inst_tokens,
        cache_kv=True,
        display_pbar=False,
        dit_modulation_precompute=args.dit_precompute,
        **gen_kwargs,
    )
    tokens_for_decode = tokenizer.post_process_tokens(out[0]).unsqueeze(0)
    print(f"[decode] {args.n_decode_runs} decode runs...")
    # warmup decode
    for _ in range(10):
        _ = tokenizer.tokens_to_audio(tokens_for_decode)
    torch.cuda.synchronize()
    decode_times = time_decode(tokenizer, tokens_for_decode,
                               args.n_decode_runs)

    # ---- Compose summary -------------------------------------------------
    cold = stats(cold_chunk_times)
    warm = stats(warm_chunk_times)
    dec = stats(decode_times)

    summary = {
        "model_path": args.model_path,
        "gpu_name": torch.cuda.get_device_name(0),
        "param_count_M": n_params / 1e6,
        "chunk_frames": chunk_length,
        "audio_secs_per_chunk": audio_secs_per_chunk,
        "n_sessions": args.n_sessions,
        "n_chunks_per_session": args.n_chunks_per_session,
        "n_warmup_sessions": args.n_warmup_sessions,
        "dit_modulation_precompute": bool(args.dit_precompute),
        # Per-chunk generation latency (cold cache, chunk 1 of session)
        "cold_chunk_s": cold,
        # Per-chunk generation latency (warm cache, chunks 2..end)
        "warm_chunk_s": warm,
        # DAC decode latency for one chunk's worth of tokens
        "dac_decode_chunk_s": dec,
    }

    # Real-time factor: wall-clock seconds per second of audio.
    # < 1 → can keep up with real-time.
    summary["rtf_cold_gen_only"] = cold["mean"] / audio_secs_per_chunk
    summary["rtf_warm_gen_only"] = warm["mean"] / audio_secs_per_chunk
    summary["rtf_warm_gen_plus_decode"] = (
        (warm["mean"] + dec["mean"]) / audio_secs_per_chunk
    )

    # Implied per-chunk latency expressed in 20ms DAC frames.
    summary["per_chunk_latency_frames_warm"] = warm["mean"] * FRAME_RATE_HZ
    summary["per_chunk_latency_frames_warm_plus_decode"] = (
        (warm["mean"] + dec["mean"]) * FRAME_RATE_HZ
    )
    # p95 worst-case (useful for buffering decisions)
    summary["p95_per_chunk_latency_frames_warm_plus_decode"] = (
        (warm["p95"] + dec["p95"]) * FRAME_RATE_HZ
    )

    print("\n" + "=" * 64)
    print("LATENCY BENCHMARK SUMMARY")
    print("=" * 64)
    print(json.dumps(summary, indent=2))

    print("\nKEY NUMBERS")
    print(f"  warm per-chunk gen: {1000*warm['mean']:.2f} +/- "
          f"{1000*warm['std']:.2f} ms  "
          f"(p50 {1000*warm['p50']:.2f}, p95 {1000*warm['p95']:.2f})")
    print(f"  dac decode:         {1000*dec['mean']:.2f} +/- "
          f"{1000*dec['std']:.2f} ms  "
          f"(p50 {1000*dec['p50']:.2f}, p95 {1000*dec['p95']:.2f})")
    print(f"  RTF (warm gen+dec): {summary['rtf_warm_gen_plus_decode']:.4f}")
    print(f"  per-chunk latency:  "
          f"{summary['per_chunk_latency_frames_warm_plus_decode']:.2f} frames "
          f"(p95 {summary['p95_per_chunk_latency_frames_warm_plus_decode']:.2f})")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
