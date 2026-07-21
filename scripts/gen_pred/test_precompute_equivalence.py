"""Equivalence test for chunk-ahead DiT modulation precompute.

Runs on one real validation batch (batch_size=1, GPU required) and checks:

  A. Batched precomputed gammas match per-frame naive gammas over all
     AdaptiveLayerNorm / AdaptiveLayerScale modules. Any difference should
     come only from GEMM reduction order.
  B. Greedy generation (temperature=0) is token-exact between the naive
     per-step conditioning path and the precompute path.
  C. Sampled generation with the same seed. Reported as a match rate, not
     asserted, since one float-level logit flip at a sampling boundary
     changes the whole continuation.
  D. With the patches installed but precompute disabled, a greedy run
     still matches the pristine pre-install run token-exactly. Doubles as
     a run-to-run determinism check. If D fails, read B as a match rate
     rather than a bug.

Exit code 0 iff A, B, D pass.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from stream_music_gen.lit_module.online_prefix_dec import (
    LitOnlinePrefixDecoderMultiOut,
)
from stream_music_gen.models.dit_precompute import precompute_dit_gammas
from stream_music_gen.utils.inference_utils import load_lit_model


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


def first_divergence(a: torch.Tensor, b: torch.Tensor):
    """First differing time index between [B, K, L] token tensors, or None."""
    if torch.equal(a, b):
        return None
    diff = (a != b).any(dim=1)  # [B, L]
    idxs = diff.nonzero()
    return int(idxs[0, 1])


def match_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a == b).float().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seq_chunks", type=int, default=5,
                        help="Chunks per generation run (5 = 5 s of audio).")
    parser.add_argument("--sampled_temperature", type=float, default=1.0)
    parser.add_argument("--data_base_dir", default=None)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required."
    torch.set_grad_enabled(False)
    device = "cuda"

    override_args = {}
    if args.data_base_dir is not None:
        override_args["data_base_dir"] = args.data_base_dir
    model, _tokenizer, (_, val_dataloader) = load_lit_model(
        args.model_path,
        lit_module_cls=LitOnlinePrefixDecoderMultiOut,
        batch_size=1,
        compile=False,
        override_args=override_args if override_args else None,
    )
    model.to(device).eval()

    if not model.use_beat_phase_dit_cond:
        print("SKIP: model has no DiT conditioning; nothing to precompute.")
        sys.exit(0)

    batch = next(iter(val_dataloader))
    gen_kwargs = _build_gen_kwargs(batch, device)
    inst_tokens = torch.tensor(batch["target_inst_token"], device=device)
    ref_dtype = next(model.parameters()).dtype
    seq_len = args.seq_chunks * model.chunk_length

    print(f"[setup] model_path: {args.model_path}")
    print(f"[setup] param dtype: {ref_dtype}, chunk_length: "
          f"{model.chunk_length}, seq_len: {seq_len}")

    def run_gen(pc: bool, temperature: float, seed: int = 1234):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            seq_len=seq_len,
            inst_tokens=inst_tokens,
            cache_kv=True,
            display_pbar=False,
            temperature=temperature,
            dit_modulation_precompute=pc,
            **gen_kwargs,
        )
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    results = {"model_path": args.model_path, "seq_len": int(seq_len),
               "param_dtype": str(ref_dtype)}

    # B1: pristine greedy run before any patch install
    out_naive_greedy, t_naive = run_gen(pc=False, temperature=0.0)
    print(f"[B1] pristine naive greedy: {t_naive:.2f}s")

    # A: gamma numerics
    attn = model.net.attn_layers
    T = model.chunk_length
    cond = model._build_dit_condition(
        gen_kwargs.get("beat_cond"),
        gen_kwargs.get("bpm_log"),
        gen_kwargs.get("time_sig_num"),
        start_frame=0,
        end_frame=T,
        ref_dtype=ref_dtype,
        time_sig_den=gen_kwargs.get("time_sig_den"),
        time_sig_change=gen_kwargs.get("time_sig_change"),
        tempo_change=gen_kwargs.get("tempo_change"),
        local_bpm_log_padded=gen_kwargs.get("local_bpm_log"),
    )
    assert cond is not None and cond.shape[1] == T
    state = precompute_dit_gammas(attn, cond)
    print(f"[A] patched modules: {len(state.aln_modules)} ALN, "
          f"{len(state.als_modules)} ALS")

    max_diff_aln = 0.0
    max_diff_als = 0.0
    for f in range(T):
        exp_f = attn.adaptive_mlp(cond[:, f : f + 1])
        for m in state.aln_modules:
            ref = m.to_gamma(exp_f) + 1.0
            d = (state.gammas[m][:, f : f + 1] - ref).abs().max().item()
            max_diff_aln = max(max_diff_aln, d)
        for m in state.als_modules:
            ref = m.to_gamma(exp_f).sigmoid()
            d = (state.gammas[m][:, f : f + 1] - ref).abs().max().item()
            max_diff_als = max(max_diff_als, d)
    state.clear()

    tol = 1e-4 if ref_dtype == torch.float32 else 2e-2
    a_pass = max_diff_aln <= tol and max_diff_als <= tol
    print(f"[A] max |gamma diff|  ALN: {max_diff_aln:.3e}  "
          f"ALS: {max_diff_als:.3e}  (tol {tol:.0e})  "
          f"{'PASS' if a_pass else 'FAIL'}")
    results.update(
        n_aln=len(state.aln_modules), n_als=len(state.als_modules),
        gamma_max_diff_aln=max_diff_aln, gamma_max_diff_als=max_diff_als,
        gamma_tol=tol, test_a_pass=a_pass,
    )

    # B2: precompute greedy run
    out_pc_greedy, t_pc = run_gen(pc=True, temperature=0.0)
    b_pass = torch.equal(out_naive_greedy, out_pc_greedy)
    b_rate = match_rate(out_naive_greedy, out_pc_greedy)
    print(f"[B] pc greedy: {t_pc:.2f}s  exact: {b_pass}  "
          f"match rate: {b_rate:.6f}  "
          f"first divergence: {first_divergence(out_naive_greedy, out_pc_greedy)}")
    results.update(
        test_b_exact=b_pass, greedy_match_rate=b_rate,
        greedy_first_divergence=first_divergence(
            out_naive_greedy, out_pc_greedy),
        t_naive_greedy_s=t_naive, t_pc_greedy_s=t_pc,
    )

    # D: fallback (patched but disabled) vs pristine
    out_naive2, t_naive2 = run_gen(pc=False, temperature=0.0)
    d_pass = torch.equal(out_naive_greedy, out_naive2)
    print(f"[D] naive-after-install greedy: {t_naive2:.2f}s  exact: {d_pass}")
    if not d_pass:
        print("[D] WARNING: baseline run-to-run nondeterminism, read "
              f"Test B as a match rate. rate: "
              f"{match_rate(out_naive_greedy, out_naive2):.6f}")
    results.update(test_d_exact=d_pass, t_naive2_greedy_s=t_naive2)

    # C: sampled, same seed
    out_naive_s, _ = run_gen(pc=False, temperature=args.sampled_temperature,
                             seed=7)
    out_pc_s, _ = run_gen(pc=True, temperature=args.sampled_temperature,
                          seed=7)
    c_rate = match_rate(out_naive_s, out_pc_s)
    c_div = first_divergence(out_naive_s, out_pc_s)
    print(f"[C] sampled (T={args.sampled_temperature}, same seed) "
          f"match rate: {c_rate:.6f}  first divergence: {c_div}")
    results.update(sampled_match_rate=c_rate,
                   sampled_first_divergence=c_div,
                   sampled_temperature=args.sampled_temperature)

    ok = a_pass and b_pass and d_pass
    results["pass"] = ok
    print("\n" + "=" * 64)
    print(f"PRECOMPUTE EQUIVALENCE: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.out_json}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
