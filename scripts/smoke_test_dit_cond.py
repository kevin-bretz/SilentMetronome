"""Smoke test for the DiT (per-layer adaptive layer-norm and scale)
beat-phase conditioning path. Checks that the model instantiates with
use_beat_phase_dit_cond=True, that a forward pass with random beat_cond
produces finite logits of the expected shape, that a tiny generate() runs
end-to-end, and that the DiT parameters are present in the parameter list.

Usage:
    python scripts/smoke_test_dit_cond.py
"""

import torch
import torch.nn.functional as F

from stream_music_gen.models.models_multi_out import (
    OnlinePrefixDecoderTransformerMultiOut,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # smoke test in fp32 for clearest signal

    # Tiny config, small enough to run on CPU in a few seconds but with
    # depth>1 so the per-layer adaptive norms are actually exercised.
    dim = 64
    depth = 4
    heads = 8  # must be divisible by project default attn_kv_heads=8
    num_tokens = 1025  # +1 for pad
    num_rvq_layers = 4
    chunk_length = 5
    max_duration_frames = 25  # 5 chunks x 5 frames each
    input_emb_dim = 8

    model = OnlinePrefixDecoderTransformerMultiOut(
        dim=dim,
        depth=depth,
        heads=heads,
        num_tokens=num_tokens,
        max_seq_len=max_duration_frames + 1,
        pad_value=0,
        num_rvq_layers=num_rvq_layers,
        shared=True,
        input_emb_dim=input_emb_dim,
        future_visibility=0,
        chunk_length=chunk_length,
        use_beat_phase_dit_cond=True,
        beat_dit_cond_dim=dim,
        beat_dit_cond_mlp_expansion=4,
    ).to(device=device, dtype=dtype)
    model.eval()

    # Step 1: parameter sanity
    has_projector = any(
        n.startswith("beat_cond_projector.") for n, _ in model.named_parameters()
    )
    has_ada_ln_to_gamma = any(
        ".to_gamma." in n
        for n, _ in model.named_parameters()
    )
    print(f"[1] beat_cond_projector params present: {has_projector}")
    print(f"[1] adaptive layer-norm/scale to_gamma params present: {has_ada_ln_to_gamma}")
    assert has_projector
    assert has_ada_ln_to_gamma

    # Step 2: forward pass
    B, T = 2, max_duration_frames
    x = torch.randint(0, num_tokens, (B, num_rvq_layers, T), device=device)
    # One instrument id per batch item (shape [B], long), matching how the
    # lit module builds dec_inst_tokens.
    inst_tokens = torch.zeros((B,), dtype=torch.long, device=device)
    input_emb = torch.randn(B, T, input_emb_dim, device=device, dtype=dtype)

    beat_cond = torch.randn(B, T, 4, device=device, dtype=dtype)
    bpm_log = torch.zeros(B, device=device, dtype=dtype)
    time_sig_num = torch.full((B,), 4, dtype=torch.long, device=device)
    time_sig_den = torch.full((B,), 4, dtype=torch.long, device=device)
    time_sig_change = torch.zeros(B, device=device, dtype=dtype)
    tempo_change = torch.zeros(B, device=device, dtype=dtype)
    local_bpm_log = torch.randn(B, T, device=device, dtype=dtype) * 0.1

    with torch.no_grad():
        logits, logits_mask, targets, extra_aux = model(
            x=x,
            inst_tokens=inst_tokens,
            input_emb=input_emb,
            beat_cond=beat_cond,
            bpm_log=bpm_log,
            time_sig_num=time_sig_num,
            time_sig_den=time_sig_den,
            time_sig_change=time_sig_change,
            tempo_change=tempo_change,
            local_bpm_log=local_bpm_log,
        )

    print(f"[2] logits shape: {tuple(logits.shape)}")
    print(f"[2] logits_mask shape: {tuple(logits_mask.shape)}")
    print(f"[2] targets shape: {tuple(targets.shape)}")
    print(f"[2] extra_aux is None (heads off): {extra_aux is None}")
    print(f"[2] logits finite: {torch.isfinite(logits[logits_mask]).all().item()}")
    assert torch.isfinite(logits[logits_mask]).all()
    assert logits.shape[2] == chunk_length
    assert extra_aux is None

    # Step 3: a tiny generate()
    # Cap to one chunk so the test runs quickly.
    gen_seq_len = chunk_length
    with torch.no_grad():
        gen = model.generate(
            seq_len=gen_seq_len,
            input_emb=input_emb[:, :gen_seq_len, :],
            inst_tokens=inst_tokens,
            beat_cond=beat_cond[:, :gen_seq_len, :],
            bpm_log=bpm_log,
            time_sig_num=time_sig_num,
            time_sig_den=time_sig_den,
            time_sig_change=time_sig_change,
            tempo_change=tempo_change,
            local_bpm_log=local_bpm_log[:, :gen_seq_len],
            cache_kv=True,
            display_pbar=False,
            temperature=1.0,
            filter_logits_fn=["top_k_multi_out"],
            filter_kwargs=[{"k": 50}],
        )
    print(f"[3] generate output shape: {tuple(gen.shape)}")
    assert gen.shape[0] == B
    assert gen.shape[1] == num_rvq_layers
    assert gen.shape[2] > 0

    # Step 4: forward without beat_cond
    # With DiT cond enabled, beat_cond is required. AdaptiveLayerNorm cannot
    # run without a condition, so verify the failure mode is loud (assertion
    # or KeyError) rather than a silent fall-through.
    threw = False
    try:
        with torch.no_grad():
            model(
                x=x,
                inst_tokens=inst_tokens,
                input_emb=input_emb,
                # beat_cond, bpm_log, time_sig_num all omitted
            )
    except Exception as e:
        threw = True
        print(f"[4] (expected) error when DiT cond on but beat_cond missing: {type(e).__name__}: {str(e)[:120]}")
    if not threw:
        print(
            "[4] WARNING: DiT-cond model accepted a forward pass with no "
            "beat_cond. This is a silent failure, AdaptiveLayerNorm "
            "fell through somehow. Investigate."
        )

    print("\nALL SMOKE TESTS PASSED" if threw else "\nSMOKE TESTS PASSED WITH WARNING")


if __name__ == "__main__":
    main()
