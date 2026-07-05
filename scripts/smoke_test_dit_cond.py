"""Smoke test for the DiT (per-layer adaptive layer-norm + scale) beat-phase
conditioning path. Verifies:
  1. The model instantiates with use_beat_phase_dit_cond=True.
  2. A training-mode forward pass with random beat_cond produces finite logits
     of the expected shape.
  3. A tiny generate() runs end-to-end without crashing.
  4. The DiT-related parameters are present and live in the parameter list.

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

    # Tiny config — small enough to run on CPU in a few seconds, but large
    # enough that depth>1 actually exercises the per-layer adaptive norms.
    dim = 64
    depth = 4
    heads = 8  # must be divisible by project default attn_kv_heads=8
    num_tokens = 1025  # +1 for pad
    num_rvq_layers = 4
    chunk_length = 5
    max_duration_frames = 25  # 5 chunks × 5 frames each
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
        use_beat_phase=False,
        use_beat_phase_dit_cond=True,
        beat_dit_cond_dim=dim,
        beat_dit_cond_mlp_expansion=4,
    ).to(device=device, dtype=dtype)
    model.eval()

    # -- Step 1: parameter sanity ---------------------------------------------
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

    # -- Step 2: forward pass --------------------------------------------------
    B, T = 2, max_duration_frames
    x = torch.randint(0, num_tokens, (B, num_rvq_layers, T), device=device)
    # inst_tokens: one instrument id per batch item (shape [B], long).
    # Mirrors the production lit module's
    # ``dec_inst_tokens = torch.tensor(batch["target_inst_token"])``.
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
        logits, logits_mask, targets, aux_pred, aux_target, c_pred, c_target = model(
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
    print(f"[2] aux_pred is None (head off): {aux_pred is None}")
    print(f"[2] logits finite: {torch.isfinite(logits[logits_mask]).all().item()}")
    assert torch.isfinite(logits[logits_mask]).all()
    assert logits.shape[2] == chunk_length
    assert aux_pred is None and aux_target is None

    # -- Step 3: a tiny generate() --------------------------------------------
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

    # -- Step 4: confirm it runs WITHOUT beat_cond too (backward compat) -----
    # When beat_cond=None and DiT cond is on, the helper returns None and we
    # should fall through to a normal decoder call. That call will still try
    # to invoke AdaptiveLayerNorm without a condition — which would error.
    # So when DiT cond is enabled, beat_cond is REQUIRED. Verify the failure
    # mode is loud (assertion or KeyError), not silent.
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
            "beat_cond. This is silent failure — it means AdaptiveLayerNorm "
            "fell through somehow. Investigate."
        )

    print("\nALL SMOKE TESTS PASSED" if threw else "\nSMOKE TESTS PASSED WITH WARNING")


def test_minimal():
    """Smoke test the MINIMAL variant: only beat_cond + local_bpm_log fed in."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    dim = 64; depth = 4; heads = 8; num_tokens = 1025
    num_rvq_layers = 4; chunk_length = 5; max_duration_frames = 25; input_emb_dim = 8

    model = OnlinePrefixDecoderTransformerMultiOut(
        dim=dim, depth=depth, heads=heads, num_tokens=num_tokens,
        max_seq_len=max_duration_frames + 1, pad_value=0,
        num_rvq_layers=num_rvq_layers, shared=True, input_emb_dim=input_emb_dim,
        future_visibility=0, chunk_length=chunk_length,
        use_beat_phase=False, use_beat_phase_dit_cond=True,
        beat_dit_cond_dim=dim, beat_dit_cond_mlp_expansion=4,
        beat_dit_cond_minimal=True,
    ).to(device=device, dtype=dtype).eval()

    # Projector must be in minimal mode
    assert model.beat_cond_projector.minimal is True
    in_lin = model.beat_cond_projector.mlp[0]
    assert in_lin.in_features == 5, f"expected 5 in-features (4 phase + 1 local_bpm), got {in_lin.in_features}"
    assert not hasattr(model.beat_cond_projector, "ts_emb"), "ts_emb should not exist in minimal mode"
    print(f"[MIN-1] projector minimal=True, in_features={in_lin.in_features}, no ts_emb tables")

    B, T = 2, max_duration_frames
    x = torch.randint(0, num_tokens, (B, num_rvq_layers, T), device=device)
    inst = torch.zeros(B, dtype=torch.long, device=device)
    ie = torch.randn(B, T, input_emb_dim, device=device, dtype=dtype)
    bc = torch.randn(B, T, 4, device=device, dtype=dtype)
    lbpm = torch.randn(B, T, device=device, dtype=dtype) * 0.1

    with torch.no_grad():
        logits, lm, _, _, _, _, _ = model(
            x=x, inst_tokens=inst, input_emb=ie,
            beat_cond=bc, local_bpm_log=lbpm,
            # Pass dummy (or omit) global signals — projector ignores them in minimal mode.
            bpm_log=torch.zeros(B, device=device, dtype=dtype),
            time_sig_num=torch.full((B,), 4, dtype=torch.long, device=device),
        )
    assert torch.isfinite(logits[lm]).all()
    print(f"[MIN-2] forward OK, logits {tuple(logits.shape)}")

    # Generate
    gen_seq_len = chunk_length
    with torch.no_grad():
        gen = model.generate(
            seq_len=gen_seq_len,
            input_emb=ie[:, :gen_seq_len, :],
            inst_tokens=inst,
            beat_cond=bc[:, :gen_seq_len, :],
            local_bpm_log=lbpm[:, :gen_seq_len],
            bpm_log=torch.zeros(B, device=device, dtype=dtype),
            time_sig_num=torch.full((B,), 4, dtype=torch.long, device=device),
            cache_kv=True, display_pbar=False, temperature=1.0,
            filter_logits_fn=["top_k_multi_out"], filter_kwargs=[{"k": 50}],
        )
    print(f"[MIN-3] generate OK, output {tuple(gen.shape)}")
    print("MINIMAL VARIANT SMOKE PASSED")


def test_aux_only():
    """Smoke test the AUX-ONLY variant: no DiT cond, only the aux head
    predicting beat_cond from hidden states."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    dim = 64; depth = 4; heads = 8; num_tokens = 1025
    num_rvq_layers = 4; chunk_length = 5; max_duration_frames = 25; input_emb_dim = 8

    model = OnlinePrefixDecoderTransformerMultiOut(
        dim=dim, depth=depth, heads=heads, num_tokens=num_tokens,
        max_seq_len=max_duration_frames + 1, pad_value=0,
        num_rvq_layers=num_rvq_layers, shared=True, input_emb_dim=input_emb_dim,
        future_visibility=0, chunk_length=chunk_length,
        use_beat_phase=False,
        use_beat_phase_dit_cond=False,
        use_beat_phase_aux_head=True,
        beat_phase_aux_head_hidden_dim=32,
    ).to(device=device, dtype=dtype).eval()

    has_aux = any(n.startswith("beat_phase_aux_head.") for n, _ in model.named_parameters())
    assert has_aux, "expected beat_phase_aux_head params"
    has_dit = any(n.startswith("beat_cond_projector.") for n, _ in model.named_parameters())
    assert not has_dit, "DiT cond should be off"
    print(f"[AUX-1] aux-head params present, no DiT cond")

    B, T = 2, max_duration_frames
    x = torch.randint(0, num_tokens, (B, num_rvq_layers, T), device=device)
    inst = torch.zeros(B, dtype=torch.long, device=device)
    ie = torch.randn(B, T, input_emb_dim, device=device, dtype=dtype)
    bc = torch.randn(B, T, 4, device=device, dtype=dtype)

    with torch.no_grad():
        logits, lm, _, aux_pred, aux_target, _, _ = model(
            x=x, inst_tokens=inst, input_emb=ie,
            beat_cond=bc,
        )
    assert aux_pred is not None and aux_target is not None
    # S = context_end_idx + chunk_length; context_end_idx is random so just check rank/last dim.
    assert aux_pred.dim() == 3 and aux_pred.shape[0] == B and aux_pred.shape[-1] == 4
    assert aux_pred.shape == aux_target.shape
    assert torch.isfinite(aux_pred).all()
    print(f"[AUX-2] forward OK, aux_pred {tuple(aux_pred.shape)}, aux_target {tuple(aux_target.shape)}")

    # Generate (aux head is not used during generate but model must still run).
    gen_seq_len = chunk_length
    with torch.no_grad():
        gen = model.generate(
            seq_len=gen_seq_len,
            input_emb=ie[:, :gen_seq_len, :],
            inst_tokens=inst,
            beat_cond=bc[:, :gen_seq_len, :],
            cache_kv=True, display_pbar=False, temperature=1.0,
            filter_logits_fn=["top_k_multi_out"], filter_kwargs=[{"k": 50}],
        )
    print(f"[AUX-3] generate OK, output {tuple(gen.shape)}")
    print("AUX-ONLY VARIANT SMOKE PASSED")


def test_aux_combined():
    """Smoke test AUX + DiT cond combined."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    dim = 64; depth = 4; heads = 8; num_tokens = 1025
    num_rvq_layers = 4; chunk_length = 5; max_duration_frames = 25; input_emb_dim = 8

    model = OnlinePrefixDecoderTransformerMultiOut(
        dim=dim, depth=depth, heads=heads, num_tokens=num_tokens,
        max_seq_len=max_duration_frames + 1, pad_value=0,
        num_rvq_layers=num_rvq_layers, shared=True, input_emb_dim=input_emb_dim,
        future_visibility=0, chunk_length=chunk_length,
        use_beat_phase=False,
        use_beat_phase_dit_cond=True,
        beat_dit_cond_dim=dim, beat_dit_cond_mlp_expansion=4,
        use_beat_phase_aux_head=True,
        beat_phase_aux_head_hidden_dim=32,
    ).to(device=device, dtype=dtype).eval()

    has_aux = any(n.startswith("beat_phase_aux_head.") for n, _ in model.named_parameters())
    has_dit = any(n.startswith("beat_cond_projector.") for n, _ in model.named_parameters())
    assert has_aux and has_dit, "both heads must be present"
    print("[AUX+DIT-1] both projector and aux head present")

    B, T = 2, max_duration_frames
    x = torch.randint(0, num_tokens, (B, num_rvq_layers, T), device=device)
    inst = torch.zeros(B, dtype=torch.long, device=device)
    ie = torch.randn(B, T, input_emb_dim, device=device, dtype=dtype)
    bc = torch.randn(B, T, 4, device=device, dtype=dtype)
    bpm = torch.zeros(B, device=device, dtype=dtype)
    tsn = torch.full((B,), 4, dtype=torch.long, device=device)

    with torch.no_grad():
        logits, lm, _, aux_pred, aux_target, _, _ = model(
            x=x, inst_tokens=inst, input_emb=ie,
            beat_cond=bc, bpm_log=bpm, time_sig_num=tsn,
        )
    assert aux_pred is not None and torch.isfinite(aux_pred).all()
    print(f"[AUX+DIT-2] forward OK, logits {tuple(logits.shape)}, aux_pred {tuple(aux_pred.shape)}")
    print("AUX + DIT COMBINED SMOKE PASSED")


def test_chroma_aux_with_minimal_dit():
    """Smoke test PHASE A: minimal DiT beat cond + chroma aux head only."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    dim = 64; depth = 4; heads = 8; num_tokens = 1025
    num_rvq_layers = 4; chunk_length = 5; max_duration_frames = 25; input_emb_dim = 8

    model = OnlinePrefixDecoderTransformerMultiOut(
        dim=dim, depth=depth, heads=heads, num_tokens=num_tokens,
        max_seq_len=max_duration_frames + 1, pad_value=0,
        num_rvq_layers=num_rvq_layers, shared=True, input_emb_dim=input_emb_dim,
        future_visibility=0, chunk_length=chunk_length,
        use_beat_phase=False,
        use_beat_phase_dit_cond=True, beat_dit_cond_minimal=True,
        use_chroma_aux_head=True, chroma_aux_head_hidden_dim=32,
    ).to(device=device, dtype=dtype).eval()

    has_chroma_aux = any(
        n.startswith("chroma_aux_head.") for n, _ in model.named_parameters()
    )
    has_chroma_proj = any(
        n.startswith("chroma_cond_projector.") for n, _ in model.named_parameters()
    )
    assert has_chroma_aux, "expected chroma_aux_head params"
    assert not has_chroma_proj, "Phase A: chroma cond should be off"
    print("[PHASE-A-1] chroma aux head present, no chroma cond projector")

    B, T = 2, max_duration_frames
    x = torch.randint(0, num_tokens, (B, num_rvq_layers, T), device=device)
    inst = torch.zeros(B, dtype=torch.long, device=device)
    ie = torch.randn(B, T, input_emb_dim, device=device, dtype=dtype)
    bc = torch.randn(B, T, 4, device=device, dtype=dtype)
    lbpm = torch.randn(B, T, device=device, dtype=dtype) * 0.1
    target_chroma = torch.rand(B, T, 12, device=device, dtype=dtype)

    with torch.no_grad():
        logits, lm, _, _, _, c_pred, c_tgt = model(
            x=x, inst_tokens=inst, input_emb=ie,
            beat_cond=bc, local_bpm_log=lbpm,
            bpm_log=torch.zeros(B, device=device, dtype=dtype),
            time_sig_num=torch.full((B,), 4, dtype=torch.long, device=device),
            target_chroma=target_chroma,
        )
    assert c_pred is not None and c_tgt is not None
    assert c_pred.shape == c_tgt.shape and c_pred.shape[-1] == 12
    assert torch.isfinite(c_pred).all()
    print(f"[PHASE-A-2] forward OK, chroma_aux_pred {tuple(c_pred.shape)}")
    print("PHASE A SMOKE PASSED")


def test_chroma_cond_aux_combined():
    """Smoke test PHASE B: minimal DiT beat cond + chroma aux + chroma cond."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    dim = 64; depth = 4; heads = 8; num_tokens = 1025
    num_rvq_layers = 4; chunk_length = 5; max_duration_frames = 25; input_emb_dim = 8

    model = OnlinePrefixDecoderTransformerMultiOut(
        dim=dim, depth=depth, heads=heads, num_tokens=num_tokens,
        max_seq_len=max_duration_frames + 1, pad_value=0,
        num_rvq_layers=num_rvq_layers, shared=True, input_emb_dim=input_emb_dim,
        future_visibility=0, chunk_length=chunk_length,
        use_beat_phase=False,
        use_beat_phase_dit_cond=True, beat_dit_cond_minimal=True,
        use_chroma_dit_cond=True, chroma_dim=12,
        use_chroma_aux_head=True, chroma_aux_head_hidden_dim=32,
    ).to(device=device, dtype=dtype).eval()

    has_chroma_proj = any(
        n.startswith("chroma_cond_projector.") for n, _ in model.named_parameters()
    )
    has_chroma_aux = any(
        n.startswith("chroma_aux_head.") for n, _ in model.named_parameters()
    )
    assert has_chroma_proj and has_chroma_aux
    print("[PHASE-B-1] chroma cond projector + aux head both present")

    B, T = 2, max_duration_frames
    x = torch.randint(0, num_tokens, (B, num_rvq_layers, T), device=device)
    inst = torch.zeros(B, dtype=torch.long, device=device)
    ie = torch.randn(B, T, input_emb_dim, device=device, dtype=dtype)
    bc = torch.randn(B, T, 4, device=device, dtype=dtype)
    lbpm = torch.randn(B, T, device=device, dtype=dtype) * 0.1
    in_chroma = torch.rand(B, T, 12, device=device, dtype=dtype)
    target_chroma = torch.rand(B, T, 12, device=device, dtype=dtype)

    with torch.no_grad():
        logits, lm, _, _, _, c_pred, c_tgt = model(
            x=x, inst_tokens=inst, input_emb=ie,
            beat_cond=bc, local_bpm_log=lbpm,
            bpm_log=torch.zeros(B, device=device, dtype=dtype),
            time_sig_num=torch.full((B,), 4, dtype=torch.long, device=device),
            input_chroma=in_chroma, target_chroma=target_chroma,
        )
    assert c_pred is not None and torch.isfinite(c_pred).all()
    assert torch.isfinite(logits[lm]).all()
    print(f"[PHASE-B-2] forward OK, logits {tuple(logits.shape)}, chroma_aux_pred {tuple(c_pred.shape)}")

    # Generate end-to-end with both cond signals
    with torch.no_grad():
        gen = model.generate(
            seq_len=chunk_length,
            input_emb=ie[:, :chunk_length, :],
            inst_tokens=inst,
            beat_cond=bc[:, :chunk_length, :],
            local_bpm_log=lbpm[:, :chunk_length],
            bpm_log=torch.zeros(B, device=device, dtype=dtype),
            time_sig_num=torch.full((B,), 4, dtype=torch.long, device=device),
            input_chroma=in_chroma[:, :chunk_length, :],
            cache_kv=True, display_pbar=False, temperature=1.0,
            filter_logits_fn=["top_k_multi_out"], filter_kwargs=[{"k": 50}],
        )
    print(f"[PHASE-B-3] generate OK, output {tuple(gen.shape)}")
    print("PHASE B SMOKE PASSED")


if __name__ == "__main__":
    main()
    print()
    test_minimal()
    print()
    test_aux_only()
    print()
    test_aux_combined()
    print()
    test_chroma_aux_with_minimal_dit()
    print()
    test_chroma_cond_aux_combined()
