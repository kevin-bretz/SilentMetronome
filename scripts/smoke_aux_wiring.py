"""Lean smoke test for new aux heads (multipitch / CQT / beat-phase-full).

No Lightning, no dataset, no wandb — just instantiate the model with the
relevant flags, run forward + backward on synthetic tensors, and assert
losses are finite. Catches shape/dtype/dead-tensor bugs before any 200K
training run is committed.

Usage:
    python scripts/smoke_aux_wiring.py
"""
import sys
import torch
import torch.nn.functional as F

from stream_music_gen.models.models_multi_out import (
    OnlinePrefixDecoderTransformerMultiOut,
)


def build_synthetic(B=2, T=100, device="cuda"):
    """B samples, T frames @ 50Hz, 1024-d input emb, 4-RVQ codes."""
    return {
        "input_emb": torch.randn(B, T, 1024, device=device),
        "output_tokens": torch.randint(0, 4096, (B, 4, T), device=device),
        "inst_tokens": torch.tensor([3, 4][:B], device=device, dtype=torch.long),
        "beat_cond": torch.randn(B, T, 4, device=device).tanh(),
        "bpm_log": torch.randn(B, device=device),
        "time_sig_num": torch.tensor([4, 3][:B], device=device, dtype=torch.long),
        "time_sig_den": torch.tensor([4, 4][:B], device=device, dtype=torch.long),
        "time_sig_change": torch.zeros(B, device=device, dtype=torch.float32),
        "tempo_change": torch.zeros(B, device=device, dtype=torch.float32),
        "local_bpm_log": torch.randn(B, T, device=device),
        "target_multipitch": (torch.rand(B, T, 128, device=device) > 0.95).float(),
        "target_velocity": torch.rand(B, T, 128, device=device),
        "target_cqt": torch.randn(B, T, 84, device=device),
        "input_cqt": torch.randn(B, T, 84, device=device),
    }


def run_phase(label, model_kwargs):
    print(f"\n========== SMOKE: {label} ==========")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    base = dict(
        dim=1024, depth=16, heads=16,
        num_tokens=4115, max_seq_len=501,
        num_rvq_layers=4, shared=True,
        input_emb_dim=1024, output_emb_dim=1024,
        future_visibility=0, chunk_length=50,
        time_sig_vocab_size=16,
    )
    base.update(model_kwargs)
    model = OnlinePrefixDecoderTransformerMultiOut(**base).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.1f}M")

    batch = build_synthetic(B=2, T=100, device=device)
    out = model(
        x=batch["output_tokens"],
        inst_tokens=batch["inst_tokens"],
        input_emb=batch["input_emb"],
        beat_cond=batch["beat_cond"],
        bpm_log=batch["bpm_log"],
        time_sig_num=batch["time_sig_num"],
        time_sig_den=batch["time_sig_den"],
        time_sig_change=batch["time_sig_change"],
        tempo_change=batch["tempo_change"],
        local_bpm_log=batch["local_bpm_log"],
        target_multipitch=batch["target_multipitch"],
        target_velocity=batch["target_velocity"],
        target_cqt=batch["target_cqt"],
        input_cqt=batch["input_cqt"],
    )
    (logits, logits_mask, targets,
     beat_aux_pred, beat_aux_target,
     chroma_aux_pred, chroma_aux_target,
     chroma_aux_dsv_preds,
     extra_aux) = out

    print(f"logits.shape={tuple(logits.shape)}  targets.shape={tuple(targets.shape)}")

    # CE loss on logits.
    ce = F.cross_entropy(logits[logits_mask], targets[logits_mask])
    print(f"ce_loss={ce.item():.4f}")
    assert torch.isfinite(ce), "CE loss not finite"

    total = ce
    if extra_aux is not None:
        print("extra_aux keys:", sorted(extra_aux.keys()))
        if "mp_pred" in extra_aux:
            mp = extra_aux["mp_pred"]
            mp_t = extra_aux["mp_target"]
            print(f"  mp_pred={tuple(mp.shape)} mp_target={tuple(mp_t.shape)}")
            bce = F.binary_cross_entropy_with_logits(mp, mp_t)
            print(f"  mp_bce={bce.item():.4f}")
            assert torch.isfinite(bce)
            total = total + bce
        if "cqt_pred" in extra_aux:
            c = extra_aux["cqt_pred"]
            ct = extra_aux["cqt_target"]
            print(f"  cqt_pred={tuple(c.shape)} cqt_target={tuple(ct.shape)}")
            mse = F.mse_loss(c, ct)
            cos = 1 - F.cosine_similarity(c, ct, dim=-1).mean()
            print(f"  cqt_mse={mse.item():.4f}  cqt_cos={cos.item():.4f}")
            assert torch.isfinite(mse) and torch.isfinite(cos)
            total = total + 0.5 * mse + 0.5 * cos
        if "input_cqt_pred" in extra_aux:
            ic = extra_aux["input_cqt_pred"]
            ict = extra_aux["input_cqt_target"]
            print(f"  input_cqt_pred={tuple(ic.shape)} input_cqt_target={tuple(ict.shape)}")
            mse_in = F.mse_loss(ic, ict)
            cos_in = 1 - F.cosine_similarity(ic, ict, dim=-1).mean()
            print(f"  input_cqt_mse={mse_in.item():.4f}  input_cqt_cos={cos_in.item():.4f}")
            assert torch.isfinite(mse_in) and torch.isfinite(cos_in)
            total = total + 0.5 * mse_in + 0.5 * cos_in
        if "beat_full_pred" in extra_aux:
            bf = extra_aux["beat_full_pred"]
            pt = extra_aux["beat_full_phase_target"]
            print(f"  beat_full_pred={tuple(bf.shape)} phase_target={tuple(pt.shape)}")
            phase_pred = bf[..., :4]
            bpm_pred = bf[..., 4]
            ts_logits = bf[..., 5:]
            phase_loss = F.mse_loss(phase_pred, pt) + (
                1 - F.cosine_similarity(phase_pred, pt, dim=-1).mean()
            )
            bpm_target = batch["bpm_log"].unsqueeze(-1).expand_as(bpm_pred)
            bpm_loss = F.mse_loss(bpm_pred, bpm_target)
            ts_target = batch["time_sig_num"].unsqueeze(-1).expand(bpm_pred.shape)
            ts_loss = F.cross_entropy(
                ts_logits.reshape(-1, ts_logits.shape[-1]), ts_target.reshape(-1)
            )
            print(f"  bf_phase={phase_loss.item():.4f}  bf_bpm={bpm_loss.item():.4f}  bf_ts={ts_loss.item():.4f}")
            assert all(torch.isfinite(x) for x in (phase_loss, bpm_loss, ts_loss))
            total = total + phase_loss + 0.1 * bpm_loss + 0.1 * ts_loss

    print(f"total_loss={total.item():.4f}")
    total.backward()

    # Verify all aux head params got gradient.
    for name in ("multipitch_aux_head", "cqt_aux_head", "input_cqt_aux_head", "beat_phase_aux_head_full"):
        if hasattr(model, name):
            head = getattr(model, name)
            grads = [p.grad for p in head.parameters() if p.grad is not None]
            assert grads, f"{name} got no grads!"
            gn = sum(g.norm().item() for g in grads)
            print(f"  {name}: |grad|={gn:.4f} ({len(grads)} tensors)")

    print(f"== {label} OK ==")
    del model
    torch.cuda.empty_cache()


def main():
    # Phase F: DiT cond + multipitch + cqt aux.
    run_phase(
        "PHASE F (cond + mp + cqt)",
        dict(
            use_beat_phase_dit_cond=True,
            beat_dit_cond_mlp_expansion=4,
            use_multipitch_aux_head=True,
            multipitch_aux_head_hidden_dim=256,
            multipitch_dim=128,
            use_cqt_aux_head=True,
            cqt_aux_head_hidden_dim=256,
            cqt_dim=84,
        ),
    )

    # Phase G: NO cond. multipitch + cqt + beat-full aux.
    run_phase(
        "PHASE G (mp + cqt + beat-full aux, no cond)",
        dict(
            use_multipitch_aux_head=True,
            multipitch_aux_head_hidden_dim=256,
            multipitch_dim=128,
            use_cqt_aux_head=True,
            cqt_aux_head_hidden_dim=256,
            cqt_dim=84,
            use_beat_phase_aux_head_full=True,
            beat_phase_aux_head_full_hidden_dim=256,
        ),
    )

    # Phase H: Phase F + input-mix CQT aux head.
    run_phase(
        "PHASE H (cond + mp + cqt + input_cqt)",
        dict(
            use_beat_phase_dit_cond=True,
            beat_dit_cond_mlp_expansion=4,
            use_multipitch_aux_head=True,
            multipitch_aux_head_hidden_dim=256,
            multipitch_dim=128,
            use_cqt_aux_head=True,
            cqt_aux_head_hidden_dim=256,
            cqt_dim=84,
            use_input_cqt_aux_head=True,
            input_cqt_aux_head_hidden_dim=256,
            input_cqt_dim=84,
        ),
    )

    # Phase CD: Phase F + cond dropout p=0.15 (CFG-style).
    run_phase(
        "PHASE CD (cond + mp + cqt + cond_dropout p=0.15)",
        dict(
            use_beat_phase_dit_cond=True,
            beat_dit_cond_mlp_expansion=4,
            cond_dropout_p=0.15,
            use_multipitch_aux_head=True,
            multipitch_aux_head_hidden_dim=256,
            multipitch_dim=128,
            use_cqt_aux_head=True,
            cqt_aux_head_hidden_dim=256,
            cqt_dim=84,
        ),
    )

    print("\n[smoke] ALL PHASES PASSED")


if __name__ == "__main__":
    main()
