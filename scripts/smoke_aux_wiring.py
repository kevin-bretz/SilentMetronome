"""Smoke test for the aux heads (multipitch / CQT / target-token future).

No Lightning, no dataset, no wandb. Instantiates the model with the
relevant flags, runs forward and backward on synthetic tensors and asserts
losses are finite and every aux head receives gradient. Catches
shape/dtype/dead-tensor bugs before committing to a long training run.

Usage:
    python scripts/smoke_aux_wiring.py
"""
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
    }


def run_case(label, model_kwargs):
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
    logits, logits_mask, targets, extra_aux = model(
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
    )

    print(f"logits.shape={tuple(logits.shape)}  targets.shape={tuple(targets.shape)}")

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
        if "tt_future_pred" in extra_aux:
            # pred: [B, num_rvq, chunk, K_off, V]; tgt: [B, num_rvq, chunk,
            # K_off]; mask: [B, num_rvq, chunk] — same convention as the
            # lit module's aux loss.
            pred = extra_aux["tt_future_pred"]
            tgt = extra_aux["tt_future_target"]
            mask = extra_aux["tt_future_logits_mask"]
            print(f"  tt_future_pred={tuple(pred.shape)} tt_future_target={tuple(tgt.shape)}")
            per_off = []
            for k in range(pred.shape[3]):
                pk = pred[:, :, :, k, :][mask]
                tk = tgt[:, :, :, k].long()[mask]
                lk = F.cross_entropy(pk, tk)
                assert torch.isfinite(lk)
                per_off.append(lk)
            tt = sum(per_off) / len(per_off)
            print(f"  tt_future_ce={tt.item():.4f} over {pred.shape[3]} offsets")
            total = total + tt

    print(f"total_loss={total.item():.4f}")
    total.backward()

    # Verify all aux head params got gradient.
    for name in ("multipitch_aux_head", "cqt_aux_head", "target_token_future_aux_head"):
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
    run_case(
        "cond + mp + cqt",
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

    run_case(
        "full system: cond + mp + cqt + tt_future",
        dict(
            use_beat_phase_dit_cond=True,
            beat_dit_cond_mlp_expansion=4,
            use_multipitch_aux_head=True,
            multipitch_aux_head_hidden_dim=256,
            multipitch_dim=128,
            use_cqt_aux_head=True,
            cqt_aux_head_hidden_dim=256,
            cqt_dim=84,
            use_target_token_future_aux_head=True,
            target_token_future_aux_head_hidden_dim=256,
            future_aux_offsets=[10, 25, 40],
        ),
    )

    run_case(
        "aux only: mp + cqt + tt_future, no cond",
        dict(
            use_multipitch_aux_head=True,
            multipitch_aux_head_hidden_dim=256,
            multipitch_dim=128,
            use_cqt_aux_head=True,
            cqt_aux_head_hidden_dim=256,
            cqt_dim=84,
            use_target_token_future_aux_head=True,
            target_token_future_aux_head_hidden_dim=256,
            future_aux_offsets=[10, 25, 40],
        ),
    )

    print("\n[smoke] all configurations passed")


if __name__ == "__main__":
    main()
