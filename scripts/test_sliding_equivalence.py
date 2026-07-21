"""Verification suite for OnlinePrefixDecoderTransformerMultiOut.generate_sliding.

Three checks, run on a real checkpoint + real test batch:

1. STRUCTURAL EQUIVALENCE: with the same seed, the first window of a
   sliding 1000-frame run must be token-identical to a plain
   generate(seq_len=500) run, because the sliding loop performs exactly
   the same ops and RNG draws until the first re-base. Compared on the
   first (500 - 2*K) frames: the last K-1 frames of the short run are
   pattern-incomplete (pad-filled at revert) while the long run fills
   them with real samples, so they legitimately differ.

2. LONG-RUN SANITY: the 1000-frame output has the right shape, is not
   pad/silence-degenerate in the second half, and decodes to audio.
   One decoded 20 s example is written next to the log for listening.

3. (optional, CKPT_FV50) same structural check on an fv=+50 checkpoint,
   exercising the future-visibility head-column bookkeeping.

Usage (compute node):
  python scripts/test_sliding_equivalence.py --model_path <fv0 ckpt> \
      [--model_path_fv50 <fv50 ckpt>] [--out_dir logs/sliding_equiv]
"""

import argparse
import os

import soundfile as sf
import torch
from lightning import seed_everything

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from stream_music_gen.dataset.token_dataset import (
    get_precomputed_token_dataloader,
)
from stream_music_gen.lit_module.online_prefix_dec import (
    LitOnlinePrefixDecoderMultiOut,
)
from stream_music_gen.utils.inference_utils import load_lit_model

GEN_KW = dict(
    cache_kv=True,
    filter_logits_fn=["top_k_multi_out"],
    filter_kwargs=[{"k": 200}],
    temperature=1.0,
    display_pbar=False,
)
K_RVQ = 4


def load(model_path, batch_size, data_base_dir=None):
    override = {
        "rms_base_dir": "stream_music_gen_data/rms_50hz",
        "audio_base_dir": "stream_music_gen_data/",
    }
    if data_base_dir:
        override["data_base_dir"] = data_base_dir
    model, tokenizer, _ = load_lit_model(
        model_path,
        lit_module_cls=LitOnlinePrefixDecoderMultiOut,
        batch_size=batch_size,
        override_args=override,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tokenizer.to(device)
    return model, tokenizer, device


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


def _truncate_frames(kw, n_frames):
    out = {}
    for key, t in kw.items():
        if torch.is_tensor(t) and t.dim() >= 2 and t.shape[1] > n_frames:
            t = t[:, :n_frames]
        out[key] = t
    return out


def get_batch(model, tokenizer, batch_size, gen_seconds, device):
    frame_rate = tokenizer.frame_rate
    # The precompute stores exactly gen_seconds (20 s / 1000 frames); an
    # fv>0 ckpt's future input past the window end is edge-padded below
    # instead of loaded (loading gen_seconds + fv hard-fails the dataset).
    duration = gen_seconds
    loader = get_precomputed_token_dataloader(
        batch_size=batch_size,
        dataset_names=model.config["dataset_names"],
        data_base_dir=model.config["data_base_dir"],
        rms_base_dir=model.config["rms_base_dir"],
        weights=None,
        duration=duration,
        num_rvq_layers=tokenizer.num_rvq_layers,
        split="test",
        num_workers=4,
        audio_base_dir="stream_music_gen_data/",
        load_audio="true",
        pattern="multilayer",
        shuffle=False,
    )
    batch = next(iter(loader))
    kw = dict(
        input_emb=batch["input_emb"].to(device),
        inst_tokens=torch.tensor(batch["target_inst_token"], device=device),
    )
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
        kw[key] = value.to(device) if value is not None else None
    if model.future_visibility > 0:
        kw = pad_future_frames(
            kw, model.future_visibility, kw["input_emb"].shape[1]
        )
    print(
        f"  batch: input_emb {tuple(kw['input_emb'].shape)}, "
        f"beat_cond "
        f"{tuple(kw['beat_cond'].shape) if kw['beat_cond'] is not None else None}"
    )
    return kw


def structural_equivalence(model, kw, short_len=500, long_len=1000, seed=123):
    # The short run gets streams truncated to its own training-era contract
    # (short_len + fv frames); the sliding run consumes the full streams.
    short_kw = _truncate_frames(kw, short_len + max(0, model.future_visibility))
    seed_everything(seed)
    short = model.generate(seq_len=short_len, **short_kw, **GEN_KW)
    seed_everything(seed)
    slid = model.generate_sliding(
        seq_len=long_len,
        window_frames=short_len,
        hop_frames=50,
        **kw,
        **GEN_KW,
    )
    n_cmp = short.shape[-1] - 2 * K_RVQ
    same = torch.equal(slid[:, :, :n_cmp], short[:, :, :n_cmp])
    print(
        f"  short {tuple(short.shape)} vs sliding {tuple(slid.shape)}; "
        f"first {n_cmp} frames identical: {same}"
    )
    if not same:
        diff = (slid[:, :, :n_cmp] != short[:, :, :n_cmp]).any(dim=1)
        first_bad = int(diff.float().argmax(dim=-1).min())
        print(f"  FIRST DIFFERING FRAME (min over batch): {first_bad}")
    return same, slid


def long_run_sanity(model, tokenizer, slid, out_dir, tag):
    B, K, T = slid.shape
    ok_shape = K == K_RVQ and T >= 990
    pad_first = float((slid[..., : T // 2] == model.pad_value).float().mean())
    pad_second = float((slid[..., T // 2 :] == model.pad_value).float().mean())
    # a healthy continuation should not collapse to the pad token
    ok_pad = pad_second < 0.5
    print(
        f"  shape {tuple(slid.shape)} ok={ok_shape}; pad-token fraction "
        f"first half {pad_first:.4f} vs second half {pad_second:.4f} "
        f"ok={ok_pad}"
    )
    tokens = tokenizer.post_process_tokens(slid[0])
    audio = tokenizer.tokens_to_audio(tokens.unsqueeze(0)).cpu().squeeze()
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"sliding_20s_{tag}.wav")
    sf.write(wav_path, audio.numpy(), tokenizer.sample_rate)
    print(f"  decoded 20 s example -> {wav_path} "
          f"({audio.shape[-1] / tokenizer.sample_rate:.2f} s)")
    return ok_shape and ok_pad


def run_suite(model_path, tag, batch_size, out_dir, data_base_dir=None):
    print(f"\n=== {tag}: {model_path}")
    model, tokenizer, device = load(model_path, batch_size, data_base_dir)
    print(f"  future_visibility = {model.future_visibility}")
    kw = get_batch(model, tokenizer, batch_size, 20, device)
    same, slid = structural_equivalence(model, kw)
    sane = long_run_sanity(model, tokenizer, slid, out_dir, tag)
    passed = same and sane
    print(f"  [{tag}] {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="fv<=0 ckpt")
    parser.add_argument("--model_path_fv50", default=None)
    parser.add_argument("--data_base_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_dir", default="logs/sliding_equiv")
    args = parser.parse_args()

    results = [
        run_suite(
            args.model_path,
            "fv0",
            args.batch_size,
            args.out_dir,
            args.data_base_dir,
        )
    ]
    if args.model_path_fv50:
        results.append(
            run_suite(
                args.model_path_fv50,
                "fv50",
                args.batch_size,
                args.out_dir,
                args.data_base_dir,
            )
        )

    if all(results):
        print("\nALL SLIDING-EQUIVALENCE CHECKS PASSED")
    else:
        raise SystemExit("\nSLIDING-EQUIVALENCE CHECKS FAILED")


if __name__ == "__main__":
    main()
