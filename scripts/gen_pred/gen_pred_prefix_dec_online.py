"""Generate model predictions using the decoder online model."""

import os
import json
import argbind
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
import soundfile as sf
from functools import partial


from lightning import seed_everything


# weird bug
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from stream_music_gen.lit_module.online_prefix_dec import (
    LitOnlinePrefixDecoderMultiOut,
)
from stream_music_gen.models.models_multi_out import (
    OnlinePrefixDecoderTransformerMultiOut,
)
from stream_music_gen.tokenizer.audio_tokenizer import BaseAudioTokenizer
from stream_music_gen.utils.inference_utils import load_lit_model
from stream_music_gen.dataset.token_dataset import (
    get_precomputed_token_dataloader,
)
from stream_music_gen.utils.audio_utils import (
    mix_with_generated_stem,
    loudness_normalize_audio,
)

GROUP = __file__
# Binding things only when this file is loaded
bind = partial(argbind.bind, group=GROUP)


def generate_prediction(
    batch: any,
    model: OnlinePrefixDecoderTransformerMultiOut,
    tokenizer: BaseAudioTokenizer,
    max_gen_seq_len: int,
    device: str,
    prompt_secs: int = 0,
    random_input: bool = False,
    zero_input: bool = False,
    temperature: float = 1.0,
    top_k: int = 200,
    filter_logits_fn=None,
    filter_kwargs=None,
    zero_beat_cond: bool = False,
    shuffle_beat_cond: bool = False,
    drop_phase: bool = False,
    drop_local_bpm: bool = False,
    drop_time_sig: bool = False,
    drop_global_bpm: bool = False,
):

    # Input Token Stems
    input_token_stems = batch["input_token_stems"]

    input_emb = batch["input_emb"].to(device)
    targets = batch["target_token"].to(device)
    dec_inst_tokens = torch.tensor(batch["target_inst_token"], device=device)

    # Beat-phase conditioning (only present if the dataset provides beat_cond.pt
    # per window). Backward-compat: if absent, model.generate falls through and
    # the BeatPhaseConditioner contributes 0.
    beat_cond = batch.get("beat_cond")
    bpm_log = batch.get("bpm_log")
    time_sig_num = batch.get("time_sig_num")
    time_sig_den = batch.get("time_sig_den")
    time_sig_change = batch.get("time_sig_change")
    tempo_change = batch.get("tempo_change")
    local_bpm_log = batch.get("local_bpm_log")
    if beat_cond is not None:
        beat_cond = beat_cond.to(device)
    if bpm_log is not None:
        bpm_log = bpm_log.to(device)
    if time_sig_num is not None:
        time_sig_num = time_sig_num.to(device)
    if time_sig_den is not None:
        time_sig_den = time_sig_den.to(device)
    if time_sig_change is not None:
        time_sig_change = time_sig_change.to(device)
    if tempo_change is not None:
        tempo_change = tempo_change.to(device)
    if local_bpm_log is not None:
        local_bpm_log = local_bpm_log.to(device)

    # Inference-time ablations on the beat-phase signal. zero_beat_cond replaces
    # the per-frame phase with all zeros (still valid sin/cos for "no phase"),
    # bpm_log with 0 (=120 BPM), and time_sig_num with 4. shuffle_beat_cond
    # permutes phase/bpm/time-sig across the batch, keeping the marginal
    # signal distribution but pairing each window with another window's grid.
    if zero_beat_cond:
        if beat_cond is not None:
            beat_cond = torch.zeros_like(beat_cond)
        if bpm_log is not None:
            bpm_log = torch.zeros_like(bpm_log)
        if time_sig_num is not None:
            time_sig_num = torch.full_like(time_sig_num, 4)
    elif shuffle_beat_cond:
        if beat_cond is not None and beat_cond.shape[0] > 1:
            perm = torch.randperm(beat_cond.shape[0], device=beat_cond.device)
            beat_cond = beat_cond[perm]
            if bpm_log is not None:
                bpm_log = bpm_log[perm]
            if time_sig_num is not None:
                time_sig_num = time_sig_num[perm]
    else:
        if drop_phase and beat_cond is not None:
            beat_cond = torch.zeros_like(beat_cond)
        if drop_local_bpm and local_bpm_log is not None:
            local_bpm_log = torch.zeros_like(local_bpm_log)
        if drop_time_sig:
            if time_sig_num is not None:
                time_sig_num = torch.full_like(time_sig_num, 4)
            if time_sig_den is not None:
                time_sig_den = torch.full_like(time_sig_den, 4)
            if time_sig_change is not None:
                time_sig_change = torch.zeros_like(time_sig_change)
        if drop_global_bpm:
            if bpm_log is not None:
                bpm_log = torch.zeros_like(bpm_log)
            if tempo_change is not None:
                tempo_change = torch.zeros_like(tempo_change)

    if prompt_secs > 0:
        raise NotImplementedError(
            "Prompt is not supported in online prefix decoder for now."
        )
    else:
        seq_out_start = None

    if random_input:
        rand_idx = torch.randperm(input_emb.shape[0])
        input_emb = input_emb[rand_idx]
    elif zero_input:
        input_emb = torch.zeros_like(input_emb)

    # Generate
    decoder_preds = model.generate(
        seq_len=max_gen_seq_len,
        seq_out_start=seq_out_start,
        input_emb=input_emb,
        inst_tokens=dec_inst_tokens,
        beat_cond=beat_cond,
        bpm_log=bpm_log,
        time_sig_num=time_sig_num,
        time_sig_den=time_sig_den,
        time_sig_change=time_sig_change,
        tempo_change=tempo_change,
        local_bpm_log=local_bpm_log,
        cache_kv=True,
        filter_logits_fn=(
            filter_logits_fn
            if filter_logits_fn is not None
            else ["top_k_multi_out"]
        ),
        filter_kwargs=(
            filter_kwargs
            if filter_kwargs is not None
            else [{"k": top_k}]
        ),
        temperature=temperature,
        display_pbar=False,
    )

    # Collect
    max_duration = model.config["max_duration"]

    input_audio_all = []
    target_audio_all = []
    pred_audio_all = []
    num_input_stems_all = []

    for i in range(targets.size(0)):
        # Number of Stems
        num_input_stems_all.append(len(batch["input_inst_id"][i]))

        # Get Target Audio
        target_audio_tokens = tokenizer.post_process_tokens(targets[i])
        target_audio_single = (
            tokenizer.tokens_to_audio(target_audio_tokens.unsqueeze(0))
            .cpu()
            .squeeze()
        )
        target_audio_single = target_audio_single[
            : max_duration * tokenizer.sample_rate
        ]
        target_audio_all.append(target_audio_single)

        # Inputs
        input_audio_single = (
            tokenizer.codec_tokens_to_audio(
                input_token_stems[i][0].unsqueeze(0)
            )
            .cpu()
            .squeeze()
        )
        input_audio_single = input_audio_single[
            : max_duration * tokenizer.sample_rate
        ]

        input_audio_all.append(input_audio_single)

        # Predicted Tokens
        pred_audio_tokens = tokenizer.post_process_tokens(decoder_preds[i])
        pred_audio_single = (
            tokenizer.tokens_to_audio(pred_audio_tokens.unsqueeze(0))
            .cpu()
            .squeeze()
        )

        # Pad generated audio with zeros
        if pred_audio_single.shape[-1] < target_audio_single.shape[-1]:
            pred_audio_single = torch.nn.functional.pad(
                pred_audio_single,
                (
                    0,
                    target_audio_single.shape[-1] - pred_audio_single.shape[-1],
                ),
            )
        pred_audio_all.append(pred_audio_single)

    return (
        input_audio_all,
        target_audio_all,
        pred_audio_all,
        num_input_stems_all,
    )


def generate_metadata(batch, idx):
    return {
        "input_inst_id": batch["input_inst_id"][idx],
        "target_inst_id": batch["target_inst_id"][idx],
        "input_token_path": batch["input_token_path"][idx],
        "target_token_path": batch["target_token_path"][idx],
        "input_audio_path": batch["input_audio_path"][idx],
        "target_audio_path": batch["target_audio_path"][idx],
    }


@bind
def main(
    model_path: str = "models/pref_dec_online_fv0_k50_beat_phase_dit_mp_cqt_aux_tt_future/step=200000.ckpt",
    save_dir: str = "models/pref_dec_online_fv0_k50_beat_phase_dit_mp_cqt_aux_tt_future/model_predictions",
    audio_base_dir: str = "stream_music_gen_data/",
    data_base_dir: str = "stream_music_gen_data/causal_dac_codes_32khz",
    rms_base_dir: str = "stream_music_gen_data/rms_50hz",
    batch_size: int = 64,
    model_name: str = "prefix_decoder_online",
    num_samples: int = -1,
    max_gen_seq_len: int = 500,
    file_id: int = 0,
    seed: Optional[int] = None,
    gen_kwargs: dict = {},
    split: str = "valid",
):
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        seed_everything(seed)

    # Load the model
    model, tokenizer, dataloaders = load_lit_model(
        model_path,
        lit_module_cls=LitOnlinePrefixDecoderMultiOut,
        batch_size=batch_size,
        override_args={
            "data_base_dir": data_base_dir,
            "rms_base_dir": rms_base_dir,
            "audio_base_dir": audio_base_dir,
        },
    )
    model.to(device)
    model.eval()
    tokenizer.to(device)
    sample_rate = tokenizer.sample_rate

    # Prefix decoder needs extra input frames for look-ahead padding
    duration = model.config["max_duration"] + max(
        0, model.future_visibility // tokenizer.frame_rate
    )

    val_dataloader = get_precomputed_token_dataloader(
        batch_size=batch_size,
        dataset_names=model.config["dataset_names"],
        data_base_dir=model.config["data_base_dir"],
        rms_base_dir=model.config["rms_base_dir"],
        weights=None,
        duration=duration,
        num_rvq_layers=tokenizer.num_rvq_layers,
        split=split,
        num_workers=8,
        audio_base_dir=audio_base_dir,
        load_audio="true",
        pattern="multilayer",
        shuffle=False,
    )

    print(f"gen_kwargs: {gen_kwargs}")
    prompt_secs = gen_kwargs.get("prompt_secs", 0)
    random_input = gen_kwargs.get("random_input", False)
    zero_input = gen_kwargs.get("zero_input", False)
    temperature = gen_kwargs.get("temperature", 1.0)
    top_k = gen_kwargs.get("top_k", 200)
    filter_logits_fn = gen_kwargs.get("filter_logits_fn", None)
    filter_kwargs = gen_kwargs.get("filter_kwargs", None)
    zero_beat_cond = gen_kwargs.get("zero_beat_cond", False)
    shuffle_beat_cond = gen_kwargs.get("shuffle_beat_cond", False)
    drop_phase = gen_kwargs.get("drop_phase", False)
    drop_local_bpm = gen_kwargs.get("drop_local_bpm", False)
    drop_time_sig = gen_kwargs.get("drop_time_sig", False)
    drop_global_bpm = gen_kwargs.get("drop_global_bpm", False)

    for batch in tqdm(val_dataloader):
        (
            input_audio_all,
            target_audio_all,
            pred_audio_all,
            num_input_stems_all,
        ) = generate_prediction(
            batch,
            model,
            tokenizer,
            max_gen_seq_len,
            device,
            prompt_secs,
            random_input,
            zero_input,
            temperature,
            top_k,
            filter_logits_fn=filter_logits_fn,
            filter_kwargs=filter_kwargs,
            zero_beat_cond=zero_beat_cond,
            shuffle_beat_cond=shuffle_beat_cond,
            drop_phase=drop_phase,
            drop_local_bpm=drop_local_bpm,
            drop_time_sig=drop_time_sig,
            drop_global_bpm=drop_global_bpm,
        )

        max_duration = model.config["max_duration"]
        for idx in range(len(input_audio_all)):
            input_audio = batch["input_audio"][idx].cpu().numpy()
            input_audio = input_audio[: max_duration * tokenizer.sample_rate]

            target_audio = batch["target_audio"][idx].cpu().numpy()
            target_audio = target_audio[: max_duration * tokenizer.sample_rate]

            pred_audio = pred_audio_all[idx].cpu().numpy()
            num_input_stems = num_input_stems_all[idx]
            target_audio_codec = target_audio_all[idx].cpu().numpy()
            input_audio_codec = input_audio_all[idx].cpu().numpy()

            input_audio = loudness_normalize_audio(
                input_audio, tokenizer.sample_rate
            )
            target_audio = loudness_normalize_audio(
                target_audio, tokenizer.sample_rate
            )
            pred_audio = loudness_normalize_audio(
                pred_audio, tokenizer.sample_rate
            )
            target_audio_codec = loudness_normalize_audio(
                target_audio_codec, tokenizer.sample_rate
            )
            input_audio_codec = loudness_normalize_audio(
                input_audio_codec, tokenizer.sample_rate
            )

            metadata = generate_metadata(batch, idx)
            metadata["num_input_stems"] = num_input_stems

            sample_dir = save_dir / f"{file_id:05d}"
            os.makedirs(sample_dir, exist_ok=True)

            # save input audio
            input_audio_path = sample_dir / "input_audio.wav"
            if not os.path.exists(input_audio_path):
                sf.write(input_audio_path, input_audio, sample_rate)
            input_audio_codec_path = sample_dir / "input_audio_codec.wav"
            if not os.path.exists(input_audio_codec_path):
                sf.write(input_audio_codec_path, input_audio_codec, sample_rate)

            # save target audio
            ground_truth_dir = sample_dir / "ground_truth"
            os.makedirs(ground_truth_dir, exist_ok=True)
            ground_truth_path = ground_truth_dir / "pred.wav"
            if not os.path.exists(ground_truth_path):
                sf.write(ground_truth_path, target_audio, sample_rate)

            ground_truth_codec_dir = sample_dir / "ground_truth_codec"
            os.makedirs(ground_truth_codec_dir, exist_ok=True)
            ground_truth_codec_path = ground_truth_codec_dir / "pred.wav"
            if not os.path.exists(ground_truth_codec_path):
                sf.write(
                    ground_truth_codec_path, target_audio_codec, sample_rate
                )

            # save pred audio
            pred_audio_dir = sample_dir / model_name
            os.makedirs(pred_audio_dir, exist_ok=True)
            pred_audio_path = pred_audio_dir / "pred.wav"
            if not os.path.exists(pred_audio_path):
                sf.write(pred_audio_path, pred_audio, sample_rate)

            # mix
            mix_gt = mix_with_generated_stem(
                input_audio, target_audio, num_input_stems, sample_rate
            )
            mix_gt_codec = mix_with_generated_stem(
                input_audio_codec,
                target_audio_codec,
                num_input_stems,
                sample_rate,
            )
            mix_pred = mix_with_generated_stem(
                input_audio, pred_audio, num_input_stems, sample_rate
            )
            mix_gt_path = ground_truth_dir / "mix.wav"
            mix_gt_codec_path = ground_truth_codec_dir / "mix.wav"
            mix_pred_path = pred_audio_dir / "mix.wav"
            if not os.path.exists(mix_gt_path):
                sf.write(mix_gt_path, mix_gt, sample_rate)
            if not os.path.exists(mix_gt_codec_path):
                sf.write(mix_gt_codec_path, mix_gt_codec, sample_rate)
            if not os.path.exists(mix_pred_path):
                sf.write(mix_pred_path, mix_pred, sample_rate)

            # save metadata
            metadata_path = sample_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            file_id += 1

        if num_samples > 0 and file_id >= num_samples:
            break


if __name__ == "__main__":
    args = argbind.parse_args(group=GROUP)
    with argbind.scope(args):
        main(num_samples=16, batch_size=8)
