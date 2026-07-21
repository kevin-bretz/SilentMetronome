"""
Lightning module for training the encoder-decoder generative
with delay pattern
"""

import argbind
import time

import wandb
import torch
import torch.nn.functional as F

from stream_music_gen.base_trainer import BaseLightningModel
from stream_music_gen.utils.lr_scheduler import LinearWarmupCosineDecay
from stream_music_gen.utils.plot_utils import audio_to_spectrogram_image
from stream_music_gen.models.models_multi_out import (
    OnlinePrefixDecoderTransformerMultiOut,
)
from stream_music_gen.dataset.token_dataset import (
    get_precomputed_token_dataloader,
)
from stream_music_gen.tokenizer import DACAudioTokenizer
from stream_music_gen.constants import MIDI_CATEGORIES
from stream_music_gen.utils.audio_utils import (
    mix_with_generated_stem,
    PRED_DB_OFFSET_LOUD,
    loudness_normalize_audio,
)

bind = argbind.bind

OnlinePrefixDecoderTransformerMultiOut = bind(
    OnlinePrefixDecoderTransformerMultiOut
)
AdamW = bind(torch.optim.AdamW)
LinearWarmupCosineDecay = bind(LinearWarmupCosineDecay)
get_dataloader = bind(get_precomputed_token_dataloader, without_prefix=True)


@bind(without_prefix=True)
class LitOnlinePrefixDecoderMultiOut(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
        max_duration: int = 10,
        time_sig_vocab_size: int = 16,
        use_beat_phase_dit_cond: bool = False,
        beat_dit_cond_dim: int = None,
        beat_dit_cond_mlp_expansion: int = 4,
        beat_phase_noise_std: float = 0.0,
        # Multipitch aux head (BCE on presence).
        use_multipitch_aux_head: bool = False,
        multipitch_aux_head_hidden_dim: int = 256,
        multipitch_aux_head_weight: float = 1.0,
        multipitch_dim: int = 128,
        # CQT aux head (MSE+cos on 84-bin log-magnitude).
        use_cqt_aux_head: bool = False,
        cqt_aux_head_hidden_dim: int = 256,
        cqt_aux_head_weight: float = 1.0,
        cqt_dim: int = 84,
        # Future target-stem token aux head (CE on patterned tokens).
        use_target_token_future_aux_head: bool = False,
        target_token_future_aux_head_hidden_dim: int = 256,
        target_token_future_aux_head_weight: float = 1.0,
        future_aux_offsets: list = None,
    ):
        """
        Lightning Module for Online Prefix Decoder, multilayer/multiout.

        Args:
            chunk_length: the length of each chunk, in frames
            chunk_start_prob: In preparing the input, the probability of the input
            as a new chunk (no input context).

        Attributes:
            inst_tokens_as_pattern_token: if we are adding inst_tokens to
            inputs/targets
        """
        super(LitOnlinePrefixDecoderMultiOut, self).__init__()

        tokenizer = DACAudioTokenizer(
            num_special_tokens=1,  # Changed from 3 to 1, no BOS
            num_instrument_tokens=len(MIDI_CATEGORIES),
            num_rvq_layers=4,
            multilayer=True,
            shared=True,
        )

        self.max_duration = max_duration
        self.frame_rate = tokenizer.frame_rate
        self.sample_rate = tokenizer.sample_rate
        self.num_rvq_layers = tokenizer.num_rvq_layers
        self.num_codebook_per_layer = tokenizer.num_codebook_per_layer
        self.max_gen_seq_len = (
            self.max_duration * self.frame_rate  # Modified, not mult. by n_rvq
        )

        # hard-code to not add (concat) inst tokens to output token
        add_inst_tokens = False

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples

        tokenizer.eval()
        self.pad_token = tokenizer.pad_token
        self.num_tokens = tokenizer.num_tokens
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.tokenizer = self.tokenizer.to(torch.device("cpu"))
        self.sample_rate = tokenizer.sample_rate

        self.use_beat_phase_dit_cond = use_beat_phase_dit_cond
        # Aux head flags + weights.
        self.use_multipitch_aux_head = bool(use_multipitch_aux_head)
        self.multipitch_aux_head_weight = float(multipitch_aux_head_weight)
        self.multipitch_dim = int(multipitch_dim)
        self.use_cqt_aux_head = bool(use_cqt_aux_head)
        self.cqt_aux_head_weight = float(cqt_aux_head_weight)
        self.cqt_dim = int(cqt_dim)
        self.use_target_token_future_aux_head = bool(
            use_target_token_future_aux_head
        )
        self.target_token_future_aux_head_weight = float(
            target_token_future_aux_head_weight
        )
        self.future_aux_offsets = tuple(
            int(d) for d in (future_aux_offsets or (10, 25, 40))
        )
        model_kwargs = dict(
            num_tokens=self.num_tokens,  # removed +2, no bos/eos
            max_seq_len=self.max_gen_seq_len + add_inst_tokens + 1,
            pad_value=self.pad_token,
            num_rvq_layers=self.num_rvq_layers,  # Added
            shared=True,
            online=True,  # Added
            inst_tokens_as_pattern_token=True,
            input_emb_dim=tokenizer.emb_dim,
            time_sig_vocab_size=time_sig_vocab_size,
            use_beat_phase_dit_cond=use_beat_phase_dit_cond,
            beat_dit_cond_dim=beat_dit_cond_dim,
            beat_dit_cond_mlp_expansion=beat_dit_cond_mlp_expansion,
            beat_phase_noise_std=beat_phase_noise_std,
            use_multipitch_aux_head=self.use_multipitch_aux_head,
            multipitch_aux_head_hidden_dim=multipitch_aux_head_hidden_dim,
            multipitch_dim=self.multipitch_dim,
            use_cqt_aux_head=self.use_cqt_aux_head,
            cqt_aux_head_hidden_dim=cqt_aux_head_hidden_dim,
            cqt_dim=self.cqt_dim,
            use_target_token_future_aux_head=self.use_target_token_future_aux_head,
            target_token_future_aux_head_hidden_dim=target_token_future_aux_head_hidden_dim,
            future_aux_offsets=list(self.future_aux_offsets),
        )
        self.model = OnlinePrefixDecoderTransformerMultiOut(**model_kwargs)

        # Compute duration before compile (compile wraps the model, hiding attrs)
        # input_delay = -future_visibility; extra input frames needed for look-ahead
        duration = self.max_duration + max(
            0, self.model.future_visibility / self.frame_rate
        )

        if compile:
            self.model = torch.compile(self.model)

        train_dataloader = get_dataloader(
            frame_rate_hz=self.frame_rate,
            duration=duration,
            num_rvq_layers=self.num_rvq_layers,
            sample_rate=self.sample_rate,
            num_codebook_per_layer=self.num_codebook_per_layer,
            split="train",
            shuffle=True,
            pattern="multilayer",
            add_inst_tokens=add_inst_tokens,  # Added
        )
        val_dataloader = get_dataloader(
            frame_rate_hz=self.frame_rate,
            duration=duration,
            num_rvq_layers=self.num_rvq_layers,
            sample_rate=self.sample_rate,
            num_codebook_per_layer=self.num_codebook_per_layer,
            split="valid",
            shuffle=False,
            pattern="multilayer",
            add_inst_tokens=add_inst_tokens,  # Added
            load_audio="true",
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def get_inputs(self, batch):
        # We will chunk and crop input inside the model class
        input_emb = batch["input_emb"]
        output_tokens = batch["target_token"]

        dec_inst_tokens = torch.tensor(
            batch["target_inst_token"], device=output_tokens.device
        )

        beat_cond = batch.get("beat_cond", None)
        bpm_log = batch.get("bpm_log", None)
        time_sig_num = batch.get("time_sig_num", None)
        time_sig_den = batch.get("time_sig_den", None)
        time_sig_change = batch.get("time_sig_change", None)
        tempo_change = batch.get("tempo_change", None)
        local_bpm_log = batch.get("local_bpm_log", None)
        target_multipitch = batch.get("target_multipitch", None)
        target_velocity = batch.get("target_velocity", None)
        target_has_multipitch = batch.get("target_has_multipitch", None)
        target_cqt = batch.get("target_cqt", None)

        return (
            input_emb,
            output_tokens,
            dec_inst_tokens,
            beat_cond,
            bpm_log,
            time_sig_num,
            time_sig_den,
            time_sig_change,
            tempo_change,
            local_bpm_log,
            target_multipitch,
            target_velocity,
            target_has_multipitch,
            target_cqt,
        )

    def _multipitch_loss(self, mp_pred, presence_target, has_mask):
        """BCE on per-frame multipitch presence. ``mp_pred``: [B, S, D]
        presence logits.
        """
        presence_logits = mp_pred
        presence_target = presence_target.to(presence_logits.dtype)

        if has_mask is None:
            return F.binary_cross_entropy_with_logits(
                presence_logits, presence_target, reduction="mean"
            )

        w = has_mask.to(presence_logits.dtype).view(-1, 1, 1)
        bce_per = F.binary_cross_entropy_with_logits(
            presence_logits, presence_target, reduction="none"
        )
        denom_b = w.sum().clamp(min=1.0) * bce_per.shape[1] * bce_per.shape[2]
        return (bce_per * w).sum() / denom_b

    def _cqt_loss(self, pred, target):
        """MSE+cos on CQT log-magnitude. No has_mask needed, every window
        has a CQT target since drums don't lack spectra.
        """
        target = target.to(pred.dtype)
        mse = F.mse_loss(pred, target)
        cos = 1.0 - F.cosine_similarity(
            pred.float(), target.float(), dim=-1, eps=1e-8
        ).mean()
        return 0.5 * mse + 0.5 * cos

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        # print(f"batch_idx: {batch_idx}, global_step: {self.global_step}")
        (
            input_emb,
            output_tokens,
            dec_inst_tokens,
            beat_cond,
            bpm_log,
            time_sig_num,
            time_sig_den,
            time_sig_change,
            tempo_change,
            local_bpm_log,
            target_multipitch,
            target_velocity,
            target_has_multipitch,
            target_cqt,
        ) = self.get_inputs(batch)

        # In x-transformers, mask is True for unmasked tokens
        (
            logits, logits_mask, targets,
            extra_aux,
        ) = self.model(
            x=output_tokens,
            input_emb=input_emb,
            inst_tokens=dec_inst_tokens,
            beat_cond=beat_cond,
            bpm_log=bpm_log,
            time_sig_num=time_sig_num,
            time_sig_den=time_sig_den,
            time_sig_change=time_sig_change,
            tempo_change=tempo_change,
            local_bpm_log=local_bpm_log,
            target_multipitch=target_multipitch,
            target_velocity=target_velocity,
            target_cqt=target_cqt,
        )

        ce_loss = F.cross_entropy(
            logits[logits_mask],  #  For delay pattern, only
            targets[logits_mask],  # compute mask on valid logits.
            ignore_index=self.pad_token,
        )

        log_d = {"train/loss": ce_loss}
        loss = ce_loss
        loss, log_d = self._add_extra_aux_losses(
            loss, log_d, extra_aux,
            target_has_multipitch, prefix="train",
        )

        # Log training loss (W&B logging included through self.log)
        self._log_dict(log_d, batch_size=output_tokens.size(0))
        return loss

    def _add_extra_aux_losses(
        self, loss, log_d, extra_aux,
        target_has_multipitch, prefix: str,
    ):
        """Add multipitch / CQT / target-token-future losses if present in
        ``extra_aux`` and the corresponding head is enabled. ``prefix`` is
        ``"train"`` or ``"val"`` for log keys.
        """
        if extra_aux is None:
            return loss, log_d
        if (
            self.use_multipitch_aux_head
            and "mp_pred" in extra_aux
            and "mp_target" in extra_aux
        ):
            mp_bce = self._multipitch_loss(
                extra_aux["mp_pred"],
                extra_aux["mp_target"],
                target_has_multipitch,
            )
            loss = loss + self.multipitch_aux_head_weight * mp_bce
            log_d[f"{prefix}/aux_mp_bce"] = mp_bce
            log_d[f"{prefix}/aux_mp_loss"] = mp_bce
        if (
            self.use_cqt_aux_head
            and "cqt_pred" in extra_aux
            and "cqt_target" in extra_aux
        ):
            cqt_loss = self._cqt_loss(
                extra_aux["cqt_pred"], extra_aux["cqt_target"]
            )
            loss = loss + self.cqt_aux_head_weight * cqt_loss
            log_d[f"{prefix}/aux_cqt_loss"] = cqt_loss
        if (
            self.use_target_token_future_aux_head
            and "tt_future_pred" in extra_aux
            and "tt_future_target" in extra_aux
            and "tt_future_logits_mask" in extra_aux
        ):
            # pred: [B, num_rvq, chunk, K_off, num_tokens]
            # tgt:  [B, num_rvq, chunk, K_off]   (long, mangled token ids)
            # mask: [B, num_rvq, chunk]          (valid prediction positions)
            pred = extra_aux["tt_future_pred"]
            tgt = extra_aux["tt_future_target"]
            mask = extra_aux["tt_future_logits_mask"]
            offsets = self.future_aux_offsets
            per_off_losses = []
            for k, delta in enumerate(offsets):
                pk = pred[:, :, :, k, :]            # [B, num_rvq, chunk, V]
                tk = tgt[:, :, :, k].long()         # [B, num_rvq, chunk]
                # Apply the same valid-position mask as the main CE loss.
                pk_valid = pk[mask]                 # [N_valid, V]
                tk_valid = tk[mask]                 # [N_valid]
                lk = F.cross_entropy(
                    pk_valid, tk_valid, ignore_index=self.pad_token,
                )
                per_off_losses.append(lk)
                log_d[f"{prefix}/aux_tt_future_d{int(delta)}"] = lk
            tt_future_total = (
                sum(per_off_losses) / len(per_off_losses)
                if per_off_losses
                else pred.new_zeros(())
            )
            loss = (
                loss + self.target_token_future_aux_head_weight * tt_future_total
            )
            log_d[f"{prefix}/aux_tt_future_loss"] = tt_future_total
        return loss, log_d

    def _get_inner_model(self):
        """Return the model, unwrapping torch.compile if present."""
        m = self.model
        return m._orig_mod if hasattr(m, "_orig_mod") else m

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        (
            input_emb,
            output_tokens,
            dec_inst_tokens,
            beat_cond,
            bpm_log,
            time_sig_num,
            time_sig_den,
            time_sig_change,
            tempo_change,
            local_bpm_log,
            target_multipitch,
            target_velocity,
            target_has_multipitch,
            target_cqt,
        ) = self.get_inputs(batch)
        (
            logits, logits_mask, targets,
            extra_aux,
        ) = self.model(
            x=output_tokens,
            input_emb=input_emb,
            inst_tokens=dec_inst_tokens,
            beat_cond=beat_cond,
            bpm_log=bpm_log,
            time_sig_num=time_sig_num,
            time_sig_den=time_sig_den,
            time_sig_change=time_sig_change,
            tempo_change=tempo_change,
            local_bpm_log=local_bpm_log,
            target_multipitch=target_multipitch,
            target_velocity=target_velocity,
            target_cqt=target_cqt,
        )
        loss = F.cross_entropy(
            logits[logits_mask],
            targets[logits_mask],
            ignore_index=self.pad_token,
        )

        # Calculate accuracy
        mask = targets != self.pad_token
        acc = (logits.argmax(dim=-1)[mask] == targets[mask]).float().mean()

        # Log validation loss (step-based logging)
        log_d = {"val/loss": loss, "val/acc": acc}
        # Aux losses (multipitch / CQT / target-token-future).
        _, log_d = self._add_extra_aux_losses(
            torch.tensor(0.0, device=loss.device), log_d, extra_aux,
            target_has_multipitch, prefix="val",
        )

        # DiT (per-layer) conditioning diagnostics. The "is the conditioner
        # alive?" signal here is whether the AdaptiveLayerNorm/Scale gamma
        # projections have moved off their zero-init. Both have
        # ``to_gamma`` weight matrices (zero-init by x_transformers).
        if self.use_beat_phase_dit_cond and batch_idx == 0:
            inner = self._get_inner_model()
            # Collect to_gamma weights from every adaptive norm + scale
            # in the decoder's layer stack. AttentionLayers stores layers
            # as ``layers``; each layer is a (norm_tuple, block, residual_fn)
            # tuple where norm_tuple = (pre_norm, post_branch_norm, post_main_norm).
            ada_ln_norms = []
            ada_scale_norms = []
            try:
                attn_layers = inner.decoder.attn_layers
                for (norm_tuple, block, residual_fn) in attn_layers.layers:
                    pre_norm = norm_tuple[0]
                    if hasattr(pre_norm, "to_gamma"):
                        ada_ln_norms.append(pre_norm.to_gamma.weight.detach().norm().item())
                    # post_branch_norm is the AdaptiveLayerScale wrapper
                    pbn = norm_tuple[1]
                    if pbn is not None and hasattr(pbn, "to_gamma"):
                        ada_scale_norms.append(pbn.to_gamma.weight.detach().norm().item())
            except Exception:
                pass
            if ada_ln_norms:
                t = torch.tensor(ada_ln_norms)
                log_d["val/dit_aln_gamma_w_mean"] = t.mean()
                log_d["val/dit_aln_gamma_w_max"] = t.max()
            if ada_scale_norms:
                t = torch.tensor(ada_scale_norms)
                log_d["val/dit_scale_gamma_w_mean"] = t.mean()
                log_d["val/dit_scale_gamma_w_max"] = t.max()
            # Projector MLP last-layer weight norm: nonzero by init, should
            # grow if conditioning is being learned.
            proj = getattr(inner, "beat_cond_projector", None)
            if proj is not None:
                log_d["val/dit_proj_mlp_out_w_norm"] = (
                    proj.mlp[-1].weight.detach().norm()
                )
        self._log_dict(log_d, batch_size=output_tokens.size(0))

        # Sample from model and log them
        # Only sample for the first validation batch
        if batch_idx == 0 and self.global_step % self.sample_interval == 0:
            self.sample_and_log(
                batch["input_audio"],
                batch["target_audio"],
                input_emb,
                output_tokens,
                dec_inst_tokens,
                batch["num_stems"],
                beat_cond=beat_cond,
                bpm_log=bpm_log,
                time_sig_num=time_sig_num,
                time_sig_den=time_sig_den,
                time_sig_change=time_sig_change,
                tempo_change=tempo_change,
                local_bpm_log=local_bpm_log,
            )

        return loss

    def log_audio(self, audio, name):
        if isinstance(audio, torch.Tensor):
            audio = audio.float().cpu().numpy()
        spectrogram = audio_to_spectrogram_image(audio, self.sample_rate)
        self.logger.experiment.log(
            {
                f"audio/{name}": wandb.Audio(
                    audio, sample_rate=self.sample_rate
                ),
                f"image/{name}": wandb.Image(spectrogram),
            }
        )

    def sample_and_log(
        self,
        input_audios: torch.Tensor,
        target_audios: torch.Tensor,
        input_emb: torch.Tensor,
        dec_inputs: torch.Tensor,
        dec_inst_tokens: torch.Tensor,
        num_stems: torch.Tensor,
        beat_cond: torch.Tensor = None,
        bpm_log: torch.Tensor = None,
        time_sig_num: torch.Tensor = None,
        time_sig_den: torch.Tensor = None,
        time_sig_change: torch.Tensor = None,
        tempo_change: torch.Tensor = None,
        local_bpm_log: torch.Tensor = None,
    ) -> None:
        # save memory
        torch.cuda.empty_cache()
        self.tokenizer = self.tokenizer.to(input_emb.device)

        # hard-code to use inst tokens
        print("Generating using Instrument Tokens:")
        print(dec_inst_tokens)
        curr_time = time.time()
        decoder_preds = self.model.generate(
            seq_len=self.max_gen_seq_len,
            seq_out_start=None,
            input_emb=input_emb,
            inst_tokens=dec_inst_tokens,
            cache_kv=True,
            filter_logits_fn=["top_k_multi_out"],  # Modified,
            filter_kwargs=[
                {"k": 200},
            ],
            beat_cond=beat_cond,
            bpm_log=bpm_log,
            time_sig_num=time_sig_num,
            time_sig_den=time_sig_den,
            time_sig_change=time_sig_change,
            tempo_change=tempo_change,
            local_bpm_log=local_bpm_log,
        )

        print(
            f"Time to generate: {time.time() - curr_time}, "
            f"shape: {decoder_preds.shape}"
        )

        # We don't log audio and image into table because there will be bugs
        table_text = wandb.Table(columns=["Index", "Array GT", "Array Gen"])
        for i in range(min(self.max_log_examples, decoder_preds.size(0))):
            array_gt_text = str(dec_inputs[i].cpu().numpy())
            array_gen_text = str(decoder_preds[i].cpu().numpy())
            table_text.add_data(i, array_gt_text, array_gen_text)

            input_audio = input_audios[i].cpu()
            target_audio = target_audios[i].cpu()
            num_input_stems = num_stems[i]

            input_audio = input_audio[: self.max_duration * self.sample_rate]
            target_audio = target_audio[: self.max_duration * self.sample_rate]

            if self.global_step == 0:
                mixed_audio_loud = mix_with_generated_stem(
                    input_audio.float().numpy(),
                    target_audio.float().numpy(),
                    num_input_stems,
                    self.sample_rate,
                    pred_db_offset=PRED_DB_OFFSET_LOUD,
                )
                self.log_audio(mixed_audio_loud, f"mixed_gt_loud_{i}")
                mixed_audio = mix_with_generated_stem(
                    input_audio.float().numpy(),
                    target_audio.float().numpy(),
                    num_input_stems,
                    self.sample_rate,
                )
                self.log_audio(mixed_audio, f"mixed_gt_{i}")

                # Only log ground truth audio for once
                self.log_audio(
                    loudness_normalize_audio(
                        input_audio.float().numpy(), self.sample_rate
                    ),
                    f"input_{i}",
                )
                self.log_audio(
                    loudness_normalize_audio(
                        target_audio.float().numpy(), self.sample_rate
                    ),
                    f"target_{i}",
                )

            pred_audio_tokens = self.tokenizer.post_process_tokens(
                decoder_preds[i]
            )
            pred_audio_single = self.tokenizer.tokens_to_audio(
                pred_audio_tokens.unsqueeze(0)
            ).cpu()
            if pred_audio_single.shape[0] < target_audio.shape[0]:
                pred_audio_single = F.pad(
                    pred_audio_single,
                    (0, target_audio.shape[0] - pred_audio_single.shape[0]),
                )
            self.log_audio(
                loudness_normalize_audio(
                    pred_audio_single.float().numpy(), self.sample_rate
                ),
                f"pred_{i}",
            )

            mixed_audio_loud = mix_with_generated_stem(
                input_audio.float().numpy(),
                pred_audio_single.float().numpy(),
                num_input_stems,
                self.sample_rate,
                pred_db_offset=PRED_DB_OFFSET_LOUD,
            )
            self.log_audio(mixed_audio_loud, f"mixed_pred_loud_{i}")
            mixed_audio = mix_with_generated_stem(
                input_audio.float().numpy(),
                pred_audio_single.float().numpy(),
                num_input_stems,
                self.sample_rate,
            )
            self.log_audio(mixed_audio, f"mixed_pred_{i}")

        self.logger.experiment.log({"array_text": table_text})

        # Free up GPU memory
        torch.cuda.empty_cache()
        self.tokenizer = self.tokenizer.to(torch.device("cpu"))

    def configure_optimizers(self):
        """
        Configures and returns the optimizer(s).
        """
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        scheduler = LinearWarmupCosineDecay(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
