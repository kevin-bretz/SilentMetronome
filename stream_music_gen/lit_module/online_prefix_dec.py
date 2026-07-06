"""
Lightning module for training the encoder-decoder generative
with delay pattern
"""

import argbind
import time
from pathlib import Path

import wandb
import torch
import torch.nn.functional as F
import numpy as np

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
        use_beat_phase: bool = False,
        time_sig_vocab_size: int = 16,
        use_beat_phase_dit_cond: bool = False,
        beat_dit_cond_dim: int = None,
        beat_dit_cond_mlp_expansion: int = 4,
        beat_dit_cond_minimal: bool = False,
        cond_dropout_p: float = 0.0,
        beat_phase_noise_std: float = 0.0,
        use_beat_phase_aux_head: bool = False,
        beat_phase_aux_head_hidden_dim: int = 256,
        beat_phase_aux_head_weight: float = 1.0,
        use_chroma_dit_cond: bool = False,
        chroma_dim: int = 12,
        chroma_dit_cond_hidden_dim: int = None,
        use_chroma_aux_head: bool = False,
        chroma_aux_head_hidden_dim: int = 256,
        chroma_aux_head_weight: float = 1.0,
        # New aux-strength knobs (default = legacy behavior).
        chroma_aux_head_linear: bool = False,
        chroma_aux_loss_type: str = "mse",  # "mse" | "cos" | "mse+cos"
        chroma_aux_horizons: list = None,    # default [0]
        chroma_aux_deep_supervision_layers: list = None,  # default empty
        chroma_aux_deep_supervision_weights: list = None,  # default equal
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
        # Input-mix CQT aux head (MSE+cos on 84-bin log-magnitude of the
        # input mix). Supervises the hidden state to retain the spectral
        # picture of the audio being conditioned on.
        use_input_cqt_aux_head: bool = False,
        input_cqt_aux_head_hidden_dim: int = 256,
        input_cqt_aux_head_weight: float = 1.0,
        input_cqt_dim: int = 84,
        # Beat-phase-full aux head: phase (MSE+cos) + bpm_log (MSE) + time_sig (CE).
        use_beat_phase_aux_head_full: bool = False,
        beat_phase_aux_head_full_hidden_dim: int = 256,
        beat_phase_aux_head_full_weight: float = 1.0,
        beat_phase_aux_full_bpm_weight: float = 0.1,
        beat_phase_aux_full_ts_weight: float = 0.1,
        # Future-mix aux heads (mp / cqt at t+δ from h_t). Require fv>0.
        use_multipitch_future_aux_head: bool = False,
        multipitch_future_aux_head_hidden_dim: int = 256,
        multipitch_future_aux_head_weight: float = 1.0,
        use_cqt_future_aux_head: bool = False,
        cqt_future_aux_head_hidden_dim: int = 256,
        cqt_future_aux_head_weight: float = 1.0,
        # Future target-stem token aux head (CE on patterned tokens).
        use_target_token_future_aux_head: bool = False,
        target_token_future_aux_head_hidden_dim: int = 256,
        target_token_future_aux_head_weight: float = 1.0,
        # Same task as above but sharing the main ``to_logits`` classifier.
        use_coupled_target_token_future_head: bool = False,
        coupled_target_token_future_head_hidden_dim: int = 256,
        coupled_target_token_future_head_weight: float = 1.0,
        future_aux_offsets: list = None,
        # KL knowledge distillation from a frozen teacher checkpoint. When
        # ``kd_teacher_ckpt`` is set, a second model (this lit module's
        # ``self.teacher``) is built with ``future_visibility=kd_teacher_fv``,
        # loaded from the ckpt, frozen, and used to provide soft targets for
        # the student via per-codebook KL on aligned target positions.
        kd_teacher_ckpt: str = "",
        kd_teacher_fv: int = 50,
        kd_beta: float = 0.5,
        kd_beta_ramp_iters: int = 10000,
        kd_temperature: float = 2.0,
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

        self.use_beat_phase = use_beat_phase
        self.use_beat_phase_dit_cond = use_beat_phase_dit_cond
        self.use_beat_phase_aux_head = use_beat_phase_aux_head
        self.beat_phase_aux_head_weight = float(beat_phase_aux_head_weight)
        self.use_chroma_dit_cond = use_chroma_dit_cond
        self.use_chroma_aux_head = use_chroma_aux_head
        self.chroma_aux_head_weight = float(chroma_aux_head_weight)
        self.chroma_aux_head_linear = bool(chroma_aux_head_linear)
        self.chroma_aux_loss_type = str(chroma_aux_loss_type)
        if self.chroma_aux_loss_type not in ("mse", "cos", "mse+cos"):
            raise ValueError(
                f"chroma_aux_loss_type must be one of "
                f"'mse', 'cos', 'mse+cos'; got '{self.chroma_aux_loss_type}'"
            )
        self.chroma_aux_horizons = tuple(
            int(h) for h in (chroma_aux_horizons or (0,))
        )
        self.chroma_aux_deep_supervision_layers = tuple(
            int(li) for li in (chroma_aux_deep_supervision_layers or ())
        )
        if chroma_aux_deep_supervision_weights:
            self.chroma_aux_deep_supervision_weights = tuple(
                float(w) for w in chroma_aux_deep_supervision_weights
            )
            if len(self.chroma_aux_deep_supervision_weights) != \
               len(self.chroma_aux_deep_supervision_layers):
                raise ValueError(
                    "chroma_aux_deep_supervision_weights length must match "
                    "chroma_aux_deep_supervision_layers length"
                )
        else:
            # Default: equal weight 1.0 per DSV layer.
            self.chroma_aux_deep_supervision_weights = (
                1.0,
            ) * len(self.chroma_aux_deep_supervision_layers)
        # New aux head flags + weights.
        self.use_multipitch_aux_head = bool(use_multipitch_aux_head)
        self.multipitch_aux_head_weight = float(multipitch_aux_head_weight)
        self.multipitch_dim = int(multipitch_dim)
        self.use_cqt_aux_head = bool(use_cqt_aux_head)
        self.cqt_aux_head_weight = float(cqt_aux_head_weight)
        self.cqt_dim = int(cqt_dim)
        self.use_input_cqt_aux_head = bool(use_input_cqt_aux_head)
        self.input_cqt_aux_head_weight = float(input_cqt_aux_head_weight)
        self.input_cqt_dim = int(input_cqt_dim)
        self.use_beat_phase_aux_head_full = bool(use_beat_phase_aux_head_full)
        self.beat_phase_aux_head_full_weight = float(
            beat_phase_aux_head_full_weight
        )
        self.beat_phase_aux_full_bpm_weight = float(
            beat_phase_aux_full_bpm_weight
        )
        self.beat_phase_aux_full_ts_weight = float(
            beat_phase_aux_full_ts_weight
        )
        self.beat_phase_aux_full_ts_vocab = int(time_sig_vocab_size)
        # Future-mix aux head flags + weights.
        self.use_multipitch_future_aux_head = bool(use_multipitch_future_aux_head)
        self.multipitch_future_aux_head_weight = float(
            multipitch_future_aux_head_weight
        )
        self.use_cqt_future_aux_head = bool(use_cqt_future_aux_head)
        self.cqt_future_aux_head_weight = float(cqt_future_aux_head_weight)
        self.use_target_token_future_aux_head = bool(
            use_target_token_future_aux_head
        )
        self.target_token_future_aux_head_weight = float(
            target_token_future_aux_head_weight
        )
        self.use_coupled_target_token_future_head = bool(
            use_coupled_target_token_future_head
        )
        self.coupled_target_token_future_head_weight = float(
            coupled_target_token_future_head_weight
        )
        self.future_aux_offsets = tuple(
            int(d) for d in (future_aux_offsets or (10, 25, 40))
        )
        # Capture explicit kwargs so we can re-build a matching teacher when
        # KD is enabled (teacher differs from student only in
        # ``future_visibility``; everything else must be identical for the
        # state_dict to load).
        model_kwargs = dict(
            num_tokens=self.num_tokens,  # removed +2, no bos/eos
            max_seq_len=self.max_gen_seq_len + add_inst_tokens + 1,
            pad_value=self.pad_token,
            num_rvq_layers=self.num_rvq_layers,  # Added
            shared=True,
            online=True,  # Added
            inst_tokens_as_pattern_token=True,
            input_emb_dim=tokenizer.emb_dim,
            use_beat_phase=use_beat_phase,
            time_sig_vocab_size=time_sig_vocab_size,
            use_beat_phase_dit_cond=use_beat_phase_dit_cond,
            beat_dit_cond_dim=beat_dit_cond_dim,
            beat_dit_cond_mlp_expansion=beat_dit_cond_mlp_expansion,
            beat_dit_cond_minimal=beat_dit_cond_minimal,
            cond_dropout_p=cond_dropout_p,
            beat_phase_noise_std=beat_phase_noise_std,
            use_beat_phase_aux_head=use_beat_phase_aux_head,
            beat_phase_aux_head_hidden_dim=beat_phase_aux_head_hidden_dim,
            use_chroma_dit_cond=use_chroma_dit_cond,
            chroma_dim=chroma_dim,
            chroma_dit_cond_hidden_dim=chroma_dit_cond_hidden_dim,
            use_chroma_aux_head=use_chroma_aux_head,
            chroma_aux_head_hidden_dim=chroma_aux_head_hidden_dim,
            chroma_aux_head_linear=self.chroma_aux_head_linear,
            chroma_aux_horizons=list(self.chroma_aux_horizons),
            chroma_aux_deep_supervision_layers=list(
                self.chroma_aux_deep_supervision_layers
            ),
            use_multipitch_aux_head=self.use_multipitch_aux_head,
            multipitch_aux_head_hidden_dim=multipitch_aux_head_hidden_dim,
            multipitch_dim=self.multipitch_dim,
            use_cqt_aux_head=self.use_cqt_aux_head,
            cqt_aux_head_hidden_dim=cqt_aux_head_hidden_dim,
            cqt_dim=self.cqt_dim,
            use_input_cqt_aux_head=self.use_input_cqt_aux_head,
            input_cqt_aux_head_hidden_dim=input_cqt_aux_head_hidden_dim,
            input_cqt_dim=self.input_cqt_dim,
            use_beat_phase_aux_head_full=self.use_beat_phase_aux_head_full,
            beat_phase_aux_head_full_hidden_dim=beat_phase_aux_head_full_hidden_dim,
            use_multipitch_future_aux_head=self.use_multipitch_future_aux_head,
            multipitch_future_aux_head_hidden_dim=multipitch_future_aux_head_hidden_dim,
            use_cqt_future_aux_head=self.use_cqt_future_aux_head,
            cqt_future_aux_head_hidden_dim=cqt_future_aux_head_hidden_dim,
            use_target_token_future_aux_head=self.use_target_token_future_aux_head,
            target_token_future_aux_head_hidden_dim=target_token_future_aux_head_hidden_dim,
            use_coupled_target_token_future_head=self.use_coupled_target_token_future_head,
            coupled_target_token_future_head_hidden_dim=coupled_target_token_future_head_hidden_dim,
            future_aux_offsets=list(self.future_aux_offsets),
        )
        self.model = OnlinePrefixDecoderTransformerMultiOut(**model_kwargs)

        # KL knowledge distillation setup. The teacher mirrors the student
        # architecturally except for ``future_visibility``, and is built
        # before ``compile`` so neither model is compiled when the teacher
        # loads its state_dict (avoids ``_orig_mod.`` reconciliation).
        self.kd_teacher_ckpt = str(kd_teacher_ckpt or "")
        self.kd_teacher_fv = int(kd_teacher_fv)
        self.kd_beta = float(kd_beta)
        self.kd_beta_ramp_iters = int(kd_beta_ramp_iters)
        self.kd_temperature = float(kd_temperature)
        self.kd_active = bool(self.kd_teacher_ckpt)
        if self.kd_active:
            teacher_kwargs = dict(model_kwargs)
            teacher_kwargs["future_visibility"] = self.kd_teacher_fv
            self.teacher = OnlinePrefixDecoderTransformerMultiOut(
                **teacher_kwargs
            )
            self._load_teacher_state_dict(self.kd_teacher_ckpt)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher.eval()
            # Cache for context_length geometry. Student predicts real target
            # tokens [c_s, c_s + chunk); teacher with same target window needs
            # c_t = c_s - 1 (see ``_compute_kd_context_length``).
            self._kd_student_fv = int(self.model.future_visibility)
            assert self._kd_student_fv <= 0 and self.kd_teacher_fv > 0, (
                "KD path assumes student fv <= 0 and teacher fv > 0; got "
                f"student_fv={self._kd_student_fv} teacher_fv={self.kd_teacher_fv}"
            )
            self._kd_logged_alignment = False

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

    def _load_teacher_state_dict(self, ckpt_path: str) -> None:
        """Load Lightning ckpt weights into ``self.teacher``. Strips the
        ``model.`` and ``model._orig_mod.`` prefixes so the inner-model state
        keys match. Hard-fails on any *unexpected* key (signals architecture
        mismatch); only tolerates missing keys (e.g., aux heads disabled in
        teacher arch).
        """
        ckpt_p = Path(ckpt_path)
        if not ckpt_p.is_file():
            raise FileNotFoundError(
                f"kd_teacher_ckpt not found: {ckpt_path}"
            )
        raw = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
        state_dict = raw.get("state_dict", raw)

        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model._orig_mod."):
                new_sd[k[len("model._orig_mod."):]] = v
            elif k.startswith("model."):
                new_sd[k[len("model."):]] = v
        if not new_sd:
            raise ValueError(
                f"No keys with 'model.' prefix in teacher ckpt {ckpt_path}"
            )

        missing, unexpected = self.teacher.load_state_dict(
            new_sd, strict=False
        )
        print(
            f"[kd-teacher] loaded {len(new_sd)} keys from {ckpt_path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if unexpected:
            print("  unexpected (first 10):")
            for k in unexpected[:10]:
                print(f"    {k}")
            raise ValueError(
                "Teacher state_dict has unexpected keys, teacher arch does "
                "not match the checkpoint. Aborting to avoid silent corruption."
            )
        if missing:
            print("  missing (first 10):")
            for k in missing[:10]:
                print(f"    {k}")

    def train(self, mode: bool = True):
        """Keep the teacher in eval mode regardless of the lit module's
        train/eval cycle. Lightning toggles ``.train()`` per epoch; without
        this override, teacher dropout would re-activate at the start of each
        epoch and shift its soft-target distribution mid-training.
        """
        super().train(mode)
        if getattr(self, "kd_active", False):
            self.teacher.eval()
        return self

    def _compute_kd_context_length(self, T: int) -> tuple:
        """Sample matched (student_c, teacher_c) such that both predict the
        same real target tokens.

        Both models share the same patterned sequence
        ``x_patterned = pattern.build(output_tokens)`` of length L_s. The
        student (fv<=0) leaves it as-is and slices targets at
        ``x_patterned[c_s+1 : c_s+1+chunk]``. The teacher (fv>0) first
        front-pads with ``fv_t - 1`` pad frames then drops the last
        ``fv_t``; its targets at ``c_t + fv_t`` map back to the **same**
        underlying x_patterned positions ``[c_t+1 : c_t+1+chunk]``. So
        position-equal targets require ``c_t = c_s``.

        Both ``c_s`` and ``c_t`` are constrained to chunk-stride values in
        the intersection of student/teacher valid ranges:
            c in {0, chunk, 2*chunk, ...} ∩ [0, T - fv_t - chunk]

        (Student range is wider: [0, T - chunk]. Teacher is the tighter
        constraint via ``max_duration_gen = T - fv_t``.)

        Returns (student_c, teacher_c) with student_c == teacher_c.
        """
        inner = self._get_inner_model()
        chunk = inner.chunk_length
        fv_t = self.kd_teacher_fv
        teacher_max = T - fv_t  # inclusive upper bound for c_t + chunk
        c_max = teacher_max - chunk  # last valid chunk-stride context
        if c_max < 0:
            raise RuntimeError(
                f"No valid KD context_length: T={T} fv_t={fv_t} chunk={chunk}"
            )
        valid = np.arange(0, c_max + 1, chunk)
        if len(valid) == 0:
            raise RuntimeError(
                f"No valid KD context_length: T={T} fv_t={fv_t} chunk={chunk}"
            )
        c = int(np.random.choice(valid))
        return c, c

    def _kd_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        logits_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position KL(teacher || student) averaged over valid mask
        positions and summed over RVQ codebooks, with Hinton-style
        temperature scaling.

        student_logits / teacher_logits: ``[B, K, S, V]``
        logits_mask:                     ``[B, K, S]`` (bool)
        """
        T = max(1e-3, float(self.kd_temperature))
        # cast to fp32 for numerically stable softmax with bf16 forward.
        s_log_probs = F.log_softmax(student_logits.float() / T, dim=-1)
        t_probs = F.softmax(teacher_logits.float() / T, dim=-1)
        # KL(t || s) = sum_v t * (log t - log s)
        kl_per_vocab = t_probs * (
            torch.log(t_probs.clamp_min(1e-12)) - s_log_probs
        )
        # Sum over vocab; keep [B, K, S].
        kl_per_position = kl_per_vocab.sum(dim=-1)
        # Mask and mean. Hinton: scale by T^2 so the gradient magnitude is
        # comparable to MLE when T -> 1.
        valid = kl_per_position[logits_mask]
        if valid.numel() == 0:
            return student_logits.new_zeros((), dtype=torch.float32)
        return valid.mean() * (T * T)

    def _kd_beta_scheduled(self) -> float:
        if self.kd_beta_ramp_iters <= 0:
            return self.kd_beta
        ramp = min(1.0, float(self.global_step) / float(self.kd_beta_ramp_iters))
        return self.kd_beta * ramp

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
        target_chroma = batch.get("target_chroma", None)
        target_has_chroma = batch.get("target_has_chroma", None)
        input_chroma = batch.get("input_chroma", None)
        target_multipitch = batch.get("target_multipitch", None)
        target_velocity = batch.get("target_velocity", None)
        target_has_multipitch = batch.get("target_has_multipitch", None)
        target_cqt = batch.get("target_cqt", None)
        input_cqt = batch.get("input_cqt", None)

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
            target_chroma,
            target_has_chroma,
            input_chroma,
            target_multipitch,
            target_velocity,
            target_has_multipitch,
            target_cqt,
            input_cqt,
        )

    def _masked_chroma_mse(self, pred, target, has_mask):
        """MSE between pred and target, summed only over windows where
        ``has_mask`` is 1.0 (drum-stem windows have target_has_chroma=0
        and contribute zero gradient). Returns scalar tensor.
        """
        if has_mask is None:
            return F.mse_loss(pred, target.to(pred.dtype))
        # has_mask: [B] -> [B, 1, 1]
        w = has_mask.to(pred.dtype).view(-1, 1, 1)
        denom = w.sum().clamp(min=1.0) * pred.shape[1] * pred.shape[2]
        sq_err = (pred - target.to(pred.dtype)).pow(2) * w
        return sq_err.sum() / denom

    def _masked_chroma_cosine(self, pred, target, has_mask):
        """``1 - cos_sim(pred, target)`` averaged over frames with
        ``has_mask`` 1.0. For multi-horizon predictions where
        last-dim = K * out_dim, we reshape to [B, T, K, out_dim] and
        compute cosine per (frame, horizon), then mean across K.
        """
        out_dim = getattr(self._get_inner_model(), "chroma_dim", 12)
        pred = pred.float()
        target = target.to(pred.dtype)
        B, T, total = pred.shape
        K = max(1, total // out_dim)
        if total != K * out_dim:
            # Defensive fallback to plain dot-product cosine.
            cos = F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
        else:
            pr = pred.reshape(B, T, K, out_dim)
            tg = target.reshape(B, T, K, out_dim)
            cos_kt = F.cosine_similarity(pr, tg, dim=-1, eps=1e-8)  # [B, T, K]
            cos = cos_kt.mean(dim=-1)  # [B, T]
        one_minus_cos = 1.0 - cos
        if has_mask is None:
            return one_minus_cos.mean()
        w = has_mask.to(pred.dtype).view(-1, 1)
        denom = w.sum().clamp(min=1.0) * T
        return (one_minus_cos * w).sum() / denom

    def _masked_chroma_loss(self, pred, target, has_mask):
        """Dispatch between MSE / cosine / hybrid per ``chroma_aux_loss_type``.
        Default ``"mse"`` returns the legacy ``_masked_chroma_mse``.
        """
        loss_type = self.chroma_aux_loss_type
        if loss_type == "mse":
            return self._masked_chroma_mse(pred, target, has_mask)
        if loss_type == "cos":
            return self._masked_chroma_cosine(pred, target, has_mask)
        if loss_type == "mse+cos":
            return 0.5 * self._masked_chroma_mse(pred, target, has_mask) \
                 + 0.5 * self._masked_chroma_cosine(pred, target, has_mask)
        raise ValueError(f"unknown chroma_aux_loss_type: {loss_type}")

    def _aggregate_chroma_aux_loss(
        self, pred, dsv_preds, target, has_mask, log_d, prefix
    ):
        """Compute chroma aux loss including deep-supervision aggregation.
        Logs per-layer DSV losses under ``{prefix}/aux_chroma_dsv_{li}_loss``.
        Returns a scalar loss tensor.
        """
        main_loss = self._masked_chroma_loss(pred, target, has_mask)
        if not dsv_preds:
            return main_loss
        layers = self.chroma_aux_deep_supervision_layers
        weights = self.chroma_aux_deep_supervision_weights
        total = main_loss
        sum_w = 1.0
        for li, w in zip(layers, weights):
            li_loss = self._masked_chroma_loss(
                dsv_preds[int(li)], target, has_mask
            )
            log_d[f"{prefix}/aux_chroma_dsv_{int(li)}_loss"] = li_loss
            total = total + float(w) * li_loss
            sum_w += float(w)
        # Normalize so DSV doesn't inflate the aux loss magnitude beyond
        # the no-DSV case (keeps ``chroma_aux_head_weight`` semantics stable).
        return total / sum_w

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

    def _beat_phase_full_loss(
        self, beat_full_pred, phase_target, bpm_log, time_sig_num
    ):
        """Phase: MSE+cos on 4-d sin/cos. Bpm: MSE on per-frame broadcast of
        window scalar. Time sig: CE on per-frame broadcast of window int.
        Returns (phase_loss, bpm_loss, ts_loss).
        """
        phase_pred = beat_full_pred[..., :4]
        bpm_pred = beat_full_pred[..., 4:5].squeeze(-1)
        ts_logits = beat_full_pred[..., 5:]
        phase_target = phase_target.to(phase_pred.dtype)
        # Phase MSE+cos.
        mse = F.mse_loss(phase_pred, phase_target)
        cos = 1.0 - F.cosine_similarity(
            phase_pred.float(), phase_target.float(), dim=-1, eps=1e-8
        ).mean()
        phase_loss = 0.5 * mse + 0.5 * cos
        # Bpm MSE, broadcasting bpm_log [B] -> [B, S].
        if bpm_log is None:
            bpm_loss = bpm_pred.new_zeros(())
        else:
            bpm_target = bpm_log.to(bpm_pred.dtype).unsqueeze(-1).expand_as(bpm_pred)
            bpm_loss = F.mse_loss(bpm_pred, bpm_target)
        # Time-sig CE, broadcasting time_sig_num [B] -> [B, S].
        if time_sig_num is None:
            ts_loss = ts_logits.new_zeros(())
        else:
            ts_target = time_sig_num.long().clamp(
                0, self.beat_phase_aux_full_ts_vocab - 1
            )
            B, S, K = ts_logits.shape
            ts_target = ts_target.unsqueeze(-1).expand(B, S)
            ts_loss = F.cross_entropy(
                ts_logits.reshape(B * S, K), ts_target.reshape(B * S)
            )
        return phase_loss, bpm_loss, ts_loss

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
            target_chroma,
            target_has_chroma,
            input_chroma,
            target_multipitch,
            target_velocity,
            target_has_multipitch,
            target_cqt,
            input_cqt,
        ) = self.get_inputs(batch)

        # KD: sample matched student/teacher context_lengths so both predict
        # the same real target tokens. Without KD, ``student_c=None`` lets
        # the model pick a random chunk-aligned context (the existing path).
        student_c = None
        teacher_c = None
        if self.kd_active:
            T_tokens = output_tokens.shape[-1]
            student_c, teacher_c = self._compute_kd_context_length(T_tokens)

        # In x-transformers, mask is True for unmasked tokens
        (
            logits, logits_mask, targets,
            beat_aux_pred, beat_aux_target,
            chroma_aux_pred, chroma_aux_target,
            chroma_aux_dsv_preds,
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
            input_chroma=input_chroma,
            target_chroma=target_chroma,
            target_multipitch=target_multipitch,
            target_velocity=target_velocity,
            target_cqt=target_cqt,
            input_cqt=input_cqt,
            context_length_override=student_c,
        )

        ce_loss = F.cross_entropy(
            logits[logits_mask],  #  For delay pattern, only
            targets[logits_mask],  # compute mask on valid logits.
            ignore_index=self.pad_token,
        )

        log_d = {"train/loss": ce_loss}
        loss = ce_loss
        if (
            self.use_beat_phase_aux_head
            and beat_aux_pred is not None
            and beat_aux_target is not None
        ):
            aux_loss = F.mse_loss(
                beat_aux_pred, beat_aux_target.to(beat_aux_pred.dtype)
            )
            loss = loss + self.beat_phase_aux_head_weight * aux_loss
            log_d["train/aux_beat_loss"] = aux_loss
        if (
            self.use_chroma_aux_head
            and chroma_aux_pred is not None
            and chroma_aux_target is not None
        ):
            chroma_aux_loss = self._aggregate_chroma_aux_loss(
                chroma_aux_pred,
                chroma_aux_dsv_preds,
                chroma_aux_target,
                target_has_chroma,
                log_d,
                prefix="train",
            )
            loss = loss + self.chroma_aux_head_weight * chroma_aux_loss
            log_d["train/aux_chroma_loss"] = chroma_aux_loss
        loss, log_d = self._add_extra_aux_losses(
            loss, log_d, extra_aux,
            target_has_multipitch, bpm_log, time_sig_num, prefix="train",
        )

        if self.kd_active:
            # Teacher forward (frozen, no_grad). Lightning's bf16-mixed
            # autocast context wraps training_step, so teacher matmuls run
            # in bf16 like the student's. ``self.teacher.eval()`` is enforced
            # via our ``train()`` override so dropout stays off.
            with torch.no_grad():
                (
                    t_logits, t_logits_mask, t_targets,
                    _t_beat_aux_pred, _t_beat_aux_target,
                    _t_chroma_aux_pred, _t_chroma_aux_target,
                    _t_chroma_aux_dsv_preds,
                    _t_extra_aux,
                ) = self.teacher(
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
                    input_chroma=input_chroma,
                    target_chroma=target_chroma,
                    target_multipitch=target_multipitch,
                    target_velocity=target_velocity,
                    target_cqt=target_cqt,
                    input_cqt=input_cqt,
                    context_length_override=teacher_c,
                )
            # Sanity-check target alignment once. ``targets`` and
            # ``t_targets`` should be element-wise equal: same RVQ-patterned
            # ids at the same prediction positions. If this trips, the
            # context_length geometry in ``_compute_kd_context_length`` is
            # off and KL would be comparing predictions for different tokens.
            if not self._kd_logged_alignment:
                if not torch.equal(targets, t_targets):
                    mismatch = (targets != t_targets).float().mean().item()
                    raise RuntimeError(
                        f"KD target misalignment ({mismatch:.4f} of positions "
                        f"disagree). student_c={student_c} teacher_c={teacher_c} "
                        f"student_shape={tuple(targets.shape)} "
                        f"teacher_shape={tuple(t_targets.shape)}"
                    )
                if not torch.equal(logits_mask, t_logits_mask):
                    raise RuntimeError(
                        "KD logits_mask misalignment between student and teacher"
                    )
                print(
                    f"[kd] alignment OK at step {int(self.global_step)}; "
                    f"student_c={student_c} teacher_c={teacher_c} "
                    f"chunk_shape={tuple(targets.shape)}"
                )
                self._kd_logged_alignment = True

            kd_loss = self._kd_kl_loss(logits, t_logits, logits_mask)
            kd_beta_now = self._kd_beta_scheduled()
            loss = loss + kd_beta_now * kd_loss
            log_d["train/kd_loss"] = kd_loss
            log_d["train/kd_beta"] = torch.tensor(
                kd_beta_now, device=loss.device, dtype=loss.dtype
            )

        # Log training loss (W&B logging included through self.log)
        self._log_dict(log_d, batch_size=output_tokens.size(0))
        return loss

    def _add_extra_aux_losses(
        self, loss, log_d, extra_aux,
        target_has_multipitch, bpm_log, time_sig_num, prefix: str,
    ):
        """Add multipitch / CQT / beat-phase-full losses if present in
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
            self.use_input_cqt_aux_head
            and "input_cqt_pred" in extra_aux
            and "input_cqt_target" in extra_aux
        ):
            input_cqt_loss = self._cqt_loss(
                extra_aux["input_cqt_pred"], extra_aux["input_cqt_target"]
            )
            loss = loss + self.input_cqt_aux_head_weight * input_cqt_loss
            log_d[f"{prefix}/aux_input_cqt_loss"] = input_cqt_loss
        if (
            self.use_multipitch_future_aux_head
            and "mp_future_pred" in extra_aux
            and "mp_future_target" in extra_aux
        ):
            pred = extra_aux["mp_future_pred"]      # [B, T, K, D]
            tgt = extra_aux["mp_future_target"]     # [B, T, K, D]
            offsets = self.future_aux_offsets
            per_off_losses = []
            for k, delta in enumerate(offsets):
                pk = pred[:, :, k, :]
                tk = tgt[:, :, k, :]
                lk = self._multipitch_loss(pk, tk, target_has_multipitch)
                per_off_losses.append(lk)
                log_d[f"{prefix}/aux_mp_future_d{int(delta)}"] = lk
            mp_future_total = (
                sum(per_off_losses) / len(per_off_losses)
                if per_off_losses
                else pred.new_zeros(())
            )
            loss = loss + self.multipitch_future_aux_head_weight * mp_future_total
            log_d[f"{prefix}/aux_mp_future_loss"] = mp_future_total
        if (
            self.use_cqt_future_aux_head
            and "cqt_future_pred" in extra_aux
            and "cqt_future_target" in extra_aux
        ):
            pred = extra_aux["cqt_future_pred"]      # [B, T, K, D]
            tgt = extra_aux["cqt_future_target"]     # [B, T, K, D]
            offsets = self.future_aux_offsets
            per_off_losses = []
            for k, delta in enumerate(offsets):
                pk = pred[:, :, k, :]
                tk = tgt[:, :, k, :]
                lk = self._cqt_loss(pk, tk)
                per_off_losses.append(lk)
                log_d[f"{prefix}/aux_cqt_future_d{int(delta)}"] = lk
            cqt_future_total = (
                sum(per_off_losses) / len(per_off_losses)
                if per_off_losses
                else pred.new_zeros(())
            )
            loss = loss + self.cqt_future_aux_head_weight * cqt_future_total
            log_d[f"{prefix}/aux_cqt_future_loss"] = cqt_future_total
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
        if (
            self.use_coupled_target_token_future_head
            and "coupled_tt_future_pred" in extra_aux
            and "coupled_tt_future_target" in extra_aux
            and "coupled_tt_future_logits_mask" in extra_aux
        ):
            # Same shapes/contract as the block above. The only difference is
            # that the prediction came through the shared main ``to_logits``
            # instead of a dedicated classifier.
            pred = extra_aux["coupled_tt_future_pred"]
            tgt = extra_aux["coupled_tt_future_target"]
            mask = extra_aux["coupled_tt_future_logits_mask"]
            offsets = self.future_aux_offsets
            per_off_losses = []
            for k, delta in enumerate(offsets):
                pk = pred[:, :, :, k, :]
                tk = tgt[:, :, :, k].long()
                pk_valid = pk[mask]
                tk_valid = tk[mask]
                lk = F.cross_entropy(
                    pk_valid, tk_valid, ignore_index=self.pad_token,
                )
                per_off_losses.append(lk)
                log_d[f"{prefix}/aux_coupled_tt_future_d{int(delta)}"] = lk
            coupled_tt_future_total = (
                sum(per_off_losses) / len(per_off_losses)
                if per_off_losses
                else pred.new_zeros(())
            )
            loss = (
                loss
                + self.coupled_target_token_future_head_weight
                * coupled_tt_future_total
            )
            log_d[f"{prefix}/aux_coupled_tt_future_loss"] = (
                coupled_tt_future_total
            )
        if (
            self.use_beat_phase_aux_head_full
            and "beat_full_pred" in extra_aux
            and "beat_full_phase_target" in extra_aux
        ):
            phase_loss, bpm_loss, ts_loss = self._beat_phase_full_loss(
                extra_aux["beat_full_pred"],
                extra_aux["beat_full_phase_target"],
                bpm_log,
                time_sig_num,
            )
            beat_full_total = (
                phase_loss
                + self.beat_phase_aux_full_bpm_weight * bpm_loss
                + self.beat_phase_aux_full_ts_weight * ts_loss
            )
            loss = loss + self.beat_phase_aux_head_full_weight * beat_full_total
            log_d[f"{prefix}/aux_beat_full_phase"] = phase_loss
            log_d[f"{prefix}/aux_beat_full_bpm"] = bpm_loss
            log_d[f"{prefix}/aux_beat_full_ts"] = ts_loss
            log_d[f"{prefix}/aux_beat_full_loss"] = beat_full_total
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
            target_chroma,
            target_has_chroma,
            input_chroma,
            target_multipitch,
            target_velocity,
            target_has_multipitch,
            target_cqt,
            input_cqt,
        ) = self.get_inputs(batch)
        (
            logits, logits_mask, targets,
            beat_aux_pred, beat_aux_target,
            chroma_aux_pred, chroma_aux_target,
            chroma_aux_dsv_preds,
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
            input_chroma=input_chroma,
            target_chroma=target_chroma,
            target_multipitch=target_multipitch,
            target_velocity=target_velocity,
            target_cqt=target_cqt,
            input_cqt=input_cqt,
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
        if (
            self.use_beat_phase_aux_head
            and beat_aux_pred is not None
            and beat_aux_target is not None
        ):
            aux_loss = F.mse_loss(
                beat_aux_pred, beat_aux_target.to(beat_aux_pred.dtype)
            )
            log_d["val/aux_beat_loss"] = aux_loss
        if (
            self.use_chroma_aux_head
            and chroma_aux_pred is not None
            and chroma_aux_target is not None
        ):
            chroma_aux_loss = self._aggregate_chroma_aux_loss(
                chroma_aux_pred,
                chroma_aux_dsv_preds,
                chroma_aux_target,
                target_has_chroma,
                log_d,
                prefix="val",
            )
            log_d["val/aux_chroma_loss"] = chroma_aux_loss
        # New aux losses (multipitch / CQT / beat-phase-full).
        _, log_d = self._add_extra_aux_losses(
            torch.tensor(0.0, device=loss.device), log_d, extra_aux,
            target_has_multipitch, bpm_log, time_sig_num, prefix="val",
        )
        # Beat-phase conditioner diagnostics, logged once per validation
        # epoch. If `beat_gate_abs_max` stays near 0 the conditioner is dead,
        # and if `beat_mlp_out_w_norm` stays at its init value the conditioner
        # MLP is not learning.
        if self.use_beat_phase and batch_idx == 0:
            bc = self._get_inner_model().beat_conditioner
            gate = bc.gate.detach()
            mlp_out_w = bc.mlp[-1].weight.detach()
            log_d["val/beat_gate_abs_mean"] = gate.abs().mean()
            log_d["val/beat_gate_abs_max"] = gate.abs().max()
            log_d["val/beat_mlp_out_w_norm"] = mlp_out_w.norm()

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
                input_chroma=input_chroma,
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
        input_chroma: torch.Tensor = None,
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
            input_chroma=input_chroma,
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
