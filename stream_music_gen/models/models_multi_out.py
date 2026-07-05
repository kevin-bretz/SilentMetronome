"""Models for online stem gen. with multiple tokens per output step"""

from typing import Tuple, Callable, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from x_transformers import TransformerWrapper, AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    join,
)
from tqdm import tqdm

from stream_music_gen.nn.transformers import Decoder, Encoder
from stream_music_gen.models.sampling import (
    top_k_multi_out,
    FILTER_LOGITS_FN,
    ComposeFilterFns,
    validate_filter_fn_kwargs,
)

# from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from stream_music_gen.models.patterns import DelayedPatternProvider
from stream_music_gen.models.dit_precompute import precompute_dit_gammas
from stream_music_gen.models.models_transformer import (
    TransformerWrapperNoInitEmb,
)


from x_transformers.x_transformers import ScaledSinusoidalEmbedding


class BaseGenerationMixin:
    """
    Base mixin class providing common generation functionality for multi-RVQ transformer models.

    This mixin contains shared logic for parameter validation, pattern processing,
    sampling, and reconstruction that can be used by different transformer architectures.
    """

    def _validate_generation_params(
        self,
        inst_tokens: Optional[torch.Tensor],
        seq_out_start: Optional[torch.Tensor],
        input_emb: Optional[torch.Tensor],
        guidance_scale: float = 1.0,
    ) -> Tuple[int, torch.device]:
        """Validate generation parameters and return batch_size and device."""
        # Model still requires some conditioning
        assert inst_tokens is not None or seq_out_start is not None
        assert (
            input_emb is not None
        ), "input_emb must be provided for generation"

        batch_size = (
            inst_tokens.shape[0]
            if inst_tokens is not None
            else seq_out_start.shape[0]
        )

        assert (
            input_emb.shape[0] == batch_size
        ), f"Batch size mismatch: input_emb has {input_emb.shape[0]} but expected {batch_size}"

        # if using CFG, we expect the batch to be doubled
        if guidance_scale != 1.0:
            assert batch_size % 2 == 0, (
                "For classifier-free guidance, prompts (and corresponding context) "
                "must have an even batch size (first half conditioned, second half null)."
            )

        device = next(self.parameters()).device
        return batch_size, device

    def _process_filter_functions(
        self, filter_logits_fn, filter_kwargs
    ) -> Tuple[Callable, dict]:
        """Process and validate filter functions."""
        # Process filter logits function
        if isinstance(filter_logits_fn, str):
            assert (
                filter_logits_fn in FILTER_LOGITS_FN
            ), f"only {join(FILTER_LOGITS_FN.keys())} are available"
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        filter_fns_is_list = validate_filter_fn_kwargs(
            filter_logits_fn, filter_kwargs
        )
        if filter_fns_is_list:
            filter_fns = ComposeFilterFns(filter_logits_fn, filter_kwargs)
            filter_logits_fn = filter_fns
            filter_kwargs = dict()

        return filter_logits_fn, filter_kwargs

    def _initialize_generation_pattern(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        seq_out_start: Optional[torch.Tensor],
        inst_tokens: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        int,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Initialize pattern and output sequence for generation."""
        pattern = self.pattern_provider.get_pattern(seq_len)

        # If prompt is not None, keep overwriting current generation with it.
        if seq_out_start is not None:
            prompt_length = seq_out_start.shape[-1]
            patterned_seq_out, _, seq_out_mask = pattern.build_pattern_sequence(
                seq_out_start,
                special_token=self.temp_token,
                keep_only_valid_steps=True,
            )

            # Use inst_tokens as pattern tokens in patterned prompts
            patterned_seq_out = self._replace_temp_token_with_inst_tokens(
                patterned_seq_out, inst_tokens
            )

            # Make out as prompt. +1 for BOS (instrument token)
            out = patterned_seq_out[:, :, : prompt_length + 1]
        else:
            # Use inst_tokens to prompt output
            out = inst_tokens.view(batch_size, 1, 1).expand(
                batch_size, self.num_rvq_layers, 1
            )
            # In here we do not count the BOS token (instrument token) to prompt length
            prompt_length = 0
            patterned_seq_out = None
            seq_out_mask = None

        # Mask needed later to remove invalid outputs after depatterning
        _, _, post_mask = pattern.build_pattern_sequence(
            torch.ones(batch_size, self.num_rvq_layers, seq_len, device=device),
            special_token=self.temp_token,
            keep_only_valid_steps=True,
        )

        return out, prompt_length, post_mask, patterned_seq_out, seq_out_mask

    def _sample_next_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        greedy: bool,
        filter_logits_fn: Callable,
        filter_kwargs: dict,
        curr_sample_step: int,
        curr_sample_length: int,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample next tokens from logits."""
        # Apply classifier-free guidance if enabled
        if guidance_scale != 1.0:
            # Expecting doubled batch: split into conditioned and null (unconditional)
            half = logits.shape[0] // 2
            logits_cond = logits[:half]
            logits_uncond = logits[half:]
            # Combine the logits using the CFG formula:
            logits = logits_uncond + guidance_scale * (
                logits_cond - logits_uncond
            )

        if greedy:
            samples = logits.argmax(dim=-1, keepdim=True)
        else:
            filtered_logits = filter_logits_fn(
                logits,
                curr_sample_step=curr_sample_step,
                curr_sample_length=curr_sample_length,
                **filter_kwargs,
            )
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            # torch.multinomial cannot handle extra batch dimensions...
            orig_shape = probs.shape[:-1]
            probs_flat = probs.view(-1, self.num_tokens)
            samples_flat = torch.multinomial(probs_flat, 1)
            samples = samples_flat.view(*orig_shape, 1)

        # If using CFG, duplicate the sampled token for both halves
        if guidance_scale != 1.0:
            samples = torch.cat([samples, samples], dim=0)

        return samples

    def _apply_pattern_constraints(
        self,
        out: torch.Tensor,
        curr_sample_step: int,
        seq_out_start: Optional[torch.Tensor],
        patterned_seq_out: Optional[torch.Tensor],
        seq_out_mask: Optional[torch.Tensor],
        post_mask: torch.Tensor,
        inst_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply pattern-based constraints and token replacement."""
        # +1 because we start with 1 token (bos / instrument token)
        curr_step = curr_sample_step + 1

        # Overwrite current output with prompt
        if seq_out_start is not None:
            out[..., curr_step][:, seq_out_mask[..., curr_step]] = (
                patterned_seq_out[..., curr_step][
                    :, seq_out_mask[..., curr_step]
                ]
            )
        else:  # No prompt, so we need to infill instrument token
            out[:, ~post_mask[..., curr_step], curr_step] = self.temp_token
            # Replace pattern token with instrument id
            out = self._replace_temp_token_with_inst_tokens(out, inst_tokens)

        return out

    def _finalize_generation(
        self, out: torch.Tensor, guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """Finalize generation by reverting pattern sequence."""
        pattern = self.pattern_provider.get_pattern(
            out.shape[-1] - 1
        )  # -1 for BOS token

        # Turn generation back into original shape
        reconstructed, rev_indexes, rev_mask = pattern.revert_pattern_sequence(
            out,
            special_token=self.pad_value,
        )

        # If using CFG, return only the conditioned half.
        if guidance_scale != 1.0:
            half = reconstructed.shape[0] // 2
            reconstructed = reconstructed[:half]

        return reconstructed


def generate_positions(length: int, delay: int, device: torch.device):
    """
    Generates input and output position tensors based on the specified length and delay.

    Args:
        length (int): The length of the sequence.
        delay (int): The delay to apply. Positive values shift the input positions backward,
                     negative values shift the output positions backward.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and output position tensors.
    """
    if delay == 0:
        input_pos = torch.arange(length, dtype=torch.long, device=device)
        output_pos = torch.arange(length, dtype=torch.long, device=device)
    elif delay > 0:
        # Calculate the number of valid positions after applying delay
        valid_length = max(length - delay, 0)
        # Create input positions with -1 padding for delay
        input_pos = (
            torch.cat(
                [
                    torch.full((delay,), -1, dtype=torch.long, device=device),
                    torch.arange(valid_length, dtype=torch.long, device=device),
                ]
            )
            if valid_length > 0
            else torch.full((length,), -1, dtype=torch.long, device=device)
        )
        output_pos = torch.arange(length, dtype=torch.long, device=device)
    else:
        # Calculate the number of valid positions after applying negative delay
        valid_length = max(length + delay, 0)
        input_pos = torch.arange(length, dtype=torch.long, device=device)
        output_pos = (
            torch.cat(
                [
                    torch.full((-delay,), -1, dtype=torch.long, device=device),
                    torch.arange(valid_length, dtype=torch.long, device=device),
                ]
            )
            if valid_length > 0
            else torch.full((length,), -1, dtype=torch.long, device=device)
        )
    return input_pos, output_pos


class DelayPatternEmbedder(nn.Module):
    """
    Embeds and sums tokens from multiple RVQ layers, like Audiocraft.

    Attributes:
        dim (int): Embedding dimension
        num_tokens (int): Number of unique tokens in vocabulary
        shared: bool - if the vocabulary is shared between
            RVQ layers in the input dataset.

    Args:
        torch.Tensor (B, K, S):
            B = batch size
            K = number of RVQ layers
            S = sequence length (after any patterning)

    Returns:
        torch.Tensor (B, S, dim)
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        num_rvq_layers: int,
        shared: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.shared = shared
        self.num_rvq_layers = num_rvq_layers
        self.num_tokens = num_tokens

        if not self.shared:
            # Need another for pattern audioLM token.
            self.emb = nn.Embedding(num_tokens, dim)
        else:
            self.emb = nn.Embedding((num_tokens) * num_rvq_layers, dim)

        # Initialize the embeddings
        nn.init.kaiming_normal_(self.emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, S] from delayed pattern
        # Apply offsets, so each codebook gets a unique set of embeddings
        if self.shared:
            offset = (
                torch.cumsum(
                    torch.tensor(
                        [0] + [self.num_tokens] * (self.num_rvq_layers - 1),
                        device=x.device,
                    ),
                    0,
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )
            x = x + offset
        else:
            raise DeprecationWarning(
                "Assume Codebooks Share the Same Vocabulary Ranges"
            )

        x = self.emb(x)  # [B, K, S, D]
        x = x.sum(dim=-3)  # [B, S, D]
        return x


class MultiOutToLogits(nn.Module):
    """
    Parallel output heads for multi-RVQ layer prediction.
    Creates separate linear projections for each RVQ layer.

    Attributes:
        num_rvq_layers (int): Number of RVQ layers/output heads
        dim (int): Input dimension
        num_tokens (int): Output dimension (vocab size)

    Args:
        Input: (B, S, dim)

    Returns:
        Output: (B, K, S, num_tokens), K = num_rvq_layers
    """

    def __init__(self, num_rvq_layers: int, dim: int, num_tokens: int):
        super().__init__()
        self.out_heads = nn.ModuleList(
            [nn.Linear(dim, num_tokens) for _ in range(num_rvq_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(x) for head in self.out_heads], dim=1)


class BeatPhaseConditioner(nn.Module):
    """Fuses a per-frame periodic beat/bar signal with per-window BPM and
    time signature, producing an additive residual for ``input_emb``.

    Zero-init *gate* (per-channel) makes the initial residual exactly 0, so a
    baseline checkpoint loaded into a beat-phase-enabled model reproduces
    baseline loss to float precision before any fine-tuning. The MLP is kept
    with normal init so that ``ln_out`` is nonzero at step 0 — this gives the
    gate a nonzero gradient in the first step, letting it actually start to
    move. (If both gate AND mlp were zero-init, nothing would update.)
    """

    def __init__(
        self,
        input_emb_dim: int,
        time_sig_vocab_size: int = 16,
        ts_emb_dim: int = 8,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_emb_dim
        self.input_emb_dim = input_emb_dim
        self.time_sig_vocab_size = time_sig_vocab_size

        self.ts_emb = nn.Embedding(time_sig_vocab_size, ts_emb_dim)
        # Per-frame signal: beat_cond (4) + bpm_log (1) + time_sig_emb (ts_emb_dim)
        in_ch = 4 + 1 + ts_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_emb_dim),
        )

        self.ln = nn.LayerNorm(input_emb_dim)
        # Per-channel gate, zero-init. Residual = ln(mlp(h)) * gate = 0 at init
        # because gate is 0. Gradient to gate in step 1 is nonzero because
        # ln(mlp(h)) is nonzero (mlp has normal init), so gate starts moving.
        self.gate = nn.Parameter(torch.zeros(input_emb_dim))

    def forward(
        self,
        beat_cond: torch.Tensor,
        bpm_log: torch.Tensor,
        time_sig_num: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            beat_cond:    [B, T, 4] float - (sin/cos beat phase, sin/cos bar phase)
            bpm_log:      [B]       float - log(bpm / 120)
            time_sig_num: [B]       long  - clamped to [0, time_sig_vocab_size-1]

        Returns:
            residual: [B, T, input_emb_dim] float - add to input_emb
        """
        B, T, _ = beat_cond.shape
        ts_idx = time_sig_num.clamp(0, self.time_sig_vocab_size - 1)
        ts = self.ts_emb(ts_idx).unsqueeze(1).expand(B, T, -1)  # [B,T,ts_emb_dim]
        bpm = bpm_log.view(B, 1, 1).expand(B, T, 1).to(beat_cond.dtype)
        h = torch.cat([beat_cond, bpm, ts.to(beat_cond.dtype)], dim=-1)
        out = self.ln(self.mlp(h)) * self.gate  # gate=0 at init => residual=0
        return out


class BeatPhaseCondProjector(nn.Module):
    """Projects all available per-window + per-frame timing info into a
    ``[B, T, dim_condition]`` tensor for x_transformers' DiT-style adaptive
    layer-norm + adaptive layer-scale conditioning (per-layer FiLM).

    Inputs (in order of stacking; padded with zeros / default ids when missing):
      Per-frame:
        - 4 ch from ``beat_cond.pt``: sin/cos of φ_beat, sin/cos of φ_bar.
        - 1 ch local_bpm_log: per-frame log(local_BPM / 120). Captures
          tempo variation within the window (rubato, accel/rallent.) and
          saves the model from having to compute phase derivatives to
          recover the local rate at which the next-token clock advances.
      Per-window (broadcast across T):
        - bpm_log scalar         (log(bpm_mean / 120), 0 ≈ 120 BPM)
        - time_sig_num embedding (categorical, vocab ≤ ``time_sig_vocab_size``)
        - time_sig_den embedding (categorical, vocab ≤ ``time_sig_den_vocab``;
          common values 2/4/8/16; clamped)
        - time_sig_change_flag   (1.0 if meter changes within window else 0.0)
        - tempo_change_flag      (1.0 if tempo varies within window else 0.0)

    Unlike ``BeatPhaseConditioner`` (additive at input), this projector does
    NOT include its own gate or LayerNorm — x_transformers' AdaptiveLayerNorm
    has zero-init ``to_gamma`` (so (γ+1)=1 ⇒ identity at step 0) and
    AdaptiveLayerScale has bias-init=-2 (sigmoid(-2)≈0.12 residual attenuation
    at step 0, per the DiT ada-ln-zero recipe).
    """

    def __init__(
        self,
        dim_condition: int,
        time_sig_vocab_size: int = 16,
        time_sig_den_vocab: int = 33,
        ts_emb_dim: int = 8,
        ts_den_emb_dim: int = 8,
        hidden_dim: Optional[int] = None,
        minimal: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim_condition
        self.dim_condition = dim_condition
        self.time_sig_vocab_size = time_sig_vocab_size
        self.time_sig_den_vocab = time_sig_den_vocab
        self.minimal = minimal

        if not self.minimal:
            self.ts_emb = nn.Embedding(time_sig_vocab_size, ts_emb_dim)
            self.ts_den_emb = nn.Embedding(time_sig_den_vocab, ts_den_emb_dim)
            # 4 (beat_cond) + 1 (per-frame local_bpm) + 1 (window bpm) +
            # 1 (ts_change) + 1 (tempo_change) + ts_emb_dim + ts_den_emb_dim
            in_ch = 4 + 1 + 1 + 1 + 1 + ts_emb_dim + ts_den_emb_dim
        else:
            # Minimal projector — only the signals the ablation showed matter:
            # 4 (beat_cond sin/cos beat+bar phase) + 1 (per-frame local_bpm_log).
            in_ch = 4 + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_condition),
        )

    def forward(
        self,
        beat_cond: torch.Tensor,
        bpm_log: torch.Tensor,
        time_sig_num: torch.Tensor,
        time_sig_den: Optional[torch.Tensor] = None,
        time_sig_change: Optional[torch.Tensor] = None,
        tempo_change: Optional[torch.Tensor] = None,
        local_bpm_log: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            beat_cond:       [B, T, 4] float - (sin/cos beat phase, sin/cos bar phase)
            bpm_log:         [B]       float - log(bpm_mean / 120)
            time_sig_num:    [B]       long  - clamped to [0, time_sig_vocab_size-1]
            time_sig_den:    [B]       long  - clamped to [0, time_sig_den_vocab-1].
                             Defaults to 4 if None (most common in slakh).
            time_sig_change: [B]       float - 1.0 if meter changes in window.
                             Defaults to 0 if None.
            tempo_change:    [B]       float - 1.0 if tempo varies in window.
                             Defaults to 0 if None.
            local_bpm_log:   [B, T]    float - per-frame log(local_BPM / 120).
                             Defaults to bpm_log broadcast across T if None.

        Returns:
            condition: [B, T, dim_condition] float
        """
        B, T, _ = beat_cond.shape
        device = beat_cond.device
        dtype = beat_cond.dtype

        if self.minimal:
            if local_bpm_log is None:
                local_bpm = bpm_log.view(B, 1, 1).expand(B, T, 1).to(dtype)
            else:
                local_bpm = local_bpm_log.to(dtype).unsqueeze(-1)
            return self.mlp(torch.cat([beat_cond, local_bpm], dim=-1))

        ts_num_idx = time_sig_num.clamp(0, self.time_sig_vocab_size - 1)
        ts_num = self.ts_emb(ts_num_idx).unsqueeze(1).expand(B, T, -1).to(dtype)

        if time_sig_den is None:
            time_sig_den = torch.full(
                (B,), 4, dtype=torch.long, device=device
            )
        ts_den_idx = time_sig_den.clamp(0, self.time_sig_den_vocab - 1)
        ts_den = (
            self.ts_den_emb(ts_den_idx).unsqueeze(1).expand(B, T, -1).to(dtype)
        )

        bpm = bpm_log.view(B, 1, 1).expand(B, T, 1).to(dtype)

        # Per-frame local BPM. If not supplied, broadcast the per-window mean
        # across all frames — the model still gets a per-frame channel even
        # when the dataloader couldn't compute true local BPM.
        if local_bpm_log is None:
            local_bpm = bpm  # [B, T, 1]
        else:
            local_bpm = local_bpm_log.to(dtype).unsqueeze(-1)  # [B, T, 1]

        if time_sig_change is None:
            ts_chg = torch.zeros(B, T, 1, dtype=dtype, device=device)
        else:
            ts_chg = (
                time_sig_change.to(dtype).view(B, 1, 1).expand(B, T, 1)
            )

        if tempo_change is None:
            tempo_chg = torch.zeros(B, T, 1, dtype=dtype, device=device)
        else:
            tempo_chg = (
                tempo_change.to(dtype).view(B, 1, 1).expand(B, T, 1)
            )

        h = torch.cat(
            [beat_cond, local_bpm, bpm, ts_chg, tempo_chg, ts_num, ts_den],
            dim=-1,
        )
        return self.mlp(h)


class BeatPhaseAuxHead(nn.Module):
    """Auxiliary head that predicts per-frame beat_cond
    ``[sin(φ_beat), cos(φ_beat), sin(φ_bar), cos(φ_bar)]`` from the
    decoder's pre-logits hidden state ``[B, S, dim]``. Trained with MSE
    against ground-truth beat_cond. Dropped at inference (zero deployment
    cost). Pushes the encoder to surface explicit beat-phase representation
    in its activations even when no explicit beat-phase conditioning is
    provided.
    """

    def __init__(self, dim: int, hidden_dim: int = 256, out_dim: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class ChromaAuxHead(nn.Module):
    """Auxiliary head that predicts per-frame target-stem chroma
    ``[out_dim]`` (pitch-class energies) from the decoder's pre-logits
    hidden state ``[B, S, dim]``. Dropped at inference.

    Parameters:
      linear:     If True, replace the 2-layer MLP with a single
                  ``Linear(dim, total_out)`` projection. A shallow head
                  pushes representational pressure onto the trunk (the
                  trunk must encode chroma-decodable features rather
                  than rely on the head's MLP to do its own decoding).
                  Standard SSL-projector trick.
      n_horizons: If >1, predict ``out_dim`` values at multiple frame
                  offsets simultaneously (concatenated along last dim).
                  Caller is responsible for building the multi-horizon
                  target. Forces the trunk to encode harmonic
                  *trajectory* rather than only the current frame.

    Reversibility: with defaults ``linear=False, n_horizons=1`` the
    parameter layout is identical to the legacy module
    (``self.mlp.{0,2}.{weight,bias}``), so existing checkpoints load
    unchanged. New configs opting in to ``linear`` or ``n_horizons>1``
    use ``self.proj`` instead, an isolated namespace.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        out_dim: int = 12,
        linear: bool = False,
        n_horizons: int = 1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.n_horizons = int(n_horizons)
        total_out = int(out_dim * self.n_horizons)
        self.linear = bool(linear)
        if self.linear:
            self.proj = nn.Linear(dim, total_out)
        elif self.n_horizons > 1:
            # Legacy MLP shape but with multi-horizon output. Use ``proj``
            # to keep the namespace separate from the legacy ckpt format.
            self.proj = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, total_out),
            )
        else:
            # Legacy single-frame, MLP head: keep ``self.mlp`` so
            # existing checkpoints load with no surgery.
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.linear or self.n_horizons > 1:
            return self.proj(hidden)
        return self.mlp(hidden)


class MultipitchAuxHead(nn.Module):
    """Predicts per-frame target-stem multipitch presence (BCE) from the
    decoder's pre-logits hidden state ``[B, S, dim]``. Output ``[B, S,
    out_dim]`` is presence logits. Dropped at inference. Standard 2-layer
    MLP — no strengthener variants here.
    """

    def __init__(self, dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class CQTAuxHead(nn.Module):
    """Predicts per-frame target-stem CQT log-magnitude ``[B, S, out_dim]``
    from the decoder's pre-logits hidden state. Trained with MSE+cosine
    against the precomputed 84-bin CQT target. Dropped at inference.
    """

    def __init__(self, dim: int, hidden_dim: int = 256, out_dim: int = 84):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class MultipitchFutureAuxHead(nn.Module):
    """Predicts target-stem multipitch presence at K future offsets from the
    decoder's hidden state at frame t. Output ``[B, S, K, out_dim]``: for
    each of K offsets δ_k, presence logits at frame t+δ_k. Only meaningful
    when ``future_visibility > 0`` so the encoder has actually seen the
    future-mix tokens that supply the answer.
    """

    def __init__(
        self, dim: int, hidden_dim: int = 256, out_dim: int = 128,
        num_offsets: int = 3,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_offsets = num_offsets
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_offsets * out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden.shape
        out = self.mlp(hidden)
        return out.view(B, S, self.num_offsets, self.out_dim)


class CQTFutureAuxHead(nn.Module):
    """Same shape as :class:`MultipitchFutureAuxHead` but for CQT log-mag.
    Output ``[B, S, K, out_dim]``: per-frame K-offset CQT prediction.
    """

    def __init__(
        self, dim: int, hidden_dim: int = 256, out_dim: int = 84,
        num_offsets: int = 3,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_offsets = num_offsets
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_offsets * out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden.shape
        out = self.mlp(hidden)
        return out.view(B, S, self.num_offsets, self.out_dim)


class TargetTokenFutureAuxHead(nn.Module):
    """Phase K — predicts target-stem DAC tokens at K future offsets from
    decoder hidden state ``h_t``. Output ``[B, S, K, num_rvq, num_tokens]``:
    for each offset δ_k, per-codebook token logits at mangled position p+δ_k.
    Loss is per-(offset, codebook) cross-entropy at valid positions only.

    Why this instead of feature-prediction aux heads (Phase J): the target
    tokens at p+δ are NOT in the input at p (they are autoregressively in
    the future), so the head cannot satisfy the loss by copying attention
    outputs. The most useful information for solving it is what the
    accompaniment is doing around frame p+δ — which lives in the future-mix
    tokens. Supervision lands directly on the generation pathway (same
    vocabulary, same Linear-to-logits shape, just shifted in time).
    """

    def __init__(
        self, dim: int, hidden_dim: int = 256, num_rvq: int = 4,
        num_tokens: int = 1024, num_offsets: int = 3,
    ):
        super().__init__()
        self.num_rvq = int(num_rvq)
        self.num_tokens = int(num_tokens)
        self.num_offsets = int(num_offsets)
        out_dim = self.num_offsets * self.num_rvq * self.num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden.shape
        out = self.mlp(hidden)
        return out.view(
            B, S, self.num_offsets, self.num_rvq, self.num_tokens
        )


class CoupledTargetTokenFutureHead(nn.Module):
    """Phase L — same target as :class:`TargetTokenFutureAuxHead` (predict
    target-stem DAC tokens at p+δ from h_t) but with the SHARED main
    ``to_logits`` classifier instead of a separate per-offset projection.

    Mechanism: a SHARED MLP trunk transforms ``hidden`` into a per-offset
    delta ``Δ_δ = trunk(hidden + offset_emb(δ))``; the per-offset hidden
    is ``h_δ = hidden + Δ_δ``; the SAME per-codebook classifier
    (``MultiOutToLogits``) used by the main next-token head is then
    applied to each ``h_δ``. Sharing the classifier across offsets is
    the coupling mechanism: gradient from "predict token at p+δ" flows
    through the same Linear weights that produce token-at-p logits, so
    any feature the trunk learns (which requires future-mix attention to
    solve) is also expressed in the main prediction circuit.

    Param count is O(dim·hidden + num_offsets·dim) regardless of how
    many offsets are requested, so K_off=50 is cheap.

    Init: ``offset_emb`` is zero-init (all offsets identical at step 0),
    and the trunk's last Linear is zero-init (``Δ_δ=0``), so ``h_δ ==
    hidden`` and logits at step 0 == baseline. After one step the
    trunk's last layer picks up gradient (because the same baseline
    logits are evaluated against different per-offset targets — the
    classifier weight backprops a non-zero gradient even though
    activations are identical), the offset_emb starts to move on the
    following step, and the head specialises from there.
    """

    def __init__(
        self, dim: int, num_offsets: int, hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_offsets = int(num_offsets)
        self.dim = int(dim)
        self.offset_emb = nn.Embedding(self.num_offsets, self.dim)
        nn.init.zeros_(self.offset_emb.weight)
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(
        self, hidden_chunk: torch.Tensor, to_logits: nn.Module,
    ) -> torch.Tensor:
        """
        Args:
            hidden_chunk: ``[B, S_chunk, dim]`` — hidden states at the
                prediction window only (saves K_off×num_rvq Linear calls).
            to_logits: parent's ``MultiOutToLogits`` (per-codebook Linear);
                takes ``[B, S, dim]`` and returns ``[B, num_rvq, S, V]``.
        Returns:
            ``[B, num_rvq, S_chunk, K_off, V]``
        """
        B, S, D = hidden_chunk.shape
        K = self.num_offsets
        # Broadcast offset embedding across (B, S): [1, 1, K, D] + [B, S, 1, D]
        # → [B, S, K, D]
        h_in = hidden_chunk.unsqueeze(-2) + self.offset_emb.weight.view(
            1, 1, K, D
        )
        h_in = h_in.reshape(B, S * K, D)
        # Shared trunk over (B, S·K, D). Zero-init last layer ⇒ all zeros
        # at step 0.
        delta = self.trunk(h_in)
        # Residual: h_δ = hidden + Δ_δ. Expand hidden over K and add.
        h_off = hidden_chunk.unsqueeze(-2).expand(B, S, K, D).reshape(
            B, S * K, D
        ) + delta
        # Single batched call into shared main to_logits.
        logits = to_logits(h_off)  # [B, num_rvq, S·K, V]
        num_rvq, V = logits.shape[1], logits.shape[3]
        # Reshape back to [B, num_rvq, S_chunk, K_off, V].
        return logits.view(B, num_rvq, S, K, V)


class BeatPhaseAuxHeadFull(nn.Module):
    """Auxiliary head predicting the full beat-phase feature set the cond
    side injects:
      - 4-d sin/cos beat+bar phase (MSE+cos target via beat_cond)
      - 1-d bpm_log (MSE target, broadcast per-frame from window scalar)
      - K-way time-signature numerator logits (CE target)
    Output layout along the last dim:
      [phase_4 | bpm_1 | ts_logits_K]
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        time_sig_vocab_size: int = 16,
    ):
        super().__init__()
        self.time_sig_vocab_size = int(time_sig_vocab_size)
        out_dim = 4 + 1 + self.time_sig_vocab_size
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class ChromaCondProjector(nn.Module):
    """Projects per-frame ``input_chroma [B, T, 12]`` into the DiT
    condition tensor ``[B, T, dim_condition]``. Output is summed with the
    beat-phase projection inside ``_build_dit_condition`` to feed both
    timing and harmonic info through a single AdaptiveLayerNorm /
    AdaptiveLayerScale path per layer. Two-layer MLP, no extras.
    """

    def __init__(
        self,
        dim_condition: int,
        chroma_dim: int = 12,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim_condition
        self.mlp = nn.Sequential(
            nn.Linear(chroma_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_condition),
        )

    def forward(self, chroma: torch.Tensor) -> torch.Tensor:
        return self.mlp(chroma)


class DecoderTransformerMultiOut(AutoregressiveWrapper, BaseGenerationMixin):
    """
    Transform Decoder with support for multiple RVQ outputs and delay patterns
    Implements staggered generation strategy for parallel RVQ layer prediction.

    Attributes:
        num_rvq_layers (int): Number of RVQ layers to predict
        shared (bool): if the token ids in each RVQ level have the same range.

        online (bool): If True, the model is in online mode and takes in concatenated
            input and output embeddings.
        future_visibility (int): Number of tokens to delay the input stream.
        input_emb_dim (int): Dimension of the input stream embeddings before concatenation.
        output_emb_dim (int): Dimension of the output stream embeddings before concatenation.

        Other args identical to models.py
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        pad_value: int = 0,
        cross_attend: bool = False,
        num_rvq_layers: int = 4,  # Modified
        shared: bool = True,  # Modified
        online: bool = False,
        future_visibility: int = 0,
        input_emb_dim: int = 128,
        output_emb_dim: int = 128,  # dimension of the output embeddings
        attention_layer_configs: Optional[dict] = None,
        external_pos: bool = False,
        cond_method: str = "add",
        use_beat_phase: bool = False,
        time_sig_vocab_size: int = 16,
        use_beat_phase_dit_cond: bool = False,
        beat_dit_cond_dim: Optional[int] = None,
        beat_dit_cond_mlp_expansion: int = 4,
        beat_dit_cond_minimal: bool = False,
        cond_dropout_p: float = 0.0,
        beat_phase_noise_std: float = 0.0,
        use_beat_phase_aux_head: bool = False,
        beat_phase_aux_head_hidden_dim: int = 256,
        use_chroma_dit_cond: bool = False,
        chroma_dim: int = 12,
        chroma_dit_cond_hidden_dim: Optional[int] = None,
        use_chroma_aux_head: bool = False,
        chroma_aux_head_hidden_dim: int = 256,
        # New (all default to legacy behavior): see ChromaAuxHead docstring
        # plus deep-supervision option below.
        chroma_aux_head_linear: bool = False,
        chroma_aux_horizons: Optional[Sequence[int]] = None,
        chroma_aux_deep_supervision_layers: Optional[Sequence[int]] = None,
        # Multipitch / CQT / beat-phase-full aux heads (all default off).
        use_multipitch_aux_head: bool = False,
        multipitch_aux_head_hidden_dim: int = 256,
        multipitch_dim: int = 128,
        use_cqt_aux_head: bool = False,
        cqt_aux_head_hidden_dim: int = 256,
        cqt_dim: int = 84,
        use_input_cqt_aux_head: bool = False,
        input_cqt_aux_head_hidden_dim: int = 256,
        input_cqt_dim: int = 84,
        use_beat_phase_aux_head_full: bool = False,
        beat_phase_aux_head_full_hidden_dim: int = 256,
        # Future-mix aux heads (mp / cqt at frame t+δ from h_t). Force the
        # encoder to use the fv lookahead. Only valid when future_visibility>0.
        use_multipitch_future_aux_head: bool = False,
        multipitch_future_aux_head_hidden_dim: int = 256,
        use_cqt_future_aux_head: bool = False,
        cqt_future_aux_head_hidden_dim: int = 256,
        # Phase K — future target-stem token prediction at t+δ from h_t.
        # Same vocabulary as the main next-token head, just shifted in time.
        # Only valid when future_visibility > 0.
        use_target_token_future_aux_head: bool = False,
        target_token_future_aux_head_hidden_dim: int = 256,
        # Phase L — same task as Phase K, but reuses the main ``to_logits``
        # classifier so future-token gradient flows through the same Linear
        # that produces the present-token logits (coupling mechanism).
        use_coupled_target_token_future_head: bool = False,
        coupled_target_token_future_head_hidden_dim: int = 256,
        future_aux_offsets: Optional[Sequence[int]] = None,
    ):
        # Skip the init of the AutoregressiveWrapper class
        super(AutoregressiveWrapper, self).__init__()

        if attention_layer_configs is None:
            attention_layer_configs = {}

        if not online:
            output_emb_dim = dim
        self.online = online

        if online and future_visibility > 0:
            max_seq_len += future_visibility

        self.output_emb = DelayPatternEmbedder(
            dim=dim,
            num_tokens=num_tokens,
            num_rvq_layers=num_rvq_layers,
            shared=shared,
        )
        self.external_pos = external_pos

        if external_pos:
            self.input_pos_enc = ScaledSinusoidalEmbedding(input_emb_dim)
            self.output_pos_enc = ScaledSinusoidalEmbedding(output_emb_dim)

        # DiT-style per-layer conditioning is gated only when ``online`` is
        # set (otherwise there's no input_emb / per-frame signal to condition
        # on in this codebase). Resolved early so we can reuse the flag in
        # the Decoder construction below.
        self.use_beat_phase_dit_cond = bool(online and use_beat_phase_dit_cond)
        self.beat_dit_cond_dim = (
            beat_dit_cond_dim if beat_dit_cond_dim is not None else dim
        )
        self.beat_dit_cond_mlp_expansion = beat_dit_cond_mlp_expansion
        self.beat_dit_cond_minimal = bool(beat_dit_cond_minimal)
        self.cond_dropout_p = float(cond_dropout_p)
        self.beat_phase_noise_std = float(beat_phase_noise_std)
        # Resolve chroma DiT cond flag here so the decoder construction
        # below can enable AdaptiveLayerNorm if either beat OR chroma is on.
        self._use_chroma_dit_cond_init = bool(online and use_chroma_dit_cond)
        self._any_dit_cond = (
            self.use_beat_phase_dit_cond or self._use_chroma_dit_cond_init
        )

        print("\n")
        print(f"{'Parameter':<15} | {'Value'}")
        print("-" * 40)
        print(f"{'Dim':<15} | {dim}")
        print(f"{'Depth':<15} | {depth}")
        print(f"{'Heads':<15} | {heads}")
        print(f"{'Attn Dropout':<15} | {attn_dropout}")
        print(f"{'FF Dropout':<15} | {ff_dropout}")
        print(f"{'CrossAttend':<15} | {cross_attend}")
        print(f"{'Num Tokens':<15} | {num_tokens}")
        print(f"{'Num RVQ Layers':<15} | {num_rvq_layers}")
        print(f"{'Max Seq Len':<15} | {max_seq_len}")
        print(f"{'External Pos':<15} | {external_pos}")
        print(f"{'Future Visibility':<15} | {future_visibility}")
        print(f"{'Input Emb Dim':<15} | {input_emb_dim}")
        print(f"{'Shared':<15} | {shared}")
        print(f"{'Online':<15} | {online}")
        print(f"{'Cond Method':<15} | {cond_method}")
        print(f"{'DiT Beat Cond':<15} | {self.use_beat_phase_dit_cond}")
        print(f"{'Cond Dropout p':<15} | {self.cond_dropout_p}")
        print(f"{'Phase Noise std':<15} | {self.beat_phase_noise_std}")

        if cond_method not in ("concat", "add", "film"):
            raise ValueError(
                f"Invalid conditioning method: {cond_method}. "
                f"Expected one of: concat, add, film"
            )

        # When DiT conditioning is enabled, route the adaptive layer-norm /
        # layer-scale flags into x_transformers via the project's Decoder
        # wrapper, which forwards ``attention_layer_configs`` to the
        # underlying x_transformers Decoder via **. The wrapper does NOT
        # accept extra kwargs directly. AdaptiveLayerNorm zero-inits its
        # gamma projection so (gamma+1)=1 ⇒ identity at step 0;
        # AdaptiveLayerScale uses bias-init=-2 ⇒ sigmoid(-2)≈0.12 residual
        # attenuation at step 0 (the DiT ada-ln-zero recipe).
        if self._any_dit_cond:
            attention_layer_configs = dict(attention_layer_configs)
            attention_layer_configs.update(
                use_adaptive_layernorm=True,
                use_adaptive_layerscale=True,
                dim_condition=self.beat_dit_cond_dim,
                adaptive_condition_mlp=True,
                adaptive_condition_mlp_expansion=self.beat_dit_cond_mlp_expansion,
            )
            # x_transformers' AdaptiveLayerNorm replaces the standard pre-norm
            # in every layer, so we must turn off the project's default
            # ``use_simple_rmsnorm=True`` (otherwise x_transformers asserts:
            # "you can only use either scalenorm, rmsnorm, ..., or adaptive
            # layernorm").
            attention_layer_configs["use_simple_rmsnorm"] = False

        self.decoder = TransformerWrapper(
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                cross_attend=cross_attend,
                attention_layer_configs=attention_layer_configs,
            ),
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            token_emb=nn.Identity(),  # Needed to circumvent the 'embedding' layer.
            to_logits=MultiOutToLogits(num_rvq_layers, dim, num_tokens),
            use_abs_pos_emb=not external_pos,
        )

        # Use Delay Pattern
        delays = list(range(num_rvq_layers))
        self.pattern_provider = DelayedPatternProvider(
            n_q=num_rvq_layers, delays=delays
        )

        if online:
            self.future_visibility = future_visibility
            if self.future_visibility <= 0:
                # The padding is for when the model is generating
                #   without hearing the input
                self.input_padding = nn.Parameter(
                    torch.randn(-future_visibility + 1, input_emb_dim)
                )
                # Initialize the input_padding with kaiming normal
                nn.init.kaiming_normal_(self.input_padding)

            # Always store conditioning method and final LN flag
            self.cond_method = cond_method
            self.use_final_ln = True  # always use final LayerNorm

            # Only create dec_input_emb for concat conditioning
            if self.cond_method == "concat":
                self.dec_input_emb = nn.Linear(
                    input_emb_dim + output_emb_dim, dim
                )

            # For methods that need per-stream norms and projection ("add", "film")
            if self.cond_method in ("add", "film"):
                # LayerNorm on raw streams
                self.ln_in = nn.LayerNorm(input_emb_dim)
                self.ln_out = nn.LayerNorm(dim)

                # Project INPUT stream to OUTPUT/Transformer dim for fusion
                self.proj_in_to_out = (
                    nn.Linear(input_emb_dim, dim)
                    if input_emb_dim != dim
                    else nn.Identity()
                )

                # Optional final LayerNorm after fusion (over dim)
                if self.use_final_ln:
                    self.ln_fuse = nn.LayerNorm(dim)

            # For "add" path only: per-channel gate over dim
            if self.cond_method == "add":
                self.add_gate = nn.Parameter(torch.zeros(dim))

            # For "film" path only: generate (gamma, beta) in dim space
            elif self.cond_method == "film":
                self.film_gen = nn.Sequential(
                    nn.Linear(2 * dim, 4 * dim),
                    nn.GELU(),
                    nn.Linear(4 * dim, 2 * dim),
                )
                _last = self.film_gen[-1]
                if isinstance(_last, nn.Linear):
                    nn.init.zeros_(_last.weight)
                    nn.init.zeros_(_last.bias)

        # Beat-phase conditioning (only when online; otherwise no input_emb path)
        self.use_beat_phase = bool(online and use_beat_phase)
        print(f"{'Use Beat Phase':<15} | {self.use_beat_phase}")
        if self.use_beat_phase:
            self.beat_conditioner = BeatPhaseConditioner(
                input_emb_dim=input_emb_dim,
                time_sig_vocab_size=time_sig_vocab_size,
            )

        # DiT-style per-layer beat-phase conditioning. The projector maps
        # (beat_cond, bpm_log, time_sig_num) → [B, T, dim_condition], which
        # x_transformers' AdaptiveLayerNorm/AdaptiveLayerScale consume per
        # layer. The decoder itself was already configured with the right
        # adaptive flags above when ``use_beat_phase_dit_cond=True``.
        if self.use_beat_phase_dit_cond:
            self.beat_cond_projector = BeatPhaseCondProjector(
                dim_condition=self.beat_dit_cond_dim,
                time_sig_vocab_size=time_sig_vocab_size,
                minimal=self.beat_dit_cond_minimal,
            )

        # Auxiliary beat-phase prediction head. Reads the decoder's
        # pre-logits hidden state and predicts beat_cond. Loss is MSE,
        # weighted into the total loss inside the lit module. Dropped at
        # inference (zero deployment cost). Useful as either a replacement
        # for explicit beat-phase conditioning (probe whether the encoder
        # can extract beat purely from audio when pushed) or as an addition
        # on top of conditioning (regulariser / representation pressure).
        self.use_beat_phase_aux_head = bool(online and use_beat_phase_aux_head)
        print(f"{'Aux Beat Head':<15} | {self.use_beat_phase_aux_head}")
        if self.use_beat_phase_aux_head:
            self.beat_phase_aux_head = BeatPhaseAuxHead(
                dim=dim,
                hidden_dim=beat_phase_aux_head_hidden_dim,
                out_dim=4,
            )

        # Chroma DiT cond projector (per-frame input-mix chroma -> condition).
        # Output is summed with the beat-phase projection in
        # ``_build_dit_condition`` so the AdaptiveLayerNorm path sees both
        # timing and harmonic info through one channel.
        self.use_chroma_dit_cond = bool(online and use_chroma_dit_cond)
        self.chroma_dim = chroma_dim
        if self.use_chroma_dit_cond:
            self.chroma_cond_projector = ChromaCondProjector(
                dim_condition=self.beat_dit_cond_dim,
                chroma_dim=chroma_dim,
                hidden_dim=chroma_dit_cond_hidden_dim,
            )
        # Chroma aux head (predicts target_chroma from pre-logits hidden state).
        self.use_chroma_aux_head = bool(online and use_chroma_aux_head)
        print(f"{'Chroma DiT Cond':<15} | {self.use_chroma_dit_cond}")
        print(f"{'Aux Chroma Head':<15} | {self.use_chroma_aux_head}")
        # Normalize to plain tuples up front so __init__ flags are simple
        # value types — easier to print, cache, and reason about.
        chroma_aux_horizons_t: Tuple[int, ...] = tuple(
            int(h) for h in (chroma_aux_horizons or (0,))
        )
        chroma_aux_dsv_layers_t: Tuple[int, ...] = tuple(
            int(li) for li in (chroma_aux_deep_supervision_layers or ())
        )
        self.chroma_aux_head_linear = bool(chroma_aux_head_linear)
        self.chroma_aux_horizons = chroma_aux_horizons_t
        self.chroma_aux_deep_supervision_layers = chroma_aux_dsv_layers_t
        if self.use_chroma_aux_head:
            n_horizons = len(chroma_aux_horizons_t)
            self.chroma_aux_head = ChromaAuxHead(
                dim=dim,
                hidden_dim=chroma_aux_head_hidden_dim,
                out_dim=chroma_dim,
                linear=self.chroma_aux_head_linear,
                n_horizons=n_horizons,
            )
            # Deep-supervision aux heads at intermediate transformer layers.
            # Empty when ``chroma_aux_deep_supervision_layers`` is unset, in
            # which case forward pass takes the legacy code path.
            if chroma_aux_dsv_layers_t:
                self.chroma_aux_dsv_heads = nn.ModuleDict({
                    str(li): ChromaAuxHead(
                        dim=dim,
                        hidden_dim=chroma_aux_head_hidden_dim,
                        out_dim=chroma_dim,
                        linear=self.chroma_aux_head_linear,
                        n_horizons=n_horizons,
                    )
                    for li in chroma_aux_dsv_layers_t
                })

        # Multipitch aux head (presence + velocity).
        self.use_multipitch_aux_head = bool(online and use_multipitch_aux_head)
        self.multipitch_dim = int(multipitch_dim)
        print(f"{'Aux MP Head':<15} | {self.use_multipitch_aux_head}")
        if self.use_multipitch_aux_head:
            self.multipitch_aux_head = MultipitchAuxHead(
                dim=dim,
                hidden_dim=multipitch_aux_head_hidden_dim,
                out_dim=self.multipitch_dim,
            )

        # CQT aux head (84-bin log-magnitude).
        self.use_cqt_aux_head = bool(online and use_cqt_aux_head)
        self.cqt_dim = int(cqt_dim)
        print(f"{'Aux CQT Head':<15} | {self.use_cqt_aux_head}")
        if self.use_cqt_aux_head:
            self.cqt_aux_head = CQTAuxHead(
                dim=dim,
                hidden_dim=cqt_aux_head_hidden_dim,
                out_dim=self.cqt_dim,
            )

        # Input-mix CQT aux head (same shape; supervised against
        # ``input_cqt.pt``). Forces the hidden state to retain a faithful
        # spectral picture of what the model is hearing — orthogonal to the
        # target-stem CQT signal which says what to generate.
        self.use_input_cqt_aux_head = bool(online and use_input_cqt_aux_head)
        self.input_cqt_dim = int(input_cqt_dim)
        print(f"{'Aux InCQT Head':<15} | {self.use_input_cqt_aux_head}")
        if self.use_input_cqt_aux_head:
            self.input_cqt_aux_head = CQTAuxHead(
                dim=dim,
                hidden_dim=input_cqt_aux_head_hidden_dim,
                out_dim=self.input_cqt_dim,
            )

        # Future aux heads come in two categories with different fv
        # requirements:
        #   • mp_future / cqt_future / coupled_tt_future predict features
        #     at frame p+δ from h_t. They REQUIRE the encoder to have
        #     attended to frame p+δ (which only happens when δ <= fv), so
        #     they're undefined for fv <= 0.
        #   • target_token_future_aux_head predicts the patterned target
        #     DAC token at p+δ from h_t. The supervision needs only the
        #     LABEL (always available in the full target stem); the
        #     encoder does NOT need future-mix input visibility for the
        #     task to be well-defined. It just becomes a harder
        #     extrapolation task. We allow it for any fv (including
        #     fv<=0) so Phase K can be run as a fair-deployability
        #     comparison without the +50 lookahead.
        encoder_lookahead_heads = (
            online and (
                use_multipitch_future_aux_head
                or use_cqt_future_aux_head
                or use_coupled_target_token_future_head
            )
        )
        any_future_aux_request = (
            encoder_lookahead_heads
            or (online and use_target_token_future_aux_head)
        )
        if encoder_lookahead_heads and future_visibility <= 0:
            raise ValueError(
                "use_multipitch_future_aux_head / use_cqt_future_aux_head / "
                "use_coupled_target_token_future_head "
                "require future_visibility > 0 (got "
                f"future_visibility={future_visibility}). Those heads "
                "predict frame t+δ from h_t and only have signal when the "
                "encoder has integrated future-mix tokens."
            )
        self.future_aux_offsets = (
            tuple(int(d) for d in (future_aux_offsets or (10, 25, 40)))
            if any_future_aux_request
            else ()
        )
        if encoder_lookahead_heads:
            max_off = max(self.future_aux_offsets)
            if max_off > future_visibility:
                raise ValueError(
                    f"max future_aux_offset ({max_off}) exceeds "
                    f"future_visibility ({future_visibility}); the encoder "
                    "hasn't seen the target frame."
                )
        self.use_multipitch_future_aux_head = bool(
            online and use_multipitch_future_aux_head
        )
        print(
            f"{'Aux MP-Fut Head':<15} | {self.use_multipitch_future_aux_head}"
        )
        if self.use_multipitch_future_aux_head:
            self.multipitch_future_aux_head = MultipitchFutureAuxHead(
                dim=dim,
                hidden_dim=multipitch_future_aux_head_hidden_dim,
                out_dim=self.multipitch_dim,
                num_offsets=len(self.future_aux_offsets),
            )
        self.use_cqt_future_aux_head = bool(
            online and use_cqt_future_aux_head
        )
        print(
            f"{'Aux CQT-Fut Head':<15} | {self.use_cqt_future_aux_head}"
        )
        if self.use_cqt_future_aux_head:
            self.cqt_future_aux_head = CQTFutureAuxHead(
                dim=dim,
                hidden_dim=cqt_future_aux_head_hidden_dim,
                out_dim=self.cqt_dim,
                num_offsets=len(self.future_aux_offsets),
            )
        self.use_target_token_future_aux_head = bool(
            online and use_target_token_future_aux_head
        )
        print(
            f"{'Aux TT-Fut Head':<15} | {self.use_target_token_future_aux_head}"
        )
        if self.use_target_token_future_aux_head:
            self.target_token_future_aux_head = TargetTokenFutureAuxHead(
                dim=dim,
                hidden_dim=target_token_future_aux_head_hidden_dim,
                num_rvq=num_rvq_layers,
                num_tokens=num_tokens,
                num_offsets=len(self.future_aux_offsets),
            )
        self.use_coupled_target_token_future_head = bool(
            online and use_coupled_target_token_future_head
        )
        print(
            f"{'Aux TT-Coupled':<15} | "
            f"{self.use_coupled_target_token_future_head}"
        )
        if self.use_coupled_target_token_future_head:
            self.coupled_target_token_future_head = CoupledTargetTokenFutureHead(
                dim=dim,
                num_offsets=len(self.future_aux_offsets),
                hidden_dim=coupled_target_token_future_head_hidden_dim,
            )
        if any_future_aux_request:
            print(
                f"{'Future Aux δ':<15} | {list(self.future_aux_offsets)}"
            )

        # Beat-phase aux head (FULL feature set: phase + bpm + time_sig).
        # Distinct from ``use_beat_phase_aux_head`` (legacy 4-d phase only).
        self.use_beat_phase_aux_head_full = bool(
            online and use_beat_phase_aux_head_full
        )
        self.beat_phase_aux_full_ts_vocab = int(time_sig_vocab_size)
        print(
            f"{'Aux Beat Full':<15} | {self.use_beat_phase_aux_head_full}"
        )
        if self.use_beat_phase_aux_head_full:
            self.beat_phase_aux_head_full = BeatPhaseAuxHeadFull(
                dim=dim,
                hidden_dim=beat_phase_aux_head_full_hidden_dim,
                time_sig_vocab_size=time_sig_vocab_size,
            )

        # Save arguments
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len + 1  # +1 for the patterning token
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.pad_value = pad_value

        self.num_rvq_layers = num_rvq_layers

        # token used to temporarily pattern things integer id. Will be replaced by inst_tokens.
        self.temp_token = self.num_tokens
        self.shared = shared

    @property
    def net(self):
        return self.decoder

    def pad_input_embs_for_delay(self, input_emb):
        assert self.future_visibility <= 0
        input_pad = self.input_padding.expand(input_emb.shape[0], -1, -1)
        input_emb = torch.cat([input_pad, input_emb], dim=1)
        if self.future_visibility < 0:
            input_emb = input_emb[:, : self.future_visibility]
        return input_emb

    def pad_beat_cond_for_delay(self, beat_cond):
        """Pad beat_cond to match pad_input_embs_for_delay by edge-repeating
        the first frame (phase continuity beats zero padding)."""
        assert self.future_visibility <= 0
        pad_len = -self.future_visibility + 1
        first = beat_cond[:, :1, :]  # [B, 1, C]
        pad = first.expand(-1, pad_len, -1)
        beat_cond = torch.cat([pad, beat_cond], dim=1)
        if self.future_visibility < 0:
            beat_cond = beat_cond[:, : self.future_visibility]
        return beat_cond

    def pad_local_bpm_for_delay(self, local_bpm_log):
        """Mirror of pad_beat_cond_for_delay for the [B, T] per-frame BPM
        tensor. Edge-repeats the first frame so the local-tempo signal stays
        aligned with the padded beat_cond and input_emb tensors.
        """
        assert self.future_visibility <= 0
        pad_len = -self.future_visibility + 1
        first = local_bpm_log[:, :1]  # [B, 1]
        pad = first.expand(-1, pad_len)
        local_bpm_log = torch.cat([pad, local_bpm_log], dim=1)
        if self.future_visibility < 0:
            local_bpm_log = local_bpm_log[:, : self.future_visibility]
        return local_bpm_log

    def pad_output_tokens_for_delay(self, output_tokens, trim_end=True):
        assert self.future_visibility > 0
        output_pad = torch.full(
            (
                output_tokens.shape[0],
                output_tokens.shape[1],
                self.future_visibility - 1,
            ),
            self.pad_value,
            device=output_tokens.device,
        )
        output_tokens = torch.cat([output_pad, output_tokens], dim=2)
        if trim_end:
            output_tokens = output_tokens[:, :, : -self.future_visibility]
        return output_tokens

    def get_input_embedding(
        self,
        output_tokens,
        input_emb,
        beat_cond: Optional[torch.Tensor] = None,
        bpm_log: Optional[torch.Tensor] = None,
        time_sig_num: Optional[torch.Tensor] = None,
    ):
        embedded_output = self.output_emb(output_tokens)

        # Beat-phase conditioning: additive residual on input_emb BEFORE
        # ln_in so the existing normalization absorbs any distribution shift.
        # At init (gate=0) this is exactly a no-op.
        # Direct attribute access (not getattr) so torch.compile can constant-
        # fold the branch instead of falling back to eager / graph-breaking.
        if self.use_beat_phase and beat_cond is not None:
            residual = self.beat_conditioner(
                beat_cond.to(input_emb.dtype),
                bpm_log.to(input_emb.dtype),
                time_sig_num,
            )
            input_emb = input_emb + residual

        # Optional external positional encodings (kept before normalization)
        if self.external_pos:
            input_pos, output_pos = generate_positions(
                length=input_emb.shape[1],
                delay=-self.future_visibility,
                device=embedded_output.device,
            )
            output_pos_emb = self.output_pos_enc(output_tokens, pos=output_pos)
            embedded_output = embedded_output + output_pos_emb
            input_pos_emb = self.input_pos_enc(input_emb, pos=input_pos)
            input_emb = input_emb + input_pos_emb

        # Backward compatibility: default to old concat behavior
        cond_method = getattr(self, "cond_method", "concat")

        if cond_method == "concat":
            concat_emb = torch.cat([input_emb, embedded_output], dim=-1)
            embedded = self.dec_input_emb(concat_emb)
            return embedded

        # "add" and "film" fuse in model dim
        if cond_method == "add":
            x_in = self.ln_in(input_emb)
            y_out = self.ln_out(embedded_output)
            x_proj = self.proj_in_to_out(x_in)
            x_fused = x_proj + y_out * self.add_gate
            if self.use_final_ln:
                x_fused = self.ln_fuse(x_fused)
            embedded = x_fused  # already [B, S, dim]
            return embedded

        if cond_method == "film":
            x_in = self.ln_in(input_emb)
            y_out = self.ln_out(embedded_output)
            x_proj = self.proj_in_to_out(x_in)
            h = torch.cat([x_proj, y_out], dim=-1)
            gamma_beta = self.film_gen(h)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            gamma = 1.0 + gamma
            x_fused = gamma * x_proj + beta
            if self.use_final_ln:
                x_fused = self.ln_fuse(x_fused)
            embedded = x_fused  # already [B, S, dim]
            return embedded

        # If an unknown method is specified, fall back to concat for safety
        concat_emb = torch.cat([input_emb, embedded_output], dim=-1)
        embedded = self.dec_input_emb(concat_emb)
        return embedded

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inst_tokens: Optional[torch.Tensor] = None,
        input_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder, used for training.

        Args:
            x (Tensor): Shape (B, K, T):

        Output:
            logits (Tensor): output logits, shape (B, K, T, num_tokens).
            logits_mask: output logits mask, such that logits[logits_mask]
                provides only valid logits. Shape (B, K, T)
        """
        B, K, T = x.shape
        pattern = self.pattern_provider.get_pattern(T)
        x, _, sequence_mask = pattern.build_pattern_sequence(
            x,
            special_token=self.temp_token,
            keep_only_valid_steps=True,
        )

        # Replace temporary patterning tokens with inst_tokens
        x = self._replace_temp_token_with_inst_tokens(x, inst_tokens)

        if self.online:
            if self.future_visibility <= 0:
                input_emb = self.pad_input_embs_for_delay(input_emb)
            else:
                x = self.pad_output_tokens_for_delay(x)

            embedded = self.get_input_embedding(x, input_emb)
        else:
            embedded = self.output_emb(x)

        logits = self.decoder(
            embedded, mask=mask, **kwargs
        )  # [B, K, S, num_tokens]

        logits = logits.permute(0, 3, 1, 2)  # [B, num_tokens, K, S]

        logits, _, logits_mask = pattern.revert_pattern_logits(
            logits, float("nan"), keep_only_valid_steps=True
        )

        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, num_tokens]

        logits_mask = logits_mask.unsqueeze(0).expand(B, *logits_mask.shape)
        return logits, logits_mask

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate(
        self,
        seq_len: int,
        seq_out_start: Optional[torch.Tensor] = None,
        input_emb: None | torch.Tensor = None,
        temperature: float = 1.0,
        filter_logits_fn: (
            str | Callable | list[str | Callable]
        ) = top_k_multi_out,
        filter_kwargs: dict | list[dict] = dict(),
        cache_kv: bool = True,
        display_pbar: bool = False,
        inst_tokens: torch.Tensor = None,
        guidance_scale: float = 1.0,  # parameter for classifier free guidance
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate function for multi-RVQ layer decoder models with delay pattern
        support.

        Args:
            seq_len (int): Number of steps to generate
            seq_out_start (torch.Tensor): Start index of the output sequence.
            input_emb (torch.Tensor): Input embeddings of the input context.
            temperature (float): Sampling temperature
            filter_logits_fn: Logit filtering function(s)
            filter_kwargs: Arguments for filtering functions
            cache_kv (bool): Cache key/value pairs
            display_pbar (bool): Show progress bar
            inst_tokens (torch.Tensor): instrument ids to use as prompt
            guidance_scale (float): Guidance scale for classifier free guidance.

        Returns:
            Tensor: Generated sequences (B, K, seq_len - 1)
        """
        # Validate parameters and get batch_size and device
        batch_size, device = self._validate_generation_params(
            inst_tokens, seq_out_start, input_emb, guidance_scale
        )

        # Process filter functions
        filter_logits_fn, filter_kwargs = self._process_filter_functions(
            filter_logits_fn, filter_kwargs
        )

        # Initialize pattern and output sequence
        out, prompt_length, post_mask, patterned_seq_out, seq_out_mask = (
            self._initialize_generation_pattern(
                seq_len, batch_size, device, seq_out_start, inst_tokens
            )
        )

        # Set flags
        greedy = temperature == 0.0

        # Initialize cache for KV caching
        cache = None

        # Handle online mode future visibility
        if self.online and self.future_visibility <= 0:
            input_emb = self.pad_input_embs_for_delay(input_emb)

        pbar = tqdm(
            range(prompt_length, seq_len),
            disable=not display_pbar,
            desc="Sampling",
        )
        for curr_sample_step in pbar:
            # Prepare model input (online-specific logic)
            if self.online:
                if self.future_visibility > 0:
                    # Here we add the left padding to the output tokens:
                    #   periods that model only need to listen but not generate.
                    # Later we will remove the padding before next step generation.
                    out = self.pad_output_tokens_for_delay(out, trim_end=False)
                if (
                    out.shape[-1] > input_emb.shape[1]
                ):  # consumed all the input embeddings
                    break
                cur_input_emb = input_emb[:, : out.shape[-1]]
                embedded = self.get_input_embedding(out, cur_input_emb)
            else:
                embedded = self.output_emb(out)  # [B, S, D]

            # Forward pass with optional KV caching
            if cache_kv:
                logits, intermediates = self.net(
                    embedded,
                    mask=None,
                    cache=cache,
                    return_intermediates=True,
                    **kwargs,
                )
                # Update cache from intermediates
                if hasattr(self.net, "can_cache_kv") and self.net.can_cache_kv:
                    cache = intermediates
            else:
                logits = self.net(
                    embedded,
                    mask=None,
                    return_intermediates=False,
                    **kwargs,
                )

            logits = logits[:, :, -1]  # Get logits for next token

            # Sample next tokens using mixin method
            samples = self._sample_next_tokens(
                logits,
                temperature,
                greedy,
                filter_logits_fn,
                filter_kwargs,
                curr_sample_step,
                out.shape[-1],
                guidance_scale,
            )

            out = torch.cat([out, samples], dim=-1)

            # Handle online mode future visibility
            if self.online and self.future_visibility > 0:
                out = out[
                    :, :, self.future_visibility - 1 :
                ]  # remove the padding for positive future visibility

            # Apply pattern constraints using mixin method
            out = self._apply_pattern_constraints(
                out,
                curr_sample_step,
                seq_out_start,
                patterned_seq_out,
                seq_out_mask,
                post_mask,
                inst_tokens,
            )

        # Finalize generation using mixin method
        return self._finalize_generation(out, guidance_scale)

    def _replace_temp_token_with_inst_tokens(
        self, x: torch.Tensor, inst_tokens: torch.Tensor
    ) -> torch.Tensor:
        temp_token_mask = x == self.temp_token
        inst_tokens_expanded = inst_tokens.view(x.shape[0], 1, 1).expand_as(x)
        x[temp_token_mask] = inst_tokens_expanded[temp_token_mask]
        return x


class EncoderDecoderTransformerMultiOut(nn.Module):
    """Encoder-decoder transformer model.

    This class is almost the same as x_transformers.XTransformer, but leaves
    flexibility to modify the network architecture and condition input method.

    Args:
        num_rvq_layers (int): Number of RVQ layers in decoder
        shared (bool): if the token ids in each RVQ level have the same range.

        duplicate_input_in_dec (bool): If True, the input stream is also
            duplicated in the decoder input, similar to the online case
        dec_output_emb_dim (int): Dimension to project the output stream to before
            concatenating with the input stream in the decoder.
            Only in use when duplicate_input_in_dec is True.
        Other args are the same as in models.py
    """

    def __init__(
        self,
        enc_dim: int = 512,
        dec_dim: int = 512,
        enc_depth: int = 6,
        dec_depth: int = 6,
        enc_heads: int = 8,
        dec_heads: int = 8,
        enc_num_tokens: int = 1024,
        dec_num_tokens: int = 1024,
        enc_max_seq_len: int = 512,
        dec_max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        pad_value: int = 0,
        input_emb_dim: int = 128,
        num_rvq_layers: int = 4,  # Modified
        shared: bool = True,
        duplicate_input_in_dec: bool = False,
        dec_output_emb_dim: int = 128,
        attention_layer_configs: Optional[dict] = None,
    ):
        super().__init__()
        if attention_layer_configs is None:
            attention_layer_configs = {}

        self.input_emb_dim = input_emb_dim
        encoder_emb = nn.Linear(input_emb_dim, enc_dim)

        self.encoder = TransformerWrapperNoInitEmb(
            attn_layers=Encoder(
                dim=enc_dim,
                depth=enc_depth,
                heads=enc_heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                attention_layer_configs=attention_layer_configs,
            ),
            num_tokens=enc_num_tokens,
            max_seq_len=enc_max_seq_len,
            return_only_embed=True,
            token_emb=encoder_emb,
        )
        self.decoder = DecoderTransformerMultiOut(
            dim=dec_dim,
            depth=dec_depth,
            heads=dec_heads,
            num_tokens=dec_num_tokens,
            max_seq_len=dec_max_seq_len,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            pad_value=pad_value,
            cross_attend=True,
            num_rvq_layers=num_rvq_layers,
            shared=shared,
            online=duplicate_input_in_dec,
            future_visibility=0,
            input_emb_dim=input_emb_dim,
            output_emb_dim=dec_dim,
            attention_layer_configs=attention_layer_configs,
        )

        # Save arguments
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_heads = enc_heads
        self.dec_heads = dec_heads
        self.enc_num_tokens = enc_num_tokens
        self.dec_num_tokens = dec_num_tokens
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.pad_value = pad_value

        self.num_rvq_layers = num_rvq_layers
        self.shared = shared

        self.duplicate_input_in_dec = duplicate_input_in_dec

    def forward(
        self,
        x_enc,
        x_dec,
        enc_mask=None,
        dec_mask=None,
        return_attn_z_loss=False,
        dec_inst_tokens=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder-decoder.

        Returns (See DecoderTransformerMultiOut):
            logits (Tensor): output logits, shape (B, K, T, num_tokens).
            logits_mask: output logits mask, such that logits[logits_mask]
                provides only valid logits. Shape (B, K, T)
            Optional attn_z_loss
        """
        if return_attn_z_loss:
            enc, cache = self.encoder(
                x_enc,
                mask=enc_mask,
                return_embeddings=True,
                return_attn_z_loss=True,
            )
            z_loss_enc = cache.attn_z_loss
            dec, cache = self.decoder(
                x_dec,
                context=enc,
                context_mask=enc_mask,
                mask=dec_mask,
                return_attn_z_loss=True,
            )
            z_loss_dec = cache.attn_z_loss
            return dec, z_loss_enc + z_loss_dec
        else:
            enc = self.encoder(x_enc, mask=enc_mask, return_embeddings=True)
            input_emb = x_enc if self.duplicate_input_in_dec else None
            dec = self.decoder(
                x_dec,
                context=enc,
                context_mask=enc_mask,
                mask=dec_mask,
                inst_tokens=dec_inst_tokens,
                input_emb=input_emb,
            )
            return dec

    @torch.no_grad()
    def generate(
        self,
        seq_in,
        seq_len,
        seq_out_start=None,
        mask=None,
        attn_mask=None,
        dec_inst_tokens=None,
        guidance_scale: float = 1.0,  # parameter for classifier free guidance
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            seq_in: Encoder input (B, T_enc)
            seq_len: Output sequence length
            mask: Encoder attention mask
            attn_mask: Encoder positional mask

        Returns:
            Tensor: Generated sequences (B, K, seq_len - 1)
        """
        encodings = self.encoder(
            seq_in, mask=mask, attn_mask=attn_mask, return_embeddings=True
        )
        input_emb = seq_in if self.duplicate_input_in_dec else None
        return self.decoder.generate(
            seq_len=seq_len,
            seq_out_start=seq_out_start,
            context=encodings,
            context_mask=mask,
            inst_tokens=dec_inst_tokens,
            input_emb=input_emb,
            guidance_scale=guidance_scale,
            **kwargs,
        )


class PrefixDecoderTransformerMultiOut(
    AutoregressiveWrapper, BaseGenerationMixin
):
    """
    Simple implementation of a prefix decoder.
    This is a offline decoder-only model that takes input as prefix and generates the output.

    Transform Decoder with support for multiple RVQ outputs and delay patterns
    Implements staggered generation strategy for parallel RVQ layer prediction.

    Attributes:
        num_rvq_layers (int): Number of RVQ layers to predict
        shared (bool): if the token ids in each RVQ level have the same range.

        online (bool): If True, the model is in online mode and takes in concatenated
            input and output embeddings.
        future_visibility (int): Number of tokens to delay the input stream.
        input_emb_dim (int): Dimension of the input stream embeddings before concatenation.
        output_emb_dim (int): Dimension of the output stream embeddings before concatenation.

        Other args identical to models.py
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        pad_value: int = 0,
        cross_attend: bool = False,
        num_rvq_layers: int = 4,  # Modified
        shared: bool = True,  # Modified
        input_emb_dim: int = 128,
        attention_layer_configs: Optional[dict] = None,
    ):

        # Skip the init of the AutoregressiveWrapper class
        super(AutoregressiveWrapper, self).__init__()

        if attention_layer_configs is None:
            attention_layer_configs = {}

        self.output_emb = DelayPatternEmbedder(
            dim=dim,
            num_tokens=num_tokens,
            num_rvq_layers=num_rvq_layers,
            shared=shared,
        )

        self.input_emb_dim = input_emb_dim
        self.input_emb_fc = nn.Linear(input_emb_dim, dim)

        print("\n")
        print(f"{'Parameter':<15} | {'Value'}")
        print("-" * 40)
        print(f"{'Dim':<15} | {dim}")
        print(f"{'Depth':<15} | {depth}")
        print(f"{'Heads':<15} | {heads}")
        print(f"{'Attn Dropout':<15} | {attn_dropout}")
        print(f"{'FF Dropout':<15} | {ff_dropout}")
        print(f"{'CrossAttend':<15} | {cross_attend}")
        print(f"{'Num Tokens':<15} | {num_tokens}")
        print(f"{'Num RVQ Layers':<15} | {num_rvq_layers}")
        print(f"{'Max Seq Len':<15} | {max_seq_len}")

        self.decoder = TransformerWrapper(
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                cross_attend=cross_attend,
                attention_layer_configs=attention_layer_configs,
            ),
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            token_emb=nn.Identity(),  # Needed to circumvent the 'embedding' layer.
            to_logits=MultiOutToLogits(num_rvq_layers, dim, num_tokens),
        )

        # Use Delay Pattern
        delays = list(range(num_rvq_layers))
        self.pattern_provider = DelayedPatternProvider(
            n_q=num_rvq_layers, delays=delays
        )

        # Save arguments
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len + 1  # +1 for the pattern token
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.pad_value = pad_value

        self.num_rvq_layers = num_rvq_layers

        # token used to temporarily pattern things integer id. Will be replaced by inst_tokens.
        self.temp_token = self.num_tokens
        self.shared = shared

    @property
    def net(self):
        return self.decoder

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inst_tokens: Optional[torch.Tensor] = None,
        input_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder, used for training.

        Args:
            x (Tensor): Shape (B, K, T):

        Output:
            logits (Tensor): output logits, shape (B, K, T, num_tokens).
            logits_mask: output logits mask, such that logits[logits_mask]
                provides only valid logits. Shape (B, K, T)
        """

        B, K, T = x.shape
        pattern = self.pattern_provider.get_pattern(T)
        x, _, sequence_mask = pattern.build_pattern_sequence(
            x,
            special_token=self.temp_token,
            keep_only_valid_steps=True,
        )

        # Replace temporary patterning tokens with inst_tokens
        x = self._replace_temp_token_with_inst_tokens(x, inst_tokens)

        output_embedded = self.output_emb(x)
        input_embedded = self.input_emb_fc(input_emb)
        prefix_length = input_emb.shape[1]

        model_input = torch.cat([input_embedded, output_embedded], dim=1)

        logits = self.decoder(
            model_input, mask=mask, **kwargs
        )  # [B, K, S, num_tokens]

        logits = logits[:, :, prefix_length:, :]

        logits = logits.permute(0, 3, 1, 2)  # [B, num_tokens, K, S]

        logits, _, logits_mask = pattern.revert_pattern_logits(
            logits, float("nan"), keep_only_valid_steps=True
        )

        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, num_tokens]

        logits_mask = logits_mask.unsqueeze(0).expand(B, *logits_mask.shape)
        return logits, logits_mask

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate(
        self,
        seq_len: int,
        seq_out_start: Optional[torch.Tensor] = None,
        input_emb: None | torch.Tensor = None,
        temperature: float = 1.0,
        filter_logits_fn: (
            str | Callable | list[str | Callable]
        ) = top_k_multi_out,
        filter_kwargs: dict | list[dict] = dict(),
        cache_kv: bool = True,
        display_pbar: bool = False,
        inst_tokens: torch.Tensor = None,
        guidance_scale: float = 1.0,  # parameter for classifier free guidance
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate function for multi-RVQ layer decoder models with delay pattern
        support.

        Args:
            seq_len (int): Number of steps to generate
            batch_size (int): Number of sequences to generate
            temperature (float): Sampling temperature
            filter_logits_fn: Logit filtering function(s)
            filter_kwargs: Arguments for filtering functions
            cache_kv (bool): Cache key/value pairs
            display_pbar (bool): Show progress bar
            inst_tokens (torch.Tensor): instrument ids to use as prompt
            guidance_scale (float): Classifier-free guidance scale

        Returns:
            Tensor: Generated sequences (B, K, seq_len - 1)
        """
        # Validate parameters and get batch_size and device
        batch_size, device = self._validate_generation_params(
            inst_tokens, seq_out_start, input_emb, guidance_scale
        )

        # Process filter functions
        filter_logits_fn, filter_kwargs = self._process_filter_functions(
            filter_logits_fn, filter_kwargs
        )

        # Initialize pattern and output sequence
        out, prompt_length, post_mask, patterned_seq_out, seq_out_mask = (
            self._initialize_generation_pattern(
                seq_len, batch_size, device, seq_out_start, inst_tokens
            )
        )

        # Set flags
        greedy = temperature == 0.0

        # Initialize cache for KV caching
        cache = None

        # Precompute input embeddings for prefix decoder
        input_embedded = self.input_emb_fc(input_emb)

        pbar = tqdm(
            range(prompt_length, seq_len),
            disable=not display_pbar,
            desc="Sampling",
        )
        for curr_sample_step in pbar:
            # Prepare model input (prefix-specific logic)
            embedded = self.output_emb(out)  # [B, S, D]
            model_input = torch.cat([input_embedded, embedded], dim=1)

            # Forward pass with optional KV caching
            if cache_kv:
                logits, intermediates = self.net(
                    model_input,
                    mask=None,
                    cache=cache,
                    return_intermediates=True,
                    **kwargs,
                )
                # Update cache from intermediates
                if hasattr(self.net, "can_cache_kv") and self.net.can_cache_kv:
                    cache = intermediates
            else:
                logits = self.net(
                    model_input,
                    mask=None,
                    return_intermediates=False,
                    **kwargs,
                )

            logits = logits[:, :, -1]  # Get logits for next token

            # Sample next tokens using mixin method
            samples = self._sample_next_tokens(
                logits,
                temperature,
                greedy,
                filter_logits_fn,
                filter_kwargs,
                curr_sample_step,
                out.shape[-1],
                guidance_scale,
            )

            out = torch.cat([out, samples], dim=-1)

            # Apply pattern constraints using mixin method
            out = self._apply_pattern_constraints(
                out,
                curr_sample_step,
                seq_out_start,
                patterned_seq_out,
                seq_out_mask,
                post_mask,
                inst_tokens,
            )

        # Finalize generation using mixin method
        return self._finalize_generation(out, guidance_scale)

    def _replace_temp_token_with_inst_tokens(
        self, x: torch.Tensor, inst_tokens: torch.Tensor
    ) -> torch.Tensor:
        temp_token_mask = x == self.temp_token
        inst_tokens_expanded = inst_tokens.view(x.shape[0], 1, 1).expand_as(x)
        x[temp_token_mask] = inst_tokens_expanded[temp_token_mask]
        return x


class OnlinePrefixDecoderTransformerMultiOut(DecoderTransformerMultiOut):
    """
    Simple implementation of an online chunked prediction model by prefix decoder.
    This is a online decoder-only model that takes input context,
    and its own previous output as prefix and generates a chunk of prediction.

    Transform Decoder with support for multiple RVQ outputs and delay patterns
    Implements staggered generation strategy for parallel RVQ layer prediction.

    Attributes:
        num_rvq_layers (int): Number of RVQ layers to predict
        shared (bool): if the token ids in each RVQ level have the same range.

        online (bool): If True, the model is in online mode and takes in concatenated
            input and output embeddings.
        future_visibility (int): Number of tokens to delay the input stream.
        input_emb_dim (int): Dimension of the input stream embeddings before concatenation.
        output_emb_dim (int): Dimension of the output stream embeddings before concatenation.

        Other args identical to models.py
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        pad_value: int = 0,
        cross_attend: bool = False,
        num_rvq_layers: int = 4,  # Modified
        shared: bool = True,  # Modified
        input_emb_dim: int = 128,
        attention_layer_configs: Optional[dict] = None,
        future_visibility: int = 0,
        output_emb_dim: int = 128,
        chunk_length: int = 10,
        chunk_start_prob: float = 0.1,
        use_beat_phase: bool = False,
        time_sig_vocab_size: int = 16,
        use_beat_phase_dit_cond: bool = False,
        beat_dit_cond_dim: Optional[int] = None,
        beat_dit_cond_mlp_expansion: int = 4,
        beat_dit_cond_minimal: bool = False,
        cond_dropout_p: float = 0.0,
        beat_phase_noise_std: float = 0.0,
        use_beat_phase_aux_head: bool = False,
        beat_phase_aux_head_hidden_dim: int = 256,
        use_chroma_dit_cond: bool = False,
        chroma_dim: int = 12,
        chroma_dit_cond_hidden_dim: Optional[int] = None,
        use_chroma_aux_head: bool = False,
        chroma_aux_head_hidden_dim: int = 256,
        chroma_aux_head_linear: bool = False,
        chroma_aux_horizons: Optional[Sequence[int]] = None,
        chroma_aux_deep_supervision_layers: Optional[Sequence[int]] = None,
        use_multipitch_aux_head: bool = False,
        multipitch_aux_head_hidden_dim: int = 256,
        multipitch_dim: int = 128,
        use_cqt_aux_head: bool = False,
        cqt_aux_head_hidden_dim: int = 256,
        cqt_dim: int = 84,
        use_input_cqt_aux_head: bool = False,
        input_cqt_aux_head_hidden_dim: int = 256,
        input_cqt_dim: int = 84,
        use_beat_phase_aux_head_full: bool = False,
        beat_phase_aux_head_full_hidden_dim: int = 256,
        use_multipitch_future_aux_head: bool = False,
        multipitch_future_aux_head_hidden_dim: int = 256,
        use_cqt_future_aux_head: bool = False,
        cqt_future_aux_head_hidden_dim: int = 256,
        use_target_token_future_aux_head: bool = False,
        target_token_future_aux_head_hidden_dim: int = 256,
        use_coupled_target_token_future_head: bool = False,
        coupled_target_token_future_head_hidden_dim: int = 256,
        future_aux_offsets: Optional[Sequence[int]] = None,
    ):
        # init with DecoderTransformerMultiOut's init
        # The max_seq_len will be larger than the actual sequence length seen by the model,
        # but that's fine.
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            pad_value=pad_value,
            cross_attend=cross_attend,
            num_rvq_layers=num_rvq_layers,
            shared=shared,
            online=True,  # Always online for this class
            future_visibility=future_visibility,
            input_emb_dim=input_emb_dim,
            output_emb_dim=output_emb_dim,
            attention_layer_configs=attention_layer_configs,
            external_pos=False,  # Always disable external position encoding for now.
            cond_method="add",  # Always use add method for now.
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
            chroma_aux_head_linear=chroma_aux_head_linear,
            chroma_aux_horizons=chroma_aux_horizons,
            chroma_aux_deep_supervision_layers=chroma_aux_deep_supervision_layers,
            use_multipitch_aux_head=use_multipitch_aux_head,
            multipitch_aux_head_hidden_dim=multipitch_aux_head_hidden_dim,
            multipitch_dim=multipitch_dim,
            use_cqt_aux_head=use_cqt_aux_head,
            cqt_aux_head_hidden_dim=cqt_aux_head_hidden_dim,
            cqt_dim=cqt_dim,
            use_input_cqt_aux_head=use_input_cqt_aux_head,
            input_cqt_aux_head_hidden_dim=input_cqt_aux_head_hidden_dim,
            input_cqt_dim=input_cqt_dim,
            use_beat_phase_aux_head_full=use_beat_phase_aux_head_full,
            beat_phase_aux_head_full_hidden_dim=beat_phase_aux_head_full_hidden_dim,
            use_multipitch_future_aux_head=use_multipitch_future_aux_head,
            multipitch_future_aux_head_hidden_dim=multipitch_future_aux_head_hidden_dim,
            use_cqt_future_aux_head=use_cqt_future_aux_head,
            cqt_future_aux_head_hidden_dim=cqt_future_aux_head_hidden_dim,
            use_target_token_future_aux_head=use_target_token_future_aux_head,
            target_token_future_aux_head_hidden_dim=target_token_future_aux_head_hidden_dim,
            use_coupled_target_token_future_head=use_coupled_target_token_future_head,
            coupled_target_token_future_head_hidden_dim=coupled_target_token_future_head_hidden_dim,
            future_aux_offsets=future_aux_offsets,
        )

        self.chunk_length = chunk_length
        self.chunk_start_prob = chunk_start_prob

    def _build_dit_condition(
        self,
        beat_cond_padded: Optional[torch.Tensor],
        bpm_log: Optional[torch.Tensor],
        time_sig_num: Optional[torch.Tensor],
        start_frame: int,
        end_frame: int,
        ref_dtype: torch.dtype,
        time_sig_den: Optional[torch.Tensor] = None,
        time_sig_change: Optional[torch.Tensor] = None,
        tempo_change: Optional[torch.Tensor] = None,
        local_bpm_log_padded: Optional[torch.Tensor] = None,
        input_chroma_padded: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Project the padded conditioning signals (beat phase + chroma)
        sliced to ``[start_frame:end_frame]`` into a condition tensor consumed
        by x_transformers' AdaptiveLayerNorm / AdaptiveLayerScale. The beat
        and chroma projections are summed when both are enabled, since the
        decoder accepts a single ``[B, T, dim_condition]`` channel per layer.

        Returns None if no conditioning is enabled or no signals are
        provided. ``local_bpm_log_padded`` and ``input_chroma_padded`` mirror
        ``beat_cond_padded`` along the time axis (same edge-repeat padding).
        """
        cond = None
        if self.use_beat_phase_dit_cond and beat_cond_padded is not None:
            bc_slice = beat_cond_padded[:, start_frame:end_frame, :]
            local_bpm_slice = (
                local_bpm_log_padded[:, start_frame:end_frame]
                if local_bpm_log_padded is not None
                else None
            )
            cond = self.beat_cond_projector(
                bc_slice.to(ref_dtype),
                bpm_log.to(ref_dtype),
                time_sig_num,
                time_sig_den=time_sig_den,
                time_sig_change=time_sig_change,
                tempo_change=tempo_change,
                local_bpm_log=local_bpm_slice,
            )
        if self.use_chroma_dit_cond and input_chroma_padded is not None:
            ch_slice = input_chroma_padded[:, start_frame:end_frame, :]
            chroma_cond = self.chroma_cond_projector(ch_slice.to(ref_dtype))
            cond = chroma_cond if cond is None else cond + chroma_cond
        return cond

    @property
    def net(self):
        return self.decoder

    def prepare_input(
        self,
        x: torch.Tensor,
        inst_tokens: torch.Tensor,
        input_emb: torch.Tensor,
        beat_cond: Optional[torch.Tensor] = None,
        bpm_log: Optional[torch.Tensor] = None,
        time_sig_num: Optional[torch.Tensor] = None,
        local_bpm_log: Optional[torch.Tensor] = None,
        input_chroma: Optional[torch.Tensor] = None,
        target_chroma: Optional[torch.Tensor] = None,
        target_multipitch: Optional[torch.Tensor] = None,
        target_velocity: Optional[torch.Tensor] = None,
        target_cqt: Optional[torch.Tensor] = None,
        input_cqt: Optional[torch.Tensor] = None,
        context_length_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, int, torch.Tensor, torch.Tensor]:
        """Prepare input embeddings and output tokens for online training.

        This method handles chunking the input sequence into smaller segments for online
        training, applies delay patterns, and prepares the input embeddings and output
        tokens for the forward pass.

        Args:
            x (torch.Tensor): Output tokens of shape [B, K, T] where B is batch size,
                K is number of RVQ layers, and T is sequence length.
            inst_tokens (torch.Tensor): Instrument tokens for each batch item.
            input_emb (torch.Tensor): Input embeddings of shape [B, T, D] where D is
                the embedding dimension.
            beat_cond (torch.Tensor, optional): Per-frame beat/bar phase,
                shape [B, T, 4]. Only used when ``use_beat_phase=True``.
            bpm_log (torch.Tensor, optional): Per-sample log(bpm/120), [B].
            time_sig_num (torch.Tensor, optional): Per-sample time signature
                numerator (long), [B].

        Returns:
            Tuple containing prepared inputs for the model forward pass.
        """

        # Decide output context length (which inner-loop we are in)
        B, K, T = x.shape
        # Apply delay pattern to the output
        pattern = self.pattern_provider.get_pattern(T)
        x_patterned, _, sequence_mask = pattern.build_pattern_sequence(
            x,
            special_token=self.temp_token,
            keep_only_valid_steps=True,
        )
        # x_patterned_full carries the un-trimmed patterned tokens so Phase K
        # can read targets at mangled position p+δ. The fv<=0 branch right-
        # pads it below when Phase K is active so target slices at p+δ stay
        # in-bounds (the encoder still gets no future-input visibility — the
        # padding is supervision-only).
        x_patterned_full = x_patterned
        # Replace temporary patterning tokens with inst_tokens
        x_patterned = self._replace_temp_token_with_inst_tokens(
            x_patterned, inst_tokens
        )

        # Here we assume the max duration of generation
        # is divisible by chunk_length.
        if T % self.chunk_length != 0:
            raise ValueError(
                f"Sequence length T={T} must be divisible by chunk_length={self.chunk_length}."
            )

        max_duration_gen = T - max(0, self.future_visibility)
        possible_context_lengths = np.arange(
            0, max_duration_gen, self.chunk_length
        )
        if context_length_override is not None:
            # KD path: caller imposes a specific context_length so student and
            # teacher predict the same target tokens. Must leave room for the
            # chunk: ``0 <= context_length <= max_duration_gen - chunk_length``.
            cl = int(context_length_override)
            if cl < 0 or cl + self.chunk_length > max_duration_gen:
                raise ValueError(
                    f"context_length_override={cl} out of range "
                    f"[0, {max_duration_gen - self.chunk_length}]"
                )
            context_length = cl
        else:
            context_length = np.random.choice(possible_context_lengths)

        # The start index is always 0.
        # Because the input is randomly chunked, we do not random the start index.
        context_end_idx = context_length

        # Adjust input and output according to future visibility, and get input and output context
        if self.future_visibility <= 0:
            # Phase K with fv<=0: right-pad the full target-token tensor with
            # pad_value so the aux-head label slice at p+δ stays in bounds.
            # This is supervision only — input_emb / beat_cond etc. are NOT
            # extended, so the encoder still has zero future-mix visibility.
            if (
                self.use_target_token_future_aux_head
                and self.future_aux_offsets
            ):
                max_off = max(self.future_aux_offsets)
                # Match the fv>0 layout: we need room for context_end_idx
                # (= context_length + 1) + δ_max - 1 + chunk_length, i.e.
                # padding by max_off + chunk_length covers the worst case.
                pad_len = max_off + self.chunk_length
                output_pad = torch.full(
                    (x_patterned_full.shape[0], x_patterned_full.shape[1], pad_len),
                    self.pad_value,
                    device=x_patterned_full.device,
                    dtype=x_patterned_full.dtype,
                )
                x_patterned_full = torch.cat(
                    [x_patterned_full, output_pad], dim=2
                )
            input_emb = self.pad_input_embs_for_delay(input_emb)
            if beat_cond is not None:
                beat_cond = self.pad_beat_cond_for_delay(beat_cond)
            if local_bpm_log is not None:
                local_bpm_log = self.pad_local_bpm_for_delay(local_bpm_log)
            # input_chroma / target_chroma have the same per-frame structure
            # as beat_cond — reuse the edge-repeat pad.
            if input_chroma is not None:
                input_chroma = self.pad_beat_cond_for_delay(input_chroma)
            if target_chroma is not None:
                target_chroma = self.pad_beat_cond_for_delay(target_chroma)
            if target_multipitch is not None:
                target_multipitch = self.pad_beat_cond_for_delay(target_multipitch)
            if target_velocity is not None:
                target_velocity = self.pad_beat_cond_for_delay(target_velocity)
            if target_cqt is not None:
                target_cqt = self.pad_beat_cond_for_delay(target_cqt)
            if input_cqt is not None:
                input_cqt = self.pad_beat_cond_for_delay(input_cqt)
            # Here we +1 to include the BOS.
            context_end_idx += 1
        else:
            # Phase K needs the un-trimmed patterned tokens so the future
            # target slice at p+δ has room beyond the chunk end. Build the
            # un-trimmed version first and slice the regular (trimmed) view
            # from it so we don't pay the pattern build twice.
            x_patterned_full = self.pad_output_tokens_for_delay(
                x_patterned, trim_end=False
            )
            x_patterned = x_patterned_full[:, :, : -self.future_visibility]
            # For future_visibility > 0,
            # BOS is included in self.future_visibility tokens in the beginning.
            context_end_idx += self.future_visibility

        input_context = input_emb[:, :context_end_idx, :]
        beat_cond_context = (
            beat_cond[:, :context_end_idx, :] if beat_cond is not None else None
        )
        output_context = x_patterned[:, :, :context_end_idx]
        output = x_patterned[
            :, :, context_end_idx : context_end_idx + self.chunk_length
        ]
        targets = x_patterned[
            :, :, context_end_idx : context_end_idx + self.chunk_length
        ]
        pred_start_idx = context_end_idx - 1
        pred_end_idx = pred_start_idx + self.chunk_length

        assert input_context.shape[1] == output_context.shape[-1]

        in_embedded = self.get_input_embedding(
            output_context,
            input_context,
            beat_cond=beat_cond_context,
            bpm_log=bpm_log,
            time_sig_num=time_sig_num,
        )
        out_embedded = self.ln_out(self.output_emb(output))
        embedded = torch.cat([in_embedded, out_embedded], dim=1)

        # sequence_mask is a triangle for the first 1 to N+1 steps
        #   because of delay pattern (e.g. 1st step only predict 1st rvq layer).
        # Here we +1 to skip the bos which is False accorss all rvq layers
        logits_mask = sequence_mask[:, pred_start_idx + 1 : pred_end_idx + 1]
        # Add batch dim
        logits_mask = logits_mask.unsqueeze(0).expand(
            input_emb.shape[0], *logits_mask.shape
        )

        assert logits_mask.shape[-1] == self.chunk_length
        assert targets.shape[-1] == self.chunk_length

        # For DiT-style per-layer conditioning, expose the padded beat_cond
        # window aligned with ``embedded``. ``embedded`` has S = context_end_idx
        # + chunk_length frames; the corresponding slice of the padded beat_cond
        # gives one condition row per frame. local_bpm_log is similarly aligned.
        beat_cond_aligned = (
            beat_cond[:, : context_end_idx + self.chunk_length, :]
            if beat_cond is not None
            else None
        )
        local_bpm_aligned = (
            local_bpm_log[:, : context_end_idx + self.chunk_length]
            if local_bpm_log is not None
            else None
        )
        input_chroma_aligned = (
            input_chroma[:, : context_end_idx + self.chunk_length, :]
            if input_chroma is not None
            else None
        )
        target_chroma_aligned = (
            target_chroma[:, : context_end_idx + self.chunk_length, :]
            if target_chroma is not None
            else None
        )
        target_multipitch_aligned = (
            target_multipitch[:, : context_end_idx + self.chunk_length, :]
            if target_multipitch is not None
            else None
        )
        target_velocity_aligned = (
            target_velocity[:, : context_end_idx + self.chunk_length, :]
            if target_velocity is not None
            else None
        )
        target_cqt_aligned = (
            target_cqt[:, : context_end_idx + self.chunk_length, :]
            if target_cqt is not None
            else None
        )
        input_cqt_aligned = (
            input_cqt[:, : context_end_idx + self.chunk_length, :]
            if input_cqt is not None
            else None
        )

        return (
            embedded,
            pred_start_idx,
            pred_end_idx,
            targets,
            logits_mask,
            beat_cond_aligned,
            local_bpm_aligned,
            input_chroma_aligned,
            target_chroma_aligned,
            target_multipitch_aligned,
            target_velocity_aligned,
            target_cqt_aligned,
            input_cqt_aligned,
            x_patterned_full,
        )

    def forward(
        self,
        x: torch.Tensor,
        inst_tokens: torch.Tensor,
        input_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        beat_cond: Optional[torch.Tensor] = None,
        bpm_log: Optional[torch.Tensor] = None,
        time_sig_num: Optional[torch.Tensor] = None,
        time_sig_den: Optional[torch.Tensor] = None,
        time_sig_change: Optional[torch.Tensor] = None,
        tempo_change: Optional[torch.Tensor] = None,
        local_bpm_log: Optional[torch.Tensor] = None,
        input_chroma: Optional[torch.Tensor] = None,
        target_chroma: Optional[torch.Tensor] = None,
        target_multipitch: Optional[torch.Tensor] = None,
        target_velocity: Optional[torch.Tensor] = None,
        target_cqt: Optional[torch.Tensor] = None,
        input_cqt: Optional[torch.Tensor] = None,
        context_length_override: Optional[int] = None,
        **kwargs,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[dict],
        Optional[dict],
    ]:
        """
        Forward pass of decoder, used for training.

        Returns 9-tuple: (logits_pred, logits_mask, targets,
                           beat_aux_pred, beat_aux_target,
                           chroma_aux_pred, chroma_aux_target,
                           chroma_aux_dsv_preds,
                           extra_aux).
        Aux fields are None when the corresponding head is disabled.
        ``chroma_aux_dsv_preds`` is None when deep supervision is off, else
        a dict {layer_idx: [B, T, K*chroma_dim]} with predictions from each
        intermediate-layer head.
        ``extra_aux`` is None when no multipitch/CQT/beat-phase-full aux head
        is enabled, else a dict with optional keys:
            mp_pred:           [B, S, multipitch_dim]      (presence logits)
            mp_target:         [B, S, multipitch_dim]      (presence binary)
            cqt_pred:          [B, S, cqt_dim]
            cqt_target:        [B, S, cqt_dim]
            beat_full_pred:    [B, S, 4 + 1 + ts_vocab]
            beat_full_phase_target: [B, S, 4]
        """
        # Phase-noise injection on beat_cond at training only. Forces the
        # encoder to use input-mix audio for beat localization rather than
        # relying entirely on the (clean) beat-phase prior. At inference
        # self.training=False ⇒ clean cond, no noise.
        if (
            beat_cond is not None
            and self.training
            and self.beat_phase_noise_std > 0.0
        ):
            beat_cond = beat_cond + torch.randn_like(beat_cond) * self.beat_phase_noise_std

        (
            embedded,
            pred_start_idx,
            pred_end_idx,
            targets,
            logits_mask,
            beat_cond_aligned,
            local_bpm_aligned,
            input_chroma_aligned,
            target_chroma_aligned,
            target_multipitch_aligned,
            target_velocity_aligned,
            target_cqt_aligned,
            input_cqt_aligned,
            x_patterned_full,
        ) = self.prepare_input(
            x,
            inst_tokens,
            input_emb,
            beat_cond=beat_cond,
            bpm_log=bpm_log,
            time_sig_num=time_sig_num,
            local_bpm_log=local_bpm_log,
            input_chroma=input_chroma,
            target_chroma=target_chroma,
            target_multipitch=target_multipitch,
            target_velocity=target_velocity,
            target_cqt=target_cqt,
            input_cqt=input_cqt,
            context_length_override=context_length_override,
        )

        # DiT-style per-layer conditioning. condition has shape
        # [B, S, dim_condition] aligned with ``embedded``. Beat and chroma
        # projections are summed inside ``_build_dit_condition`` when both
        # are enabled.
        decoder_extra = {}
        if self.use_beat_phase_dit_cond or self.use_chroma_dit_cond:
            cond_len = (
                beat_cond_aligned.shape[1]
                if beat_cond_aligned is not None
                else (
                    input_chroma_aligned.shape[1]
                    if input_chroma_aligned is not None
                    else None
                )
            )
            if cond_len is not None:
                condition = self._build_dit_condition(
                    beat_cond_aligned,
                    bpm_log,
                    time_sig_num,
                    start_frame=0,
                    end_frame=cond_len,
                    ref_dtype=embedded.dtype,
                    time_sig_den=time_sig_den,
                    time_sig_change=time_sig_change,
                    tempo_change=tempo_change,
                    local_bpm_log_padded=local_bpm_aligned,
                    input_chroma_padded=input_chroma_aligned,
                )
                if condition is not None:
                    if self.training and self.cond_dropout_p > 0.0:
                        keep_mask = (
                            torch.rand(
                                condition.shape[0], device=condition.device
                            )
                            >= self.cond_dropout_p
                        ).to(condition.dtype)
                        condition = condition * keep_mask.view(-1, 1, 1)
                    decoder_extra["condition"] = condition

        # When ANY aux head is on, ask x_transformers to return both logits
        # and the pre-logits hidden state ``[B, S, dim]``. When chroma deep
        # supervision is active, also return per-layer intermediates.
        any_aux = (
            self.use_beat_phase_aux_head
            or self.use_chroma_aux_head
            or self.use_multipitch_aux_head
            or self.use_cqt_aux_head
            or self.use_input_cqt_aux_head
            or self.use_beat_phase_aux_head_full
            or self.use_multipitch_future_aux_head
            or self.use_cqt_future_aux_head
            or self.use_target_token_future_aux_head
            or self.use_coupled_target_token_future_head
        )
        need_layer_hiddens = bool(
            self.use_chroma_aux_head and self.chroma_aux_deep_supervision_layers
        )
        layer_hiddens = None
        if any_aux:
            if need_layer_hiddens:
                (logits, hidden), intermediates = self.decoder(
                    embedded,
                    mask=mask,
                    return_logits_and_embeddings=True,
                    return_intermediates=True,
                    **decoder_extra,
                    **kwargs,
                )
                # ``layer_hiddens`` is a list of [B, S, dim] tensors, one per
                # transformer layer (pre-final-norm). We index by layer idx
                # specified in ``chroma_aux_deep_supervision_layers``.
                layer_hiddens = intermediates.layer_hiddens
            else:
                logits, hidden = self.decoder(
                    embedded,
                    mask=mask,
                    return_logits_and_embeddings=True,
                    **decoder_extra,
                    **kwargs,
                )
            beat_aux_pred = (
                self.beat_phase_aux_head(hidden)
                if self.use_beat_phase_aux_head
                else None
            )
            chroma_aux_pred = (
                self.chroma_aux_head(hidden)
                if self.use_chroma_aux_head
                else None
            )
            mp_aux_pred = (
                self.multipitch_aux_head(hidden)
                if self.use_multipitch_aux_head
                else None
            )
            cqt_aux_pred = (
                self.cqt_aux_head(hidden)
                if self.use_cqt_aux_head
                else None
            )
            input_cqt_aux_pred = (
                self.input_cqt_aux_head(hidden)
                if self.use_input_cqt_aux_head
                else None
            )
            beat_full_pred = (
                self.beat_phase_aux_head_full(hidden)
                if self.use_beat_phase_aux_head_full
                else None
            )
            mp_future_pred = (
                self.multipitch_future_aux_head(hidden)
                if self.use_multipitch_future_aux_head
                else None
            )
            cqt_future_pred = (
                self.cqt_future_aux_head(hidden)
                if self.use_cqt_future_aux_head
                else None
            )
            tt_future_pred = (
                self.target_token_future_aux_head(hidden)
                if self.use_target_token_future_aux_head
                else None
            )
            # Phase L: invoke the SHARED main ``to_logits`` on per-offset
            # trunk(hidden_chunk) outputs. Only the chunk window is needed
            # for the loss, so we slice first to keep K_off×num_rvq Linear
            # calls cheap. ``self.decoder.to_logits`` is the
            # MultiOutToLogits passed in at construction.
            if self.use_coupled_target_token_future_head:
                h_chunk = hidden[:, pred_start_idx:pred_end_idx, :]
                coupled_tt_future_pred = self.coupled_target_token_future_head(
                    h_chunk, self.decoder.to_logits,
                )  # [B, num_rvq, S_chunk, K_off, V]
            else:
                coupled_tt_future_pred = None
        else:
            logits = self.decoder(
                embedded, mask=mask, **decoder_extra, **kwargs
            )
            beat_aux_pred = None
            chroma_aux_pred = None
            mp_aux_pred = None
            cqt_aux_pred = None
            input_cqt_aux_pred = None
            beat_full_pred = None
            mp_future_pred = None
            cqt_future_pred = None
            tt_future_pred = None
            coupled_tt_future_pred = None

        beat_aux_target = (
            beat_cond_aligned if self.use_beat_phase_aux_head else None
        )

        # Build chroma aux target. For multi-horizon prediction
        # (``len(chroma_aux_horizons) > 1``), concatenate horizon-shifted
        # targets along the last dim and slice the prediction to T_valid.
        # Default (horizons == (0,)) is the legacy single-frame path.
        chroma_aux_target = None
        chroma_aux_dsv_preds: Optional[dict] = None
        if self.use_chroma_aux_head:
            if target_chroma_aligned is not None:
                horizons = self.chroma_aux_horizons
                if len(horizons) == 1 and horizons[0] == 0:
                    chroma_aux_target = target_chroma_aligned
                else:
                    max_h = max(horizons)
                    T_full = target_chroma_aligned.shape[1]
                    T_valid = T_full - max_h
                    if T_valid <= 0:
                        # Sequence too short for the requested horizons —
                        # fall back to single-frame to avoid empty tensors.
                        chroma_aux_target = target_chroma_aligned
                    else:
                        shifted = [
                            target_chroma_aligned[:, h:h + T_valid, :]
                            for h in horizons
                        ]
                        chroma_aux_target = torch.cat(shifted, dim=-1)
                        if chroma_aux_pred is not None:
                            chroma_aux_pred = chroma_aux_pred[:, :T_valid, :]
            # Run deep-supervision heads at each requested layer index.
            if (
                layer_hiddens is not None
                and self.chroma_aux_deep_supervision_layers
            ):
                chroma_aux_dsv_preds = {}
                for li in self.chroma_aux_deep_supervision_layers:
                    head = self.chroma_aux_dsv_heads[str(li)]
                    pred_li = head(layer_hiddens[li])
                    if (
                        chroma_aux_target is not None
                        and pred_li.shape[1] != chroma_aux_target.shape[1]
                    ):
                        # Slice DSV preds to T_valid as well when multi-horizon.
                        pred_li = pred_li[:, :chroma_aux_target.shape[1], :]
                    chroma_aux_dsv_preds[int(li)] = pred_li

        # Build extra_aux dict for multipitch / CQT / beat-phase-full heads.
        # All targets are already aligned with hidden along the time axis.
        extra_aux: Optional[dict] = None
        if (
            self.use_multipitch_aux_head
            or self.use_cqt_aux_head
            or self.use_input_cqt_aux_head
            or self.use_beat_phase_aux_head_full
            or self.use_multipitch_future_aux_head
            or self.use_cqt_future_aux_head
            or self.use_target_token_future_aux_head
            or self.use_coupled_target_token_future_head
        ):
            extra_aux = {}
            if self.use_multipitch_aux_head and mp_aux_pred is not None:
                extra_aux["mp_pred"] = mp_aux_pred
                if target_multipitch_aligned is not None:
                    extra_aux["mp_target"] = target_multipitch_aligned
            if self.use_cqt_aux_head and cqt_aux_pred is not None:
                extra_aux["cqt_pred"] = cqt_aux_pred
                if target_cqt_aligned is not None:
                    extra_aux["cqt_target"] = target_cqt_aligned
            if (
                self.use_input_cqt_aux_head
                and input_cqt_aux_pred is not None
            ):
                extra_aux["input_cqt_pred"] = input_cqt_aux_pred
                if input_cqt_aligned is not None:
                    extra_aux["input_cqt_target"] = input_cqt_aligned
            if (
                self.use_beat_phase_aux_head_full
                and beat_full_pred is not None
            ):
                extra_aux["beat_full_pred"] = beat_full_pred
                if beat_cond_aligned is not None:
                    extra_aux["beat_full_phase_target"] = beat_cond_aligned
            # Future aux: predict mp / cqt at frame t+δ_k from h_t.
            # T_valid = T - max(offsets) so every offset has a target frame.
            # Pred is [B, S, K, D] from the head; sliced to [:, :T_valid].
            # Target stacked from offset-shifted slices of aligned features.
            if self.future_aux_offsets and (
                self.use_multipitch_future_aux_head
                or self.use_cqt_future_aux_head
            ):
                offsets = self.future_aux_offsets
                max_off = max(offsets)
                if (
                    self.use_multipitch_future_aux_head
                    and mp_future_pred is not None
                    and target_multipitch_aligned is not None
                ):
                    T_full = target_multipitch_aligned.shape[1]
                    T_valid = T_full - max_off
                    if T_valid > 0:
                        mp_target_future = torch.stack(
                            [
                                target_multipitch_aligned[:, k:k + T_valid, :]
                                for k in offsets
                            ],
                            dim=2,
                        )  # [B, T_valid, K, D]
                        extra_aux["mp_future_pred"] = (
                            mp_future_pred[:, :T_valid, :, :]
                        )
                        extra_aux["mp_future_target"] = mp_target_future
                if (
                    self.use_cqt_future_aux_head
                    and cqt_future_pred is not None
                    and target_cqt_aligned is not None
                ):
                    T_full = target_cqt_aligned.shape[1]
                    T_valid = T_full - max_off
                    if T_valid > 0:
                        cqt_target_future = torch.stack(
                            [
                                target_cqt_aligned[:, k:k + T_valid, :]
                                for k in offsets
                            ],
                            dim=2,
                        )  # [B, T_valid, K, D]
                        extra_aux["cqt_future_pred"] = (
                            cqt_future_pred[:, :T_valid, :, :]
                        )
                        extra_aux["cqt_future_target"] = cqt_target_future
            # Phase K: predict target-stem patterned tokens at p+δ from h_t.
            # Hidden positions [pred_start_idx, pred_end_idx) are the same
            # slice the main next-token head predicts from; for offset δ the
            # mangled-position target is at context_end_idx + δ - 1 + i for
            # i in [0, chunk_length). x_patterned_full has the un-trimmed
            # patterned tokens so positions p+δ stay in bounds for δ<=fv.
            if (
                self.future_aux_offsets
                and self.use_target_token_future_aux_head
                and tt_future_pred is not None
                and x_patterned_full is not None
            ):
                offsets = self.future_aux_offsets
                context_end_idx = pred_start_idx + 1
                # tt_future_pred is [B, S, K_off, num_rvq, num_tokens]; slice
                # to the prediction window so it matches logits_mask shape.
                pred_window = tt_future_pred[
                    :, pred_start_idx:pred_end_idx, :, :, :
                ]
                # Rearrange to [B, num_rvq, chunk, K_off, num_tokens] so the
                # codebook axis matches `targets`/`logits_mask` layout.
                pred_window = pred_window.permute(0, 3, 1, 2, 4).contiguous()
                # Gather per-offset target slices from x_patterned_full.
                tt_target_slices = []
                for delta in offsets:
                    start = context_end_idx + int(delta) - 1
                    end = start + self.chunk_length
                    tt_target_slices.append(
                        x_patterned_full[:, :, start:end]
                    )
                # Stack along a new last axis: [B, num_rvq, chunk, K_off].
                tt_future_target = torch.stack(tt_target_slices, dim=-1)
                extra_aux["tt_future_pred"] = pred_window
                extra_aux["tt_future_target"] = tt_future_target
                extra_aux["tt_future_logits_mask"] = logits_mask
            # Phase L: coupled head shares Phase K's targets+mask (same task,
            # same vocabulary, same valid positions); only the prediction
            # pathway differs (shared main ``to_logits`` instead of separate
            # classifier). Build its own keys so both can be enabled side-by-
            # side and weighted independently.
            if (
                self.future_aux_offsets
                and self.use_coupled_target_token_future_head
                and coupled_tt_future_pred is not None
                and x_patterned_full is not None
            ):
                offsets = self.future_aux_offsets
                context_end_idx = pred_start_idx + 1
                # Match Phase K layout: pred axis order is
                # [B, num_rvq, S_chunk, K_off, V] — already produced that way
                # by CoupledTargetTokenFutureHead. No permute needed.
                coupled_target_slices = []
                for delta in offsets:
                    start = context_end_idx + int(delta) - 1
                    end = start + self.chunk_length
                    coupled_target_slices.append(
                        x_patterned_full[:, :, start:end]
                    )
                coupled_tt_future_target = torch.stack(
                    coupled_target_slices, dim=-1
                )  # [B, num_rvq, chunk, K_off]
                extra_aux["coupled_tt_future_pred"] = coupled_tt_future_pred
                extra_aux["coupled_tt_future_target"] = coupled_tt_future_target
                extra_aux["coupled_tt_future_logits_mask"] = logits_mask

        logits_pred = logits[:, :, pred_start_idx:pred_end_idx, :]
        assert logits_pred.shape[2] > 0
        assert logits_mask.shape[2] == logits_pred.shape[2]
        assert logits_pred.shape[2] == targets.shape[2]
        return (
            logits_pred, logits_mask, targets,
            beat_aux_pred, beat_aux_target,
            chroma_aux_pred, chroma_aux_target,
            chroma_aux_dsv_preds,
            extra_aux,
        )

    def _apply_pattern_constraints(
        self,
        out: torch.Tensor,
        curr_sample_step: int,
        curr_global_step: int,
        post_mask: torch.Tensor,
        inst_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply pattern-based constraints for online prefix decoder generation.

        The indexing is a bit different than that in decoder online model and prefix decoder.
        """
        # +1 because we start with 1 token (bos / instrument token)
        replace_mask = ~post_mask[..., curr_global_step + 1]
        if replace_mask.any():
            out[:, replace_mask, curr_sample_step] = self.temp_token
            # Replace pattern token with instrument id
            out = self._replace_temp_token_with_inst_tokens(out, inst_tokens)
        return out

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate_chunk(
        self,
        global_start_idx: int,
        context_emb: torch.Tensor,
        generate_len: int,
        inst_tokens: torch.Tensor,
        post_mask: torch.Tensor,
        temperature: float = 1.0,
        filter_logits_fn: (
            str | Callable | list[str | Callable]
        ) = top_k_multi_out,
        filter_kwargs: dict | list[dict] = dict(),
        cache_kv: bool = True,
        beat_cond_padded: Optional[torch.Tensor] = None,
        bpm_log: Optional[torch.Tensor] = None,
        time_sig_num: Optional[torch.Tensor] = None,
        time_sig_den: Optional[torch.Tensor] = None,
        time_sig_change: Optional[torch.Tensor] = None,
        tempo_change: Optional[torch.Tensor] = None,
        local_bpm_log_padded: Optional[torch.Tensor] = None,
        input_chroma_padded: Optional[torch.Tensor] = None,
        dit_modulation_precompute: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inner-loop generate function for multi-RVQ layer decoder models with delay pattern
        support. This function generates a chunk of output tokens given the input context.
        The input context contains the input tokens and the output tokens before the chunk.
        The output context contains the output tokens after the chunk.

        Args:
            global_start_idx (int): Global start index of the chunk.
            context_emb (torch.Tensor): Embeddings of the input context,
                contains both input and previous output.
            generate_len (int): Number of steps to generate.
            inst_tokens (torch.Tensor): Instrument tokens for each batch item.
            post_mask (torch.Tensor): Mask for the post-patterning tokens.
                Mainly for indicating where are the instrument tokens (act as BOS) at the
                beginning, and thus need to fill in instrument tokens instead of using
                the the sampled output token.
            temperature (float): Sampling temperature.
            filter_logits_fn: Logit filtering function(s).
            filter_kwargs: Arguments for filtering functions.
            cache_kv (bool): Cache key/value pairs.
            dit_modulation_precompute (bool): Inference-only optimisation.
                Precompute the DiT adaptive-norm modulations for the whole
                chunk as batched GEMMs instead of per-step launches. Output
                is equivalent up to GEMM-reduction-order float noise.
            **kwargs: Additional arguments for the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Generated output tokens and model input.
        """
        # Set flags
        greedy = temperature == 0.0

        # Initialize cache for KV caching
        cache = None

        # No need to precompute input embeddings as it is prepared by the outer loop.
        out = torch.zeros(
            context_emb.shape[0],
            self.num_rvq_layers,
            0,
            device=context_emb.device,
            dtype=torch.long,
        )

        model_input = context_emb
        # Track the model_input length BEFORE this iteration so we can build
        # the right per-frame DiT condition slice.
        context_len = context_emb.shape[1]

        any_dit_cond = (
            self.use_beat_phase_dit_cond or self.use_chroma_dit_cond
        ) and (beat_cond_padded is not None or input_chroma_padded is not None)

        # Chunk-ahead modulation precompute (inference-only): the KV-cached
        # steps s >= 1 consume the DiT condition for frames
        # [context_len, context_len + generate_len - 1). The conditioning
        # signals for those frames are already known here, so run the
        # projector, the adaptive MLP, and every per-layer to_gamma ONCE as
        # batched GEMMs and cache the resulting modulations; the AR loop
        # then applies per-frame slices instead of launching every
        # conditioning linear once per step.
        pc_state = None
        pc_cond_chunk = None
        if (
            dit_modulation_precompute
            and any_dit_cond
            and cache_kv
            and generate_len > 1
        ):
            pc_cond_chunk = self._build_dit_condition(
                beat_cond_padded,
                bpm_log,
                time_sig_num,
                start_frame=context_len,
                end_frame=context_len + generate_len - 1,
                ref_dtype=model_input.dtype,
                time_sig_den=time_sig_den,
                time_sig_change=time_sig_change,
                tempo_change=tempo_change,
                local_bpm_log_padded=local_bpm_log_padded,
                input_chroma_padded=input_chroma_padded,
            )
            if (
                pc_cond_chunk is not None
                and pc_cond_chunk.shape[1] == generate_len - 1
            ):
                pc_state = precompute_dit_gammas(
                    self.net.attn_layers, pc_cond_chunk
                )
            else:
                # Conditioning signals shorter than the chunk (sequence
                # end); fall back to the per-step path.
                pc_cond_chunk = None

        try:
            for curr_sample_step in range(generate_len):
                # Build the DiT condition slice that aligns with whatever
                # ``self.net`` will actually process at this step:
                #   * iter 0 OR cache_kv=False: the full model_input is
                #     processed (length = context_len + curr_sample_step).
                #     Pass condition for all those frames.
                #   * iter ≥ 1 with cache: x_transformers slices x to its
                #     last ``cache_age=1`` frame internally; we must mirror
                #     that and pass condition for ONLY that one frame.
                extra_call_kwargs = dict(kwargs)
                if any_dit_cond:
                    if cache_kv and cache is not None:
                        if pc_state is not None:
                            # Cached-gamma fast path. The raw cond slice is
                            # still passed so x_transformers' need_condition
                            # assert holds; its (1-frame) adaptive_mlp
                            # output is ignored by the patched forwards.
                            local_idx = curr_sample_step - 1
                            pc_state.activate(local_idx)
                            cond = pc_cond_chunk[
                                :, local_idx : local_idx + 1
                            ]
                        else:
                            cur_frame_idx = (
                                context_len + curr_sample_step - 1
                            )
                            cond = self._build_dit_condition(
                                beat_cond_padded,
                                bpm_log,
                                time_sig_num,
                                start_frame=cur_frame_idx,
                                end_frame=cur_frame_idx + 1,
                                ref_dtype=model_input.dtype,
                                time_sig_den=time_sig_den,
                                time_sig_change=time_sig_change,
                                tempo_change=tempo_change,
                                local_bpm_log_padded=local_bpm_log_padded,
                                input_chroma_padded=input_chroma_padded,
                            )
                    else:
                        if pc_state is not None:
                            pc_state.deactivate()
                        full_len = context_len + curr_sample_step
                        cond = self._build_dit_condition(
                            beat_cond_padded,
                            bpm_log,
                            time_sig_num,
                            start_frame=0,
                            end_frame=full_len,
                            ref_dtype=model_input.dtype,
                            time_sig_den=time_sig_den,
                            time_sig_change=time_sig_change,
                            tempo_change=tempo_change,
                            local_bpm_log_padded=local_bpm_log_padded,
                            input_chroma_padded=input_chroma_padded,
                        )
                    if cond is not None:
                        extra_call_kwargs["condition"] = cond

                # Forward pass with optional KV caching
                if cache_kv:
                    logits, intermediates = self.net(
                        model_input,
                        mask=None,
                        cache=cache,
                        return_intermediates=True,
                        **extra_call_kwargs,
                    )
                    # Update cache from intermediates
                    if (
                        hasattr(self.net, "can_cache_kv")
                        and self.net.can_cache_kv
                    ):
                        cache = intermediates
                else:
                    logits = self.net(
                        model_input,
                        mask=None,
                        return_intermediates=False,
                        **extra_call_kwargs,
                    )

                logits = logits[:, :, -1]  # Get logits for next token

                # Sample next tokens using mixin method
                samples = self._sample_next_tokens(
                    logits,
                    temperature,
                    greedy,
                    filter_logits_fn,
                    filter_kwargs,
                    curr_sample_step,
                    out.shape[-1],
                    guidance_scale=1.0,  # disable guidance for now
                )

                # append the samples to the output
                out = torch.cat([out, samples], dim=-1)

                # Apply pattern constraints using mixin method
                out = self._apply_pattern_constraints(
                    out,
                    curr_sample_step,
                    curr_global_step=global_start_idx + curr_sample_step,
                    post_mask=post_mask,
                    inst_tokens=inst_tokens,
                )

                # append the samples to the model input
                out_embedded = self.ln_out(self.output_emb(out))
                model_input = torch.cat([model_input, out_embedded], dim=1)
        finally:
            # Never leave stale cached gammas active: a later naive-path
            # generate on the same (patched) model must fall back cleanly.
            if pc_state is not None:
                pc_state.clear()

        return out

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate(
        self,
        seq_len: int,
        seq_out_start: Optional[torch.Tensor] = None,
        input_emb: None | torch.Tensor = None,
        temperature: float = 1.0,
        filter_logits_fn: (
            str | Callable | list[str | Callable]
        ) = top_k_multi_out,
        filter_kwargs: dict | list[dict] = dict(),
        cache_kv: bool = True,
        display_pbar: bool = False,
        inst_tokens: torch.Tensor = None,
        beat_cond: Optional[torch.Tensor] = None,
        bpm_log: Optional[torch.Tensor] = None,
        time_sig_num: Optional[torch.Tensor] = None,
        time_sig_den: Optional[torch.Tensor] = None,
        time_sig_change: Optional[torch.Tensor] = None,
        tempo_change: Optional[torch.Tensor] = None,
        local_bpm_log: Optional[torch.Tensor] = None,
        input_chroma: Optional[torch.Tensor] = None,
        dit_modulation_precompute: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate function for Online Prefix Decoder that
            predicts a chunk of multi-RVQ audio tokens in delay pattern.

        Args:
            seq_len (int): Number of steps to generate.
            seq_out_start (torch.Tensor): Start index of the output sequence.
            input_emb (torch.Tensor): Input embeddings of the input context.
            temperature (float): Sampling temperature
            filter_logits_fn: Logit filtering function(s)
            filter_kwargs: Arguments for filtering functions
            cache_kv (bool): Cache key/value pairs
            display_pbar (bool): Show progress bar
            inst_tokens (torch.Tensor): instrument ids to use as prompt
            beat_cond (torch.Tensor, optional): [B, T, 4] per-frame beat/bar
                phase (sin/cos). Required when use_beat_phase=True.
            bpm_log (torch.Tensor, optional): [B] log(bpm/120).
            time_sig_num (torch.Tensor, optional): [B] long time-sig numerator.

        Returns:
            Tensor: Generated sequences (B, K, seq_len - 1)
        """
        # NOTE: we do not support prompting for now.
        if seq_out_start is not None:
            raise NotImplementedError("Prompting is not supported yet.")

        # Validate parameters and get batch_size and device.
        # seq_out_start: we do not support CFG for now.
        batch_size, device = self._validate_generation_params(
            inst_tokens, seq_out_start, input_emb, guidance_scale=1.0
        )

        # Process filter functions
        filter_logits_fn, filter_kwargs = self._process_filter_functions(
            filter_logits_fn, filter_kwargs
        )

        # Initialize pattern and output sequence
        (
            output_tokens,
            prompt_length,
            post_mask,
            patterned_seq_out,
            seq_out_mask,
        ) = self._initialize_generation_pattern(
            seq_len, batch_size, device, seq_out_start, inst_tokens
        )

        # Handle online mode future visibility
        if self.future_visibility <= 0:
            input_emb = self.pad_input_embs_for_delay(input_emb)
            if beat_cond is not None:
                beat_cond = self.pad_beat_cond_for_delay(beat_cond)
            if local_bpm_log is not None:
                local_bpm_log = self.pad_local_bpm_for_delay(local_bpm_log)
            if input_chroma is not None:
                input_chroma = self.pad_beat_cond_for_delay(input_chroma)
        else:
            output_tokens = self.pad_output_tokens_for_delay(
                output_tokens, trim_end=False
            )

        start_indices = np.arange(0, seq_len, self.chunk_length)
        gen_lengths = np.minimum(self.chunk_length, seq_len - start_indices)

        display_pbar = True

        pbar = tqdm(
            range(len(start_indices)),
            disable=not display_pbar,
            desc="Generating chunks",
        )

        for i in pbar:
            start_idx = start_indices[i]
            gen_length = gen_lengths[i]
            cur_len = output_tokens.shape[-1]
            chunk_beat_cond = (
                beat_cond[:, :cur_len, :] if beat_cond is not None else None
            )
            input_context = self.get_input_embedding(
                output_tokens,
                input_emb[:, :cur_len, :],
                beat_cond=chunk_beat_cond,
                bpm_log=bpm_log,
                time_sig_num=time_sig_num,
            )

            chunk_output_tokens = self.generate_chunk(
                start_idx,
                input_context,
                gen_length,
                inst_tokens,
                post_mask,
                temperature,
                filter_logits_fn,
                filter_kwargs,
                cache_kv,
                beat_cond_padded=beat_cond,
                bpm_log=bpm_log,
                time_sig_num=time_sig_num,
                time_sig_den=time_sig_den,
                time_sig_change=time_sig_change,
                tempo_change=tempo_change,
                local_bpm_log_padded=local_bpm_log,
                input_chroma_padded=input_chroma,
                dit_modulation_precompute=dit_modulation_precompute,
                **kwargs,
            )
            output_tokens = torch.cat(
                [output_tokens, chunk_output_tokens], dim=-1
            )

        # Handle positive future visibility
        if self.future_visibility > 0:
            delay_amount = self.future_visibility
            output_tokens = output_tokens[
                :, :, delay_amount:
            ]  # remove the padding for positive future visibility

        return self._finalize_generation(output_tokens)
