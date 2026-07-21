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


class BeatPhaseCondProjector(nn.Module):
    """Projects all available per-window + per-frame timing info into a
    ``[B, T, dim_condition]`` tensor for x_transformers' DiT-style adaptive
    layer-norm + adaptive layer-scale conditioning (per-layer FiLM).

    Stacking order (missing signals are padded with zeros / default ids) is
    per-frame beat_cond (4 ch, sin/cos beat and bar phase) and local_bpm_log
    (1 ch, log(local_BPM / 120)), then per-window bpm_log, time_sig_num
    embedding, time_sig_den embedding, time_sig_change flag and tempo_change
    flag broadcast across T. The per-frame local BPM channel captures tempo
    variation within the window without requiring the model to differentiate
    the phase signal.

    This projector has no gate or LayerNorm of its own, since the modulation
    is applied per layer downstream. x_transformers' AdaptiveLayerNorm
    zero-inits ``to_gamma`` (identity at step 0) and AdaptiveLayerScale uses
    bias-init=-2 (sigmoid(-2), about 0.12 residual attenuation at step 0),
    following the DiT ada-ln-zero recipe.
    """

    def __init__(
        self,
        dim_condition: int,
        time_sig_vocab_size: int = 16,
        time_sig_den_vocab: int = 33,
        ts_emb_dim: int = 8,
        ts_den_emb_dim: int = 8,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim_condition
        self.dim_condition = dim_condition
        self.time_sig_vocab_size = time_sig_vocab_size
        self.time_sig_den_vocab = time_sig_den_vocab

        self.ts_emb = nn.Embedding(time_sig_vocab_size, ts_emb_dim)
        self.ts_den_emb = nn.Embedding(time_sig_den_vocab, ts_den_emb_dim)
        # 4 (beat_cond) + 1 (per-frame local_bpm) + 1 (window bpm) +
        # 1 (ts_change) + 1 (tempo_change) + ts_emb_dim + ts_den_emb_dim
        in_ch = 4 + 1 + 1 + 1 + 1 + ts_emb_dim + ts_den_emb_dim
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

        # If per-frame local BPM is not supplied, broadcast the per-window
        # mean so the model still gets a per-frame channel.
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


class MultipitchAuxHead(nn.Module):
    """Predicts per-frame target-stem multipitch presence (BCE) from the
    decoder's pre-logits hidden state ``[B, S, dim]``. Output ``[B, S,
    out_dim]`` is presence logits. Dropped at inference.
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


class TargetTokenFutureAuxHead(nn.Module):
    """Predicts target-stem DAC tokens at K future offsets from the decoder
    hidden state ``h_t``. Output is ``[B, S, K, num_rvq, num_tokens]``, per
    offset δ_k the per-codebook token logits at mangled position p+δ_k.
    Loss is per-(offset, codebook) cross-entropy at valid positions only.

    Unlike feature-prediction future heads, the target tokens at p+δ are
    not in the input at p (they are autoregressively in the future), so the
    head cannot satisfy the loss by copying attention outputs. The most
    useful information for solving it is what the accompaniment is doing
    around frame p+δ, which lives in the future-mix tokens. Supervision
    lands directly on the generation pathway, same vocabulary and same
    Linear-to-logits shape, just shifted in time.
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
        time_sig_vocab_size: int = 16,
        use_beat_phase_dit_cond: bool = False,
        beat_dit_cond_dim: Optional[int] = None,
        beat_dit_cond_mlp_expansion: int = 4,
        beat_phase_noise_std: float = 0.0,
        # Multipitch / CQT aux heads (all default off).
        use_multipitch_aux_head: bool = False,
        multipitch_aux_head_hidden_dim: int = 256,
        multipitch_dim: int = 128,
        use_cqt_aux_head: bool = False,
        cqt_aux_head_hidden_dim: int = 256,
        cqt_dim: int = 84,
        # Future target-stem token prediction at t+δ from h_t. Same
        # vocabulary as the main next-token head, just shifted in time.
        use_target_token_future_aux_head: bool = False,
        target_token_future_aux_head_hidden_dim: int = 256,
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

        # DiT-style per-layer conditioning requires ``online``, since the
        # offline path has no input_emb / per-frame signal to condition on.
        # Resolved early because the Decoder construction below needs it.
        self.use_beat_phase_dit_cond = bool(online and use_beat_phase_dit_cond)
        self.beat_dit_cond_dim = (
            beat_dit_cond_dim if beat_dit_cond_dim is not None else dim
        )
        self.beat_dit_cond_mlp_expansion = beat_dit_cond_mlp_expansion
        self.beat_phase_noise_std = float(beat_phase_noise_std)

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
        print(f"{'Phase Noise std':<15} | {self.beat_phase_noise_std}")

        if cond_method not in ("concat", "add", "film"):
            raise ValueError(
                f"Invalid conditioning method: {cond_method}. "
                f"Expected one of: concat, add, film"
            )

        # Route the adaptive layer-norm / layer-scale flags into
        # x_transformers via the Decoder wrapper, which only forwards
        # ``attention_layer_configs`` (it does not accept extra kwargs
        # directly). AdaptiveLayerNorm zero-inits its gamma projection
        # (identity at step 0) and AdaptiveLayerScale uses bias-init=-2,
        # the DiT ada-ln-zero recipe.
        if self.use_beat_phase_dit_cond:
            attention_layer_configs = dict(attention_layer_configs)
            attention_layer_configs.update(
                use_adaptive_layernorm=True,
                use_adaptive_layerscale=True,
                dim_condition=self.beat_dit_cond_dim,
                adaptive_condition_mlp=True,
                adaptive_condition_mlp_expansion=self.beat_dit_cond_mlp_expansion,
            )
            # AdaptiveLayerNorm replaces the standard pre-norm in every
            # layer, so the default ``use_simple_rmsnorm=True`` must be
            # turned off or x_transformers asserts on the norm choice.
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

        # DiT-style per-layer beat-phase conditioning. The projector maps
        # (beat_cond, bpm_log, time_sig_num) to [B, T, dim_condition], which
        # AdaptiveLayerNorm/AdaptiveLayerScale consume per layer. The decoder
        # was already configured with the adaptive flags above.
        if self.use_beat_phase_dit_cond:
            self.beat_cond_projector = BeatPhaseCondProjector(
                dim_condition=self.beat_dit_cond_dim,
                time_sig_vocab_size=time_sig_vocab_size,
            )

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

        # Future target-token aux head. It only needs the label, which is
        # always available in the full target stem, so it stays well-defined
        # at any fv. Without lookahead it just becomes a harder extrapolation
        # task, which allows a fair comparison without future-input
        # visibility.
        any_future_aux_request = online and use_target_token_future_aux_head
        self.future_aux_offsets = (
            tuple(int(d) for d in (future_aux_offsets or (10, 25, 40)))
            if any_future_aux_request
            else ()
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
        if any_future_aux_request:
            print(
                f"{'Future Aux δ':<15} | {list(self.future_aux_offsets)}"
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
    ):
        embedded_output = self.output_emb(output_tokens)

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
        time_sig_vocab_size: int = 16,
        use_beat_phase_dit_cond: bool = False,
        beat_dit_cond_dim: Optional[int] = None,
        beat_dit_cond_mlp_expansion: int = 4,
        beat_phase_noise_std: float = 0.0,
        use_multipitch_aux_head: bool = False,
        multipitch_aux_head_hidden_dim: int = 256,
        multipitch_dim: int = 128,
        use_cqt_aux_head: bool = False,
        cqt_aux_head_hidden_dim: int = 256,
        cqt_dim: int = 84,
        use_target_token_future_aux_head: bool = False,
        target_token_future_aux_head_hidden_dim: int = 256,
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
            time_sig_vocab_size=time_sig_vocab_size,
            use_beat_phase_dit_cond=use_beat_phase_dit_cond,
            beat_dit_cond_dim=beat_dit_cond_dim,
            beat_dit_cond_mlp_expansion=beat_dit_cond_mlp_expansion,
            beat_phase_noise_std=beat_phase_noise_std,
            use_multipitch_aux_head=use_multipitch_aux_head,
            multipitch_aux_head_hidden_dim=multipitch_aux_head_hidden_dim,
            multipitch_dim=multipitch_dim,
            use_cqt_aux_head=use_cqt_aux_head,
            cqt_aux_head_hidden_dim=cqt_aux_head_hidden_dim,
            cqt_dim=cqt_dim,
            use_target_token_future_aux_head=use_target_token_future_aux_head,
            target_token_future_aux_head_hidden_dim=target_token_future_aux_head_hidden_dim,
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
    ) -> Optional[torch.Tensor]:
        """Project the padded beat-phase conditioning signals sliced to
        ``[start_frame:end_frame]`` into a condition tensor consumed by
        x_transformers' AdaptiveLayerNorm / AdaptiveLayerScale.

        Returns None if no conditioning is enabled or no signals are
        provided. ``local_bpm_log_padded`` mirrors ``beat_cond_padded``
        along the time axis (same edge-repeat padding).
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
        target_multipitch: Optional[torch.Tensor] = None,
        target_velocity: Optional[torch.Tensor] = None,
        target_cqt: Optional[torch.Tensor] = None,
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
                shape [B, T, 4]. Only used when
                ``use_beat_phase_dit_cond=True``.
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
        # x_patterned_full keeps the un-trimmed patterned tokens so the
        # target-token future aux head can read labels at mangled position
        # p+δ. The fv<=0 branch right-pads it below so those slices stay
        # in bounds. The padding is supervision-only, the encoder gets no
        # future-input visibility from it.
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
            # Caller imposes a specific context_length instead of sampling
            # one. Must leave room for the chunk.
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
            # With fv<=0, right-pad the full target-token tensor with
            # pad_value so the aux-head label slice at p+δ stays in bounds.
            # Supervision only, input_emb / beat_cond etc. are not extended,
            # so the encoder still has zero future-mix visibility.
            if (
                self.use_target_token_future_aux_head
                and self.future_aux_offsets
            ):
                max_off = max(self.future_aux_offsets)
                # Padding by max_off + chunk_length covers the worst case,
                # room for context_end_idx + δ_max - 1 + chunk_length.
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
            # The per-frame feature targets have the same structure as
            # beat_cond, so the edge-repeat pad is reused.
            if target_multipitch is not None:
                target_multipitch = self.pad_beat_cond_for_delay(target_multipitch)
            if target_velocity is not None:
                target_velocity = self.pad_beat_cond_for_delay(target_velocity)
            if target_cqt is not None:
                target_cqt = self.pad_beat_cond_for_delay(target_cqt)
            # Here we +1 to include the BOS.
            context_end_idx += 1
        else:
            # The future-token aux head needs the un-trimmed patterned
            # tokens so the target slice at p+δ has room beyond the chunk
            # end. Build the un-trimmed version first and slice the trimmed
            # view from it to avoid paying the pattern build twice.
            x_patterned_full = self.pad_output_tokens_for_delay(
                x_patterned, trim_end=False
            )
            x_patterned = x_patterned_full[:, :, : -self.future_visibility]
            # For future_visibility > 0,
            # BOS is included in self.future_visibility tokens in the beginning.
            context_end_idx += self.future_visibility

        input_context = input_emb[:, :context_end_idx, :]
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
        # window aligned with ``embedded`` (S = context_end_idx +
        # chunk_length frames, one condition row per frame). local_bpm_log
        # is aligned the same way.
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

        return (
            embedded,
            pred_start_idx,
            pred_end_idx,
            targets,
            logits_mask,
            beat_cond_aligned,
            local_bpm_aligned,
            target_multipitch_aligned,
            target_velocity_aligned,
            target_cqt_aligned,
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
        target_multipitch: Optional[torch.Tensor] = None,
        target_velocity: Optional[torch.Tensor] = None,
        target_cqt: Optional[torch.Tensor] = None,
        context_length_override: Optional[int] = None,
        **kwargs,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        Optional[dict],
    ]:
        """
        Forward pass of decoder, used for training.

        Returns 4-tuple: (logits_pred, logits_mask, targets, extra_aux).
        ``extra_aux`` is None when no aux head is enabled, else a dict with
        optional keys:
            mp_pred:           [B, S, multipitch_dim]      (presence logits)
            mp_target:         [B, S, multipitch_dim]      (presence binary)
            cqt_pred:          [B, S, cqt_dim]
            cqt_target:        [B, S, cqt_dim]
            tt_future_pred:    [B, num_rvq, chunk, K_off, num_tokens]
            tt_future_target:  [B, num_rvq, chunk, K_off]
            tt_future_logits_mask: [B, num_rvq, chunk]
        """
        # Phase-noise injection on beat_cond at training only. Pushes the
        # encoder to use input-mix audio for beat localization rather than
        # relying entirely on the clean beat-phase prior. No noise at
        # inference since self.training is False.
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
            target_multipitch_aligned,
            target_velocity_aligned,
            target_cqt_aligned,
            x_patterned_full,
        ) = self.prepare_input(
            x,
            inst_tokens,
            input_emb,
            beat_cond=beat_cond,
            bpm_log=bpm_log,
            time_sig_num=time_sig_num,
            local_bpm_log=local_bpm_log,
            target_multipitch=target_multipitch,
            target_velocity=target_velocity,
            target_cqt=target_cqt,
            context_length_override=context_length_override,
        )

        # DiT-style per-layer conditioning. condition has shape
        # [B, S, dim_condition] aligned with ``embedded``.
        decoder_extra = {}
        if self.use_beat_phase_dit_cond and beat_cond_aligned is not None:
            condition = self._build_dit_condition(
                beat_cond_aligned,
                bpm_log,
                time_sig_num,
                start_frame=0,
                end_frame=beat_cond_aligned.shape[1],
                ref_dtype=embedded.dtype,
                time_sig_den=time_sig_den,
                time_sig_change=time_sig_change,
                tempo_change=tempo_change,
                local_bpm_log_padded=local_bpm_aligned,
            )
            if condition is not None:
                decoder_extra["condition"] = condition

        # When any aux head is on, ask x_transformers to return both logits
        # and the pre-logits hidden state [B, S, dim].
        any_aux = (
            self.use_multipitch_aux_head
            or self.use_cqt_aux_head
            or self.use_target_token_future_aux_head
        )
        if any_aux:
            logits, hidden = self.decoder(
                embedded,
                mask=mask,
                return_logits_and_embeddings=True,
                **decoder_extra,
                **kwargs,
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
            tt_future_pred = (
                self.target_token_future_aux_head(hidden)
                if self.use_target_token_future_aux_head
                else None
            )
        else:
            logits = self.decoder(
                embedded, mask=mask, **decoder_extra, **kwargs
            )
            mp_aux_pred = None
            cqt_aux_pred = None
            tt_future_pred = None

        # Build extra_aux dict for multipitch / CQT / target-token-future
        # heads. All targets are already aligned with hidden along the time
        # axis.
        extra_aux: Optional[dict] = None
        if any_aux:
            extra_aux = {}
            if self.use_multipitch_aux_head and mp_aux_pred is not None:
                extra_aux["mp_pred"] = mp_aux_pred
                if target_multipitch_aligned is not None:
                    extra_aux["mp_target"] = target_multipitch_aligned
            if self.use_cqt_aux_head and cqt_aux_pred is not None:
                extra_aux["cqt_pred"] = cqt_aux_pred
                if target_cqt_aligned is not None:
                    extra_aux["cqt_target"] = target_cqt_aligned
            # Target-token future aux, predict target-stem patterned tokens
            # at p+δ from h_t. Hidden positions [pred_start_idx,
            # pred_end_idx) are the same slice the main next-token head
            # predicts from. For offset δ the mangled-position target is at
            # context_end_idx + δ - 1 + i for i in [0, chunk_length).
            # x_patterned_full has the un-trimmed patterned tokens so those
            # positions stay in bounds.
            if (
                self.future_aux_offsets
                and self.use_target_token_future_aux_head
                and tt_future_pred is not None
                and x_patterned_full is not None
            ):
                offsets = self.future_aux_offsets
                context_end_idx = pred_start_idx + 1
                # tt_future_pred is [B, S, K_off, num_rvq, num_tokens].
                # Slice to the prediction window to match logits_mask shape.
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
                # Stack along a new last axis, [B, num_rvq, chunk, K_off].
                tt_future_target = torch.stack(tt_target_slices, dim=-1)
                extra_aux["tt_future_pred"] = pred_window
                extra_aux["tt_future_target"] = tt_future_target
                extra_aux["tt_future_logits_mask"] = logits_mask

        logits_pred = logits[:, :, pred_start_idx:pred_end_idx, :]
        assert logits_pred.shape[2] > 0
        assert logits_mask.shape[2] == logits_pred.shape[2]
        assert logits_pred.shape[2] == targets.shape[2]
        return (
            logits_pred, logits_mask, targets,
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
        dit_modulation_precompute: bool = True,
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
            self.use_beat_phase_dit_cond and beat_cond_padded is not None
        )

        # Chunk-ahead modulation precompute, inference only. The KV-cached
        # steps s >= 1 consume the DiT condition for frames
        # [context_len, context_len + generate_len - 1). Those conditioning
        # signals are already known here, so the projector, the adaptive
        # MLP and every per-layer to_gamma run once as batched GEMMs and
        # the resulting modulations are cached. The AR loop then applies
        # per-frame slices instead of launching every conditioning linear
        # once per step.
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
                # end), fall back to the per-step path.
                pc_cond_chunk = None

        try:
            for curr_sample_step in range(generate_len):
                # Build the DiT condition slice that aligns with what
                # ``self.net`` actually processes this step. On iter 0 or
                # with cache_kv=False the full model_input is processed, so
                # pass condition for all those frames. On later iters with
                # cache, x_transformers slices x to its last frame
                # internally, so pass condition for only that one frame.
                extra_call_kwargs = dict(kwargs)
                if any_dit_cond:
                    if cache_kv and cache is not None:
                        if pc_state is not None:
                            # Cached-gamma fast path. The raw cond slice is
                            # still passed so x_transformers' need_condition
                            # assert holds. Its one-frame adaptive_mlp
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
            # Never leave stale cached gammas active. A later naive-path
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
        dit_modulation_precompute: bool = True,
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
                phase (sin/cos). Required when use_beat_phase_dit_cond=True.
            bpm_log (torch.Tensor, optional): [B] log(bpm/120).
            time_sig_num (torch.Tensor, optional): [B] long time-sig numerator.

        Returns:
            Tensor: Generated sequences (B, K, seq_len - 1)
        """
        # Prompting is not supported.
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
            input_context = self.get_input_embedding(
                output_tokens,
                input_emb[:, :cur_len, :],
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

    def generate_sliding(
        self,
        seq_len: int,
        window_frames: int = 500,
        hop_frames: Optional[int] = None,
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
        dit_modulation_precompute: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Sliding-window variant of ``generate`` for sequences longer than
        the trained context window.

        The transformer uses learned absolute positions sized for
        ``max_seq_len`` (~10 s), so ``generate(seq_len=1000)`` is impossible
        directly. This method streams past that limit: once the running
        context reaches ``window_frames``, the oldest ``hop_frames`` frames
        are dropped and the remainder is re-based to position 0. Because the
        outer chunk loop rebuilds the full fused context every chunk (KV
        cache is per-chunk) and the delay pattern is shift-invariant, the
        re-base is pure tensor slicing: the head columns (BOS, plus the
        future-visibility padding when fv>0) are kept and frames
        ``[head, head+offset)`` are removed. Input/beat tensors are sliced
        with the same offset, so the per-chunk input-visibility semantics
        are exactly those of ``generate``; the only approximation is the
        truncated history inherent to any sliding window.

        For ``seq_len <= window_frames`` this delegates to ``generate``
        unchanged; for longer sequences the first ``window_frames`` frames
        are op-for-op (and RNG-draw-for-draw) identical to ``generate``.
        """
        if seq_len <= window_frames:
            return self.generate(
                seq_len=seq_len,
                seq_out_start=seq_out_start,
                input_emb=input_emb,
                temperature=temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
                cache_kv=cache_kv,
                display_pbar=display_pbar,
                inst_tokens=inst_tokens,
                beat_cond=beat_cond,
                bpm_log=bpm_log,
                time_sig_num=time_sig_num,
                time_sig_den=time_sig_den,
                time_sig_change=time_sig_change,
                tempo_change=tempo_change,
                local_bpm_log=local_bpm_log,
                dit_modulation_precompute=dit_modulation_precompute,
                **kwargs,
            )

        if seq_out_start is not None:
            raise NotImplementedError("Prompting is not supported yet.")
        if hop_frames is None:
            hop_frames = self.chunk_length
        assert window_frames % self.chunk_length == 0, (
            f"window_frames ({window_frames}) must be a multiple of "
            f"chunk_length ({self.chunk_length})"
        )
        assert hop_frames % self.chunk_length == 0, (
            f"hop_frames ({hop_frames}) must be a multiple of "
            f"chunk_length ({self.chunk_length})"
        )
        assert 0 < hop_frames < window_frames

        batch_size, device = self._validate_generation_params(
            inst_tokens, seq_out_start, input_emb, guidance_scale=1.0
        )
        filter_logits_fn, filter_kwargs = self._process_filter_functions(
            filter_logits_fn, filter_kwargs
        )
        (
            output_tokens,
            prompt_length,
            post_mask,
            patterned_seq_out,
            seq_out_mask,
        ) = self._initialize_generation_pattern(
            seq_len, batch_size, device, seq_out_start, inst_tokens
        )

        if self.future_visibility <= 0:
            input_emb = self.pad_input_embs_for_delay(input_emb)
            if beat_cond is not None:
                beat_cond = self.pad_beat_cond_for_delay(beat_cond)
            if local_bpm_log is not None:
                local_bpm_log = self.pad_local_bpm_for_delay(local_bpm_log)
            # Patterned output has 1 head column (BOS); the padded
            # input/beat tensors carry a matching pad column at index 0.
            head_cols = 1
            cond_head = 1
        else:
            output_tokens = self.pad_output_tokens_for_delay(
                output_tokens, trim_end=False
            )
            # (fv - 1) pad columns + BOS; input/beat tensors are unpadded
            # and frame-indexed (the model reads them fv frames ahead), so
            # they re-base with a plain shift.
            head_cols = self.future_visibility
            cond_head = 0

        def _slice_frames(t, offset):
            """Drop frames [cond_head, cond_head+offset) along dim 1."""
            if t is None or offset == 0:
                return t
            if cond_head == 0:
                return t[:, offset:]
            return torch.cat([t[:, :cond_head], t[:, cond_head + offset :]], dim=1)

        start_indices = np.arange(0, seq_len, self.chunk_length)
        gen_lengths = np.minimum(self.chunk_length, seq_len - start_indices)

        pbar = tqdm(
            range(len(start_indices)),
            disable=not display_pbar,
            desc="Generating chunks (sliding)",
        )

        window_offset = 0
        for i in pbar:
            start_idx = int(start_indices[i])
            gen_length = int(gen_lengths[i])

            # Advance the window so context + new chunk fit in window_frames.
            while (start_idx + gen_length) - window_offset > window_frames:
                window_offset += hop_frames

            if window_offset == 0:
                local_out = output_tokens
                local_input = input_emb
                local_beat = beat_cond
                local_bpm = local_bpm_log
            else:
                local_out = torch.cat(
                    [
                        output_tokens[:, :, :head_cols],
                        output_tokens[:, :, head_cols + window_offset :],
                    ],
                    dim=-1,
                )
                local_input = _slice_frames(input_emb, window_offset)
                local_beat = _slice_frames(beat_cond, window_offset)
                local_bpm = _slice_frames(local_bpm_log, window_offset)

            cur_len = local_out.shape[-1]
            chunk_beat_cond = (
                local_beat[:, :cur_len, :] if local_beat is not None else None
            )
            input_context = self.get_input_embedding(
                local_out,
                local_input[:, :cur_len, :],
                beat_cond=chunk_beat_cond,
                bpm_log=bpm_log,
                time_sig_num=time_sig_num,
            )

            chunk_output_tokens = self.generate_chunk(
                start_idx,  # global: post_mask covers the full seq_len pattern
                input_context,
                gen_length,
                inst_tokens,
                post_mask,
                temperature,
                filter_logits_fn,
                filter_kwargs,
                cache_kv,
                beat_cond_padded=local_beat,
                bpm_log=bpm_log,
                time_sig_num=time_sig_num,
                time_sig_den=time_sig_den,
                time_sig_change=time_sig_change,
                tempo_change=tempo_change,
                local_bpm_log_padded=local_bpm,
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
