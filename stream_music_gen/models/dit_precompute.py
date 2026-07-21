"""Chunk-ahead precompute of DiT adaptive-norm modulations (inference-only).

With adaptive layer-norm / layer-scale conditioning, every KV-cached
autoregressive step launches the adaptive MLP plus one ``to_gamma`` linear
per conditioned module (~66 kernels for a 16-layer decoder) on a single
frame. Those modulations depend only on the conditioning signals, which are
known for the whole upcoming chunk before generation starts, so they can be
computed once per chunk as batched GEMMs and applied per step as a slice +
multiply.

Usage (see ``OnlinePrefixDecoderTransformerMultiOut.generate_chunk``):

    state = precompute_dit_gammas(attn_layers, cond_chunk)  # [B, T, dim_cond]
    for s in range(T):
        state.activate(s)
        ...  # forward one cached step; patched norms use cached gammas
    state.clear()

Patching is done by overriding ``forward`` on the module *instances*
(state_dict keys are untouched, so checkpoint save/load is unaffected).
When ``state.active`` is False the patched forwards delegate to the original
x-transformers implementations, so training, prefill, and non-precompute
generation are bit-identical to an unpatched model. Caveat: patched modules
hold closures and are no longer picklable/deepcopy-able — eval-only.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from x_transformers.x_transformers import (
    AdaptiveLayerNorm,
    AdaptiveLayerScale,
)


class DiTPrecomputeState:
    """Cached per-frame modulations for the chunk currently being generated.

    ``gammas`` maps each patched module to a ``[B, T_chunk, dim]`` tensor
    with the module's activation folded in (ALN: ``to_gamma(c) + 1``;
    ALS: ``sigmoid(to_gamma(c))``). ``idx`` selects the frame within the
    chunk; ``active`` gates whether patched forwards use the cache.
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.idx: int = 0
        self.gammas: Dict[nn.Module, torch.Tensor] = {}
        self.aln_modules: List[AdaptiveLayerNorm] = []
        self.als_modules: List[AdaptiveLayerScale] = []

    def activate(self, idx: int) -> None:
        self.active = True
        self.idx = idx

    def deactivate(self) -> None:
        self.active = False

    def clear(self) -> None:
        """Deactivate and drop cached tensors (call when the chunk ends)."""
        self.active = False
        self.gammas = {}


def _make_aln_forward(module: AdaptiveLayerNorm, state: DiTPrecomputeState):
    orig_forward = type(module).forward

    def patched_forward(x, *, condition=None):
        if state.active:
            gamma = state.gammas[module][:, state.idx : state.idx + 1]
            return module.ln(x) * gamma
        return orig_forward(module, x, condition=condition)

    return patched_forward


def _make_als_forward(module: AdaptiveLayerScale, state: DiTPrecomputeState):
    orig_forward = type(module).forward

    def patched_forward(x, *, condition=None, **kwargs):
        if not state.active:
            return orig_forward(module, x, condition=condition, **kwargs)
        out = module.fn(x, **kwargs)
        gamma = state.gammas[module][:, state.idx : state.idx + 1]
        if isinstance(out, torch.Tensor):
            return out * gamma
        out, *rest = out
        return (out * gamma, *rest)

    return patched_forward


def install_dit_precompute(attn_layers: nn.Module) -> DiTPrecomputeState:
    """Idempotently patch all adaptive-norm modules under ``attn_layers``.

    Returns the (possibly pre-existing) state object stored on
    ``attn_layers._dit_pc_state``.
    """
    state = getattr(attn_layers, "_dit_pc_state", None)
    if state is not None:
        return state

    assert getattr(attn_layers, "need_condition", False), (
        "attn_layers has no adaptive conditioning; nothing to precompute"
    )
    assert hasattr(attn_layers, "adaptive_mlp"), (
        "expected adaptive_condition_mlp=True on the conditioned decoder"
    )

    state = DiTPrecomputeState()
    for m in attn_layers.modules():
        if isinstance(m, AdaptiveLayerNorm):
            state.aln_modules.append(m)
            m.forward = _make_aln_forward(m, state)
        elif isinstance(m, AdaptiveLayerScale):
            state.als_modules.append(m)
            m.forward = _make_als_forward(m, state)
    assert state.aln_modules or state.als_modules, (
        "no AdaptiveLayerNorm/AdaptiveLayerScale modules found to patch"
    )
    attn_layers._dit_pc_state = state
    return state


def precompute_dit_gammas(
    attn_layers: nn.Module, cond_chunk: torch.Tensor
) -> DiTPrecomputeState:
    """Batch-compute and cache the modulations for a whole chunk.

    ``cond_chunk`` is the raw ``[B, T_chunk, dim_condition]`` condition (the
    same tensor the per-step path would feed frame-by-frame). Runs the
    adaptive MLP and every ``to_gamma`` once over all frames. The returned
    state is deactivated; the caller activates it per step.
    """
    state = install_dit_precompute(attn_layers)
    expanded = attn_layers.adaptive_mlp(cond_chunk)
    gammas: Dict[nn.Module, torch.Tensor] = {}
    for m in state.aln_modules:
        gammas[m] = m.to_gamma(expanded) + 1.0
    for m in state.als_modules:
        gammas[m] = m.to_gamma(expanded).sigmoid()
    state.gammas = gammas
    state.deactivate()
    return state
