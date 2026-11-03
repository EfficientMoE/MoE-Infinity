"""Fused GDN / KDA chunked forward for inference (#304).

Drop-in replacements for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``
and ``fla.ops.kda.chunk_kda`` that keep fla's numerics but cut the DRAM
round trips of the fla pipeline on sm_120:

GDN  fla:   l2norm(q), l2norm(k), cumsum(g), kkt+solve -> A, w/u(A),
            h (writes every chunk state), o (reads them back)
     here:  l2norm(q), cumsum(g), prep (k-norm + kkt + solve + w/u, A stays
            in registers), h+o (states never leave registers)
KDA  fla's cumsum / intra (w, u, kg, Aqk) followed by the fused h+o.

Calls the fused path only for inference on CUDA bf16 inputs with
K = V = 128, H == HV and no ``cu_seqlens``; everything else goes to the
original implementation (``fallback``).
"""

from __future__ import annotations

import os
from typing import Callable

import torch

ENV_DISABLE = "MOE_LINEAR_ATTN_FUSED"
# Opt-in: L2-blocked GDN prefill with the JIT-built CUDA recurrence
# (gdn_blocked / csrc/gdn_recur.cu). First use compiles with nvcc (~1 min).
ENV_CUDA = "MOE_LINEAR_ATTN_CUDA"
# Below this many tokens per call the fused GDN path is not faster than fla
# on sm_120 (2K: 209 vs 198 us; 9K: 861 vs 1128 us), so leave short
# prefills on fla.
MIN_FUSED_TOKENS = 4096


def fused_enabled() -> bool:
    return os.environ.get(ENV_DISABLE, "1") != "0"


def _supported(q, k, v, g, beta, initial_state, kwargs, gate_dims) -> bool:
    if not fused_enabled():
        return False
    for key in ("cu_seqlens", "cu_seq_lens_q", "seq_idx", "cp_context"):
        if kwargs.get(key) is not None:
            return False
    if kwargs.get("state_v_first") or kwargs.get("use_gate_in_kernel"):
        return False
    if kwargs.get("use_beta_sigmoid_in_kernel") or kwargs.get("safe_gate"):
        return False
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, g, beta)
    ):
        return False
    if not (q.is_cuda and q.dtype == torch.bfloat16):
        return False
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False
    if q.dim() != 4 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        return False
    if q.shape[-1] != 128 or v.shape[-1] != 128:
        return False
    if gate_dims == 3 and q.shape[0] * q.shape[1] < MIN_FUSED_TOKENS:
        return False
    if g is None or g.dim() != gate_dims or beta is None or beta.dim() != 3:
        return False
    if initial_state is not None and (
        initial_state.dtype != torch.float32
        or initial_state.shape != (q.shape[0], q.shape[2], 128, 128)
    ):
        return False
    return True


def chunk_gated_delta_rule_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    from moe_infinity.kernel.gdn_fused import delta_rule_fwd_h_o
    from moe_infinity.kernel.gdn_prep import gdn_fwd_prep

    if scale is None:
        scale = k.shape[-1] ** -0.5
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    gc = chunk_local_cumsum(g.float(), chunk_size=64, scale=RCP_LN2)
    if os.environ.get(ENV_CUDA, "0") == "1":
        from moe_infinity.kernel import gdn_recur_cuda
        from moe_infinity.kernel.gdn_blocked import (
            chunk_gated_delta_rule_blocked,
        )

        if gdn_recur_cuda.available():
            return chunk_gated_delta_rule_blocked(
                q,
                k,
                v,
                gc,
                beta.contiguous(),
                scale,
                initial_state=None
                if initial_state is None
                else initial_state.contiguous(),
                output_final_state=output_final_state,
                norm_qk=use_qk_l2norm_in_kernel,
                use_cuda=True,
            )
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)[0]
    w, u, kn = gdn_fwd_prep(
        k, v, gc, beta.contiguous(), use_qk_l2norm_in_kernel
    )
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    return delta_rule_fwd_h_o(
        q,
        kn,
        w,
        u,
        gc,
        scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def chunk_kda_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    from moe_infinity.kernel.gdn_fused import delta_rule_fwd_h_o

    if scale is None:
        scale = k.shape[-1] ** -0.5
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)[0]
        k = l2norm_fwd(k)[0]
    gc = chunk_local_cumsum(g.float(), chunk_size=64, scale=RCP_LN2)
    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=gc, beta=beta.contiguous(), scale=scale
    )
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    return delta_rule_fwd_h_o(
        q,
        kg,
        w,
        u,
        gc,
        scale,
        Aqk=Aqk,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def make_fused(kind: str, fallback: Callable) -> Callable:
    """Wrap ``fallback`` (fla / transformers callable with the HF calling
    convention ``fn(q, k, v, g=..., beta=..., ...)``) with the fused path."""
    fused = chunk_kda_fused if kind == "kda" else chunk_gated_delta_rule_fused
    gate_dims = 4 if kind == "kda" else 3

    def fn(query, key, value, g=None, beta=None, *args, **kwargs):
        if args:
            return fallback(query, key, value, g, beta, *args, **kwargs)
        init = kwargs.get("initial_state")
        if _supported(query, key, value, g, beta, init, kwargs, gate_dims):
            return fused(
                query,
                key,
                value,
                g,
                beta,
                scale=kwargs.get("scale"),
                initial_state=init,
                output_final_state=bool(kwargs.get("output_final_state")),
                use_qk_l2norm_in_kernel=bool(
                    kwargs.get("use_qk_l2norm_in_kernel")
                ),
            )
        return fallback(query, key, value, g=g, beta=beta, **kwargs)

    fn.__name__ = f"moe_fused_{kind}_chunk"
    fn.__qualname__ = fn.__name__
    fn.__wrapped__ = fallback  # type: ignore[attr-defined]
    fn._moe_backend = "fused"  # type: ignore[attr-defined]
    fn._moe_device_safe = getattr(fallback, "_moe_device_safe", False)  # type: ignore[attr-defined]
    return fn
