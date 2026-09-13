# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Registers MoE-Infinity's engine operations into moe-store's EngineOps
registry so the model wrappers (moe_store.wrappers) can call fused kernels
and spec-decode facilities without importing the engine."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_registered = False


def register_engine_ops() -> None:
    global _registered
    if _registered:
        return

    from moe_store.hooks import register_engine_ops as _register

    from moe_infinity.kernel import topk_softmax
    from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm
    from moe_infinity.spec_decode import _route_ahead_ctx
    from moe_infinity.spec_decode._nvtx import nvtx_phase
    from moe_infinity.spec_decode._prefetch_route import (
        union_experts_from_mask,
    )

    _register(
        topk_softmax=topk_softmax,
        fused_mxfp4_gemm=fused_mxfp4_gemm,
        union_experts_from_mask=union_experts_from_mask,
        nvtx_phase=nvtx_phase,
        route_ahead_ctx=_route_ahead_ctx,
    )

    try:
        from moe_infinity import _v4_fp4
    except ImportError as error:
        logger.warning(
            "native _v4_fp4 extension unavailable (%s); DeepSeek-V4 "
            "Path-B FP4 kernels will raise if used",
            error,
        )
    else:
        _register(v4_fp4_ext=_v4_fp4)

    _registered = True
