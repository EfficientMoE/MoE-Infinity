# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, Sequence

from moe_infinity.memory.adaptive_precision_policy import (
    ExpertKey,
    PrecisionPlan,
)
from moe_infinity.runtime.expert_precision import (
    CandidateVariantSpec,
    ExecutionKind,
    ExpertFormat,
)

_UNIFORM_FORMAT = ExpertFormat.FP8_E4M3_BLOCK128
_UNIFORM_EXECUTION = ExecutionKind.FP8_DEQUANT_BF16_GEMM


def uniform_fp8_targets(
    variants: Sequence[CandidateVariantSpec],
) -> Dict[ExpertKey, ExpertFormat]:
    if not variants:
        raise ValueError(
            "uniform FP8 residency requires at least one candidate variant"
        )

    converter_version = variants[0].converter_version
    targets: Dict[ExpertKey, ExpertFormat] = {}
    for variant in variants:
        if variant.format is not _UNIFORM_FORMAT:
            raise ValueError(
                "uniform FP8 residency requires every expert in "
                f"{_UNIFORM_FORMAT.value}; got {variant.format.value} for "
                f"layer {variant.layer_id} expert {variant.expert_id}"
            )
        if variant.execution is not _UNIFORM_EXECUTION:
            raise ValueError(
                "uniform FP8 residency requires every expert to use "
                f"{_UNIFORM_EXECUTION.value}; got {variant.execution.value} "
                f"for layer {variant.layer_id} expert {variant.expert_id}"
            )
        if variant.converter_version != converter_version:
            raise ValueError(
                "uniform FP8 residency requires one converter version across "
                f"all experts; got {variant.converter_version!r} and "
                f"{converter_version!r}"
            )
        key = ExpertKey(variant.layer_id, variant.expert_id)
        if key in targets:
            raise ValueError(
                "uniform FP8 residency requires one variant per expert; "
                f"duplicate layer {variant.layer_id} expert {variant.expert_id}"
            )
        targets[key] = _UNIFORM_FORMAT
    return targets


def load_time_fp8_plan(
    variants: Sequence[CandidateVariantSpec],
    *,
    generation: int,
) -> PrecisionPlan:
    if generation < 0:
        raise ValueError(f"generation must be non-negative; got {generation}")

    targets = uniform_fp8_targets(variants)
    accounted_bytes = sum(variant.aligned_bytes for variant in variants)
    return PrecisionPlan(
        epoch=generation,
        targets=MappingProxyType(dict(targets)),
        transitions=(),
        evictions=(),
        accounted_bytes=accounted_bytes,
    )


def maybe_apply_uniform_fp8(
    dispatcher,
    variants: Sequence[CandidateVariantSpec],
    *,
    mode: str,
    generation: int,
) -> bool:
    if mode != "uniform_fp8":
        return False
    # Build the plan first: on a non-uniform manifest this raises before the
    # dispatcher is touched, so the caller must not fall through to the online
    # adaptive policy. The exception is intentionally left uncaught.
    plan = load_time_fp8_plan(variants, generation=generation)
    publish_uniform_fp8_plan(dispatcher, variants, plan)
    return True


def publish_uniform_fp8_plan(
    dispatcher,
    variants: Sequence[CandidateVariantSpec],
    plan: PrecisionPlan,
) -> None:
    generations = {
        (
            ExpertKey(variant.layer_id, variant.expert_id),
            variant.format,
        ): plan.epoch
        for variant in variants
    }
    if not dispatcher.set_precision_targets(
        plan.as_native_targets(generations), plan.epoch
    ):
        raise RuntimeError(
            "uniform FP8 residency target publication was rejected"
        )
