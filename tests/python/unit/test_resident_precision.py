import pytest

from moe_infinity.memory.adaptive_precision_policy import ExpertKey
from moe_infinity.runtime.expert_precision import (
    CandidateVariantSpec,
    ExecutionKind,
    ExpertFormat,
)
from moe_infinity.runtime.resident_precision import (
    load_time_fp8_plan,
    uniform_fp8_targets,
)


def _variant(
    layer,
    expert,
    *,
    fmt=ExpertFormat.FP8_E4M3_BLOCK128,
    execution=ExecutionKind.FP8_DEQUANT_BF16_GEMM,
    version="adaptive-expert-v1",
    aligned=100,
):
    return CandidateVariantSpec(
        layer_id=layer,
        expert_id=expert,
        format=fmt,
        execution=execution,
        tensor_ids=(1,),
        tensor_roles=("gate.weight",),
        payload_bytes=aligned,
        aligned_bytes=aligned,
        workspace_bytes=0,
        source_format=ExpertFormat.BF16,
        converter_version=version,
    )


def test_targets_are_fp8_for_every_expert():
    variants = (_variant(0, 1), _variant(2, 4, aligned=50))
    targets = uniform_fp8_targets(variants)
    assert targets == {
        ExpertKey(0, 1): ExpertFormat.FP8_E4M3_BLOCK128,
        ExpertKey(2, 4): ExpertFormat.FP8_E4M3_BLOCK128,
    }


def test_plan_has_no_transitions_and_sums_aligned_bytes():
    plan = load_time_fp8_plan((_variant(0, 1), _variant(0, 2)), generation=3)
    assert plan.epoch == 3
    assert plan.transitions == ()
    assert plan.evictions == ()
    assert plan.accounted_bytes == 200
    assert set(plan.targets.values()) == {ExpertFormat.FP8_E4M3_BLOCK128}


def test_mixed_format_raises():
    mixed = (
        _variant(0, 1),
        _variant(
            0, 2, fmt=ExpertFormat.BF16, execution=ExecutionKind.BF16_GEMM
        ),
    )
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(mixed)


def test_int4_raises():
    mixed = (
        _variant(0, 1),
        _variant(
            0,
            2,
            fmt=ExpertFormat.MARLIN_INT4_GROUP128,
            execution=ExecutionKind.MARLIN_W4A16,
        ),
    )
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(mixed)


def test_converter_mismatch_raises():
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(
            (
                _variant(0, 1, version="adaptive-expert-v1"),
                _variant(0, 2, version="adaptive-expert-v2"),
            )
        )


def test_execution_mismatch_raises():
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(
            (_variant(0, 1), _variant(0, 2, execution=ExecutionKind.BF16_GEMM))
        )


def test_duplicate_expert_raises():
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets((_variant(0, 1), _variant(0, 1, aligned=40)))


def test_empty_variants_raise():
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(())


def test_negative_generation_raises():
    with pytest.raises(ValueError):
        load_time_fp8_plan((_variant(0, 1),), generation=-1)
