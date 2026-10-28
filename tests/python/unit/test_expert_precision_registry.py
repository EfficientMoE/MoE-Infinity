from types import SimpleNamespace

import pytest
import torch

from moe_infinity.runtime.adaptive_precision_allowlist import (
    ReleasedAdaptiveEntry,
    is_released,
)
from moe_infinity.runtime.expert_precision import (
    ExecutionKind,
    ExpertFormat,
    resolve_model_precision_capabilities,
)
from moe_infinity.utils import ArcherConfig


def test_adaptive_precision_is_disabled_by_default(tmp_path):
    cfg = ArcherConfig(
        offload_path=str(tmp_path),
        use_native_engine=False,
        kv_cache_memory_ratio=0.0,
    )
    assert cfg.adaptive_expert_precision is False
    assert cfg.adaptive_hbm_budget_bytes == 0


def test_general_converter_candidate_separates_storage_and_execution():
    model = SimpleNamespace(model_type="qwen3_moe", quantization_config=None)
    caps = resolve_model_precision_capabilities(model, extension_names=set())
    assert caps.protected_reason is None
    assert (
        caps.formats[ExpertFormat.FP8_E4M3_BLOCK128].execution
        is ExecutionKind.FP8_DEQUANT_BF16_GEMM
    )
    assert (
        caps.formats[ExpertFormat.FP8_E4M3_BLOCK128].output_dtype
        is torch.bfloat16
    )


def test_converter_candidate_is_not_implicitly_released():
    entry = ReleasedAdaptiveEntry(
        checkpoint_fingerprint="a" * 64,
        format=ExpertFormat.FP8_E4M3_BLOCK128,
        converter_version="adaptive-expert-v1",
        quality_attestation_sha256="b" * 64,
    )
    assert is_released(entry) is False


@pytest.mark.parametrize(
    ("model_type", "reason"),
    [
        ("gpt_oss", "protected:gpt_oss_mxfp4"),
        ("glm_moe_dsa", "protected:glm_fp8"),
        ("deepseek_v4", "protected:deepseek_v4_fp4"),
        ("deepseek_v41", "protected:deepseek_v4_fp4"),
    ],
)
def test_model_specific_low_bit_paths_are_protected(model_type, reason):
    model = SimpleNamespace(
        model_type=model_type, quantization_config={"quant_method": "fp8"}
    )
    caps = resolve_model_precision_capabilities(
        model, extension_names={"_v4_fp4"}
    )
    assert caps.protected_reason == reason
    assert all(cap.protected for cap in caps.formats.values())


def test_deepseek_v41_never_resolves_to_glm_fp8_identity():
    """model_type must win before the generic quant_method fp8 fallback.

    The V4.1 checkpoint carries quantization_config.quant_method == "fp8"
    (32x32 blocks, ue8m0 scales, fp4 experts); resolving it through the
    GLM_FP8_BLOCK128 path would be a silent format mismatch.
    """
    config = SimpleNamespace(
        model_type="deepseek_v41",
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
    )
    caps = resolve_model_precision_capabilities(
        config, extension_names={"_v4_fp4"}
    )
    assert caps.protected_reason == "protected:deepseek_v4_fp4"
    assert ExpertFormat.DEEPSEEK_V4_FP4 in caps.formats
    assert ExpertFormat.GLM_FP8_BLOCK128 not in caps.formats
    assert all(cap.protected for cap in caps.formats.values())


def test_invalid_adaptive_budget_is_rejected(tmp_path):
    with pytest.raises(
        ValueError, match="adaptive_hbm_budget_bytes must be positive"
    ):
        ArcherConfig.load_from_json(
            {
                "offload_path": str(tmp_path),
                "use_native_engine": False,
                "adaptive_expert_precision": True,
                "adaptive_hbm_budget_bytes": 0,
            }
        )
