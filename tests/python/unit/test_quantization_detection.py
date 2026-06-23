# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Tests for moe_infinity.utils.quantization — detection, validation, and cast decisions.

TDD RED phase: these tests are written BEFORE the implementation exists.
"""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs):
    """Build a minimal mock PretrainedConfig with the given attributes."""
    defaults = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "torch_dtype": torch.bfloat16,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_local_experts": 8,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_checkpoint_dir_with_file(filename, content_dict):
    """Create a temp dir containing a single JSON file with the given dict."""
    d = tempfile.mkdtemp()
    filepath = os.path.join(d, filename)
    with open(filepath, "w") as f:
        json.dump(content_dict, f)
    return d


def _make_checkpoint_dir_with_gguf():
    """Create a temp dir containing a .gguf file."""
    d = tempfile.mkdtemp()
    # Create a dummy gguf file
    with open(os.path.join(d, "model-q4_0.gguf"), "wb") as f:
        f.write(b"\x00" * 64)
    return d


# ===========================================================================
# TestQuantizationDetection
# ===========================================================================


class TestQuantizationDetection:
    """Tests for detect_quantization() function."""

    def test_unquantized_model_returns_none(self):
        """Standard fp16/bf16 config with no quant markers → None."""
        from moe_infinity.utils.quantization import detect_quantization

        config = _make_config()
        result = detect_quantization(config, "")
        assert result is None

    def test_gptq_from_config_quantization_config(self):
        """config.quantization_config with quant_method='gptq' → QuantizationInfo(method='gptq')."""
        from moe_infinity.utils.quantization import detect_quantization

        config = _make_config(
            quantization_config={
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
            }
        )
        result = detect_quantization(config, "")
        assert result is not None
        assert result.method == "gptq"
        assert result.supported is True
        assert result.bits == 4
        assert result.group_size == 128

    def test_awq_from_config_quantization_config(self):
        """config.quantization_config with quant_method='awq' → QuantizationInfo(method='awq')."""
        from moe_infinity.utils.quantization import detect_quantization

        config = _make_config(
            quantization_config={
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
            }
        )
        result = detect_quantization(config, "")
        assert result is not None
        assert result.method == "awq"
        assert result.supported is True

    def test_gptq_from_quantize_config_json(self):
        """No config.quantization_config but quantize_config.json exists → GPTQ detected."""
        from moe_infinity.utils.quantization import detect_quantization

        config = _make_config()  # no quantization_config attr
        ckpt_dir = _make_checkpoint_dir_with_file(
            "quantize_config.json",
            {"quant_method": "gptq", "bits": 4, "group_size": 128},
        )
        try:
            result = detect_quantization(config, ckpt_dir)
            assert result is not None
            assert result.method == "gptq"
            assert result.supported is True
            assert result.source == "quantize_config.json"
        finally:
            import shutil

            shutil.rmtree(ckpt_dir)

    def test_awq_from_quant_config_json(self):
        """No config.quantization_config but quant_config.json exists → AWQ detected."""
        from moe_infinity.utils.quantization import detect_quantization

        config = _make_config()
        ckpt_dir = _make_checkpoint_dir_with_file(
            "quant_config.json",
            {"quant_method": "awq", "w_bit": 4, "q_group_size": 128},
        )
        try:
            result = detect_quantization(config, ckpt_dir)
            assert result is not None
            assert result.method == "awq"
            assert result.supported is True
            assert result.source == "quant_config.json"
        finally:
            import shutil

            shutil.rmtree(ckpt_dir)

    def test_hqq_from_quantization_config_json(self):
        """quantization_config.json with type='hqq' → QuantizationInfo(method='hqq', supported=False)."""
        from moe_infinity.utils.quantization import detect_quantization

        config = _make_config()
        ckpt_dir = _make_checkpoint_dir_with_file(
            "quantization_config.json",
            {"quant_method": "hqq", "type": "hqq"},
        )
        try:
            result = detect_quantization(config, ckpt_dir)
            assert result is not None
            assert result.method == "hqq"
            assert result.supported is False
        finally:
            import shutil

            shutil.rmtree(ckpt_dir)

    def test_bnb_detected_as_unsupported(self):
        """quant_method='bitsandbytes' → QuantizationInfo(method='bitsandbytes', supported=False)."""
        from moe_infinity.utils.quantization import detect_quantization

        config = _make_config(
            quantization_config={
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
            }
        )
        result = detect_quantization(config, "")
        assert result is not None
        assert result.method == "bitsandbytes"
        assert result.supported is False

    def test_gguf_checkpoint_detected_as_unsupported(self):
        """Checkpoint dir contains only .gguf files → QuantizationInfo(method='gguf', supported=False)."""
        from moe_infinity.utils.quantization import detect_quantization

        config = _make_config()
        ckpt_dir = _make_checkpoint_dir_with_gguf()
        try:
            result = detect_quantization(config, ckpt_dir)
            assert result is not None
            assert result.method == "gguf"
            assert result.supported is False
        finally:
            import shutil

            shutil.rmtree(ckpt_dir)

    def test_exl2_detected_as_unsupported(self):
        """quant_method='exl2' → QuantizationInfo(method='exl2', supported=False)."""
        from moe_infinity.utils.quantization import detect_quantization

        config = _make_config(quantization_config={"quant_method": "exl2"})
        result = detect_quantization(config, "")
        assert result is not None
        assert result.method == "exl2"
        assert result.supported is False


# ===========================================================================
# TestQuantizationValidation
# ===========================================================================


class TestQuantizationValidation:
    """Tests for validate_quantization_support()."""

    def test_gptq_passes_validation(self):
        """GPTQ QuantizationInfo does not raise."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            validate_quantization_support,
        )

        info = QuantizationInfo(
            method="gptq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        # Should not raise
        validate_quantization_support(info)

    def test_awq_passes_validation(self):
        """AWQ QuantizationInfo does not raise."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            validate_quantization_support,
        )

        info = QuantizationInfo(
            method="awq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        validate_quantization_support(info)

    def test_hqq_raises_clear_error(self):
        """HQQ raises ValueError with message mentioning 'hqq' and 'not supported'."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            validate_quantization_support,
        )

        info = QuantizationInfo(
            method="hqq",
            supported=False,
            bits=4,
            group_size=None,
            config_dict={},
            source="config",
        )
        with pytest.raises(ValueError, match="(?i)hqq"):
            validate_quantization_support(info)

    def test_bnb_raises_clear_error(self):
        """bitsandbytes raises ValueError with actionable message."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            validate_quantization_support,
        )

        info = QuantizationInfo(
            method="bitsandbytes",
            supported=False,
            bits=4,
            group_size=None,
            config_dict={},
            source="config",
        )
        with pytest.raises(ValueError, match="(?i)bitsandbytes"):
            validate_quantization_support(info)

    def test_gguf_raises_clear_error(self):
        """GGUF raises ValueError suggesting llama.cpp or Ollama as alternatives."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            validate_quantization_support,
        )

        info = QuantizationInfo(
            method="gguf",
            supported=False,
            bits=None,
            group_size=None,
            config_dict={},
            source="file",
        )
        with pytest.raises(ValueError, match="(?i)(llama.cpp|ollama)"):
            validate_quantization_support(info)

    def test_exl2_raises_clear_error(self):
        """EXL2 raises ValueError suggesting ExLlamaV2."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            validate_quantization_support,
        )

        info = QuantizationInfo(
            method="exl2",
            supported=False,
            bits=None,
            group_size=None,
            config_dict={},
            source="config",
        )
        with pytest.raises(ValueError, match="(?i)exl2"):
            validate_quantization_support(info)

    def test_unsupported_includes_model_name_in_error(self):
        """Error message includes the model name for user debugging."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            validate_quantization_support,
        )

        info = QuantizationInfo(
            method="hqq",
            supported=False,
            bits=4,
            group_size=None,
            config_dict={},
            source="config",
        )
        with pytest.raises(ValueError, match="my-cool-model"):
            validate_quantization_support(info, model_name="my-cool-model")


# ===========================================================================
# TestTensorCastDecision
# ===========================================================================


class TestTensorCastDecision:
    """Tests for should_cast_tensor() and get_quant_dtype_for_tensor()."""

    def test_no_quant_always_casts(self):
        """quant_info=None → always returns True (existing behavior preserved)."""
        from moe_infinity.utils.quantization import should_cast_tensor

        assert (
            should_cast_tensor("model.layers.0.self_attn.q_proj.weight", None)
            is True
        )
        assert should_cast_tensor("anything.qweight", None) is True

    def test_gptq_skips_cast_for_qweight(self):
        """GPTQ: tensor named '*.qweight' → should_cast=False."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            should_cast_tensor,
        )

        info = QuantizationInfo(
            method="gptq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        assert (
            should_cast_tensor(
                "model.layers.0.block_sparse_moe.experts.0.w1.qweight", info
            )
            is False
        )

    def test_gptq_skips_cast_for_qzeros(self):
        """GPTQ: '*.qzeros' → should_cast=False."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            should_cast_tensor,
        )

        info = QuantizationInfo(
            method="gptq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        assert (
            should_cast_tensor(
                "model.layers.0.block_sparse_moe.experts.0.w1.qzeros", info
            )
            is False
        )

    def test_gptq_skips_cast_for_scales(self):
        """GPTQ: '*.scales' → should_cast=False."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            should_cast_tensor,
        )

        info = QuantizationInfo(
            method="gptq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        assert (
            should_cast_tensor(
                "model.layers.0.block_sparse_moe.experts.0.w1.scales", info
            )
            is False
        )

    def test_gptq_skips_cast_for_g_idx(self):
        """GPTQ: '*.g_idx' → should_cast=False."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            should_cast_tensor,
        )

        info = QuantizationInfo(
            method="gptq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        assert (
            should_cast_tensor(
                "model.layers.0.block_sparse_moe.experts.0.w1.g_idx", info
            )
            is False
        )

    def test_gptq_casts_non_quant_tensors(self):
        """GPTQ: '*.weight' (not qweight) → should_cast=True."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            should_cast_tensor,
        )

        info = QuantizationInfo(
            method="gptq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        assert (
            should_cast_tensor("model.layers.0.self_attn.q_proj.weight", info)
            is True
        )
        assert should_cast_tensor("model.embed_tokens.weight", info) is True

    def test_awq_skips_cast_for_qweight(self):
        """AWQ: '*.qweight' → should_cast=False."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            should_cast_tensor,
        )

        info = QuantizationInfo(
            method="awq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        assert (
            should_cast_tensor(
                "model.layers.0.mlp.experts.0.gate_proj.qweight", info
            )
            is False
        )

    def test_awq_skips_cast_for_qzeros(self):
        """AWQ: '*.qzeros' → should_cast=False."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            should_cast_tensor,
        )

        info = QuantizationInfo(
            method="awq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        assert (
            should_cast_tensor(
                "model.layers.0.mlp.experts.0.gate_proj.qzeros", info
            )
            is False
        )

    def test_awq_skips_cast_for_scales(self):
        """AWQ: '*.scales' → should_cast=False."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            should_cast_tensor,
        )

        info = QuantizationInfo(
            method="awq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        assert (
            should_cast_tensor(
                "model.layers.0.mlp.experts.0.gate_proj.scales", info
            )
            is False
        )

    def test_awq_casts_non_quant_tensors(self):
        """AWQ: regular tensors → should_cast=True."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            should_cast_tensor,
        )

        info = QuantizationInfo(
            method="awq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        assert (
            should_cast_tensor("model.layers.0.self_attn.q_proj.weight", info)
            is True
        )

    def test_quant_dtype_preserves_int32_for_qweight(self):
        """get_quant_dtype: qweight → returns original tensor dtype (int32), not model dtype."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            get_quant_dtype_for_tensor,
        )

        info = QuantizationInfo(
            method="gptq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        tensor = torch.zeros(1, dtype=torch.int32)
        result = get_quant_dtype_for_tensor("w1.qweight", tensor, info)
        assert result == torch.int32

    def test_quant_dtype_preserves_fp16_for_scales(self):
        """get_quant_dtype: scales → returns original tensor dtype (float16)."""
        from moe_infinity.utils.quantization import (
            QuantizationInfo,
            get_quant_dtype_for_tensor,
        )

        info = QuantizationInfo(
            method="gptq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={},
            source="config",
        )
        tensor = torch.zeros(1, dtype=torch.float16)
        result = get_quant_dtype_for_tensor("w1.scales", tensor, info)
        assert result == torch.float16

    def test_quant_dtype_returns_none_for_regular_tensor(self):
        """get_quant_dtype: regular weight with no quant → returns None (use model dtype)."""
        from moe_infinity.utils.quantization import get_quant_dtype_for_tensor

        tensor = torch.zeros(1, dtype=torch.float32)
        result = get_quant_dtype_for_tensor("w1.weight", tensor, None)
        assert result is None
