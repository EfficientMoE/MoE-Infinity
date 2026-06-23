from dataclasses import dataclass
from typing import Optional

import pytest
import torch


@dataclass
class QuantizationConfig:
    quant_method: str
    bits: int = 4
    group_size: int = 128


@dataclass
class ModelConfig:
    model_type: str
    quantization_config: Optional[QuantizationConfig]


def make_gptq_config():
    return ModelConfig(
        model_type="mixtral",
        quantization_config=QuantizationConfig(
            quant_method="gptq", bits=4, group_size=128
        ),
    )


def make_non_quantized_config():
    return ModelConfig(model_type="mixtral", quantization_config=None)


def make_mxfp4_config():
    return ModelConfig(
        model_type="gpt_oss",
        quantization_config=QuantizationConfig(quant_method="mxfp4"),
    )


class TestGPTQDetection:
    def test_positive(self):
        from moe_infinity.utils.gptq import is_gptq_quantized

        assert is_gptq_quantized(make_gptq_config()) is True

    def test_negative_none(self):
        from moe_infinity.utils.gptq import is_gptq_quantized

        assert is_gptq_quantized(make_non_quantized_config()) is False

    def test_negative_mxfp4(self):
        from moe_infinity.utils.gptq import is_gptq_quantized

        assert is_gptq_quantized(make_mxfp4_config()) is False

    def test_no_quantization_attr(self):
        from moe_infinity.utils.gptq import is_gptq_quantized

        @dataclass
        class BareConfig:
            model_type: str

        assert is_gptq_quantized(BareConfig(model_type="mixtral")) is False

    def test_dict_quantization_config(self):
        from moe_infinity.utils.gptq import is_gptq_quantized

        @dataclass
        class DictConfig:
            quantization_config: dict

        cfg = DictConfig(
            quantization_config={"quant_method": "gptq", "bits": 4}
        )
        assert is_gptq_quantized(cfg) is True


class TestGPTQBitsAndGroupSize:
    def test_bits(self):
        from moe_infinity.utils.gptq import get_gptq_bits

        assert get_gptq_bits(make_gptq_config()) == 4

    def test_group_size(self):
        from moe_infinity.utils.gptq import get_gptq_group_size

        assert get_gptq_group_size(make_gptq_config()) == 128

    def test_defaults_when_no_config(self):
        from moe_infinity.utils.gptq import get_gptq_bits, get_gptq_group_size

        assert get_gptq_bits(make_non_quantized_config()) == 4
        assert get_gptq_group_size(make_non_quantized_config()) == 128


class TestGPTQPackedTensor:
    def test_qweight_detected(self):
        from moe_infinity.utils.gptq import is_gptq_packed_tensor

        assert (
            is_gptq_packed_tensor("model.layers.0.experts.0.w1.qweight") is True
        )

    def test_qzeros_detected(self):
        from moe_infinity.utils.gptq import is_gptq_packed_tensor

        assert (
            is_gptq_packed_tensor("model.layers.0.experts.0.w1.qzeros") is True
        )

    def test_scales_not_packed(self):
        from moe_infinity.utils.gptq import is_gptq_packed_tensor

        assert (
            is_gptq_packed_tensor("model.layers.0.experts.0.w1.scales") is False
        )

    def test_g_idx_not_packed(self):
        from moe_infinity.utils.gptq import is_gptq_packed_tensor

        assert (
            is_gptq_packed_tensor("model.layers.0.experts.0.w1.g_idx") is False
        )

    def test_regular_weight_not_packed(self):
        from moe_infinity.utils.gptq import is_gptq_packed_tensor

        assert (
            is_gptq_packed_tensor("model.layers.0.experts.0.w1.weight") is False
        )


class TestGPTQComponent:
    def test_all_components_detected(self):
        from moe_infinity.utils.gptq import is_gptq_component

        for suffix in ("qweight", "qzeros", "scales", "g_idx"):
            name = f"model.layers.0.experts.0.w1.{suffix}"
            assert is_gptq_component(name) is True, f"Failed for {suffix}"

    def test_regular_weight_not_component(self):
        from moe_infinity.utils.gptq import is_gptq_component

        assert is_gptq_component("model.layers.0.experts.0.w1.weight") is False


class TestGPTQDtypePreservation:
    def test_packed_tensors_stay_int32(self):
        from moe_infinity.utils.gptq import is_gptq_packed_tensor

        state_dict = {
            "model.layers.0.experts.0.w1.qweight": torch.randint(
                0, 2**31, (64, 128), dtype=torch.int32
            ),
            "model.layers.0.experts.0.w1.qzeros": torch.randint(
                0, 2**31, (1, 16), dtype=torch.int32
            ),
            "model.layers.0.experts.0.w1.scales": torch.randn(
                1, 128, dtype=torch.float16
            ),
            "model.layers.0.experts.0.w1.g_idx": torch.arange(
                512, dtype=torch.int32
            ),
            "model.layers.0.gate.weight": torch.randn(
                8, 512, dtype=torch.float16
            ),
        }

        dtype_cls = torch.float16
        for k, v in state_dict.items():
            if is_gptq_packed_tensor(k):
                state_dict[k] = v.to("cpu")
            else:
                state_dict[k] = v.to(dtype_cls).to("cpu")

        assert (
            state_dict["model.layers.0.experts.0.w1.qweight"].dtype
            == torch.int32
        )
        assert (
            state_dict["model.layers.0.experts.0.w1.qzeros"].dtype
            == torch.int32
        )
        assert (
            state_dict["model.layers.0.experts.0.w1.scales"].dtype
            == torch.float16
        )
        assert (
            state_dict["model.layers.0.experts.0.w1.g_idx"].dtype
            == torch.float16
        )
        assert state_dict["model.layers.0.gate.weight"].dtype == torch.float16

    def test_bandwidth_ratio(self):
        in_features = 4096
        out_features = 14336
        bits = 4
        pack_factor = 32 // bits

        fp16_bytes = in_features * out_features * 2
        gptq_qweight_bytes = (in_features // pack_factor) * out_features * 4
        group_size = 128
        num_groups = in_features // group_size
        gptq_scales_bytes = num_groups * out_features * 2
        gptq_qzeros_bytes = num_groups * (out_features // pack_factor) * 4
        gptq_total = gptq_qweight_bytes + gptq_scales_bytes + gptq_qzeros_bytes

        ratio = fp16_bytes / gptq_total
        assert ratio > 3.0, f"Expected >3x savings, got {ratio:.1f}x"
