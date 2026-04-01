# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnannotatedClassAttribute=false, reportUninitializedInstanceVariable=false, reportPrivateUsage=false, reportPrivateLocalImportUsage=false, reportUnusedImport=false, reportUnusedCallResult=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportExplicitAny=false, reportAny=false, reportArgumentType=false, reportOperatorIssue=false, reportImplicitStringConcatenation=false, reportUnnecessaryComparison=false, reportUnreachable=false, reportMissingTypeArgument=false, reportDeprecated=false, reportGeneralTypeIssues=false
import importlib
import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest
import torch

if "flash_attn" not in sys.modules:
    flash_attn_stub = types.ModuleType("flash_attn")
    flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None
    )
    sys.modules["flash_attn"] = flash_attn_stub
elif getattr(sys.modules["flash_attn"], "__spec__", None) is None:
    sys.modules["flash_attn"].__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None
    )

from moe_infinity.runtime.model_offload import OffloadEngine
from moe_infinity.utils.hf_config import parse_expert_id
from moe_infinity.utils.quantization import QuantizationInfo


def _gptq_info() -> QuantizationInfo:
    return QuantizationInfo(
        method="gptq",
        supported=True,
        bits=4,
        group_size=128,
        config_dict={"quant_method": "gptq", "bits": 4},
        source="config.quantization_config",
    )


def _engine_for_gptq(with_config_attr: bool = True) -> OffloadEngine:
    engine = OffloadEngine.__new__(OffloadEngine)
    engine.dtype_cls = torch.bfloat16
    engine.quant_method = None
    engine._quant_info = _gptq_info()
    engine.config = (
        SimpleNamespace(quantization_config={"quant_method": "gptq", "bits": 4})
        if with_config_attr
        else SimpleNamespace()
    )
    return engine


class TestGPTQTensorHandling:
    def test_qweight_not_cast_to_float(self):
        engine = _engine_for_gptq()
        state_dict = {
            "model.layers.0.block_sparse_moe.experts.0.w1.qweight": torch.ones(
                (2, 2), dtype=torch.int32
            )
        }

        OffloadEngine._cast_state_dict_tensors(engine, state_dict)

        assert (
            state_dict[
                "model.layers.0.block_sparse_moe.experts.0.w1.qweight"
            ].dtype
            == torch.int32
        )

    def test_scales_preserved_as_float16(self):
        engine = _engine_for_gptq()
        state_dict = {
            "model.layers.0.block_sparse_moe.experts.0.w1.scales": torch.ones(
                (2, 2), dtype=torch.float16
            )
        }

        OffloadEngine._cast_state_dict_tensors(engine, state_dict)

        assert (
            state_dict[
                "model.layers.0.block_sparse_moe.experts.0.w1.scales"
            ].dtype
            == torch.float16
        )

    def test_g_idx_preserved_as_int32(self):
        engine = _engine_for_gptq()
        state_dict = {
            "model.layers.0.block_sparse_moe.experts.0.w1.g_idx": torch.ones(
                (4,), dtype=torch.int32
            )
        }

        OffloadEngine._cast_state_dict_tensors(engine, state_dict)

        assert (
            state_dict[
                "model.layers.0.block_sparse_moe.experts.0.w1.g_idx"
            ].dtype
            == torch.int32
        )

    def test_non_expert_weight_still_cast(self):
        engine = _engine_for_gptq()
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(
                (2, 2), dtype=torch.float32
            )
        }

        OffloadEngine._cast_state_dict_tensors(engine, state_dict)

        assert (
            state_dict["model.layers.0.self_attn.q_proj.weight"].dtype
            == torch.bfloat16
        )

    def test_gptq_config_detection_from_config_attr(self, monkeypatch):
        engine = _engine_for_gptq(with_config_attr=True)
        model = object()

        seen = {}

        class FakeGPTQQuantizer:
            @staticmethod
            def from_dict(cfg):
                seen["cfg"] = dict(cfg)

                class _Q:
                    def convert_model(self, m):
                        return m

                return _Q()

        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "optimum.gptq":
                return SimpleNamespace(GPTQQuantizer=FakeGPTQQuantizer)
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        OffloadEngine._apply_quantized_model_conversion(engine, model)

        assert seen["cfg"]["quant_method"] == "gptq"

    def test_gptq_quantizer_called_after_model_creation(self, monkeypatch):
        engine = _engine_for_gptq(with_config_attr=False)
        model = object()

        calls = {"from_dict": 0, "convert_model": 0}

        class FakeGPTQQuantizer:
            @staticmethod
            def from_dict(cfg):
                calls["from_dict"] += 1

                class _Q:
                    def convert_model(self, m):
                        calls["convert_model"] += 1
                        return m

                return _Q()

        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "optimum.gptq":
                return SimpleNamespace(GPTQQuantizer=FakeGPTQQuantizer)
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        OffloadEngine._apply_quantized_model_conversion(engine, model)

        assert calls == {"from_dict": 1, "convert_model": 1}

    def test_gptq_exllama_disabled(self, monkeypatch):
        engine = _engine_for_gptq(with_config_attr=False)
        model = object()

        class FakeGPTQQuantizer:
            @staticmethod
            def from_dict(cfg):
                class _Q:
                    def convert_model(self, m):
                        return m

                return _Q()

        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "optimum.gptq":
                return SimpleNamespace(GPTQQuantizer=FakeGPTQQuantizer)
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        OffloadEngine._apply_quantized_model_conversion(engine, model)

        assert engine.config.quantization_config["use_exllama"] is False
        assert engine.config.quantization_config["disable_exllama"] is True

    def test_gptq_missing_optimum_raises_clear_error(self, monkeypatch):
        engine = _engine_for_gptq(with_config_attr=False)

        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "optimum.gptq":
                raise ImportError("optimum missing")
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        with pytest.raises(
            ImportError, match="optimum.*pip install optimum\\[gptq\\]"
        ):
            OffloadEngine._apply_quantized_model_conversion(engine, object())


class TestGPTQExpertKeyMapping:
    def test_mixtral_gptq_expert_keys_parsed(self):
        config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=4,
            num_local_experts=8,
        )
        key = "model.layers.0.block_sparse_moe.experts.0.w1.qweight"
        assert parse_expert_id(key, config) == (0, 0)

    def test_qwen_gptq_expert_keys_parsed(self):
        config = SimpleNamespace(
            architectures=["Qwen3MoeForCausalLM"],
            num_hidden_layers=4,
            num_experts=8,
        )
        key = "model.layers.1.mlp.experts.0.gate_proj.qweight"
        assert parse_expert_id(key, config) == (1, 0)

    def test_deepseek_gptq_expert_keys_parsed(self):
        config = SimpleNamespace(
            architectures=["DeepseekV3ForCausalLM"],
            num_hidden_layers=4,
            n_routed_experts=8,
        )
        key = "model.layers.2.mlp.experts.0.gate_proj.qweight"
        assert parse_expert_id(key, config) == (2, 0)
