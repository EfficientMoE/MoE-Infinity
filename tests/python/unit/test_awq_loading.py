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


def _awq_info() -> QuantizationInfo:
    return QuantizationInfo(
        method="awq",
        supported=True,
        bits=4,
        group_size=128,
        config_dict={"quant_method": "awq", "w_bit": 4},
        source="quant_config.json",
    )


def _engine_for_awq() -> OffloadEngine:
    engine = OffloadEngine.__new__(OffloadEngine)
    engine.dtype_cls = torch.bfloat16
    engine.quant_method = None
    engine._quant_info = _awq_info()
    engine.config = SimpleNamespace()
    return engine


class TestAWQTensorHandling:
    def test_awq_qweight_not_cast_to_float(self):
        engine = _engine_for_awq()
        state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.qweight": torch.ones(
                (2, 2), dtype=torch.int32
            )
        }

        OffloadEngine._cast_state_dict_tensors(engine, state_dict)

        assert (
            state_dict["model.layers.0.mlp.experts.0.gate_proj.qweight"].dtype
            == torch.int32
        )

    def test_awq_scales_preserved(self):
        engine = _engine_for_awq()
        state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.scales": torch.ones(
                (2, 2), dtype=torch.float16
            )
        }

        OffloadEngine._cast_state_dict_tensors(engine, state_dict)

        assert (
            state_dict["model.layers.0.mlp.experts.0.gate_proj.scales"].dtype
            == torch.float16
        )

    def test_awq_qzeros_preserved(self):
        engine = _engine_for_awq()
        state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.qzeros": torch.ones(
                (2, 2), dtype=torch.int32
            )
        }

        OffloadEngine._cast_state_dict_tensors(engine, state_dict)

        assert (
            state_dict["model.layers.0.mlp.experts.0.gate_proj.qzeros"].dtype
            == torch.int32
        )

    def test_non_quant_tensor_still_cast(self):
        engine = _engine_for_awq()
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


class TestAWQModelConversion:
    def test_awq_detected_triggers_conversion(self, monkeypatch):
        engine = _engine_for_awq()
        model = SimpleNamespace(converted=False)
        calls = {"replace": 0}

        def replace_linear_modules(m):
            calls["replace"] += 1
            m.converted = True
            return m

        fake_awq = SimpleNamespace(
            replace_linear_modules=replace_linear_modules
        )
        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "awq":
                return fake_awq
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        converted_model = OffloadEngine._apply_quantized_model_conversion(
            engine, model
        )

        assert calls["replace"] == 1
        assert converted_model.converted is True
        assert engine.quant_method == "awq"

    def test_awq_missing_autoawq_raises_clear_error(self, monkeypatch):
        engine = _engine_for_awq()
        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "awq":
                raise ImportError("no autoawq")
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        with pytest.raises(ImportError, match="autoawq"):
            OffloadEngine._apply_quantized_model_conversion(engine, object())

    def test_awq_conversion_preserves_model_structure(self, monkeypatch):
        engine = _engine_for_awq()
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=object())])
        )

        def replace_linear_modules(m):
            return m

        fake_awq = SimpleNamespace(
            replace_linear_modules=replace_linear_modules
        )
        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "awq":
                return fake_awq
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        converted_model = OffloadEngine._apply_quantized_model_conversion(
            engine, model
        )

        assert hasattr(converted_model, "model")
        assert hasattr(converted_model.model, "layers")
        assert hasattr(converted_model.model.layers[0], "mlp")


class TestAWQExpertKeyMapping:
    def test_awq_mixtral_expert_keys(self):
        config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=4,
            num_local_experts=8,
        )
        key = "model.layers.0.block_sparse_moe.experts.0.w1.qweight"
        assert parse_expert_id(key, config) == (0, 0)

    def test_awq_qwen_expert_keys(self):
        config = SimpleNamespace(
            architectures=["Qwen3MoeForCausalLM"],
            num_hidden_layers=4,
            num_experts=8,
        )
        key = "model.layers.1.mlp.experts.0.gate_proj.qweight"
        assert parse_expert_id(key, config) == (1, 0)

    def test_awq_deepseek_expert_keys(self):
        config = SimpleNamespace(
            architectures=["DeepseekV3ForCausalLM"],
            num_hidden_layers=4,
            n_routed_experts=8,
        )
        key = "model.layers.2.mlp.experts.0.gate_proj.qweight"
        assert parse_expert_id(key, config) == (2, 0)
