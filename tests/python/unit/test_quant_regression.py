# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnannotatedClassAttribute=false, reportUninitializedInstanceVariable=false, reportPrivateUsage=false, reportPrivateLocalImportUsage=false, reportUnusedImport=false, reportUnusedCallResult=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportExplicitAny=false, reportAny=false, reportArgumentType=false, reportOperatorIssue=false, reportImplicitStringConcatenation=false, reportUnnecessaryComparison=false, reportUnreachable=false, reportMissingTypeArgument=false, reportDeprecated=false, reportGeneralTypeIssues=false
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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

from moe_infinity.runtime.model_offload import (
    OffloadEngine,
    _write_model_signature,
)
from moe_infinity.utils.hf_config import parse_expert_id
from moe_infinity.utils.quantization import detect_quantization


class TestFullPrecisionRegression:
    def test_fp16_mixtral_config_no_quant_detected(self):
        config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            model_type="mixtral",
            torch_dtype=torch.float16,
            num_hidden_layers=2,
            hidden_size=64,
            num_local_experts=4,
        )
        assert detect_quantization(config, "") is None

    def test_fp16_tensor_cast_unchanged(self):
        engine = OffloadEngine.__new__(OffloadEngine)
        engine.dtype_cls = torch.float16
        engine._quant_info = None

        state_dict = {
            "model.embed_tokens.weight": torch.ones(
                (2, 2), dtype=torch.float32
            ),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(
                (2, 2), dtype=torch.float32
            ),
        }
        OffloadEngine._cast_state_dict_tensors(engine, state_dict)

        assert state_dict["model.embed_tokens.weight"].dtype == torch.float16
        assert (
            state_dict["model.layers.0.self_attn.q_proj.weight"].dtype
            == torch.float16
        )

    def test_deepseek_v3_fp8_path_unchanged(self):
        engine = OffloadEngine.__new__(OffloadEngine)
        engine.dtype_cls = torch.float8_e4m3fn
        engine._quant_info = None

        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(
                (4, 4), dtype=torch.float16
            )
        }
        OffloadEngine._cast_state_dict_tensors(engine, state_dict)

        assert (
            state_dict["model.layers.0.self_attn.q_proj.weight"].dtype
            == torch.float8_e4m3fn
        )

    def test_name_id_map_format_unchanged(self):
        class _DummyArcherEngine:
            def is_tensor_offloaded(self, _):
                return False

            def offload(self, tensor, tensor_id):
                assert tensor is not None
                assert isinstance(tensor_id, int)

        engine = OffloadEngine.__new__(OffloadEngine)
        engine.param_id = 0
        engine.name_id_map = {}
        engine.archer_engine = _DummyArcherEngine()

        state_dict = {
            "model.embed_tokens.weight": torch.ones((1,), dtype=torch.float16),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(
                (1,), dtype=torch.float16
            ),
        }
        OffloadEngine._offload_state_dict(
            engine, state_dict, empty_state_dict={}
        )

        assert set(engine.name_id_map.keys()) == set(state_dict.keys())
        assert all(isinstance(v, int) for v in engine.name_id_map.values())

    def test_model_signature_unchanged(self, tmp_path: Path):
        config = SimpleNamespace(
            model_type="mixtral",
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=2,
            hidden_size=64,
            vocab_size=32000,
            intermediate_size=128,
            num_local_experts=4,
            torch_dtype="torch.float16",
        )
        _write_model_signature(str(tmp_path), "fp16-model", config)

        signature_path = tmp_path / "model_signature.json"
        with signature_path.open("r") as f:
            payload = json.load(f)

        assert set(payload.keys()) == {
            "model_name",
            "config_fingerprint",
            "signature_version",
        }
        assert payload["model_name"] == "fp16-model"
        assert payload["signature_version"] == 1


class TestQuantizedKeyPatternParsing:
    def test_standard_weight_key_still_works(self):
        config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=4,
            num_local_experts=8,
        )
        key = "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        assert parse_expert_id(key, config) == (0, 0)

    def test_qweight_key_works(self):
        config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=4,
            num_local_experts=8,
        )
        key = "model.layers.0.block_sparse_moe.experts.0.w1.qweight"
        assert parse_expert_id(key, config) == (0, 0)

    def test_scales_key_works(self):
        config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=4,
            num_local_experts=8,
        )
        key = "model.layers.0.block_sparse_moe.experts.0.w1.scales"
        assert parse_expert_id(key, config) == (0, 0)

    def test_g_idx_key_works(self):
        config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=4,
            num_local_experts=8,
        )
        key = "model.layers.0.block_sparse_moe.experts.0.w1.g_idx"
        assert parse_expert_id(key, config) == (0, 0)

    def test_non_expert_key_returns_none(self):
        config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=4,
            num_local_experts=8,
        )
        assert parse_expert_id("model.embed_tokens.weight", config) == (
            None,
            None,
        )
