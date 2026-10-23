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

from moe_store.parsing.hf_config import parse_expert_id
from moe_store.parsing.quantization import QuantizationInfo

from moe_infinity.runtime.model_offload import OffloadEngine, _GPTQLinear


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
            if name == "optimum.gptq.quantizer":
                return SimpleNamespace(
                    GPTQQuantizer=FakeGPTQQuantizer,
                    is_gptqmodel_available=lambda: True,
                )
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
            if name == "optimum.gptq.quantizer":
                return SimpleNamespace(
                    GPTQQuantizer=FakeGPTQQuantizer,
                    is_gptqmodel_available=lambda: True,
                )
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
            if name == "optimum.gptq.quantizer":
                return SimpleNamespace(
                    GPTQQuantizer=FakeGPTQQuantizer,
                    is_gptqmodel_available=lambda: True,
                )
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        OffloadEngine._apply_quantized_model_conversion(engine, model)

        assert engine.config.quantization_config["use_exllama"] is False
        assert engine.config.quantization_config["disable_exllama"] is True

    def test_gptq_missing_optimum_raises_clear_error(self, monkeypatch):
        engine = _engine_for_gptq(with_config_attr=False)

        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "optimum.gptq.quantizer":
                raise ImportError("optimum missing")
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        with pytest.raises(
            ImportError, match="optimum.*pip install optimum gptqmodel"
        ):
            OffloadEngine._apply_quantized_model_conversion(engine, object())

    @pytest.mark.parametrize("legacy_optimum", [False, True])
    def test_gptq_missing_backend_fails_before_from_dict(
        self, monkeypatch, legacy_optimum
    ):
        def from_dict(cfg):
            pytest.fail("from_dict must not run without a backend")

        module = SimpleNamespace(
            GPTQQuantizer=SimpleNamespace(from_dict=from_dict),
            is_gptqmodel_available=lambda: False,
        )
        if legacy_optimum:
            module.is_auto_gptq_available = lambda: False

        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "optimum.gptq.quantizer":
                return module
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        with pytest.raises(
            ImportError, match="pip install optimum gptqmodel"
        ) as exc:
            OffloadEngine._apply_quantized_model_conversion(
                _engine_for_gptq(), object()
            )

        assert "No GPTQ backend" in str(exc.value.__cause__)

    def test_gptq_legacy_auto_gptq_backend_still_converts(self, monkeypatch):
        model = object()
        module = SimpleNamespace(
            GPTQQuantizer=SimpleNamespace(
                from_dict=lambda cfg: SimpleNamespace(convert_model=lambda m: m)
            ),
            is_gptqmodel_available=lambda: False,
            is_auto_gptq_available=lambda: True,
        )
        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "optimum.gptq.quantizer":
                return module
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        assert (
            OffloadEngine._apply_quantized_model_conversion(
                _engine_for_gptq(), model
            )
            is model
        )

    def test_gptq_incompatible_backend_keeps_original_cause(self, monkeypatch):
        incompatible = ImportError("gptqmodel version is too old")
        original_import_module = importlib.import_module

        def _fake_import(name, package=None):
            if name == "optimum.gptq.quantizer":
                raise incompatible
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        with pytest.raises(ImportError, match="compatible GPTQ backend") as exc:
            OffloadEngine._apply_quantized_model_conversion(
                _engine_for_gptq(), object()
            )

        assert exc.value.__cause__ is incompatible


def _native_model_and_engine(
    bits=4, checkpoint_format="gptq", dtype=torch.float32
):
    model = torch.nn.Module()
    model.base_model_prefix = "model"
    model.model = torch.nn.Module()
    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    expert = torch.nn.Module()
    expert.up_proj = torch.nn.Linear(32, 32, bias=True, dtype=dtype)
    block.mlp.experts = torch.nn.ModuleList([expert])
    block.mlp.shared_expert_gate = torch.nn.Linear(32, 1, bias=False)
    block.mlp.shared_expert = torch.nn.Linear(32, 32, bias=False)
    model.model.layers = torch.nn.ModuleList([block])
    engine = _engine_for_gptq()
    engine.config.quantization_config.update(
        bits=bits,
        checkpoint_format=checkpoint_format,
        desc_act=True,
        dynamic={"-:.*shared_expert.*": {}},
    )
    name = "model.layers.0.mlp.experts.0.up_proj"
    engine.name_id_map = {
        f"{name}.{key}": index
        for index, key in enumerate(
            ("qweight", "qzeros", "scales", "g_idx", "bias")
        )
    }
    return model, engine, name


def _pack_rows(values, bits):
    """Independent scalar packer: each column contains consecutive input rows."""
    pack = 32 // bits
    words = [
        [
            sum(int(values[row + i, col]) << (bits * i) for i in range(pack))
            for col in range(values.shape[1])
        ]
        for row in range(0, values.shape[0], pack)
    ]
    return torch.tensor(words, dtype=torch.int64).to(torch.int32)


class TestNativeGPTQConversion:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16]
    )
    def test_packed_forward_without_backend_imports(
        self, monkeypatch, bits, checkpoint_format, dtype
    ):
        model, engine, name = _native_model_and_engine(
            bits, checkpoint_format, dtype
        )
        model.eval()
        gate = model.model.layers[0].mlp.shared_expert_gate
        shared = model.model.layers[0].mlp.shared_expert
        original_import = importlib.import_module

        def no_backend_imports(module, package=None):
            if module.startswith(("optimum", "gptqmodel", "auto_gptq")):
                pytest.fail(f"native conversion imported {module}")
            return original_import(module, package)

        monkeypatch.setattr(importlib, "import_module", no_backend_imports)
        assert engine._apply_quantized_model_conversion(model) is model
        layer = model.get_submodule(name)
        assert isinstance(layer, _GPTQLinear)
        assert not layer.training
        assert model.model.layers[0].mlp.shared_expert_gate is gate
        assert model.model.layers[0].mlp.shared_expert is shared
        assert all(buf.numel() == 1 for buf in layer.buffers())
        assert layer.g_idx.dtype == torch.int32
        assert layer.scales.dtype == dtype

        # Include signed packed words, nonsequential act-order groups, and
        # the GPTQ v1 zero offset. No dense weight is retained by the layer.
        mask = (1 << bits) - 1
        rng = torch.Generator().manual_seed(270)
        integers = torch.randint(mask + 1, (32, 32), generator=rng)
        offset = int(checkpoint_format == "gptq")
        zeros = torch.randint(mask + 1, (4, 32), generator=rng) + offset
        groups = torch.arange(32, dtype=torch.int32).flip(0) % 4
        scales = torch.randint(1, 5, (4, 32), generator=rng).to(dtype) / 32
        bias = torch.arange(32).to(dtype) / 16
        # Archer hydrates the existing buffer objects by replacing .data.
        for key, tensor in {
            "qweight": _pack_rows(integers, bits),
            "qzeros": _pack_rows((zeros - offset).T, bits).T.contiguous(),
            "scales": scales,
            "g_idx": groups,
            "bias": bias,
        }.items():
            getattr(layer, key).data = tensor
        x = (torch.arange(2 * 3 * 32).reshape(2, 3, 32) % 7).to(dtype) / 8
        dense = (integers - zeros[groups.long()]).to(dtype) * scales[
            groups.long()
        ]
        expected = x @ dense + bias
        torch.testing.assert_close(layer(x), expected)
        assert set(layer.state_dict()) == {
            "qweight",
            "qzeros",
            "scales",
            "g_idx",
            "bias",
        }

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_symmetric_uses_midpoint_zero_and_ignores_qzeros(self, bits):
        # Published Qwen3.5 GPTQ-Int4: sym=True, no checkpoint_format, and
        # qzeros hold the midpoint itself. Applying the v1 "+1" offset to
        # them shifts every weight by one quantization step.
        model, engine, name = _native_model_and_engine(bits)
        engine.config.quantization_config.pop("checkpoint_format")
        engine.config.quantization_config["sym"] = True
        engine._apply_quantized_model_conversion(model)
        layer = model.get_submodule(name)

        mask = (1 << bits) - 1
        midpoint = 1 << (bits - 1)
        rng = torch.Generator().manual_seed(270)
        integers = torch.randint(mask + 1, (32, 32), generator=rng)
        groups = torch.arange(32, dtype=torch.int32) // 8
        scales = torch.randint(1, 5, (4, 32), generator=rng).float() / 32
        stored_zeros = torch.full((4, 32), midpoint)
        for key, tensor in {
            "qweight": _pack_rows(integers, bits),
            "qzeros": _pack_rows(stored_zeros.T, bits).T.contiguous(),
            "scales": scales,
            "g_idx": groups,
            "bias": torch.zeros(32),
        }.items():
            getattr(layer, key).data = tensor
        x = (torch.arange(3 * 32).reshape(3, 32) % 5).float() / 4
        dense = (integers - midpoint).float() * scales[groups.long()]
        torch.testing.assert_close(layer(x), x @ dense)

    def test_topology_contains_all_packed_buffers(self):
        model, engine, name = _native_model_and_engine()
        second = torch.nn.Module()
        second.up_proj = torch.nn.Linear(32, 32, bias=True)
        model.model.layers[0].mlp.experts.append(second)
        for index, key in enumerate(
            ("qweight", "qzeros", "scales", "g_idx", "bias")
        ):
            engine.name_id_map[
                f"model.layers.0.mlp.experts.1.up_proj.{key}"
            ] = index + 5
        engine._apply_quantized_model_conversion(model)
        topo = engine.get_topology(model)
        assert topo == [
            ("model.layers.0.mlp.experts", [list(range(5)), list(range(5, 10))])
        ]

    def test_missing_component_rejected_before_any_replacement(self):
        model, engine, name = _native_model_and_engine()
        expert = model.model.layers[0].mlp.experts[0]
        expert.down_proj = torch.nn.Linear(32, 32)
        second_name = name.removesuffix("up_proj") + "down_proj"
        for key in ("qweight", "qzeros", "scales"):
            engine.name_id_map[f"{second_name}.{key}"] = 10
        original = model.get_submodule(name)
        with pytest.raises(ValueError, match="Incomplete GPTQ.*g_idx"):
            engine._apply_quantized_model_conversion(model)
        assert model.get_submodule(name) is original

    def test_checkpoint_without_base_prefix_maps_buffers(self):
        model, engine, name = _native_model_and_engine()
        del engine.name_id_map[f"{name}.bias"]
        engine.name_id_map = {
            key.removeprefix("model."): value
            for key, value in engine.name_id_map.items()
        }
        engine._apply_quantized_model_conversion(model)
        assert isinstance(model.get_submodule(name), _GPTQLinear)
        assert model.get_submodule(name).bias is None
        assert set(engine.name_id_map) == {
            f"{name}.{key}" for key in ("qweight", "qzeros", "scales", "g_idx")
        }

    @pytest.mark.parametrize(
        "config",
        [
            {"bits": 3},
            {"checkpoint_format": "marlin"},
            {"pack_dtype": "int16"},
            {"dynamic": {".*up_proj": {"bits": 8}}},
        ],
    )
    def test_unsupported_config_retains_optimum_guard(
        self, monkeypatch, config
    ):
        model, engine, name = _native_model_and_engine()
        engine.config.quantization_config.update(config)
        original = model.get_submodule(name)
        original_import = importlib.import_module

        def no_backend(module, package=None):
            if module == "optimum.gptq.quantizer":
                return SimpleNamespace(
                    GPTQQuantizer=SimpleNamespace(
                        from_dict=lambda cfg: pytest.fail("missing backend")
                    ),
                    is_gptqmodel_available=lambda: False,
                )
            return original_import(module, package)

        monkeypatch.setattr(importlib, "import_module", no_backend)
        with pytest.raises(ImportError, match="compatible GPTQ backend"):
            engine._apply_quantized_model_conversion(model)
        assert model.get_submodule(name) is original


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
