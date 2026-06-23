# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnannotatedClassAttribute=false, reportUninitializedInstanceVariable=false, reportPrivateUsage=false, reportPrivateLocalImportUsage=false, reportUnusedImport=false, reportUnusedCallResult=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportExplicitAny=false, reportAny=false, reportArgumentType=false, reportOperatorIssue=false, reportImplicitStringConcatenation=false, reportUnnecessaryComparison=false, reportUnreachable=false, reportMissingTypeArgument=false, reportDeprecated=false, reportGeneralTypeIssues=false
import importlib
from types import SimpleNamespace

import pytest

from moe_infinity.entrypoints.big_modeling import MoE
from moe_infinity.utils.quantization import (
    QuantizationInfo,
    validate_quantization_support,
)


def _unsupported_info(method: str) -> QuantizationInfo:
    return QuantizationInfo(
        method=method,
        supported=False,
        bits=4,
        group_size=128,
        config_dict={"quant_method": method},
        source="config.quantization_config",
    )


class TestUnsupportedQuantizationErrors:
    def test_hqq_error_mentions_format_name(self):
        with pytest.raises(ValueError, match="(?i)hqq"):
            validate_quantization_support(_unsupported_info("hqq"))

    def test_hqq_error_suggests_full_precision(self):
        with pytest.raises(ValueError, match="(?i)full-precision"):
            validate_quantization_support(_unsupported_info("hqq"))

    def test_bnb_error_mentions_bitsandbytes(self):
        with pytest.raises(ValueError, match="(?i)bitsandbytes"):
            validate_quantization_support(_unsupported_info("bitsandbytes"))

    def test_gguf_error_suggests_alternatives(self):
        with pytest.raises(ValueError, match="(?i)(llama.cpp|ollama)"):
            validate_quantization_support(_unsupported_info("gguf"))

    def test_exl2_error_suggests_exllamav2(self):
        with pytest.raises(ValueError, match="(?i)exllamav2"):
            validate_quantization_support(_unsupported_info("exl2"))

    def test_error_raised_before_model_download(self, monkeypatch, tmp_path):
        import huggingface_hub
        from transformers import AutoConfig

        quant_utils = importlib.import_module("moe_infinity.utils.quantization")

        model_config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            model_type="mixtral",
            torch_dtype="float16",
            num_hidden_layers=1,
            num_local_experts=2,
            hidden_size=16,
            max_position_embeddings=32,
        )

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: model_config,
        )
        monkeypatch.setattr(
            quant_utils,
            "detect_quantization",
            lambda config, checkpoint_path: _unsupported_info("hqq"),
        )
        monkeypatch.setattr(
            quant_utils,
            "validate_quantization_support",
            lambda info, model_name="": (_ for _ in ()).throw(
                ValueError("HQQ not supported")
            ),
        )

        def _snapshot_should_not_run(*args, **kwargs):
            raise AssertionError("snapshot_download should not be called")

        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            _snapshot_should_not_run,
        )

        with pytest.raises(ValueError, match="HQQ not supported"):
            MoE(
                "remote/mixtral-hqq-model",
                config={
                    "offload_path": str(tmp_path),
                    "use_native_engine": False,
                },
            )

    def test_error_raised_after_download_for_file_detected_quant(
        self, monkeypatch, tmp_path
    ):
        import huggingface_hub
        from transformers import AutoConfig

        quant_utils = importlib.import_module("moe_infinity.utils.quantization")

        model_config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            model_type="mixtral",
            torch_dtype="float16",
            num_hidden_layers=1,
            num_local_experts=2,
            hidden_size=16,
            max_position_embeddings=32,
        )

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: model_config,
        )
        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            lambda *args, **kwargs: str(tmp_path),
        )

        detect_calls = {"count": 0}

        def _detect(config, checkpoint_path):
            detect_calls["count"] += 1
            if checkpoint_path:
                return _unsupported_info("gguf")
            return None

        monkeypatch.setattr(quant_utils, "detect_quantization", _detect)
        monkeypatch.setattr(
            quant_utils,
            "validate_quantization_support",
            lambda info, model_name="": (
                (_ for _ in ()).throw(ValueError("GGUF not supported"))
                if info is not None
                else None
            ),
        )

        with pytest.raises(ValueError, match="GGUF not supported"):
            MoE(
                "remote/mixtral-file-quant-model",
                config={
                    "offload_path": str(tmp_path),
                    "use_native_engine": False,
                },
            )

        assert detect_calls["count"] >= 2

    def test_supported_quant_does_not_error(self):
        gptq = QuantizationInfo(
            method="gptq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={"quant_method": "gptq"},
            source="config",
        )
        awq = QuantizationInfo(
            method="awq",
            supported=True,
            bits=4,
            group_size=128,
            config_dict={"quant_method": "awq"},
            source="config",
        )

        validate_quantization_support(gptq)
        validate_quantization_support(awq)
