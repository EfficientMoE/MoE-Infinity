import importlib.machinery
import inspect
import os
import sys
import types

import pytest


def _ensure_flash_attn_stub_has_spec() -> None:
    flash_attn_module = sys.modules.get("flash_attn")
    if flash_attn_module is None:
        flash_attn_module = types.ModuleType("flash_attn")
        flash_attn_module.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn", loader=None
        )
        sys.modules["flash_attn"] = flash_attn_module
    elif getattr(flash_attn_module, "__spec__", None) is None:
        flash_attn_module.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn", loader=None
        )


_ensure_flash_attn_stub_has_spec()


def test_gpt_oss_all_components_importable():
    from moe_infinity.common.constants import MODEL_MAPPING_NAMES
    from moe_infinity.models.gpt_oss import SyncGptOssMLP
    from moe_infinity.utils.hf_config import parse_expert_id, parse_moe_param
    from moe_infinity.utils.mxfp4 import (
        get_mxfp4_modules_to_not_convert,
        identify_mxfp4_pairs,
        is_mxfp4_quantized,
    )

    assert "gptoss" in MODEL_MAPPING_NAMES
    assert SyncGptOssMLP is not None
    assert callable(is_mxfp4_quantized)
    assert callable(get_mxfp4_modules_to_not_convert)
    assert callable(identify_mxfp4_pairs)
    assert callable(parse_moe_param)
    assert callable(parse_expert_id)


def test_gpt_oss_monkey_patch_module_accessible():
    import transformers.models.gpt_oss.modeling_gpt_oss as mod

    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    assert hasattr(mod, "GptOssMLP")
    assert mod.GptOssMLP is not SyncGptOssMLP


def test_gpt_oss_120b_config_parseable():
    from unittest.mock import MagicMock

    from moe_infinity.utils.hf_config import parse_moe_param

    config_120b = MagicMock()
    config_120b.architectures = ["GptOssForCausalLM"]
    config_120b.model_type = "gpt_oss"
    config_120b.num_hidden_layers = 94
    config_120b.num_local_experts = 128
    config_120b.num_experts_per_tok = 4

    layers, experts, enc_layers = parse_moe_param(config_120b)
    assert layers == 94
    assert experts == 128
    assert enc_layers == 0


def test_gpt_oss_model_offload_has_gptoss_patches():
    import moe_infinity.runtime.model_offload as mod

    source = inspect.getsource(mod)
    assert "gpt_oss.modeling_gpt_oss" in source
    assert "GptOssMLP = SyncGptOssMLP" in source
    assert "_old_gpt_oss_mlp" in source


@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.slow
@pytest.mark.integration
def test_gpt_oss_20b_e2e():
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    checkpoint = "openai/gpt-oss-20b"
    offload_path = os.path.expanduser("~/moe-infinity-gpt-oss-e2e")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = MoE(
        checkpoint,
        {
            "offload_path": offload_path,
            "device_memory_ratio": 0.75,
        },
    )

    prompt = "Explain quantum mechanics in one sentence."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    output_ids = model.generate(input_ids, max_new_tokens=32)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assert len(output_text) > len(prompt)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.integration
def test_gpt_oss_120b_config():
    from transformers import AutoConfig

    from moe_infinity.common.constants import MODEL_MAPPING_NAMES
    from moe_infinity.utils.hf_config import parse_moe_param

    config = AutoConfig.from_pretrained(
        "openai/gpt-oss-120b", trust_remote_code=True
    )
    layers, experts, enc_layers = parse_moe_param(config)
    assert layers > 0
    assert experts > 0
    assert enc_layers == 0

    architectures = config.architectures or []
    assert architectures
    arch_str = architectures[0].lower()
    matched = next(
        (key for key in MODEL_MAPPING_NAMES if key in arch_str), None
    )
    assert matched == "gptoss"
