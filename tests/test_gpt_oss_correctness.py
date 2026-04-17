import importlib.machinery
import os
import sys
import types
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch


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


def test_sync_gpt_oss_mlp_forward_shape():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    config = MagicMock()
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_local_experts = 4
    config.num_experts_per_tok = 2

    mlp = SyncGptOssMLP(config)
    torch.nn.init.normal_(mlp.experts.gate_up_proj)
    torch.nn.init.normal_(mlp.experts.down_proj)
    torch.nn.init.zeros_(mlp.experts.gate_up_proj_bias)
    torch.nn.init.zeros_(mlp.experts.down_proj_bias)
    torch.nn.init.normal_(mlp.router.weight)
    torch.nn.init.zeros_(mlp.router.bias)

    hidden = torch.randn(1, 3, 64)
    with torch.no_grad():
        output, router_logits = mlp(hidden)

    assert output.shape == (1, 3, 64)
    assert router_logits.shape == (3, 4)


def test_sync_gpt_oss_mlp_router_logits_shape():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    config = MagicMock()
    config.hidden_size = 32
    config.intermediate_size = 32
    config.num_local_experts = 8
    config.num_experts_per_tok = 2

    mlp = SyncGptOssMLP(config)
    for param in mlp.parameters():
        torch.nn.init.normal_(param, std=0.01)

    hidden = torch.randn(2, 5, 32)
    with torch.no_grad():
        _, router_logits = mlp(hidden)

    assert router_logits.shape == (10, 8)


def test_gptoss_architecture_in_constants():
    from moe_infinity.common.constants import MODEL_MAPPING_NAMES

    assert "gptoss" in MODEL_MAPPING_NAMES
    arch_str = "gptossforcausallm"
    matched = next(
        (key for key in MODEL_MAPPING_NAMES if key in arch_str), None
    )
    assert matched == "gptoss"


def test_gptoss_parse_moe_param():
    from moe_infinity.utils.hf_config import parse_moe_param

    config = MagicMock()
    config.architectures = ["GptOssForCausalLM"]
    config.num_hidden_layers = 24
    config.num_local_experts = 32

    layers, experts, enc_layers = parse_moe_param(config)
    assert layers == 24
    assert experts == 32
    assert enc_layers == 0


@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.slow
def test_gpt_oss_20b_forward_parity():
    from transformers import AutoTokenizer, GptOssForCausalLM

    from moe_infinity import MoE

    checkpoint = "openai/gpt-oss-20b"
    offload_path = os.path.expanduser("~/moe-infinity-gpt-oss-parity")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer("What is 2+2?", return_tensors="pt").to("cuda:0")

    ref_model = cast(
        torch.nn.Module,
        GptOssForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        ),
    )
    ref_model = ref_model.to("cuda:0")
    with torch.no_grad():
        ref_out = ref_model(**inputs).logits.cpu()
    del ref_model

    model = MoE(
        checkpoint,
        {
            "offload_path": offload_path,
            "device_memory_ratio": 0.75,
        },
    )
    with torch.no_grad():
        moe_out = model.model(**inputs).logits.cpu()

    assert torch.allclose(ref_out, moe_out, atol=1e-2), (
        f"Parity check failed. Max diff: {(ref_out - moe_out).abs().max():.4f}"
    )
