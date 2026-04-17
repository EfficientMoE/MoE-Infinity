import importlib.machinery
import sys
import types
from unittest.mock import MagicMock

import torch
import torch.nn as nn

flash_attn_mod = sys.modules.get("flash_attn")
if flash_attn_mod is None:
    flash_attn_mod = types.ModuleType("flash_attn")
    flash_attn_mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None
    )
    sys.modules["flash_attn"] = flash_attn_mod
elif getattr(flash_attn_mod, "__spec__", None) is None:
    flash_attn_mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None
    )


def make_gpt_oss_config():
    cfg = MagicMock()
    cfg.architectures = ["GptOssForCausalLM"]
    cfg.model_type = "gpt_oss"
    cfg.num_hidden_layers = 2
    cfg.num_local_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 64
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.vocab_size = 256
    return cfg


def test_sync_gpt_oss_mlp_is_imported_in_model_offload():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    assert issubclass(SyncGptOssMLP, nn.Module)


def test_packed_expert_slice_shape():
    e, h, i = 4, 64, 64
    gate_up_proj = torch.randn(e, h, 2 * i)
    down_proj = torch.randn(e, i, h)

    expert_idx = 2
    gate_up_slice = gate_up_proj[expert_idx]
    down_slice = down_proj[expert_idx]

    assert gate_up_slice.shape == (
        h,
        2 * i,
    ), f"Expected ({h}, {2 * i}), got {gate_up_slice.shape}"
    assert down_slice.shape == (
        i,
        h,
    ), f"Expected ({i}, {h}), got {down_slice.shape}"
    assert torch.equal(gate_up_slice, gate_up_proj[2])


def test_sync_gpt_oss_mlp_isinstance_detection():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    config = make_gpt_oss_config()
    mlp = SyncGptOssMLP(config)

    assert isinstance(mlp, SyncGptOssMLP), "isinstance check must work"
    assert isinstance(mlp, nn.Module), "Must be nn.Module subclass"


def test_sync_gpt_oss_mlp_in_model_offload_imports():
    import moe_infinity.models.gpt_oss as gpt_oss_mod

    assert hasattr(gpt_oss_mod, "SyncGptOssMLP"), (
        "SyncGptOssMLP must exist in module"
    )
