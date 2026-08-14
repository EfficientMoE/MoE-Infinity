import importlib.machinery
import sys
import types
from unittest.mock import MagicMock, Mock

import torch

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


def _deterministic_mlp():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    torch.manual_seed(137)
    mlp = SyncGptOssMLP(make_gpt_oss_config())
    for parameter in mlp.parameters():
        if parameter.numel():
            torch.nn.init.normal_(parameter, std=0.01)
    mlp.layer_id = 1
    return mlp


def test_sync_gpt_oss_mlp_instantiation():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    config = make_gpt_oss_config()
    mlp = SyncGptOssMLP(config)

    assert hasattr(mlp, "expert_executor"), "Must have expert_executor"
    assert hasattr(mlp, "archer_tracer"), "Must have archer_tracer"
    assert hasattr(mlp, "archer_engine"), "Must have archer_engine"
    assert hasattr(mlp, "expert_tensor_ids"), "Must have expert_tensor_ids"
    assert hasattr(mlp, "layer_id"), "Must have layer_id"

    assert hasattr(mlp, "experts"), "Must have experts submodule"
    assert hasattr(
        mlp.experts, "gate_up_proj"
    ), "Must have experts.gate_up_proj parameter"
    assert hasattr(
        mlp.experts, "down_proj"
    ), "Must have experts.down_proj parameter"
    assert hasattr(mlp, "router"), "Must have router"

    E, H, I = 4, 64, 64
    assert mlp.experts.gate_up_proj.shape == (
        E,
        H,
        2 * I,
    ), f"Expected ({E}, {H}, {2 * I}), got {mlp.experts.gate_up_proj.shape}"
    assert mlp.experts.down_proj.shape == (
        E,
        I,
        H,
    ), f"Expected ({E}, {I}, {H}), got {mlp.experts.down_proj.shape}"


def test_sync_gpt_oss_mlp_swiglu_activation():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    config = make_gpt_oss_config()
    mlp = SyncGptOssMLP(config)

    gate = torch.tensor([1.0, 8.0, -3.0])
    up = torch.tensor([0.5, -9.0, 2.0])
    alpha = 1.702

    gate_clamped = gate.clamp(max=7.0)
    up_clamped = up.clamp(-7.0, 7.0)
    expected = (up_clamped + 1) * (
        gate_clamped * torch.sigmoid(gate_clamped * alpha)
    )

    result = mlp._swiglu(gate, up)
    assert torch.allclose(
        result, expected, atol=1e-6
    ), f"SwiGLU mismatch: {result} vs {expected}"

    assert gate_clamped[1] == 7.0, "Gate must be clamped to 7.0"
    assert up_clamped[1] == -7.0, "Up must be clamped to -7.0"


def test_sync_gpt_oss_mlp_router_has_bias():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    config = make_gpt_oss_config()
    mlp = SyncGptOssMLP(config)

    assert hasattr(mlp.router, "bias"), "Router must have bias attribute"
    assert mlp.router.bias is not None, "Router bias must not be None"
    assert mlp.router.bias.shape == (
        config.num_local_experts,
    ), f"Router bias shape must be ({config.num_local_experts},)"


def test_sync_gpt_oss_mlp_dispatches_when_executor_is_injected():
    mlp = _deterministic_mlp()
    hidden = torch.randn(1, 2, 64)
    expected = torch.randn(2, 64)
    executor = Mock()
    executor.wait_dispatch_local.return_value = expected
    mlp.expert_executor = executor

    output, router_logits = mlp(hidden)

    assert output.shape == hidden.shape
    assert torch.equal(output.view(2, 64), expected)
    executor.dispatch_local.assert_called_once()
    args = executor.dispatch_local.call_args
    assert args.args[0] == 1
    assert args.args[1].shape == (2, 64)
    assert args.args[2].shape == (2, 4)
    assert args.args[3].shape == (2, 4)
    assert torch.equal(args.kwargs["router_logits"], router_logits)
    executor.wait_dispatch_local.assert_called_once_with()


def test_sync_gpt_oss_mlp_resident_path_matches_existing_forward():
    mlp = _deterministic_mlp()
    hidden = torch.randn(1, 2, 64)

    with torch.no_grad():
        output, router_logits = mlp(hidden)
        hidden_flat = hidden.view(-1, hidden.shape[-1])
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, mlp.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
        expected = torch.zeros_like(hidden_flat)
        for expert_idx in range(mlp.num_experts):
            token_positions, topk_positions = torch.where(
                selected_experts == expert_idx
            )
            if token_positions.numel() == 0:
                continue
            expert_output = mlp._expert_forward(
                hidden_flat[token_positions], expert_idx
            )
            weights = routing_weights[token_positions, topk_positions].to(
                hidden.dtype
            )
            expected[token_positions] += weights.unsqueeze(-1) * expert_output

    assert torch.equal(output, expected.view_as(hidden))
