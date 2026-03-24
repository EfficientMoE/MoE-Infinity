import pytest
import torch
import torch.nn.functional as F

from moe_infinity.models.deepseek import DeepseekMoEBlock, DeepseekMoEGate
from moe_infinity.models.modeling_deepseek_v2.configuration_deepseek import (
    DeepseekV2Config,
)
from moe_infinity.models.modeling_deepseek_v2.modeling_deepseek import (
    DeepseekV2MoE,
    MoEGate,
)
from tests.python.ops.conftest import BF16_ATOL, BF16_RTOL, requires_cuda


def _build_v2_config(**overrides):
    config = DeepseekV2Config(
        n_routed_experts=64,
        num_experts_per_tok=6,
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        scoring_func="softmax",
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        hidden_size=256,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_shared_experts=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _sort_idx_and_weight(topk_idx, topk_weight):
    sorted_idx, order = topk_idx.sort(dim=-1)
    sorted_weight = topk_weight.gather(1, order)
    return sorted_idx, sorted_weight


class _LocalExpertExecutor:
    """Mock expert executor that runs experts in Python (no C++ backend)."""

    def __init__(self, experts):
        self.experts = experts
        self._hidden = None
        self._mask = None
        self._weights = None

    def dispatch_local(
        self, layer_id, hidden_states, router_mask, router_weights
    ):
        del layer_id
        self._hidden = hidden_states
        self._mask = router_mask
        self._weights = router_weights

    def wait_dispatch_local(self):
        output = torch.zeros(
            self._hidden.shape, device=self._hidden.device, dtype=torch.float32
        )
        for expert_id, expert in enumerate(self.experts):
            token_mask = self._mask[:, expert_id].bool()
            if not token_mask.any():
                continue
            expert_out = expert(self._hidden[token_mask]).to(torch.float32)
            weights = self._weights[token_mask, expert_id].unsqueeze(-1)
            output[token_mask] += expert_out * weights
        return output


@requires_cuda
def test_v2_gate_equivalence(seed_everything):
    """V2-Lite config: native MoEGate and simplified DeepseekMoEGate must
    produce identical routing because topk_method='greedy' with
    routed_scaling_factor=1.0 and norm_topk_prob=False."""
    del seed_everything
    config = _build_v2_config(
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
    )

    native_gate = MoEGate(config).cuda().bfloat16().eval()
    simplified_gate = DeepseekMoEGate(config).cuda().bfloat16().eval()

    with torch.no_grad():
        simplified_gate.weight.copy_(torch.randn_like(simplified_gate.weight))
        native_gate.weight.copy_(simplified_gate.weight)

    hidden_states = torch.randn(
        3, 5, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )

    with torch.no_grad():
        simplified_logits = simplified_gate(hidden_states)
        simplified_scores = F.softmax(
            simplified_logits, dim=-1, dtype=torch.float32
        )
        simplified_weight, simplified_idx = torch.topk(
            simplified_scores,
            k=config.num_experts_per_tok,
            dim=-1,
            sorted=False,
        )

        native_idx, native_weight, aux_loss = native_gate(hidden_states)

    assert aux_loss is None
    simplified_idx_s, simplified_weight_s = _sort_idx_and_weight(
        simplified_idx, simplified_weight
    )
    native_idx_s, native_weight_s = _sort_idx_and_weight(
        native_idx, native_weight
    )

    assert torch.equal(simplified_idx_s, native_idx_s)
    torch.testing.assert_close(
        simplified_weight_s, native_weight_s, rtol=BF16_RTOL, atol=BF16_ATOL
    )


@requires_cuda
def test_v2_gate_group_limited_greedy(seed_everything):
    """V2-full config: simplified gate MUST differ from native gate because
    group_limited_greedy constrains which expert groups can be selected."""
    del seed_everything
    config = _build_v2_config(
        hidden_size=1,
        topk_method="group_limited_greedy",
        n_group=8,
        topk_group=3,
        num_experts_per_tok=6,
        norm_topk_prob=True,
        routed_scaling_factor=16.0,
    )

    native_gate = MoEGate(config).cuda().bfloat16().eval()
    simplified_gate = DeepseekMoEGate(config).cuda().bfloat16().eval()

    experts_per_group = config.n_routed_experts // config.n_group
    crafted_weight = torch.full(
        (config.n_routed_experts, 1),
        -30.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    for group_id in range(config.n_group):
        crafted_weight[group_id * experts_per_group, 0] = 8.0 - group_id

    with torch.no_grad():
        native_gate.weight.copy_(crafted_weight)
        simplified_gate.weight.copy_(crafted_weight)

    hidden_states = torch.ones(2, 4, 1, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        simplified_logits = simplified_gate(hidden_states)
        simplified_scores = F.softmax(
            simplified_logits, dim=-1, dtype=torch.float32
        )
        simplified_weight, simplified_idx = torch.topk(
            simplified_scores,
            k=config.num_experts_per_tok,
            dim=-1,
            sorted=False,
        )
        native_idx, native_weight, _ = native_gate(hidden_states)

    simplified_idx_s, _ = _sort_idx_and_weight(
        simplified_idx, simplified_weight
    )
    native_idx_s, _ = _sort_idx_and_weight(native_idx, native_weight)

    assert not torch.equal(simplified_idx_s, native_idx_s)


@requires_cuda
@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_v2_routing_mask_conversion(seed_everything, norm_topk_prob):
    """Verify __prepare_expert_route scatter conversion produces correct masks."""
    del seed_everything
    config = _build_v2_config(
        topk_method="group_limited_greedy",
        n_group=8,
        topk_group=3,
        num_experts_per_tok=6,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=16.0,
    )
    block = DeepseekMoEBlock(config).cuda().bfloat16().eval()

    hidden_states = torch.randn(
        2, 3, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )

    with torch.no_grad():
        topk_idx, topk_weight, _ = block.gate(hidden_states)
        router_mask, routing_weights_mask = (
            block._DeepseekMoEBlock__prepare_expert_route(hidden_states)
        )

    manual_mask = torch.zeros_like(router_mask)
    manual_mask.scatter_(1, topk_idx, True)
    manual_weights = torch.zeros_like(routing_weights_mask)
    manual_weights.scatter_(1, topk_idx, topk_weight.to(torch.float32))

    assert torch.equal(router_mask, manual_mask)
    torch.testing.assert_close(
        routing_weights_mask, manual_weights, rtol=BF16_RTOL, atol=BF16_ATOL
    )

    per_row_selected = router_mask.sum(dim=-1)
    assert torch.all(per_row_selected == config.num_experts_per_tok)

    per_row_weight_sum = routing_weights_mask.sum(dim=-1)
    if norm_topk_prob:
        torch.testing.assert_close(
            per_row_weight_sum,
            torch.ones_like(per_row_weight_sum),
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )
    else:
        assert torch.all(per_row_weight_sum > 0)

    assert torch.equal(routing_weights_mask.ne(0), router_mask)


@requires_cuda
def test_v2_moe_block_forward_no_offload(seed_everything):
    """Full MoE block with mock executor must match native DeepseekV2MoE."""
    del seed_everything
    config = _build_v2_config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=96,
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
    )

    native_moe = DeepseekV2MoE(config).cuda().bfloat16().eval()
    sync_moe = DeepseekMoEBlock(config).cuda().bfloat16().eval()

    sync_moe.load_state_dict(native_moe.state_dict(), strict=True)
    sync_moe.layer_id = 0
    sync_moe.expert_executor = _LocalExpertExecutor(sync_moe.experts)

    hidden_states = torch.randn(
        2, 7, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        native_out = native_moe(hidden_states)
        sync_out = sync_moe(hidden_states)

    torch.testing.assert_close(
        sync_out, native_out, rtol=BF16_RTOL, atol=BF16_ATOL
    )
