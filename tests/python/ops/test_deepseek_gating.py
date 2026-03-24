import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]

from moe_infinity.models.modeling_deepseek_v3.configuration_deepseek import (
    DeepseekV3Config,
)
from moe_infinity.models.modeling_deepseek_v3.modeling_deepseek import (
    MoEGate,
)
from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    seed_everything,
)


def reference_moe_gate(
    hidden_states,
    weight,
    bias,
    n_group,
    topk_group,
    top_k,
    norm_topk_prob,
    scaling_factor,
):
    bsz, seq_len, h = hidden_states.shape
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(hidden_states.float(), weight.float())
    scores = logits.sigmoid()
    scores_for_choice = scores + bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(scores.shape[0], n_group, -1)
        .topk(2, dim=-1)[0]
        .sum(-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand_as(scores_for_choice.view(scores.shape[0], n_group, -1))
        .reshape(scores.shape[0], -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)

    if top_k > 1 and norm_topk_prob:
        topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * scaling_factor
    return topk_idx, topk_weight


def _build_config(
    *, topk_group=3, routed_scaling_factor=2.5, norm_topk_prob=True
):
    return DeepseekV3Config(
        n_routed_experts=64,
        num_experts_per_tok=4,
        n_group=8,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        scoring_func="sigmoid",
        routed_scaling_factor=routed_scaling_factor,
        topk_method="noaux_tc",
        hidden_size=256,
    )


def _sort_idx_and_weight(topk_idx, topk_weight):
    sorted_idx, order = topk_idx.sort(dim=-1)
    sorted_weight = topk_weight.gather(1, order)
    return sorted_idx, sorted_weight


@requires_cuda
def test_deepseek_gate_grouped_topk_matches_reference(seed_everything):
    config = _build_config(topk_group=3, routed_scaling_factor=2.5)
    gate = MoEGate(config).cuda().bfloat16().eval()

    with torch.no_grad():
        gate.weight.copy_(torch.randn_like(gate.weight))
        gate.e_score_correction_bias.zero_()

    hidden_states = torch.randn(
        3, 5, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )

    with torch.no_grad():
        custom_idx, custom_weight = gate(hidden_states)
        ref_idx, ref_weight = reference_moe_gate(
            hidden_states,
            gate.weight.detach(),
            gate.e_score_correction_bias.detach(),
            config.n_group,
            config.topk_group,
            config.num_experts_per_tok,
            config.norm_topk_prob,
            config.routed_scaling_factor,
        )

    custom_idx_sorted, custom_weight_sorted = _sort_idx_and_weight(
        custom_idx, custom_weight
    )
    ref_idx_sorted, ref_weight_sorted = _sort_idx_and_weight(
        ref_idx, ref_weight
    )

    assert torch.equal(custom_idx_sorted, ref_idx_sorted)
    torch.testing.assert_close(
        custom_weight_sorted,
        ref_weight_sorted,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )


@requires_cuda
def test_deepseek_gate_bias_correction_matches_reference(seed_everything):
    config = _build_config(topk_group=1, routed_scaling_factor=2.5)
    gate = MoEGate(config).cuda().bfloat16().eval()

    experts_per_group = config.n_routed_experts // config.n_group
    bias = torch.full(
        (config.n_routed_experts,), -5.0, device="cuda", dtype=torch.bfloat16
    )
    bias[:experts_per_group] = 5.0

    with torch.no_grad():
        gate.weight.copy_(torch.randn_like(gate.weight))
        gate.e_score_correction_bias.copy_(bias)

    hidden_states = torch.randn(
        2, 7, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )

    with torch.no_grad():
        custom_idx, custom_weight = gate(hidden_states)
        ref_idx, ref_weight = reference_moe_gate(
            hidden_states,
            gate.weight.detach(),
            gate.e_score_correction_bias.detach(),
            config.n_group,
            config.topk_group,
            config.num_experts_per_tok,
            config.norm_topk_prob,
            config.routed_scaling_factor,
        )

    custom_idx_sorted, custom_weight_sorted = _sort_idx_and_weight(
        custom_idx, custom_weight
    )
    ref_idx_sorted, ref_weight_sorted = _sort_idx_and_weight(
        ref_idx, ref_weight
    )

    assert torch.equal(custom_idx_sorted, ref_idx_sorted)
    torch.testing.assert_close(
        custom_weight_sorted,
        ref_weight_sorted,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    assert torch.all(custom_idx_sorted < experts_per_group)


@requires_cuda
def test_deepseek_gate_scaling_and_epsilon_matches_reference(seed_everything):
    config = _build_config(topk_group=3, routed_scaling_factor=2.5)
    gate = MoEGate(config).cuda().bfloat16().eval()

    with torch.no_grad():
        gate.weight.copy_(torch.randn_like(gate.weight))
        gate.e_score_correction_bias.copy_(
            torch.randn_like(gate.e_score_correction_bias)
        )

    hidden_states = torch.randn(
        4, 3, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )

    with torch.no_grad():
        custom_idx, custom_weight = gate(hidden_states)
        ref_idx, ref_weight = reference_moe_gate(
            hidden_states,
            gate.weight.detach(),
            gate.e_score_correction_bias.detach(),
            config.n_group,
            config.topk_group,
            config.num_experts_per_tok,
            config.norm_topk_prob,
            config.routed_scaling_factor,
        )

    custom_idx_sorted, custom_weight_sorted = _sort_idx_and_weight(
        custom_idx, custom_weight
    )
    ref_idx_sorted, ref_weight_sorted = _sort_idx_and_weight(
        ref_idx, ref_weight
    )

    assert torch.equal(custom_idx_sorted, ref_idx_sorted)
    torch.testing.assert_close(
        custom_weight_sorted,
        ref_weight_sorted,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    expected_sum = torch.full_like(
        custom_weight.sum(-1), config.routed_scaling_factor
    )
    torch.testing.assert_close(
        custom_weight.sum(-1),
        expected_sum,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )

    with torch.no_grad():
        gate.weight.copy_(-torch.rand_like(gate.weight).abs() * 2_000.0)
        gate.e_score_correction_bias.zero_()

    hidden_states_eps = (
        torch.rand(
            3, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16
        ).abs()
        * 2_000.0
    )

    with torch.no_grad():
        custom_idx_eps, custom_weight_eps = gate(hidden_states_eps)
        ref_idx_eps, ref_weight_eps = reference_moe_gate(
            hidden_states_eps,
            gate.weight.detach(),
            gate.e_score_correction_bias.detach(),
            config.n_group,
            config.topk_group,
            config.num_experts_per_tok,
            config.norm_topk_prob,
            config.routed_scaling_factor,
        )

    custom_idx_sorted_eps, custom_weight_sorted_eps = _sort_idx_and_weight(
        custom_idx_eps, custom_weight_eps
    )
    ref_idx_sorted_eps, ref_weight_sorted_eps = _sort_idx_and_weight(
        ref_idx_eps, ref_weight_eps
    )

    assert torch.equal(custom_idx_sorted_eps, ref_idx_sorted_eps)
    torch.testing.assert_close(
        custom_weight_sorted_eps,
        ref_weight_sorted_eps,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    assert torch.isfinite(custom_weight_eps).all()
