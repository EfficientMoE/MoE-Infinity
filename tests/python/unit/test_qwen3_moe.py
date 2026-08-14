import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from moe_infinity.models import Qwen3MoEBlock


class _RecordingExecutor:
    def __init__(self):
        self.hidden_states: torch.Tensor | None = None
        self.router_logits: torch.Tensor | None = None

    def dispatch_local(
        self,
        layer_id: int | None,
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        router_weights: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        prefetcher: object = None,
    ) -> None:
        del layer_id, router_mask, router_weights, prefetcher
        self.hidden_states = hidden_states
        self.router_logits = router_logits

    def wait_dispatch_local(self) -> torch.Tensor:
        assert self.hidden_states is not None
        return self.hidden_states.clone()


def test_qwen3_block_returns_tensor_and_routes_logits_via_executor():
    config = Qwen3MoeConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    block = Qwen3MoEBlock(config)
    executor = _RecordingExecutor()
    block.expert_executor = executor
    block.layer_id = 3
    hidden_states = torch.randn(2, 5, config.hidden_size)

    output = block(hidden_states)

    assert isinstance(output, torch.Tensor)
    assert output.shape == hidden_states.shape
    assert executor.router_logits is not None
    assert executor.router_logits.shape == (
        hidden_states.shape[0] * hidden_states.shape[1],
        config.num_experts,
    )
