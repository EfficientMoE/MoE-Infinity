"""
Pytest fixtures for GPT-OSS unit tests.

These fixtures provide synthetic (no-download) test data for GPT-OSS TDD.
All tensor shapes match the real gpt-oss-20b model scaled down:
  - num_experts: 4 (real: 32)
  - hidden_size: 64 (real: 2880)
  - intermediate_size: 64 (real: 2880)
  - num_hidden_layers: 2 (real: 24)
"""

from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def gpt_oss_config():
    """Minimal mock GptOssConfig for unit testing without downloading."""
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
    cfg.max_position_embeddings = 512
    cfg.torch_dtype = torch.bfloat16
    return cfg


@pytest.fixture
def gpt_oss_packed_weights():
    """
    Synthetic packed expert weight tensors matching GPT-OSS format.

    Real shapes (scaled down for testing):
      gate_up_proj: (num_experts, hidden_size, 2 * intermediate_size)
      gate_up_proj_bias: (num_experts, 2 * intermediate_size)
      down_proj: (num_experts, intermediate_size, hidden_size)
      down_proj_bias: (num_experts, hidden_size)
    """
    E, H, I = 4, 64, 64  # num_experts, hidden_size, intermediate_size
    return {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(E, H, 2 * I),
        "model.layers.0.mlp.experts.gate_up_proj_bias": torch.randn(E, 2 * I),
        "model.layers.0.mlp.experts.down_proj": torch.randn(E, I, H),
        "model.layers.0.mlp.experts.down_proj_bias": torch.randn(E, H),
        "model.layers.0.mlp.router.weight": torch.randn(E, H),
        "model.layers.0.mlp.router.bias": torch.randn(E),
        "model.layers.1.mlp.experts.gate_up_proj": torch.randn(E, H, 2 * I),
        "model.layers.1.mlp.experts.gate_up_proj_bias": torch.randn(E, 2 * I),
        "model.layers.1.mlp.experts.down_proj": torch.randn(E, I, H),
        "model.layers.1.mlp.experts.down_proj_bias": torch.randn(E, H),
        "model.layers.1.mlp.router.weight": torch.randn(E, H),
        "model.layers.1.mlp.router.bias": torch.randn(E),
    }


@pytest.fixture
def gpt_oss_router_output():
    """
    Synthetic GptOssTopKRouter output: (router_scores, router_indices).

    Matches output of GptOssTopKRouter.forward():
      router_scores: (batch * seq_len, num_experts) — sparse softmax scores
      router_indices: (batch * seq_len, top_k) — selected expert indices
    """
    batch_seq, num_experts, top_k = 8, 4, 2
    # Sparse scores: zeros except for selected experts
    router_scores = torch.zeros(batch_seq, num_experts)
    router_indices = torch.randint(0, num_experts, (batch_seq, top_k))
    # Fill selected positions with random weights summing to 1
    for i in range(batch_seq):
        weights = torch.rand(top_k)
        weights = weights / weights.sum()
        router_scores[i, router_indices[i]] = weights
    return router_scores, router_indices
