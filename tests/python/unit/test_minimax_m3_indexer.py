# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")

from transformers.models.minimax_m3_vl import (  # noqa: E402
    modeling_minimax_m3_vl as minimax,
)
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (  # noqa: E402
    MiniMaxM3VLTextConfig,
)

from moe_infinity.kernel.minimax_m3_sparse_attention import (  # noqa: E402
    indexer_attention_forward,
)

SEQ_LEN = 17
BLOCK_SIZE = 4


def _build_tiny_fixture():
    torch.manual_seed(0)
    config = MiniMaxM3VLTextConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        index_n_heads=2,
        index_head_dim=16,
        index_block_size=BLOCK_SIZE,
        index_topk_blocks=2,
        index_local_blocks=1,
        layer_types=["minimax_m3_sparse"],
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        vocab_size=128,
        intermediate_size=64,
    )
    config._attn_implementation = "eager"
    module = (
        minimax.MiniMaxM3VLAttention(config, 0).to(dtype=torch.bfloat16).eval()
    )
    hidden_states = torch.randn(
        1, SEQ_LEN, config.hidden_size, dtype=torch.bfloat16
    )
    rotary = minimax.MiniMaxM3VLRotaryEmbedding(config)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    cos, sin = rotary(hidden_states, position_ids)
    return config, module, hidden_states, (cos, sin), position_ids


def _unmasked_block_argmax(module, hidden_states, position_embeddings):
    indexer = module.indexer
    batch, q_len, _ = hidden_states.shape
    idx_q = indexer.q_proj(hidden_states).view(
        batch, q_len, -1, indexer.head_dim
    )
    idx_q = indexer.q_norm(idx_q).transpose(1, 2)
    idx_k = indexer.k_proj(hidden_states).view(
        batch, q_len, 1, indexer.head_dim
    )
    idx_k = indexer.k_norm(idx_k).transpose(1, 2)
    cos, sin = position_embeddings
    idx_q, idx_k = minimax.apply_rotary_pos_emb(
        idx_q, idx_k, cos[..., : indexer.head_dim], sin[..., : indexer.head_dim]
    )
    scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))
    k_len = idx_k.shape[2]
    num_key_blocks = -(-k_len // BLOCK_SIZE)
    pad = num_key_blocks * BLOCK_SIZE - k_len
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    scores = scores.view(
        batch, indexer.num_heads, q_len, num_key_blocks, BLOCK_SIZE
    )
    block_scores = scores.amax(dim=-1).amax(dim=1)
    return block_scores.argmax(dim=-1)[0]


def test_tiny_kernel_matches_hf_eager():
    config, module, hidden_states, position_embeddings, position_ids = (
        _build_tiny_fixture()
    )

    with torch.no_grad():
        eager_out, _ = module(
            hidden_states, position_embeddings, None, position_ids=position_ids
        )
        kernel_out, _ = indexer_attention_forward(
            module,
            hidden_states,
            position_embeddings,
            None,
            position_ids=position_ids,
        )
        block_indices = module.indexer(
            hidden_states, position_embeddings, None, position_ids
        )

    assert kernel_out.shape == eager_out.shape
    assert kernel_out.shape == (1, SEQ_LEN, config.hidden_size)
    assert torch.allclose(kernel_out, eager_out, rtol=1e-2, atol=1e-2)

    assert bool((block_indices == -1).any())

    q_block = position_ids[0] // BLOCK_SIZE
    for query in range(SEQ_LEN):
        selected = block_indices[0, query]
        selected = selected[selected >= 0]
        assert bool((selected <= q_block[query]).all())

    unmasked_best = _unmasked_block_argmax(
        module, hidden_states, position_embeddings
    )
    future_queries = [
        query
        for query in range(SEQ_LEN)
        if int(unmasked_best[query]) > int(q_block[query])
    ]
    assert future_queries
    for query in future_queries:
        selected = block_indices[0, query].tolist()
        assert int(unmasked_best[query]) not in selected


def test_kernel_flag_defaults_off_and_is_noop(monkeypatch):
    pytest.importorskip("moe_store.wrappers")
    from moe_infinity.boundary import transformers_model_overrides as overrides

    config, module, hidden_states, position_embeddings, position_ids = (
        _build_tiny_fixture()
    )
    assert not getattr(config, "enable_minimax_m3_indexer_kernel", False)

    with torch.no_grad():
        reference_out, _ = module(
            hidden_states, position_embeddings, None, position_ids=position_ids
        )

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("kernel entry point called while flag is off")

    monkeypatch.setattr(
        "moe_infinity.kernel.minimax_m3_sparse_attention."
        "minimax_m3_sparse_attention",
        _raise_if_called,
    )

    overrides.install_transformers_model_overrides()
    try:
        with torch.no_grad():
            noop_out, _ = module(
                hidden_states,
                position_embeddings,
                None,
                position_ids=position_ids,
            )
    finally:
        overrides.restore_context_transformers_model_overrides()
        overrides.restore_runtime_transformers_model_overrides()

    assert torch.equal(noop_out, reference_out)
