"""Decode steps replayed from CUDA graphs on a resident-expert model.

A tiny random-weight Qwen3-MoE (every expert resident, HF ``batched_mm``
experts, ``Qwen3PagedAttention`` shims) served by ``ContinuousBatchingEngine``
over a storage-bound ``PagedAttentionBackend``: greedy tokens must be the same
with graphs on, graphs off, and HF ``generate`` on a ``DynamicCache``.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from tests.python.ops.conftest import requires_cuda

BLOCK_SIZE = 16
NUM_KV_BLOCKS = 128
NEW_TOKENS = 12
PROMPTS = (
    [3, 9, 27, 4, 15],
    [8, 2, 44, 19, 7, 31],
    [11, 5, 60],
)


def _paged_kernel_available() -> bool:
    from moe_infinity.kernel.paged_attention_ops import HAS_PAGED_ATTN

    return HAS_PAGED_ATTN


def _config():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )

    config = Qwen3MoeConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=2,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=512,
        max_position_embeddings=1024,
    )
    config._experts_implementation = "batched_mm"
    return config


def _paged_model(config, device: torch.device) -> torch.nn.Module:
    import transformers.models.qwen3_moe.modeling_qwen3_moe as hf_qwen3
    from moe_store.wrappers.qwen3_paged_attention import Qwen3PagedAttention

    original = hf_qwen3.Qwen3MoeAttention
    hf_qwen3.Qwen3MoeAttention = Qwen3PagedAttention
    try:
        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = hf_qwen3.Qwen3MoeForCausalLM(config)
    finally:
        hf_qwen3.Qwen3MoeAttention = original
    return model.to(device, torch.bfloat16).eval()


def _hf_tokens(config, state_dict, device, prompt: list[int]) -> list[int]:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeForCausalLM,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Qwen3MoeForCausalLM(config)
    model.load_state_dict(state_dict)
    model = model.to(device, torch.bfloat16).eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.generate(
            torch.tensor([prompt], device=device),
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            pad_token_id=0,
        )
    return out[0, len(prompt) :].tolist()


def _serve(model, config, device, prompts, *, graphs: bool):
    from moe_infinity.runtime.attention_backend import PagedAttentionBackend
    from moe_infinity.runtime.attention_types import DecodeGraphCapability
    from moe_infinity.runtime.paged_kv_storage import (
        PagedKVStorage,
        PagedKVStorageSpec,
    )
    from moe_infinity.serving.engine import ContinuousBatchingEngine
    from moe_infinity.serving.factory import generate_tokens

    storage = PagedKVStorage(
        PagedKVStorageSpec(
            num_layers=config.num_hidden_layers,
            num_blocks=NUM_KV_BLOCKS,
            block_size=BLOCK_SIZE,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
    )
    backend = PagedAttentionBackend(storage=storage, use_flashinfer=False)

    class ResidentEngine:
        expert_layer_modules: list[object] = []

        def get_attention_backend(self) -> PagedAttentionBackend:
            return backend

        def decode_graph_capability(self) -> DecodeGraphCapability:
            return DecodeGraphCapability(True, "eligible")

    engine = ContinuousBatchingEngine(
        model=model,
        engine=ResidentEngine(),
        config={
            "num_kv_blocks": NUM_KV_BLOCKS,
            "max_batch_size": 4,
            "block_size": BLOCK_SIZE,
            "num_layers": config.num_hidden_layers,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "dtype": "bfloat16",
            "eos_token_id": None,
            "enable_decode_cuda_graphs": graphs,
            "decode_cuda_graph_batch_sizes": (1, 2, 4),
            "decode_cuda_graph_context_sizes": (32, 64),
        },
    )
    try:
        tokens = generate_tokens(
            engine, [list(p) for p in prompts], max_new_tokens=NEW_TOKENS
        )
        return tokens, engine.cuda_graph_runner.stats()
    finally:
        engine.cuda_graph_runner.close()


@requires_cuda
def test_replayed_decode_step_has_no_host_sync() -> None:
    if not _paged_kernel_available():
        pytest.skip("native paged attention extension not built")
    from benchmarks.serving.decode_cuda_graph_fixture import (
        build_resident_fixture,
    )

    fixture = build_resident_fixture(
        batch_sizes=(1, 4), context_sizes=(32,), warmup_iters=1
    )
    try:
        batch = fixture.make_batch(3, 32)
        expected = fixture.graph_runner.try_execute(batch)  # captures
        assert expected is not None
        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            replayed = fixture.graph_runner.try_execute(batch)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        assert replayed is not None
        torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
        assert fixture.graph_runner.stats()["replays"] == 2
    finally:
        fixture.close()


@requires_cuda
@pytest.mark.parametrize("num_prompts", [1, 3])
def test_resident_decode_replays_graph_with_identical_tokens(
    num_prompts: int,
) -> None:
    if not _paged_kernel_available():
        pytest.skip("native paged attention extension not built")
    device = torch.device("cuda", 0)
    config = _config()
    prompts = PROMPTS[:num_prompts]

    # Each engine binds the shims to its own storage, so each gets a model.
    eager_model = _paged_model(config, device)
    eager_tokens, eager_stats = _serve(
        eager_model, config, device, prompts, graphs=False
    )
    graph_tokens, graph_stats = _serve(
        _paged_model(config, device), config, device, prompts, graphs=True
    )
    reference = [
        _hf_tokens(config, eager_model.state_dict(), device, prompt)
        for prompt in prompts
    ]

    assert eager_tokens == reference
    assert graph_tokens == eager_tokens
    assert eager_stats["replays"] == 0
    # The first token of every request comes from prefill; every later
    # step is a decode step and must have been replayed.
    assert graph_stats["replays"] == NEW_TOKENS - 1
    assert graph_stats["captures"] >= 1
    assert graph_stats["capture_failures"] == 0
    assert graph_stats["fallback_reasons"] == {}


@requires_cuda
def test_grouped_resident_experts_replay_with_identical_tokens() -> None:
    """Experts behind the grouped decode kernel's device pointer table."""
    if not _paged_kernel_available():
        pytest.skip("native paged attention extension not built")
    from moe_infinity.serving.resident_experts import (
        grouped_decode_op,
        install_grouped_resident_experts,
    )

    if grouped_decode_op() is None:
        pytest.skip("moe_infinity._store built without grouped_moe_decode")
    device = torch.device("cuda", 0)
    config = _config()
    runs = []
    for graphs in (False, True):
        model = _paged_model(config, device)
        assert len(install_grouped_resident_experts(model)) == (
            config.num_hidden_layers
        )
        runs.append(_serve(model, config, device, PROMPTS, graphs=graphs))
    (eager_tokens, _), (graph_tokens, graph_stats) = runs

    assert graph_tokens == eager_tokens
    assert graph_stats["replays"] == NEW_TOKENS - 1
    assert graph_stats["fallback_reasons"] == {}
