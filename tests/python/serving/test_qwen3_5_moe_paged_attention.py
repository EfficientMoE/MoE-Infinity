from __future__ import annotations

import importlib
import logging
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeTextConfig,
)
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_qwen3_5
from transformers.models.qwen3_moe.configuration_qwen3_moe import (
    Qwen3MoeConfig,
)

from moe_infinity.boundary.hybrid_attention_layout import (
    linear_attention_layer_indices,
    paged_kv_layer_count,
    paged_kv_layer_indices,
)
from moe_infinity.boundary.qwen3_5_moe_paged_attention import (
    Qwen3_5MoePagedAttention,
)
from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import KVCacheSpec
from moe_infinity.serving.batch import BatchMetadata
from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.linear_attention_state import (
    LinearAttentionStateCache,
)
from moe_infinity.serving.model_runner import (
    HYBRID_CHUNKED_PREFILL_DISABLED_REASON,
    HYBRID_PREFIX_REUSE_DISABLED_REASON,
    ModelRunner,
    paged_layer_index,
)
from moe_infinity.serving.sequence import SamplingParams, SequenceStatus

BLOCK_SIZE = 4
LAYER_TYPES = [
    "linear_attention",
    "full_attention",
    "linear_attention",
    "full_attention",
]


def _text_config() -> Qwen3_5MoeTextConfig:
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        layer_types=list(LAYER_TYPES),
        eos_token_id=None,
        pad_token_id=0,
        bos_token_id=1,
    )
    config._attn_implementation = "eager"
    return config


def _vl_config() -> Qwen3_5MoeConfig:
    config = Qwen3_5MoeConfig(
        text_config=_text_config().to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
    )
    config._attn_implementation = "eager"
    config.get_text_config()._attn_implementation = "eager"
    return config


@contextmanager
def _paged_attention_override() -> Iterator[None]:
    original = hf_qwen3_5.Qwen3_5MoeAttention
    hf_qwen3_5.Qwen3_5MoeAttention = Qwen3_5MoePagedAttention
    try:
        yield
    finally:
        hf_qwen3_5.Qwen3_5MoeAttention = original


def _build_model(kind: str, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    with _paged_attention_override(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "causal":
            model = Qwen3_5MoeForCausalLM(_text_config())
        else:
            model = Qwen3_5MoeForConditionalGeneration(_vl_config())
    return model.eval()


class _ServingEngine:
    def __init__(self, backend: PagedAttentionBackend) -> None:
        self.backend = backend
        self.kv_cache = SimpleNamespace(block_size=BLOCK_SIZE)
        self.expert_layer_modules: list[object] = []

    def get_attention_backend(self) -> PagedAttentionBackend:
        return self.backend


def _cpu_backend(num_blocks: int = 64, num_layers: int = 2):
    return PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            block_size=BLOCK_SIZE,
        ),
        num_gpu_blocks=num_blocks,
        device=torch.device("cpu"),
        num_layers=num_layers,
    )


def _runner(model: torch.nn.Module) -> ModelRunner:
    return ModelRunner(
        model, _ServingEngine(_cpu_backend()), device=torch.device("cpu")
    )


class _Sequences:
    """Minimal scheduler stand-in: per-sequence tokens and block tables."""

    def __init__(self) -> None:
        self.tokens: dict[int, list[int]] = {}
        self.computed: dict[int, int] = {}
        self.blocks: dict[int, list[int]] = {}
        self._next_block = 0

    def add(self, seq_id: int, prompt: list[int]) -> None:
        self.tokens[seq_id] = list(prompt)
        self.computed[seq_id] = 0
        self.blocks[seq_id] = list(
            range(self._next_block, self._next_block + 8)
        )
        self._next_block += 8

    def batch(self, seq_ids: list[int]) -> BatchMetadata:
        input_ids: list[int] = []
        seq_lengths: list[int] = []
        context_lengths: list[int] = []
        for seq_id in seq_ids:
            context = self.computed[seq_id]
            query = self.tokens[seq_id][context:]
            input_ids.extend(query)
            seq_lengths.append(len(query))
            context_lengths.append(context)
        return BatchMetadata(
            seq_ids=list(seq_ids),
            input_token_ids=input_ids,
            seq_lengths=seq_lengths,
            context_lengths=context_lengths,
            is_prefill=[context == 0 for context in context_lengths],
            block_tables=[self.blocks[seq_id] for seq_id in seq_ids],
            sampling_params=[SamplingParams() for _ in seq_ids],
        )

    def advance(self, batch: BatchMetadata, logits: torch.Tensor) -> None:
        for row, seq_id in enumerate(batch.seq_ids):
            last = batch.query_offsets[row + 1] - 1
            self.computed[seq_id] += batch.query_lengths[row]
            self.tokens[seq_id].append(int(logits[last].argmax()))


def _hf_greedy(
    model: torch.nn.Module, prompt: list[int], steps: int
) -> tuple[list[int], torch.Tensor]:
    input_ids = torch.tensor([prompt])
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.generate(
            input_ids,
            max_new_tokens=steps,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=0,
        )
    tokens = out.sequences[0, len(prompt) :].tolist()
    return tokens, torch.stack([step[0] for step in out.logits])


PROMPTS = {
    11: [3, 9, 27, 4, 15],
    12: [8, 2, 44, 19, 7, 31, 5, 60, 12],
}


def _assert_matches_hf(
    model: torch.nn.Module,
    seqs: _Sequences,
    step_logits: dict[int, list[torch.Tensor]],
    steps: int,
) -> None:
    for seq_id, prompt in PROMPTS.items():
        tokens, logits = _hf_greedy(model, prompt, steps)
        assert seqs.tokens[seq_id][len(prompt) :] == tokens
        torch.testing.assert_close(
            torch.stack(step_logits[seq_id]), logits, rtol=1e-4, atol=1e-4
        )


def _record(
    step_logits: dict[int, list[torch.Tensor]],
    batch: BatchMetadata,
    logits: torch.Tensor,
) -> None:
    for row, seq_id in enumerate(batch.seq_ids):
        step_logits.setdefault(seq_id, []).append(
            logits[batch.query_offsets[row + 1] - 1]
        )


def test_layout_resolves_text_and_vl_configs() -> None:
    for config in (_text_config(), _vl_config()):
        assert paged_kv_layer_indices(config) == (1, 3)
        assert linear_attention_layer_indices(config) == (0, 2)
        assert paged_kv_layer_count(config, 4) == 2


def test_layout_leaves_non_hybrid_models_unchanged() -> None:
    qwen3 = Qwen3MoeConfig(num_hidden_layers=3)
    assert paged_kv_layer_indices(qwen3) is None
    assert linear_attention_layer_indices(qwen3) == ()
    assert paged_kv_layer_count(qwen3, 3) == 3
    # Mixed sliding/full attention (e.g. GPT-OSS) is not a hybrid layout.
    sliding = SimpleNamespace(
        model_type="gpt_oss",
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
    )
    assert paged_kv_layer_count(sliding, 2) == 2
    assert paged_layer_index(SimpleNamespace(layer_idx=5)) == 5


def test_overrides_install_and_restore_swap_only_full_attention() -> None:
    overrides = importlib.import_module(
        "moe_infinity.boundary.transformers_model_overrides"
    )
    original = hf_qwen3_5.Qwen3_5MoeAttention
    original_delta_net = hf_qwen3_5.Qwen3_5MoeGatedDeltaNet
    original_vision = hf_qwen3_5.Qwen3_5MoeVisionAttention
    installed = False
    try:
        overrides.install_transformers_model_overrides()
        installed = True
        assert hf_qwen3_5.Qwen3_5MoeAttention is Qwen3_5MoePagedAttention
        assert hf_qwen3_5.Qwen3_5MoeGatedDeltaNet is original_delta_net
        assert hf_qwen3_5.Qwen3_5MoeVisionAttention is original_vision
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Qwen3_5MoeForConditionalGeneration(_vl_config())

        overrides.restore_context_transformers_model_overrides()
        assert hf_qwen3_5.Qwen3_5MoeAttention is original
    finally:
        if installed:
            overrides.restore_context_transformers_model_overrides()
            overrides.restore_runtime_transformers_model_overrides()
        hf_qwen3_5.Qwen3_5MoeAttention = original
    assert hf_qwen3_5.Qwen3_5MoeAttention is original

    layers = model.model.language_model.layers
    for layer_idx, layer in enumerate(layers):
        if LAYER_TYPES[layer_idx] == "full_attention":
            assert type(layer.self_attn) is Qwen3_5MoePagedAttention
        else:
            assert type(layer.linear_attn) is original_delta_net
            assert not hasattr(layer, "self_attn")
    vision_types = {type(block.attn) for block in model.model.visual.blocks}
    assert vision_types == {original_vision}


@pytest.mark.parametrize("kind", ["causal", "vl"])
def test_runner_matches_shim_and_dense_paged_layers(kind: str) -> None:
    runner = _runner(_build_model(kind))
    assert runner._get_paged_attention_classes() == [Qwen3_5MoePagedAttention]
    registrations = runner._collect_qwen_layer_registrations()
    assert [layer_idx for layer_idx, _ in registrations] == [0, 1]
    assert runner._resolve_expected_layers() == 2
    modules = runner._get_qwen3_paged_attention_modules()
    assert [module.layer_idx for module in modules] == [1, 3]
    assert [module.paged_layer_idx for module in modules] == [0, 1]


def test_eager_fallback_is_identical_to_hf_attention() -> None:
    config = _text_config()
    torch.manual_seed(0)
    reference = hf_qwen3_5.Qwen3_5MoeAttention(config, layer_idx=3).eval()
    shim = Qwen3_5MoePagedAttention(config, layer_idx=3).eval()
    shim.load_state_dict(reference.state_dict())
    rotary = hf_qwen3_5.Qwen3_5MoeTextRotaryEmbedding(config)

    hidden = torch.randn(2, 5, config.hidden_size)
    position_ids = torch.arange(5).expand(2, -1)
    position_embeddings = rotary(hidden, position_ids)
    mask = torch.full((5, 5), float("-inf")).triu(1)[None, None]

    def run(module, cache):
        with torch.no_grad():
            return module(
                hidden_states=hidden,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                past_key_values=cache,
            )

    ref_cache = DynamicCache(config=config)
    shim_cache = DynamicCache(config=config)
    ref_out, ref_weights = run(reference, ref_cache)
    shim_out, shim_weights = run(shim, shim_cache)
    assert torch.equal(ref_out, shim_out)
    assert torch.equal(ref_weights, shim_weights)
    assert torch.equal(ref_cache.layers[3].keys, shim_cache.layers[3].keys)
    assert torch.equal(ref_cache.layers[3].values, shim_cache.layers[3].values)


@pytest.mark.parametrize("kind", ["causal", "vl"])
def test_paged_decode_matches_hf_generate(kind: str) -> None:
    model = _build_model(kind)
    runner = _runner(model)
    seqs = _Sequences()
    for seq_id, prompt in PROMPTS.items():
        seqs.add(seq_id, prompt)

    steps = 6
    step_logits: dict[int, list[torch.Tensor]] = {}
    for _ in range(steps):
        batch = seqs.batch(list(PROMPTS))
        logits = runner.execute(batch)
        assert logits.shape == (batch.total_tokens, 64)
        _record(step_logits, batch, logits)
        seqs.advance(batch, logits)

    _assert_matches_hf(model, seqs, step_logits, steps)


def test_mixed_prefill_and_decode_rows_and_reexecution() -> None:
    model = _build_model("causal", seed=1)
    runner = _runner(model)
    seqs = _Sequences()
    first, second = PROMPTS
    seqs.add(first, PROMPTS[first])
    step_logits: dict[int, list[torch.Tensor]] = {}

    batch = seqs.batch([first])
    logits = runner.execute(batch)
    _record(step_logits, batch, logits)
    seqs.advance(batch, logits)

    # The second request prefills in the same batch as the first decodes.
    seqs.add(second, PROMPTS[second])
    steps = 5
    for step in range(steps):
        rows = [first, second] if step < steps - 1 else [second]
        batch = seqs.batch(rows)
        logits = runner.execute(batch)
        # The engine re-runs a step's batch; recurrent state must not advance.
        again = runner.execute(batch)
        torch.testing.assert_close(again, logits, rtol=0, atol=0)
        _record(step_logits, batch, logits)
        seqs.advance(batch, logits)

    tokens, _ = _hf_greedy(model, PROMPTS[first], steps)
    assert seqs.tokens[first][len(PROMPTS[first]) :] == tokens
    tokens, _ = _hf_greedy(model, PROMPTS[second], steps)
    assert seqs.tokens[second][len(PROMPTS[second]) :] == tokens


def test_runner_rejects_missing_linear_state_and_rich_execution() -> None:
    runner = _runner(_build_model("causal"))
    batch = BatchMetadata(
        seq_ids=[5],
        input_token_ids=[7],
        seq_lengths=[1],
        context_lengths=[4],
        is_prefill=[False],
        block_tables=[[0, 1]],
        sampling_params=[SamplingParams()],
    )
    with pytest.raises(RuntimeError, match="no linear-attention state"):
        runner.execute(batch)
    with pytest.raises(RuntimeError, match="hybrid linear-attention"):
        runner.execute_rich(batch)


def test_linear_attention_state_cache_lifecycle() -> None:
    config = _text_config()
    state = LinearAttentionStateCache(config, (0, 2))
    cache = DynamicCache(config=config)
    for layer_idx in (0, 2):
        cache.update_conv_state(torch.randn(2, 6, 4), layer_idx)
        cache.update_recurrent_state(torch.randn(2, 4, 8, 8), layer_idx)
    state.commit([1, 2], [0, 0], [5, 3], cache)
    assert state.has_state(1, 5) and state.has_state(2, 3)

    restored = state.build_cache([2, 1], [3, 5])
    assert restored.has_previous_state(0)
    torch.testing.assert_close(
        restored.layers[2].recurrent_states[0],
        cache.layers[2].recurrent_states[1],
    )
    with pytest.raises(ValueError, match="mix new and continuing"):
        state.build_cache([1, 3], [5, 0])

    state.commit([1], [5], [1], restored_single(state, 1, 5))
    assert state.has_state(1, 5) and state.has_state(1, 6)
    state.retain([1])
    assert 2 not in state and 1 in state
    state.release([1])
    assert len(state) == 0


def restored_single(
    state: LinearAttentionStateCache, seq_id: int, num_tokens: int
) -> DynamicCache:
    return state.build_cache([seq_id], [num_tokens])


class _FlashInferStoreBackend:
    def __init__(self) -> None:
        self.registrations: list[object] = []
        self.created: list[int] = []
        self.block_store = SimpleNamespace(num_blocks=8)

    def _flashinfer_enabled(self) -> bool:
        return True

    def create_layered_store(self, *, layer_count: int) -> object:
        self.created.append(layer_count)
        return self.block_store

    def register_layers(self, registrations: list[object]) -> None:
        self.registrations = list(registrations)


def test_prefix_and_chunked_prefill_report_hybrid_reason() -> None:
    model = _build_model("vl")
    backend = _FlashInferStoreBackend()
    runner = ModelRunner(
        model,
        SimpleNamespace(
            get_attention_backend=lambda: backend, expert_layer_modules=[]
        ),
        device=torch.device("cpu"),
    )
    capability = runner.get_prefix_reuse_capability()
    assert capability.supported is False
    assert capability.reason == HYBRID_PREFIX_REUSE_DISABLED_REASON
    # The layered store is still created and registered for the paged layers.
    assert backend.created == [2]
    assert [r.layer_idx for r in backend.registrations] == [0, 1]
    assert runner.supports_chunked_prefill() is False
    assert (
        runner.chunked_prefill_unavailable_reason()
        == HYBRID_CHUNKED_PREFILL_DISABLED_REASON
    )


def _engine_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": 4,
        "max_tokens_per_step": 64,
        "block_size": BLOCK_SIZE,
        "num_layers": 2,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype": "float32",
        "eos_token_id": None,
        "model_memory_bytes": 0,
        "num_kv_blocks": 32,
    }
    config.update(overrides)
    return config


def _hybrid_engine(
    monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> ContinuousBatchingEngine:
    model = _build_model("vl")
    backend = _cpu_backend(num_blocks=32)
    # Exercise the FlashInfer-only layered store creation while the
    # engine binds it, then execute on the CPU SDPA path.
    monkeypatch.setattr(backend, "_flashinfer_enabled", lambda: True)
    engine = ContinuousBatchingEngine(
        model=model,
        engine=SimpleNamespace(
            get_attention_backend=lambda: backend, expert_layer_modules=[]
        ),
        config=_engine_config(**overrides),
    )
    monkeypatch.delattr(backend, "_flashinfer_enabled")
    return engine


def test_engine_binds_layered_store_for_hybrid_layout(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _hybrid_engine(monkeypatch, enable_prefix_caching=True)
    assert "failed to bind layered paged KV store" not in caplog.text
    assert "no paged-attention shim matched" not in caplog.text
    backend = engine.model_runner.get_attention_backend()
    assert backend.block_store.num_layers == 2
    assert sorted(backend._layer_registry) == [0, 1]
    assert engine.prefix_cache is None
    assert (
        engine._prefix_cache_disabled_reason
        == HYBRID_PREFIX_REUSE_DISABLED_REASON
    )


def test_engine_reports_chunked_prefill_fallback_for_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _hybrid_engine(monkeypatch, enable_chunked_prefill=True)
    stats = engine.get_stats()
    assert stats["chunked_prefill_active"] is False
    assert (
        stats["chunked_prefill_fallback_reason"]
        == HYBRID_CHUNKED_PREFILL_DISABLED_REASON
    )


def test_engine_greedy_decode_matches_hf_and_releases_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _hybrid_engine(monkeypatch)
    model = engine.model
    steps = 5
    for seq_id, prompt in PROMPTS.items():
        engine.add_request(
            f"req-{seq_id}",
            list(prompt),
            SamplingParams(temperature=0.0, max_tokens=steps),
        )
    for _ in range(4 * steps):
        engine.step()
        if not engine.has_pending_requests():
            break
    assert not engine.has_pending_requests()

    for seq_id, prompt in PROMPTS.items():
        expected, _ = _hf_greedy(model, prompt, steps)
        assert engine.get_request_n_outputs(f"req-{seq_id}") == [expected]

    state = engine.model_runner._get_linear_attention_state()
    assert state is not None
    engine.model_runner.retain_linear_attention_state(
        seq_id
        for seq_id, sequence in engine._sequences.items()
        if sequence.status
        not in (SequenceStatus.FINISHED, SequenceStatus.CANCELLED)
    )
    assert len(state) == 0


def test_serving_configs_size_kv_by_full_attention_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    srv = importlib.import_module(
        "moe_infinity.entrypoints.openai.api_server_v2"
    )
    from moe_infinity.serving.factory import build_serving_config

    model = SimpleNamespace(config=_vl_config(), dtype=torch.float32)
    monkeypatch.setattr(
        sys,
        "argv",
        ["api_server_v2.py", "--model", "m", "--offload-dir", "/tmp/offload"],
    )
    config = srv._build_engine_config(srv.parse_args(), model)
    assert config["num_layers"] == 2
    assert config["num_kv_heads"] == 2
    assert config["head_dim"] == 8

    moe = SimpleNamespace(
        model=model, engine=object(), engine_config=SimpleNamespace()
    )
    assert build_serving_config(moe)["num_layers"] == 2
