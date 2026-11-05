"""Mixtral paged-attention shim: serving decode keeps KV history (#318)."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers import MixtralConfig, MixtralForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.mixtral import modeling_mixtral as hf_mixtral

from moe_infinity.boundary.mixtral_paged_attention import (
    MixtralPagedAttention,
)
from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import KVCacheSpec
from moe_infinity.serving.batch import BatchMetadata
from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.model_runner import ModelRunner
from moe_infinity.serving.sequence import SamplingParams

BLOCK_SIZE = 4
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8
VOCAB = 64

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA required"
        ),
    ),
]


def _config(**overrides: object) -> MixtralConfig:
    kwargs: dict[str, object] = {
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 16,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 128,
        "sliding_window": None,
        "eos_token_id": None,
        "pad_token_id": 0,
        "bos_token_id": 1,
    }
    kwargs.update(overrides)
    config = MixtralConfig(**kwargs)
    config._attn_implementation = "eager"
    return config


@contextmanager
def _paged_attention_override() -> Iterator[None]:
    original = hf_mixtral.MixtralAttention
    hf_mixtral.MixtralAttention = MixtralPagedAttention
    try:
        yield
    finally:
        hf_mixtral.MixtralAttention = original


def _build_model(seed: int = 0, device: str = "cpu") -> torch.nn.Module:
    torch.manual_seed(seed)
    with _paged_attention_override(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MixtralForCausalLM(_config())
    return model.to(device).eval()


def _backend(device: str, num_blocks: int = 64) -> PagedAttentionBackend:
    return PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            block_size=BLOCK_SIZE,
        ),
        num_gpu_blocks=num_blocks,
        device=torch.device(device),
        num_layers=NUM_LAYERS,
    )


class _ServingEngine:
    def __init__(self, backend: object) -> None:
        self.backend = backend
        self.kv_cache = SimpleNamespace(block_size=BLOCK_SIZE)
        self.expert_layer_modules: list[object] = []

    def get_attention_backend(self) -> object:
        return self.backend


def _runner(model: torch.nn.Module, device: str = "cpu") -> ModelRunner:
    return ModelRunner(
        model,
        _ServingEngine(_backend(device)),
        device=torch.device(device),
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
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt], device=device)
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


def test_eager_fallback_is_identical_to_hf_attention() -> None:
    config = _config()
    torch.manual_seed(0)
    reference = hf_mixtral.MixtralAttention(config, layer_idx=1).eval()
    shim = MixtralPagedAttention(config, layer_idx=1).eval()
    shim.load_state_dict(reference.state_dict())
    rotary = hf_mixtral.MixtralRotaryEmbedding(config)

    hidden = torch.randn(2, 5, config.hidden_size)
    position_embeddings = rotary(hidden, torch.arange(5).expand(2, -1))
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
    ref_out, _ = run(reference, ref_cache)
    shim_out, _ = run(shim, shim_cache)
    assert torch.equal(ref_out, shim_out)
    assert torch.equal(ref_cache.layers[1].keys, shim_cache.layers[1].keys)


def test_overrides_install_and_restore_mixtral_attention() -> None:
    overrides = importlib.import_module(
        "moe_infinity.boundary.transformers_model_overrides"
    )
    original = hf_mixtral.MixtralAttention
    installed = False
    try:
        overrides.install_transformers_model_overrides()
        installed = True
        assert hf_mixtral.MixtralAttention is MixtralPagedAttention
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MixtralForCausalLM(_config())
        overrides.restore_context_transformers_model_overrides()
        assert hf_mixtral.MixtralAttention is original
    finally:
        if installed:
            overrides.restore_context_transformers_model_overrides()
            overrides.restore_runtime_transformers_model_overrides()
        hf_mixtral.MixtralAttention = original
    assert {type(layer.self_attn) for layer in model.model.layers} == {
        MixtralPagedAttention
    }


def test_runner_detects_shim_and_registers_layers() -> None:
    runner = _runner(_build_model())
    assert runner._get_paged_attention_classes() == [MixtralPagedAttention]
    assert [
        layer_idx for layer_idx, _ in runner._collect_qwen_layer_registrations()
    ] == list(range(NUM_LAYERS))
    assert runner.paged_kv_history_unsupported_reason() is None
    runner.require_paged_kv_history()


@pytest.mark.parametrize("device", DEVICES)
def test_paged_decode_matches_hf_generate(device: str) -> None:
    model = _build_model(device=device)
    runner = _runner(model, device)
    seqs = _Sequences()
    for seq_id, prompt in PROMPTS.items():
        seqs.add(seq_id, prompt)

    steps = 6
    step_logits: dict[int, list[torch.Tensor]] = {}
    for _ in range(steps):
        batch = seqs.batch(list(PROMPTS))
        logits = runner.execute(batch)
        assert logits.shape == (batch.total_tokens, VOCAB)
        for row, seq_id in enumerate(batch.seq_ids):
            step_logits.setdefault(seq_id, []).append(
                logits[batch.query_offsets[row + 1] - 1]
            )
        seqs.advance(batch, logits)

    for seq_id, prompt in PROMPTS.items():
        tokens, logits = _hf_greedy(model, prompt, steps)
        assert seqs.tokens[seq_id][len(prompt) :] == tokens
        torch.testing.assert_close(
            torch.stack(step_logits[seq_id]), logits, rtol=1e-4, atol=1e-4
        )


def _engine_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": 4,
        "max_tokens_per_step": 64,
        "block_size": BLOCK_SIZE,
        "num_layers": NUM_LAYERS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "dtype": "float32",
        "eos_token_id": None,
        "model_memory_bytes": 0,
        "num_kv_blocks": 32,
    }
    config.update(overrides)
    return config


def _engine(
    monkeypatch: pytest.MonkeyPatch, device: str, **overrides: object
) -> ContinuousBatchingEngine:
    model = _build_model(seed=1, device=device)
    backend = _backend(device, num_blocks=32)
    monkeypatch.setattr(
        ContinuousBatchingEngine,
        "_resolve_device",
        staticmethod(lambda model: torch.device(device)),
    )
    return ContinuousBatchingEngine(
        model=model,
        engine=SimpleNamespace(
            get_attention_backend=lambda: backend, expert_layer_modules=[]
        ),
        config=_engine_config(**overrides),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_engine_greedy_decode_matches_hf(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    engine = _engine(monkeypatch, device)
    engine.model_runner.require_paged_kv_history()
    steps = 6
    for seq_id, prompt in PROMPTS.items():
        engine.add_request(
            f"req-{seq_id}",
            list(prompt),
            SamplingParams(temperature=0.0, max_tokens=steps),
        )
    for _ in range(8 * steps):
        engine.step()
        if not engine.has_pending_requests():
            break
    assert not engine.has_pending_requests()
    for seq_id, prompt in PROMPTS.items():
        expected, _ = _hf_greedy(engine.model, prompt, steps)
        assert engine.get_request_n_outputs(f"req-{seq_id}") == [expected]


def test_sliding_window_shorter_than_context_is_unsupported() -> None:
    assert MixtralPagedAttention.paged_unsupported_reason(_config()) is None
    assert (
        MixtralPagedAttention.paged_unsupported_reason(
            _config(sliding_window=128)
        )
        is None
    )
    reason = MixtralPagedAttention.paged_unsupported_reason(
        _config(sliding_window=16)
    )
    assert reason is not None and "sliding_window=16" in reason

    torch.manual_seed(0)
    with _paged_attention_override(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MixtralForCausalLM(_config(sliding_window=16)).eval()
    runner = _runner(model)
    with pytest.raises(RuntimeError, match="sliding_window=16"):
        runner.require_paged_kv_history()


def test_kv_cache_spec_for_config() -> None:
    assert MixtralPagedAttention.get_kv_cache_spec_for_config(_config()) == {
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
    }
