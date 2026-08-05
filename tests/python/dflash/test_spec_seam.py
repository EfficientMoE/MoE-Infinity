import torch
from transformers import GPT2Config, GPT2LMHeadModel

from moe_infinity.engine.generation_loop import (
    GenerationEngine,
    GenerationResult,
    SpecDecodeStrategy,
)
from moe_infinity.engine.types import SamplingParams, SequenceStatus
from moe_infinity.memory.kv_cache_manager import KVCacheManager
from moe_infinity.runtime.attention_types import KVCacheSpec

VOCAB = 128
PROMPT = [3, 11, 42, 7, 90]
MAX_TOKENS = 16

# Captured from the pre-refactor GenerationEngine (seed 7 tiny GPT-2 fixture below);
# the seam refactor must leave the standard path byte-identical to this.
BASELINE_IDS = [61, 61, 108, 108, 108, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]


def _build_forward():
    torch.manual_seed(7)
    cfg = GPT2Config(
        vocab_size=VOCAB,
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=64,
        n_ctx=64,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = GPT2LMHeadModel(cfg).eval()
    ctx: list[int] = []

    def forward(token_ids, attention_metadata):
        ctx.extend(token_ids)
        ids = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            out = model(ids)
        return out.logits[0, -1:, :]

    return forward


def _make_engine(spec_strategy=None, **engine_kwargs):
    spec = KVCacheSpec(
        num_kv_heads=2, head_dim=8, dtype=torch.float32, block_size=4
    )
    mgr = KVCacheManager(num_gpu_blocks=64, num_cpu_blocks=16, block_size=4)
    return GenerationEngine(
        kv_cache_manager=mgr,
        kv_spec=spec,
        num_layers=2,
        vocab_size=VOCAB,
        model_forward_fn=_build_forward(),
        eos_token_id=2,
        spec_strategy=spec_strategy,
        **engine_kwargs,
    )


def _greedy_params():
    return SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=MAX_TOKENS)


class _ExplodingStrategy:
    def run(self, *, engine, prompt_token_ids, sampling_params, request_id=None):
        raise RuntimeError("spec strategy must not be called on this path")


class _RecordingStrategy:
    def __init__(self, ids):
        self._ids = ids
        self.calls = []

    def run(self, *, engine, prompt_token_ids, sampling_params, request_id=None):
        self.calls.append(
            {
                "engine": engine,
                "prompt_token_ids": prompt_token_ids,
                "sampling_params": sampling_params,
                "request_id": request_id,
            }
        )
        return list(self._ids)


def test_spec_off_default_matches_prerefactor_baseline():
    engine = _make_engine()
    result = engine.generate(
        prompt_token_ids=list(PROMPT), sampling_params=_greedy_params()
    )
    assert isinstance(result, GenerationResult)
    assert result.output_token_ids == BASELINE_IDS
    assert result.finish_reason == SequenceStatus.FINISHED_LENGTH


def test_spec_off_explicit_none_matches_prerefactor_baseline():
    engine = _make_engine(spec_strategy=None)
    result = engine.generate(
        prompt_token_ids=list(PROMPT), sampling_params=_greedy_params()
    )
    assert result.output_token_ids == BASELINE_IDS
    assert result.finish_reason == SequenceStatus.FINISHED_LENGTH


def test_nongreedy_params_bypass_spec_strategy():
    non_greedy = [
        SamplingParams(temperature=0.7, max_tokens=8),
        SamplingParams(temperature=1.0, top_p=0.9, max_tokens=8),
        SamplingParams(temperature=1.0, top_k=10, max_tokens=8),
    ]
    for params in non_greedy:
        torch.manual_seed(99)
        guarded = _make_engine(spec_strategy=_ExplodingStrategy())
        guarded_result = guarded.generate(
            prompt_token_ids=list(PROMPT), sampling_params=params
        )

        torch.manual_seed(99)
        reference = _make_engine()
        reference_result = reference.generate(
            prompt_token_ids=list(PROMPT), sampling_params=params
        )

        assert guarded_result.output_token_ids == reference_result.output_token_ids
        assert guarded_result.finish_reason == reference_result.finish_reason


def test_greedy_delegates_to_spec_strategy():
    canned_ids = [5, 6, 7, 8]
    strategy = _RecordingStrategy(canned_ids)
    engine = _make_engine(spec_strategy=strategy)
    result = engine.generate(
        prompt_token_ids=list(PROMPT), sampling_params=_greedy_params()
    )
    assert len(strategy.calls) == 1
    call = strategy.calls[0]
    assert call["engine"] is engine
    assert call["prompt_token_ids"] == PROMPT
    assert result.output_token_ids == canned_ids
    assert isinstance(result, GenerationResult)


def test_spec_strategy_protocol_runtime_shape():
    assert hasattr(SpecDecodeStrategy, "run")
