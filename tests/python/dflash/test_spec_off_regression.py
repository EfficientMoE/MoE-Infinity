"""Task 10: spec-off byte-identity regression.

The full DFlash integration (T1 seam + ``_generate_standard``, T2 rich
forward helper, T3/T4 ops + loader, T6/T7 state machine, T8
``MoE.generate(..., speculative_draft=...)``) must leave the standard,
non-speculative decode path byte-identical to the pre-integration baseline.

Proof (A) pins ``GenerationEngine.generate(spec_strategy=None)`` on a
seeded pure-CPU GPT-2 model to the same literal Task 1 captured before the
seam refactor -- device-independent, so the literal is portable.

Proof (B) drives ``MoE.generate`` without ``speculative_draft`` through the
real sync path on the tiny gpt-oss target. Since ``_native_model_forward``
runs on the model device (``cuda:0`` when present), tiny-target ids are
hardware-dependent; (B) therefore asserts equality against the in-process
``_generate_standard`` baseline (and non-sticky ``spec_strategy is None``)
rather than a hard-coded literal, which stays in the CPU-only proof (A).
"""

from __future__ import annotations

import os
import sys
import warnings

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    build_tiny_target,
    set_determinism,
)

from moe_infinity.engine.generation_loop import (  # noqa: E402
    GenerationEngine,
    GenerationResult,
)
from moe_infinity.engine.types import (  # noqa: E402
    SamplingParams,
    SequenceStatus,
)
from moe_infinity.entrypoints.big_modeling import MoE  # noqa: E402
from moe_infinity.memory.kv_cache_manager import KVCacheManager  # noqa: E402
from moe_infinity.runtime.attention_types import KVCacheSpec  # noqa: E402
from moe_infinity.spec_decode.dflash import _resolve_stop_ids  # noqa: E402

GPT2_VOCAB = 128
GPT2_PROMPT = [3, 11, 42, 7, 90]
GPT2_MAX_TOKENS = 16

# Captured from the pre-refactor GenerationEngine standard loop (seed-7 tiny
# GPT-2 fixture below). This is the identical baseline pinned by Task 1's
# ``test_spec_seam.py``; the full DFlash integration must not move it.
BASELINE_IDS = [61, 61, 108, 108, 108, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]


def _build_cpu_gpt2_forward():
    """A deterministic, CPU-only model_forward_fn (no device moves)."""
    torch.manual_seed(7)
    cfg = GPT2Config(
        vocab_size=GPT2_VOCAB,
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


def _gpt2_engine(spec_strategy=None):
    return GenerationEngine(
        kv_cache_manager=KVCacheManager(
            num_gpu_blocks=64, num_cpu_blocks=16, block_size=4
        ),
        kv_spec=KVCacheSpec(
            num_kv_heads=2, head_dim=8, dtype=torch.float32, block_size=4
        ),
        num_layers=2,
        vocab_size=GPT2_VOCAB,
        model_forward_fn=_build_cpu_gpt2_forward(),
        eos_token_id=2,
        spec_strategy=spec_strategy,
    )


def _greedy_params(max_tokens=GPT2_MAX_TOKENS):
    return SamplingParams(
        temperature=0.0, top_p=1.0, top_k=0, max_tokens=max_tokens
    )


def test_spec_off_default_matches_committed_baseline():
    """Default engine (no strategy) == committed pre-integration literal."""
    result = _gpt2_engine().generate(
        prompt_token_ids=list(GPT2_PROMPT), sampling_params=_greedy_params()
    )
    assert isinstance(result, GenerationResult)
    assert result.output_token_ids == BASELINE_IDS
    assert result.finish_reason == SequenceStatus.FINISHED_LENGTH


def test_spec_off_explicit_none_matches_committed_baseline():
    """Explicit ``spec_strategy=None`` == committed baseline (byte-ident)."""
    result = _gpt2_engine(spec_strategy=None).generate(
        prompt_token_ids=list(GPT2_PROMPT), sampling_params=_greedy_params()
    )
    assert result.output_token_ids == BASELINE_IDS
    assert result.finish_reason == SequenceStatus.FINISHED_LENGTH


def test_spec_off_standard_path_is_deterministic():
    """The seeded standard path is bit-reproducible (validates the literal)."""
    first = _gpt2_engine().generate(
        prompt_token_ids=list(GPT2_PROMPT), sampling_params=_greedy_params()
    )
    second = _gpt2_engine().generate(
        prompt_token_ids=list(GPT2_PROMPT), sampling_params=_greedy_params()
    )
    assert first.output_token_ids == second.output_token_ids == BASELINE_IDS


OSS_PROMPT = [3, 7, 11, 2, 5]
OSS_MAX_NEW_TOKENS = 16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _tiny_moe_shell(seed: int = 0):
    """MoE shell around the tiny gpt-oss target + a real native engine.

    Mirrors production wiring (engine holds
    ``model_forward_fn=shell._native_model_forward``) without the offload
    runtime, matching ``test_engine_wire.py``'s shell.
    """
    set_determinism(seed)
    target = build_tiny_target(seed=seed).to(DEVICE)
    shell = MoE.__new__(MoE)
    shell.model = target
    shell.use_native_engine = True
    shell.max_seq_length = 64
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    shell._configure_hook = lambda input_ids: None

    stop_ids = _resolve_stop_ids(target, None)
    eos_token_id = stop_ids[0] if stop_ids else -1

    engine = GenerationEngine(
        kv_cache_manager=KVCacheManager(
            num_gpu_blocks=64, num_cpu_blocks=16, block_size=4
        ),
        kv_spec=KVCacheSpec(
            num_kv_heads=2, head_dim=8, dtype=torch.float32, block_size=4
        ),
        num_layers=int(target.config.num_hidden_layers),
        vocab_size=int(target.config.vocab_size),
        model_forward_fn=shell._native_model_forward,
        eos_token_id=eos_token_id,
        max_seq_length=64,
    )
    shell._native_generation_engine = engine
    return shell, engine


def _moe_generate(shell, **kwargs):
    input_ids = torch.tensor([OSS_PROMPT], dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return shell.generate(input_ids, **kwargs)


def test_moe_generate_without_drafter_matches_standard_path():
    """No drafter => _generate_standard; output == in-process baseline."""
    shell, engine = _tiny_moe_shell()

    shell._cached_past_key_values = None
    baseline = engine.generate(
        prompt_token_ids=list(OSS_PROMPT),
        sampling_params=SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=OSS_MAX_NEW_TOKENS,
        ),
    )
    assert engine.spec_strategy is None

    standard_calls = []
    orig_standard = engine._generate_standard

    def standard_spy(*args, **kwargs):
        standard_calls.append((args, kwargs))
        return orig_standard(*args, **kwargs)

    engine._generate_standard = standard_spy

    out = _moe_generate(
        shell, do_sample=False, max_new_tokens=OSS_MAX_NEW_TOKENS
    )

    assert engine.spec_strategy is None
    assert len(standard_calls) == 1
    assert out[0].tolist() == OSS_PROMPT + baseline.output_token_ids
    new_ids = out[0, len(OSS_PROMPT) :].tolist()
    assert 0 < len(new_ids) <= OSS_MAX_NEW_TOKENS


def test_moe_generate_without_drafter_is_deterministic():
    """Two independent no-drafter runs agree (standard path is stable)."""
    shell_a, _ = _tiny_moe_shell()
    out_a = _moe_generate(
        shell_a, do_sample=False, max_new_tokens=OSS_MAX_NEW_TOKENS
    )
    shell_b, _ = _tiny_moe_shell()
    out_b = _moe_generate(
        shell_b, do_sample=False, max_new_tokens=OSS_MAX_NEW_TOKENS
    )
    assert out_a[0].tolist() == out_b[0].tolist()
