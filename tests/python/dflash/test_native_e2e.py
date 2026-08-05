"""Task 9: tiny-model E2E losslessness parity (native DFlash == plain greedy).

The definitive autonomous correctness gate for DFlash losslessness -- no 120B,
no network, no GPU requirement. For a set of FIXED prompts it drives the REAL
sync engine path end to end:

    MoE.generate(..., speculative_draft=spec)
        -> GenerationEngine.generate (greedy gate applies)
        -> spec.run -> DFlashSpeculator.generate  (via MoE._native_model_forward_rich)

and asserts the emitted token-id sequence is EXACTLY identical to
``plain_greedy_decode`` on the same tiny target over >= 64 new tokens for EVERY
prompt (the tiny CPU model is bit-reproducible, so token identity -- not just
agreement rate -- is the gate; on the MXFP4 120B this becomes agreement-rate
parity). A first-divergence index is reported on failure.

A second check proves the drafter is actually exercised: mean per-step
acceptance (accepted drafts, ``NativeStepTrace.accept``) is > 0 on at least one
prompt, so parity is a real draft->verify->accept loss-free path and not a
degenerate accept-0 fallback that trivially equals plain greedy.

Device follows the sibling tests (``cuda:0`` when available, else ``cpu``):
``MoE._native_model_forward_rich`` moves inputs to ``cuda:0`` whenever CUDA is
present, so the target must live there too. The tiny fixtures are
bit-reproducible on both, verified identical.
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
    plain_greedy_decode,
    set_determinism,
)

from moe_infinity.engine.generation_loop import GenerationEngine  # noqa: E402
from moe_infinity.entrypoints.big_modeling import MoE  # noqa: E402
from moe_infinity.memory.kv_cache_manager import KVCacheManager  # noqa: E402
from moe_infinity.runtime.attention_types import KVCacheSpec  # noqa: E402
from moe_infinity.spec_decode import (  # noqa: E402
    DFlashSpeculator,
    read_dflash_config,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 64
UNREACHABLE_EOS = -1

# Fixed prompts spanning several lengths; token ids stay inside the tiny vocab.
PROMPTS = [
    [3, 7, 11, 2, 5],
    [1, 2, 3],
    [10, 20, 30, 40],
    [5],
    [8, 16, 24, 32, 40, 48],
    [42, 17, 33, 9],
]

_EVIDENCE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".sisyphus", "evidence"
    )
)


def _build_engine_shell(seed: int = 0):
    """MoE shell around the tiny target with a real ``GenerationEngine``.

    Mirrors the production wiring (engine ``model_forward_fn`` bound to
    ``shell._native_model_forward``) without an offload runtime, exactly like
    ``test_engine_wire.py``. ``eos_token_id`` is unreachable so the engine
    never forces an early stop; the tiny target config's own ``eos_token_id``
    is ``None``, matching ``plain_greedy_decode(eos_token_id=None)``.
    """
    set_determinism(seed)
    target = build_tiny_target(seed=seed).to(DEVICE)
    shell = MoE.__new__(MoE)
    shell.model = target
    shell.use_native_engine = True
    shell.max_seq_length = 256
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    shell._configure_hook = lambda input_ids: None

    engine = GenerationEngine(
        kv_cache_manager=KVCacheManager(
            num_gpu_blocks=256, num_cpu_blocks=64, block_size=4
        ),
        kv_spec=KVCacheSpec(
            num_kv_heads=2, head_dim=8, dtype=torch.float32, block_size=4
        ),
        num_layers=int(target.config.num_hidden_layers),
        vocab_size=int(target.config.vocab_size),
        model_forward_fn=shell._native_model_forward,
        eos_token_id=UNREACHABLE_EOS,
        max_seq_length=256,
    )
    shell._native_generation_engine = engine
    return shell, target


def _native_generate(shell, spec, prompt):
    input_ids = torch.tensor([prompt], dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        out = shell.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            speculative_draft=spec,
        )
    new_ids = out[0, len(prompt) :].tolist()
    accepts = [rec.accept for rec in spec.step_trace]
    return new_ids, accepts


def _decode_all():
    """Decode every prompt both ways; return per-prompt plain/native/accepts."""
    shell, target = _build_engine_shell(seed=0)
    drafter = build_tiny_drafter(target, seed=1).to(DEVICE)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    spec = DFlashSpeculator.from_models(shell, drafter, config=config, device=DEVICE)

    results = []
    for prompt in PROMPTS:
        plain = plain_greedy_decode(
            target,
            torch.tensor([prompt], dtype=torch.long, device=DEVICE),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        plain_new = plain[0, len(prompt) :].tolist()
        native_new, accepts = _native_generate(shell, spec, prompt)
        results.append(
            {
                "prompt": prompt,
                "plain_new": plain_new,
                "native_new": native_new,
                "accepts": accepts,
            }
        )
    return results


def _first_divergence(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _write_evidence(filename, lines):
    os.makedirs(_EVIDENCE_DIR, exist_ok=True)
    path = os.path.join(_EVIDENCE_DIR, filename)
    with open(path, "w") as handle:
        handle.write(f"device={DEVICE} max_new_tokens={MAX_NEW_TOKENS}\n")
        handle.write("\n".join(lines) + "\n")
    return path


@pytest.fixture(scope="module")
def decoded():
    return _decode_all()


def test_native_equals_plain_greedy_token_identical(decoded):
    """QA (happy): native DFlash == plain greedy, token-identical, >= 64 ids."""
    lines = []
    failures = []
    for record in decoded:
        prompt = record["prompt"]
        plain_new = record["plain_new"]
        native_new = record["native_new"]
        first_div = _first_divergence(plain_new, native_new)
        identical = plain_new == native_new
        length_ok = len(native_new) >= 64
        lines.append(
            f"prompt={str(prompt):<26} new_tokens={len(native_new)} "
            f"len>=64={length_ok} identical={identical} "
            f"first_divergence={first_div}"
        )
        if not (identical and length_ok):
            failures.append((prompt, first_div, plain_new, native_new))

    _write_evidence("task-9-native-parity.txt", lines)
    print("\n".join(lines))

    if failures:
        prompt, first_div, plain_new, native_new = failures[0]
        raise AssertionError(
            f"native DFlash diverged from plain greedy for prompt {prompt} "
            f"at new-token index {first_div} "
            f"({len(failures)} of {len(decoded)} prompts failed):\n"
            f"  plain ={plain_new}\n  native={native_new}"
        )


def test_acceptance_length_is_non_degenerate(decoded):
    """QA (sanity): the drafter is exercised -- mean accept > 0 on >= 1 prompt."""
    lines = []
    per_prompt_mean = []
    total_accept = 0
    for record in decoded:
        prompt = record["prompt"]
        accepts = record["accepts"]
        step_total = sum(accepts)
        total_accept += step_total
        mean = step_total / len(accepts) if accepts else 0.0
        per_prompt_mean.append(mean)
        lines.append(
            f"prompt={str(prompt):<26} steps={len(accepts)} "
            f"total_accept={step_total} mean_accept={mean:.4f}"
        )

    prompts_with_accept = sum(1 for mean in per_prompt_mean if mean > 0)
    lines.append(
        f"aggregate total_accept={total_accept} "
        f"prompts_with_accept={prompts_with_accept}/{len(decoded)} "
        f"max_mean_accept={max(per_prompt_mean):.4f}"
    )

    _write_evidence("task-9-accept-nonzero.txt", lines)
    print("\n".join(lines))

    assert any(mean > 0 for mean in per_prompt_mean), (
        "DFlash accepted zero drafts on every prompt (degenerate accept-0 "
        f"loop -- drafter never exercised); per-prompt means={per_prompt_mean}"
    )
