"""E2E: OLMoE decode through the paged shim matches plain transformers.

`docs/model-compatibility.md` marks OLMoE continuous serving `not validated`
by its own definition -- "validated means the named scope has a repository
harness". Prior evidence for OlmoePagedAttention was a synthetic-backend
unit test and a manual, non-repo comparison run on real hardware
(teacher-forced against plain transformers through `api_server_v2`; see the
project's own logbook). This file is that harness: it drives a real
`allenai/OLMoE-1B-7B-0924-Instruct` checkpoint through
`ContinuousBatchingEngine` -- sequential requests, then several concurrent
ones of different lengths -- and checks every generated token against a
teacher-forced reference forward on plain `transformers`.

The reference runs on CPU, deterministically, and independently of the
GPU memory the served model occupies (offloaded experts plus the paged KV
cache already use most of a 24 GB card at the ratios below); this mirrors
the project's own choice of a CPU reference as the legal way to make an
exact-token comparison meaningful.

Agreement counts a token correct if it is the reference's greedy choice,
or if the reference's logit for it sits within one bf16 unit-in-the-last-
-place of the top-1 logit -- the tie rule this project settled on after
measuring that run-to-run engine noise and cross-GPU reference noise both
land within that margin (see `fase2_criterio.md` v1.1 in the project's
working notes). Below that margin the reference itself is not more
informative than the engine.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for E2E tests",
    ),
]

MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"
SEQUENTIAL_PROMPTS = [
    "The capital of France is",
    "Water freezes at",
    "The three primary colors are",
]
CONCURRENT_PROMPT = "Explain in one sentence why water expands when it freezes:"
# Cycled to fill higher-concurrency batches (test_olmoe_decode_matches_reference_
# at_concurrency below); reusing these four rather than inventing new prompt text
# keeps every case teacher-forceable against the same reference methodology.
ALL_PROMPTS = [*SEQUENTIAL_PROMPTS, CONCURRENT_PROMPT]
MAX_NEW_TOKENS = 16
# Same value used for both MoE() and the engine config below, as
# api_server_v2 does at both its call sites (see _build_engine_config).
DEVICE_MEMORY_RATIO = 0.5
AGREEMENT_THRESHOLD = 0.99


def _resolve_int_attr(config: object, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    return None


def _resolve_dtype(model: object) -> torch.dtype:
    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype):
        return model_dtype

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            parameter_source = parameters()
            if isinstance(parameter_source, Iterator):
                first_param = next(parameter_source, None)
            elif isinstance(parameter_source, Iterable):
                first_param = next(iter(parameter_source), None)
            else:
                first_param = None
        except Exception:
            first_param = None
        if isinstance(first_param, torch.Tensor):
            return first_param.dtype

    cfg = getattr(model, "config", None)
    cfg_dtype = getattr(cfg, "torch_dtype", None)
    if isinstance(cfg_dtype, torch.dtype):
        return cfg_dtype
    if isinstance(cfg_dtype, str):
        mapping = {
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        normalized = cfg_dtype.replace("torch.", "")
        if normalized in mapping:
            return mapping[normalized]

    return torch.float16


def _build_engine_config(
    model: object,
    kv_cache_ratio: float,
    device_memory_ratio: float,
    max_batch_size: int = 8,
) -> dict[str, object]:
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise RuntimeError("model.config is required")

    num_layers = _resolve_int_attr(
        model_config, "num_hidden_layers", "num_layers", "n_layer"
    )
    num_attention_heads = _resolve_int_attr(
        model_config, "num_attention_heads", "n_head"
    )
    num_kv_heads = _resolve_int_attr(
        model_config, "num_key_value_heads", "num_kv_heads", "n_head_kv"
    )
    hidden_size = _resolve_int_attr(model_config, "hidden_size", "n_embd")
    head_dim = _resolve_int_attr(model_config, "head_dim")
    eos_token_id = _resolve_int_attr(model_config, "eos_token_id")

    if num_layers is None or num_attention_heads is None:
        raise RuntimeError("unable to resolve model layer/head config")
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads
    if head_dim is None:
        if hidden_size is None:
            raise RuntimeError("unable to resolve model head_dim")
        head_dim = hidden_size // max(1, num_attention_heads)

    config: dict[str, object] = {
        # api_server_v2 passes the identical ratio to MoE() and to this
        # engine config at both its call sites; using a different, larger
        # value here (this helper's origin, test_kv_parity.py, defaults to
        # 0.75 for a different model) OOM'd the expert cache on a 24 GB
        # card even though 0.5 alone works for this checkpoint. Caller
        # passes the same value used for MoE(), matching production.
        "device_memory_ratio": device_memory_ratio,
        "kv_cache_ratio": kv_cache_ratio,
        "max_batch_size": max_batch_size,
        "max_tokens_per_step": 2048,
        "block_size": 16,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": _resolve_dtype(model),
    }
    if isinstance(eos_token_id, int):
        config["eos_token_id"] = eos_token_id
    return config


def _require_local_model_dir(model_name: str) -> str:
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            ignore_patterns=["flax*", "tf*"],
        )
    except Exception as exc:
        pytest.skip(f"Model not cached locally ({model_name}): {exc}")


@contextlib.contextmanager
def _real_flash_attn_absence():
    """`tests/conftest.py` stubs `flash_attn` into `sys.modules` as a
    `MagicMock` so CPU-only unit tests can import optional CUDA modules.
    That stub makes `MoE`'s own availability probe
    (`entrypoints/big_modeling.py`, ``try: import flash_attn``) succeed,
    which then asks transformers for real FlashAttention2 and fails --
    a test-harness artifact of running under the repo's shared conftest,
    not something the served model ever exhibits (`api_server_v2` never
    imports `tests/conftest.py`). Remove the stub for the duration of model
    construction so this test sees what serving actually sees; architectures
    excluded from the probe elsewhere (deepseek, qwen3, ...) do not need
    this because their availability is forced False regardless.
    """
    stub = sys.modules.get("flash_attn")
    removed = isinstance(stub, MagicMock)
    if removed:
        del sys.modules["flash_attn"]
    try:
        yield
    finally:
        if removed:
            sys.modules["flash_attn"] = stub


def _has_long_repeat(token_ids: list[int], limit: int = 8) -> bool:
    run = best = 1
    for a, b in zip(token_ids, token_ids[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best > limit


def _run_requests(
    engine: Any,
    prompt_token_ids_list: list[list[int]],
    max_new_tokens: int,
    id_prefix: str = "olmoe-parity",
) -> dict[int, list[int]]:
    """`id_prefix` must be unique per call against the SAME engine instance:
    completed request ids stay registered on the engine (`add_request`
    rejects a repeat), so calling this twice with the default prefix on one
    engine -- as the sequential test does, one request at a time -- collides
    on the second call."""
    from moe_infinity.serving.sequence import SamplingParams

    for idx, prompt_ids in enumerate(prompt_token_ids_list):
        engine.add_request(
            request_id=f"{id_prefix}-{idx}",
            prompt_token_ids=list(prompt_ids),
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=max_new_tokens
            ),
        )

    outputs = engine.run_until_done()
    return {
        idx: outputs[f"{id_prefix}-{idx}"]
        for idx in range(len(prompt_token_ids_list))
    }


_REFERENCE_SUBPROCESS_SCRIPT = r"""
import json, math, sys
import torch
from transformers import AutoModelForCausalLM

def _bf16_ulp(x):
    if x == 0.0:
        return 0.0
    return (2.0 ** math.floor(math.log2(abs(x)))) * (2.0 ** -7)

payload = json.loads(sys.stdin.read())
model = AutoModelForCausalLM.from_pretrained(
    payload["model_dir"],
    dtype=torch.bfloat16,
    local_files_only=True,
    attn_implementation="eager",
).eval()

results = []
for prompt_ids, generated_ids in payload["cases"]:
    full = torch.tensor([[*prompt_ids, *generated_ids]])
    with torch.no_grad():
        logits = model(full).logits[0].float()
    agree = 0
    for t, token_id in enumerate(generated_ids):
        pos = len(prompt_ids) + t - 1
        row = logits[pos]
        top1 = row.max().item()
        if row[token_id].item() >= top1 - _bf16_ulp(top1):
            agree += 1
    results.append([agree, len(generated_ids)])
print(json.dumps(results))
"""


def _reference_agreement_subprocess(
    model_dir: str, cases: list[tuple[list[int], list[int]]]
) -> list[tuple[int, int]]:
    """Teacher-forced agreement against plain `transformers`, computed in a
    FRESH interpreter that never imports `moe_infinity`.

    `MoE(...)` monkey-patches `transformers.models.olmoe.modeling_olmoe.
    OlmoeSparseMoeBlock` at module scope for the life of the process
    (`runtime/model_offload.py`'s `setup_archer_hooks`). A "plain"
    `AutoModelForCausalLM` built in the SAME process after that inherits
    the patched class instead of the real one and is not a reference at
    all -- measured: it raises `AttributeError: 'SyncOlmoeMoEBlock' object
    has no attribute 'expert_executor'`, since that block only works once
    wired up by `archer_from_pretrained`. A subprocess sidesteps the global
    patch entirely, the same way every manual measurement in this
    project's working notes ran the reference in a separate container."""
    payload = json.dumps({"model_dir": model_dir, "cases": cases})
    proc = subprocess.run(
        [sys.executable, "-c", _REFERENCE_SUBPROCESS_SCRIPT],
        input=payload,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(f"reference subprocess failed:\n{proc.stderr[-4000:]}")
    return [(agree, total) for agree, total in json.loads(proc.stdout)]


@pytest.fixture(scope="module")
def olmoe_engine_bundle(tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("transformers")
    pytest.importorskip("moe_infinity")

    from transformers import AutoTokenizer

    from moe_infinity import MoE
    from moe_infinity.serving.engine import ContinuousBatchingEngine

    local_model_dir = _require_local_model_dir(MODEL_NAME)
    offload_path = Path(tmp_path_factory.mktemp("e2e_olmoe_parity")) / "offload"
    offload_path.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir, local_files_only=True
        )
        with _real_flash_attn_absence():
            moe_model = MoE(
                local_model_dir,
                {
                    "offload_path": str(offload_path),
                    "device_memory_ratio": DEVICE_MEMORY_RATIO,
                },
            )
    except Exception as exc:
        pytest.skip(f"Unable to initialize OLMoE for parity test: {exc}")

    def build_engine(
        kv_cache_ratio: float = 0.15, max_batch_size: int = 8
    ) -> Any:
        return ContinuousBatchingEngine(
            model=moe_model.model,
            engine=moe_model.engine,
            config=_build_engine_config(
                moe_model.model,
                kv_cache_ratio=kv_cache_ratio,
                device_memory_ratio=DEVICE_MEMORY_RATIO,
                max_batch_size=max_batch_size,
            ),
            tokenizer=tokenizer,
        )

    return tokenizer, build_engine, local_model_dir


def test_olmoe_decode_matches_reference_sequential(olmoe_engine_bundle) -> None:
    """One request at a time: the decode-context regression (defect 1)
    showed up here first -- every decode step after the first was off by
    one token of context."""
    tokenizer, build_engine, local_model_dir = olmoe_engine_bundle
    engine = build_engine()

    cases: list[tuple[list[int], list[int]]] = []
    try:
        for seq_idx, prompt in enumerate(SEQUENTIAL_PROMPTS):
            prompt_ids = tokenizer(prompt)["input_ids"]
            outputs = _run_requests(
                engine,
                [prompt_ids],
                MAX_NEW_TOKENS,
                id_prefix=f"seq-{seq_idx}",
            )
            generated = outputs[0]
            assert not _has_long_repeat(
                generated
            ), f"decode collapsed for {prompt!r}: {generated}"
            cases.append((prompt_ids, generated))
    finally:
        # Free the paged KV cache's device tensors before the next test in
        # this module builds its own engine against the same GPU budget.
        engine.shutdown()

    agree_total, token_total = (
        sum(values)
        for values in zip(
            *_reference_agreement_subprocess(local_model_dir, cases)
        )
    )
    agreement = agree_total / token_total
    assert agreement >= AGREEMENT_THRESHOLD, (
        f"sequential greedy agreement {agreement:.4f} "
        f"({agree_total}/{token_total}) below {AGREEMENT_THRESHOLD}"
    )


def test_olmoe_decode_matches_reference_concurrent(olmoe_engine_bundle) -> None:
    """Several requests of different prompt length in flight together: the
    mixed prefill+decode split (defect 2) and the packed multi-sequence
    prefill (defect 3) only showed up here, never with a single request."""
    tokenizer, build_engine, local_model_dir = olmoe_engine_bundle
    engine = build_engine()

    prompts = [*SEQUENTIAL_PROMPTS, CONCURRENT_PROMPT]
    prompt_ids_list = [tokenizer(p)["input_ids"] for p in prompts]
    cases: list[tuple[list[int], list[int]]] = []
    try:
        outputs = _run_requests(engine, prompt_ids_list, MAX_NEW_TOKENS)

        for idx, prompt_ids in enumerate(prompt_ids_list):
            generated = outputs[idx]
            assert not _has_long_repeat(
                generated
            ), f"decode collapsed for {prompts[idx]!r}: {generated}"
            cases.append((prompt_ids, generated))
    finally:
        engine.shutdown()

    agree_total, token_total = (
        sum(values)
        for values in zip(
            *_reference_agreement_subprocess(local_model_dir, cases)
        )
    )
    agreement = agree_total / token_total
    assert agreement >= AGREEMENT_THRESHOLD, (
        f"concurrent greedy agreement {agreement:.4f} "
        f"({agree_total}/{token_total}) below {AGREEMENT_THRESHOLD}"
    )


@pytest.mark.parametrize("concurrency", [8, 16, 32])
def test_olmoe_decode_matches_reference_at_concurrency(
    olmoe_engine_bundle, concurrency: int
) -> None:
    """Beyond the 4-request concurrent case above: `max_batch_size` is
    raised to match `concurrency` so a batch this size can actually be
    packed into one prefill instead of being chunked by the engine's
    default cap of 8 -- the scenario defects 2 and 3 were about."""
    tokenizer, build_engine, local_model_dir = olmoe_engine_bundle
    engine = build_engine(max_batch_size=concurrency)

    prompts = [ALL_PROMPTS[i % len(ALL_PROMPTS)] for i in range(concurrency)]
    prompt_ids_list = [tokenizer(p)["input_ids"] for p in prompts]
    cases: list[tuple[list[int], list[int]]] = []
    try:
        outputs = _run_requests(
            engine,
            prompt_ids_list,
            MAX_NEW_TOKENS,
            id_prefix=f"conc{concurrency}",
        )

        for idx, prompt_ids in enumerate(prompt_ids_list):
            generated = outputs[idx]
            assert not _has_long_repeat(
                generated
            ), f"decode collapsed for {prompts[idx]!r}: {generated}"
            cases.append((prompt_ids, generated))
    finally:
        engine.shutdown()

    agree_total, token_total = (
        sum(values)
        for values in zip(
            *_reference_agreement_subprocess(local_model_dir, cases)
        )
    )
    agreement = agree_total / token_total
    assert agreement >= AGREEMENT_THRESHOLD, (
        f"concurrency={concurrency} greedy agreement {agreement:.4f} "
        f"({agree_total}/{token_total}) below {AGREEMENT_THRESHOLD}"
    )
