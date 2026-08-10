"""Task 11: GPU-gated gpt-oss-120b native DFlash validation harness.

This module is the DOCUMENTED, GPU-only correctness/throughput harness for the
native draft->verify->rollback DFlash path (T6/T8) on the real 120B target. It
is *not* part of the autonomous CPU suite: the whole module is guarded by a
single ``skipif`` that reports it as SKIPPED (never failed/errored) unless ALL
of the following hold:

  1. ``MOE_DFLASH_GPU`` is set in the environment (opt-in flag), AND
  2. CUDA is available, AND
  3. both checkpoints -- ``openai/gpt-oss-120b`` (target) and
     ``z-lab/gpt-oss-120b-DFlash`` (drafter) -- are present in the HuggingFace
     cache under ``$HF_HOME`` (falling back to the standard HF resolution when
     ``HF_HOME`` is unset).

The autonomous assertion for this task is exactly that: with no GPU flag the
test is reported SKIPPED, so it runs clean in normal CI without the ~60GB
checkpoints or a GPU. The conditions are checked cheapest-first so the common
(no-flag) path never touches CUDA or the filesystem past ``os.environ``.

When enabled it loads the resident 120B target + drafter and, over a fixed
128-token continuation for each prompt, records to
``.sisyphus/evidence/gpu-gated/task-11-120b-results.json``:

  * token **agreement-rate** -- native DFlash greedy vs plain greedy;
  * plain-decode **self-consistency** -- pairwise agreement of repeat plain
    greedy runs (the MXFP4 FP-near-tie floor the agreement-rate must clear);
  * **acceptance-length histogram** -- tokens advanced per verify step
    (``NativeStepTrace.accept + 1``) read from ``spec.step_trace``;
  * decode **tok/s** -- DFlash vs no-spec (single-stream).

Documented pass conditions (RFC Phase 0: mean accept ~= 3.66, single-stream
~1.18-1.32x):

  * agreement-rate >= plain self-consistency (losslessness on MXFP4 is measured
    as agreement, NOT string identity);
  * mean acceptance length in the sanity band ~3-5;
  * tok/s recorded for both paths.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pytest
import torch

TARGET_REPO = "openai/gpt-oss-120b"
DRAFTER_REPO = "z-lab/gpt-oss-120b-DFlash"

CONTINUATION_TOKENS = 128
SELF_CONSISTENCY_RUNS = 3
PROMPTS = (
    "The capital of France is",
    "In a shocking turn of events, scientists discovered that",
    "def fibonacci(n):\n    ",
)

# Mean acceptance length == tokens advanced per verify (trace.accept + 1).
# RFC Phase 0 reports mean accept ~= 3.66; keep a slightly widened sanity band
# so a real GPU run is not brittle against FP-near-tie flips.
ACCEPT_LEN_LO = 2.0
ACCEPT_LEN_HI = 6.0
# Native greedy may flip FP near-ties vs plain greedy on MXFP4, so the gate is
# agreement-rate >= plain self-consistency (with a tiny float tolerance).
AGREEMENT_EPS = 1e-9

# tests/python/dflash/test_gpu_120b.py -> repo root is parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = REPO_ROOT / ".sisyphus" / "evidence" / "gpu-gated"
RESULTS_PATH = EVIDENCE_DIR / "task-11-120b-results.json"


def _hf_home() -> Path:
    """Resolve the HF cache root the same way the ``huggingface_hub`` does.

    Honors ``$HF_HOME`` first (as the task specifies), then
    ``$HUGGINGFACE_HUB_CACHE`` (which points directly at the ``hub`` dir), then
    ``$XDG_CACHE_HOME``/``~/.cache`` -- so the checkpoint probe still works when
    ``HF_HOME`` is unset in the environment.
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub:
        return Path(hub).expanduser().parent
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "huggingface"


def _checkpoint_present(repo_id: str) -> bool:
    """True iff ``repo_id`` has a non-empty snapshot in the HF hub cache."""
    folder = "models--" + repo_id.replace("/", "--")
    snapshots = _hf_home() / "hub" / folder / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(child.is_dir() for child in snapshots.iterdir())


def _skip_reason() -> Optional[str]:
    """Return why the harness must skip, or ``None`` when it may run.

    Cheapest-first: the opt-in flag gates before CUDA, which gates before the
    filesystem probe -- so the normal (no-flag) run skips without importing
    heavy deps or touching the checkpoint cache.
    """
    if not os.environ.get("MOE_DFLASH_GPU"):
        return "MOE_DFLASH_GPU unset (GPU-gated 120B DFlash harness)"
    if not torch.cuda.is_available():
        return "CUDA unavailable (GPU-gated 120B DFlash harness)"
    missing = [
        repo
        for repo in (TARGET_REPO, DRAFTER_REPO)
        if not _checkpoint_present(repo)
    ]
    if missing:
        return "checkpoints not present in $HF_HOME: " + ", ".join(missing)
    return None


SKIP_REASON = _skip_reason()

pytestmark = pytest.mark.skipif(
    SKIP_REASON is not None, reason=SKIP_REASON or "gpu-gated"
)


def _load_resident_target():
    """Load the 120B target resident by default (offload is a tunable knob).

    ``device_memory_ratio`` defaults high (experts resident) per the
    Reconciliation Contract; ``MOE_DFLASH_MEM_RATIO`` /
    ``MOE_DFLASH_OFFLOAD`` let a GPU operator retune without editing the test.
    """
    from moe_infinity import MoE

    offload_path = os.environ.get(
        "MOE_DFLASH_OFFLOAD",
        str(_hf_home() / "moe-infinity" / "gpt-oss-120b-dflash"),
    )
    ratio = float(os.environ.get("MOE_DFLASH_MEM_RATIO", "0.9"))
    return MoE(
        TARGET_REPO,
        {"offload_path": offload_path, "device_memory_ratio": ratio},
    )


def _greedy(model, input_ids, spec=None) -> list[int]:
    """Greedy-decode ``CONTINUATION_TOKENS`` new ids; return only the new ids."""
    kwargs: dict = {
        "do_sample": False,
        "max_new_tokens": CONTINUATION_TOKENS,
    }
    if spec is not None:
        kwargs["speculative_draft"] = spec
    out = model.generate(input_ids, **kwargs)
    return [int(t) for t in out[0, input_ids.shape[1] :].tolist()]


def _agreement(a: list[int], b: list[int]) -> float:
    """Token agreement-rate over the shared prefix length of ``a`` and ``b``."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


def test_120b_native_dflash_validation() -> None:
    """Agreement-rate parity + acceptance-length + tok/s on the resident 120B.

    Writes the full metric set to ``RESULTS_PATH`` and asserts the documented
    pass conditions. Only reached on a GPU with both checkpoints cached; the
    autonomous run skips this entire module via ``pytestmark``.
    """
    from transformers import AutoTokenizer

    from moe_infinity.spec_decode import DFlashSpeculator

    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_REPO, trust_remote_code=True
    )
    model = _load_resident_target()
    spec = DFlashSpeculator(model, DRAFTER_REPO)

    per_prompt: list[dict] = []
    accept_lengths: list[int] = []

    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda:0")

        t0 = time.perf_counter()
        plain = _greedy(model, input_ids)
        plain_dt = time.perf_counter() - t0
        plain_tok_s = len(plain) / plain_dt if plain_dt > 0 else 0.0

        spec.step_trace = []
        t0 = time.perf_counter()
        dflash = _greedy(model, input_ids, spec=spec)
        dflash_dt = time.perf_counter() - t0
        dflash_tok_s = len(dflash) / dflash_dt if dflash_dt > 0 else 0.0

        # Acceptance length == tokens advanced per verify step (accept + 1).
        step_lengths = [int(tr.accept) + 1 for tr in spec.step_trace]
        accept_lengths.extend(step_lengths)
        mean_step = (
            sum(step_lengths) / len(step_lengths) if step_lengths else 0.0
        )

        repeats = [
            _greedy(model, input_ids) for _ in range(SELF_CONSISTENCY_RUNS)
        ]
        sc_scores = [_agreement(plain, r) for r in repeats]
        self_consistency = sum(sc_scores) / len(sc_scores)

        per_prompt.append(
            {
                "prompt": prompt,
                "agreement_rate": _agreement(dflash, plain),
                "self_consistency": self_consistency,
                "mean_accept_len": mean_step,
                "num_steps": len(step_lengths),
                "plain_tok_s": plain_tok_s,
                "dflash_tok_s": dflash_tok_s,
                "speedup": (
                    dflash_tok_s / plain_tok_s if plain_tok_s > 0 else 0.0
                ),
            }
        )

    histogram: dict[int, int] = {}
    for length in accept_lengths:
        histogram[length] = histogram.get(length, 0) + 1
    mean_accept = (
        sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
    )

    summary = {
        "target": TARGET_REPO,
        "drafter": DRAFTER_REPO,
        "continuation_tokens": CONTINUATION_TOKENS,
        "self_consistency_runs": SELF_CONSISTENCY_RUNS,
        "mean_accept_len": mean_accept,
        "acceptance_histogram": {
            str(k): histogram[k] for k in sorted(histogram)
        },
        "per_prompt": per_prompt,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    # Pass condition 1: agreement-rate >= plain self-consistency per prompt.
    for row in per_prompt:
        assert (
            row["agreement_rate"] >= row["self_consistency"] - AGREEMENT_EPS
        ), (
            f"agreement {row['agreement_rate']:.4f} < self-consistency "
            f"{row['self_consistency']:.4f} for prompt {row['prompt']!r}"
        )

    # Pass condition 2: mean acceptance length in the documented sanity band.
    assert accept_lengths, "no DFlash steps recorded (spec.step_trace empty)"
    assert ACCEPT_LEN_LO <= mean_accept <= ACCEPT_LEN_HI, (
        f"mean acceptance length {mean_accept:.2f} outside sanity band "
        f"[{ACCEPT_LEN_LO}, {ACCEPT_LEN_HI}]"
    )

    # Pass condition 3: tok/s recorded for both paths.
    for row in per_prompt:
        assert row["plain_tok_s"] > 0.0, "plain decode tok/s not recorded"
        assert row["dflash_tok_s"] > 0.0, "DFlash decode tok/s not recorded"
