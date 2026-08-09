"""GPU-gated real-model DFlash validation on the gpt-oss-20b pair.

Opt-in via ``MOE_DFLASH_GPU=1`` with ``openai/gpt-oss-20b`` +
``z-lab/gpt-oss-20b-DFlash`` present in the HF cache. Asserts the drafter-driven
contract loads (block_size=8, not the 120b default of 10), that native DFlash
greedy is token-identical to plain greedy (losslessness), and that acceptance
length + speedup are positive. Without the flag this collects and skips cleanly
(no CUDA, no filesystem, no model load).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pytest
import torch

TARGET_REPO = "openai/gpt-oss-20b"
DRAFTER_REPO = "z-lab/gpt-oss-20b-DFlash"
CONTINUATION_TOKENS = 32
PROMPT = "Explain in one paragraph why the sky appears blue."


def _hf_home() -> Path:
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        val = os.environ.get(var)
        if val:
            base = Path(val)
            return base / "hub" if var == "XDG_CACHE_HOME" else base
    return Path.home() / ".cache" / "huggingface"


def _checkpoint_present(repo: str) -> bool:
    hub = _hf_home()
    hub = hub if hub.name == "hub" else hub / "hub"
    return (hub / f"models--{repo.replace('/', '--')}").is_dir()


def _skip_reason() -> Optional[str]:
    if not os.environ.get("MOE_DFLASH_GPU"):
        return "MOE_DFLASH_GPU unset (GPU-gated 20B DFlash harness)"
    if not torch.cuda.is_available():
        return "CUDA unavailable (GPU-gated 20B DFlash harness)"
    missing = [
        r for r in (TARGET_REPO, DRAFTER_REPO) if not _checkpoint_present(r)
    ]
    if missing:
        return "checkpoints not present in $HF_HOME: " + ", ".join(missing)
    return None


SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(
    SKIP_REASON is not None, reason=SKIP_REASON or "gpu-gated"
)


def _greedy(model, input_ids, spec=None) -> list[int]:
    kwargs: dict = {"do_sample": False, "max_new_tokens": CONTINUATION_TOKENS}
    if spec is not None:
        kwargs["speculative_draft"] = spec
    out = model.generate(input_ids, **kwargs)
    return [int(t) for t in out[0, input_ids.shape[1] :].tolist()]


def test_20b_native_dflash_losslessness() -> None:
    from transformers import AutoTokenizer

    from moe_infinity import MoE
    from moe_infinity.spec_decode import DFlashSpeculator

    offload = os.environ.get(
        "MOE_DFLASH_OFFLOAD", "/tmp/opencode/moe-offload/gpt-oss-20b"
    )
    os.makedirs(offload, exist_ok=True)
    ratio = float(os.environ.get("MOE_DFLASH_MEM_RATIO", "0.9"))

    tok = AutoTokenizer.from_pretrained(TARGET_REPO, trust_remote_code=True)
    model = MoE(
        TARGET_REPO, {"offload_path": offload, "device_memory_ratio": ratio}
    )
    spec = DFlashSpeculator(model, DRAFTER_REPO)

    assert spec.config.block_size == 8
    assert spec.config.target_layer_ids == [1, 6, 11, 16, 21]

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")

    t0 = time.time()
    plain = _greedy(model, ids)
    plain_s = time.time() - t0

    spec.step_trace = []
    t0 = time.time()
    dfl = _greedy(model, ids, spec=spec)
    dfl_s = time.time() - t0

    assert (
        dfl == plain
    ), "native DFlash greedy must be token-identical to plain greedy"

    trace = getattr(spec, "step_trace", []) or []
    accepts = [getattr(tr, "accept", 0) + 1 for tr in trace]
    mean_accept = sum(accepts) / len(accepts) if accepts else 0.0
    assert mean_accept >= 1.0
    assert plain_s > 0 and dfl_s > 0
