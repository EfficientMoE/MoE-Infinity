"""GPU-gated serving-vs-sync DFlash losslessness on gpt-oss-20b."""

from __future__ import annotations

import os
import time
from math import ceil
from pathlib import Path

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


def _skip_reason() -> str | None:
    if not os.environ.get("MOE_DFLASH_GPU"):
        return "MOE_DFLASH_GPU unset (GPU-gated serving DFlash harness)"
    if not torch.cuda.is_available():
        return "CUDA unavailable (GPU-gated serving DFlash harness)"
    missing = [
        r for r in (TARGET_REPO, DRAFTER_REPO) if not _checkpoint_present(r)
    ]
    if missing:
        return "checkpoints not present in $HF_HOME: " + ", ".join(missing)
    return None


def _model_int(config: object, *names: str) -> int:
    get_text = getattr(config, "get_text_config", None)
    text_config = (
        get_text()
        if callable(get_text)
        else getattr(config, "text_config", None)
    )
    for candidate in (config, text_config):
        if candidate is None:
            continue
        for name in names:
            value = getattr(candidate, name, None)
            if isinstance(value, int):
                return value
    raise RuntimeError(f"unable to resolve any of {names!r} from model config")


SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(
    SKIP_REASON is not None, reason=SKIP_REASON or "gpu-gated"
)


def test_serving_dflash_matches_sync_generate() -> None:
    from transformers import AutoTokenizer

    from moe_infinity import MoE
    from moe_infinity.serving.engine import ContinuousBatchingEngine
    from moe_infinity.serving.sequence import SamplingParams
    from moe_infinity.spec_decode import DFlashSpeculator

    offload = os.environ.get(
        "MOE_DFLASH_OFFLOAD", "/tmp/opencode/moe-offload/gpt-oss-20b"
    )
    os.makedirs(offload, exist_ok=True)
    ratio = float(os.environ.get("MOE_DFLASH_MEM_RATIO", "0.9"))

    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_REPO, trust_remote_code=True
    )
    model = MoE(
        TARGET_REPO, {"offload_path": offload, "device_memory_ratio": ratio}
    )
    spec = DFlashSpeculator(model, DRAFTER_REPO)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda:0")

    sync_output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=CONTINUATION_TOKENS,
        speculative_draft=spec,
    )
    sync_tokens = [
        int(token) for token in sync_output[0, input_ids.shape[1] :].tolist()
    ]

    model_config = model.model.config
    block_size = 16
    serving_config: dict[str, object] = {
        "device_memory_ratio": ratio,
        "kv_cache_ratio": 0.01,
        "max_batch_size": 1,
        "max_tokens_per_step": 2048,
        "block_size": block_size,
        "num_layers": _model_int(
            model_config, "num_hidden_layers", "num_layers"
        ),
        "num_kv_heads": _model_int(
            model_config,
            "num_key_value_heads",
            "num_kv_heads",
            "num_attention_heads",
        ),
        "head_dim": _model_int(model_config, "head_dim"),
        "dtype": str(getattr(model.model, "dtype", torch.bfloat16)).replace(
            "torch.", ""
        ),
        "eos_token_id": getattr(model_config, "eos_token_id", None),
        "num_kv_blocks": max(1, ceil(input_ids.shape[1] / block_size)),
    }
    serving = ContinuousBatchingEngine(
        model=model.model,
        engine=model.engine,
        config=serving_config,
        tokenizer=tokenizer,
        speculative_draft=spec,
    )
    serving.add_request(
        request_id="serving-dflash",
        prompt_token_ids=[int(token) for token in input_ids[0].tolist()],
        sampling_params=SamplingParams(
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            max_tokens=CONTINUATION_TOKENS,
        ),
    )

    started = time.perf_counter()
    serving_result = serving.run_until_done()
    elapsed = time.perf_counter() - started
    serving_tokens = serving_result["serving-dflash"]
    assert isinstance(serving_tokens, list)
    assert serving_tokens == sync_tokens

    tok_s = len(serving_tokens) / elapsed
    message = (
        f"SERVING_DFLASH_TOKEN_IDENTICAL=True tokens={len(serving_tokens)}; "
        f"elapsed_s={elapsed:.3f}; tok_s={tok_s:.2f}"
    )
    print(message, flush=True)
    assert tok_s > 0
