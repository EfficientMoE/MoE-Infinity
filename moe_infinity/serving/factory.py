"""Supported in-process entry points for the continuous-batching engine.

This is the replacement for the deprecated ``MoE.generate()``: it runs the same
``ContinuousBatchingEngine`` that powers the OpenAI-compatible server, fully
in-process and synchronously, returning token lists instead of speaking HTTP.
"""

from __future__ import annotations

from typing import Optional

import torch

from .engine import ContinuousBatchingEngine
from .sequence import SamplingParams


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


def build_serving_config(moe: object, **overrides: object) -> dict[str, object]:
    """Derive a ``ContinuousBatchingEngine`` config dict from a loaded ``MoE``.

    Keyword overrides win over derived defaults. Raises ``RuntimeError`` when a
    required model-config field cannot be resolved.
    """
    model = moe.model
    model_config = model.config
    config: dict[str, object] = {
        "device_memory_ratio": float(
            getattr(moe.engine_config, "device_memory_ratio", 0.75)
        ),
        "kv_cache_ratio": 0.25,
        "max_batch_size": 1,
        "block_size": 16,
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
        "dtype": str(getattr(model, "dtype", torch.bfloat16)).replace(
            "torch.", ""
        ),
        "eos_token_id": getattr(model_config, "eos_token_id", None),
    }
    config.update(overrides)
    return config


def create_serving_engine(
    moe: object,
    tokenizer: Optional[object] = None,
    speculative_draft: Optional[object] = None,
    **overrides: object,
) -> ContinuousBatchingEngine:
    """Build an in-process ``ContinuousBatchingEngine`` for a loaded ``MoE``."""
    return ContinuousBatchingEngine(
        model=moe.model,
        engine=moe.engine,
        config=build_serving_config(moe, **overrides),
        tokenizer=tokenizer,
        speculative_draft=speculative_draft,
    )


def generate_tokens(
    engine: ContinuousBatchingEngine,
    prompts_token_ids: list[list[int]],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> list[list[int]]:
    """Run prompts to completion synchronously; returns NEW tokens per prompt.

    The prompt is NOT re-prepended (unlike the deprecated ``MoE.generate()``).
    """
    for index, prompt in enumerate(prompts_token_ids):
        engine.add_request(
            request_id=f"inproc-{index}",
            prompt_token_ids=[int(t) for t in prompt],
            sampling_params=SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_new_tokens,
            ),
        )
    results = engine.run_until_done()
    return [
        list(results[f"inproc-{index}"])
        for index in range(len(prompts_token_ids))
    ]
