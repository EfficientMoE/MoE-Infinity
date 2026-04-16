from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for E2E tests",
    ),
]


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
    model: object, kv_cache_ratio: float
) -> dict[str, object]:
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise RuntimeError("model.config is required")

    num_layers = _resolve_int_attr(
        model_config,
        "num_hidden_layers",
        "num_layers",
        "n_layer",
    )
    num_attention_heads = _resolve_int_attr(
        model_config,
        "num_attention_heads",
        "n_head",
    )
    num_kv_heads = _resolve_int_attr(
        model_config,
        "num_key_value_heads",
        "num_kv_heads",
        "n_head_kv",
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
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": kv_cache_ratio,
        "max_batch_size": 64,
        "max_tokens_per_step": 4096,
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


@pytest.fixture(scope="module")
def integrity_bundle(model_name: str, tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("transformers")
    pytest.importorskip("moe_infinity")

    from transformers import AutoTokenizer

    from moe_infinity import MoE
    from moe_infinity.serving.engine import ContinuousBatchingEngine

    local_model_dir = _require_local_model_dir(model_name)
    offload_path = Path(tmp_path_factory.mktemp("e2e_integrity")) / "offload"
    offload_path.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        moe_model = MoE(
            local_model_dir,
            {
                "offload_path": str(offload_path),
                "device_memory_ratio": 0.75,
            },
        )
    except Exception as exc:
        pytest.skip(f"Unable to initialize model for integrity test: {exc}")

    def build_engine(kv_cache_ratio: float) -> Any:
        return ContinuousBatchingEngine(
            model=moe_model.model,
            engine=moe_model.engine,
            config=_build_engine_config(
                moe_model.model,
                kv_cache_ratio=kv_cache_ratio,
            ),
            tokenizer=tokenizer,
        )

    return tokenizer, build_engine


def _run_and_capture_logits(
    engine: Any,
    prompt_batches: list[list[int]],
    max_new_tokens: int,
) -> tuple[torch.Tensor, int]:
    from moe_infinity.serving.sequence import SamplingParams

    target_seq_id = 0
    swap_count = [0]
    captured_logits: list[torch.Tensor] = []

    original_swap_out = engine.kv_cache.swap_out
    original_extract = engine._extract_last_token_logits

    def counting_swap_out(seq_id: int) -> None:
        swap_count[0] += 1
        original_swap_out(seq_id)

    def capture_last_token_logits(
        logits: torch.Tensor, batch: Any
    ) -> torch.Tensor:
        step_logits = original_extract(logits, batch)
        if target_seq_id in batch.seq_ids:
            target_index = batch.seq_ids.index(target_seq_id)
            captured_logits.append(
                step_logits[target_index]
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
        return step_logits

    engine.eos_token_id = None
    with (
        patch.object(
            engine.kv_cache,
            "swap_out",
            side_effect=counting_swap_out,
        ),
        patch.object(
            engine,
            "_extract_last_token_logits",
            side_effect=capture_last_token_logits,
        ),
    ):
        for req_idx, prompt_ids in enumerate(prompt_batches):
            engine.add_request(
                request_id=f"integrity-{req_idx}",
                prompt_token_ids=list(prompt_ids),
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                ),
            )
        _ = engine.run_until_done()

    assert len(captured_logits) >= max_new_tokens
    logits_tensor = torch.stack(captured_logits[:max_new_tokens], dim=0)
    assert torch.isfinite(logits_tensor).all()
    return logits_tensor, swap_count[0]


def test_kv_integrity_logits_match_after_swap(integrity_bundle) -> None:
    tokenizer, build_engine = integrity_bundle

    baseline_engine = build_engine(kv_cache_ratio=0.8)
    pressure_engine = build_engine(kv_cache_ratio=0.05)

    high_blocks = baseline_engine.kv_cache.num_blocks
    low_blocks = pressure_engine.kv_cache.num_blocks
    block_size = pressure_engine.kv_cache.block_size

    request_count = 4
    blocks_per_request = max(1, low_blocks // request_count + 1)
    required_blocks = request_count * blocks_per_request

    while required_blocks >= high_blocks and request_count > 1:
        request_count -= 1
        blocks_per_request = max(1, low_blocks // request_count + 1)
        required_blocks = request_count * blocks_per_request

    if required_blocks >= high_blocks:
        pytest.skip(
            f"Cannot form no-swap vs swap workload with available KV blocks: "
            f"high={high_blocks}, low={low_blocks}, required={required_blocks}"
        )

    fill_ids = tokenizer("hello", add_special_tokens=False).input_ids
    fill_id = int(fill_ids[0]) if fill_ids else 1
    prompt_len = block_size * blocks_per_request
    prompt_batches = [[fill_id] * prompt_len for _ in range(request_count)]

    torch.manual_seed(20260331)
    torch.cuda.manual_seed_all(20260331)
    logits_no_swap, baseline_swaps = _run_and_capture_logits(
        baseline_engine,
        prompt_batches,
        max_new_tokens=8,
    )

    torch.manual_seed(20260331)
    torch.cuda.manual_seed_all(20260331)
    logits_with_swap, pressure_swaps = _run_and_capture_logits(
        pressure_engine,
        prompt_batches,
        max_new_tokens=8,
    )

    assert baseline_swaps == 0
    assert pressure_swaps > 0
    assert torch.isfinite(logits_no_swap).all()
    assert torch.isfinite(logits_with_swap).all()
    assert torch.allclose(
        logits_no_swap,
        logits_with_swap,
        rtol=1e-3,
        atol=1e-5,
    )
