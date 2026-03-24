import gc
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_infinity import MoE


def _get_attn_impl(model_name: str) -> str:
    if "deepseek" in model_name.lower():
        return "eager"
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "eager"


@pytest.fixture(
    scope="session",
    params=[
        "deepseek-ai/DeepSeek-V2-Lite",
        "Qwen/Qwen3-30B-A3B",
    ],
)
def consistency_outputs(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — consistency test requires GPU")

    model_name = request.param
    prompt = os.environ.get(
        "MOE_CONSISTENCY_PROMPT", "The capital of France is"
    )
    attn_impl = _get_attn_impl(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError(
            f"[{model_name}] tokenizer.eos_token_id is required for deterministic generation"
        )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    hf_model.eval()
    with torch.no_grad():
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits.detach().cpu()
        hf_gen = hf_model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
        hf_gen_ids = hf_gen.detach().cpu()

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    offload_base = tmp_path_factory.mktemp("offload")
    offload_dir = Path(offload_base) / model_name.replace("/", "_")
    config = {
        "offload_path": str(offload_dir),
        "device_memory_ratio": 0.75,
    }

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        moe_model = MoE(model_name, config)
    finally:
        torch.set_default_dtype(default_dtype)

    with torch.no_grad():
        _ = moe_model(input_ids)

    with torch.no_grad():
        moe_output = moe_model(input_ids)
        moe_logits = moe_output.logits.detach().cpu()
        moe_gen = moe_model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
        moe_gen_ids = moe_gen.detach().cpu()

    del moe_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "tokenizer": tokenizer,
        "hf_logits": hf_logits,
        "hf_gen_ids": hf_gen_ids,
        "moe_logits": moe_logits,
        "moe_gen_ids": moe_gen_ids,
    }


@pytest.mark.cuda
def test_forward_consistency(consistency_outputs: Dict[str, Any]) -> None:
    model_name = consistency_outputs["model_name"]
    hf_logits = consistency_outputs["hf_logits"]
    moe_logits = consistency_outputs["moe_logits"]

    abs_diff = (moe_logits - hf_logits).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    hf_top = hf_logits.argmax(dim=-1)
    moe_top = moe_logits.argmax(dim=-1)
    print(
        f"\n[{model_name}] Forward — max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )
    print(f"[{model_name}] Argmax match: {torch.equal(hf_top, moe_top)}")
    print(
        f"[{model_name}] Logits shape: hf={hf_logits.shape}, moe={moe_logits.shape}"
    )

    assert (
        hf_logits.shape == moe_logits.shape
    ), f"Shape mismatch: {hf_logits.shape} vs {moe_logits.shape}"
    assert torch.equal(hf_top, moe_top), (
        f"[{model_name}] Argmax mismatch at positions: "
        f"{(hf_top != moe_top).nonzero(as_tuple=True)}"
    )
    torch.testing.assert_close(
        moe_logits,
        hf_logits,
        rtol=1e-3,
        atol=1e-3,
        msg=lambda msg: f"[{model_name}] Logits not close. Max diff: {max_diff:.6f}\n{msg}",
    )


@pytest.mark.cuda
def test_generation_consistency(consistency_outputs: Dict[str, Any]) -> None:
    model_name = consistency_outputs["model_name"]
    tokenizer = consistency_outputs["tokenizer"]
    hf_gen_ids = consistency_outputs["hf_gen_ids"]
    moe_gen_ids = consistency_outputs["moe_gen_ids"]

    hf_text = tokenizer.decode(hf_gen_ids[0], skip_special_tokens=True)
    moe_text = tokenizer.decode(moe_gen_ids[0], skip_special_tokens=True)
    print(f"\n[{model_name}] HF  output: {hf_text!r}")
    print(f"[{model_name}] MoE output: {moe_text!r}")

    assert torch.equal(hf_gen_ids, moe_gen_ids), (
        f"[{model_name}] Generation mismatch!\n"
        f"HF  tokens: {hf_gen_ids[0].tolist()}\n"
        f"MoE tokens: {moe_gen_ids[0].tolist()}\n"
        f"First diff: {(hf_gen_ids != moe_gen_ids).nonzero(as_tuple=True)}"
    )
