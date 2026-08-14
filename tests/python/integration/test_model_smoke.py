import os

import pytest

pytestmark = pytest.mark.gpu

MODELS = [
    (
        "deepseek_v2",
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "MOE_DEEPSEEK_V2_SMOKE",
        {"trust_remote_code": True, "device_memory_ratio": 0.75},
    ),
    (
        "mixtral",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "MOE_MIXTRAL_SMOKE",
        {"trust_remote_code": False, "device_memory_ratio": 0.75},
    ),
    (
        "qwen3",
        "Qwen/Qwen3-30B-A3B",
        "MOE_QWEN3_SMOKE",
        {"trust_remote_code": False, "device_memory_ratio": 0.75},
    ),
    (
        "qwen3_5",
        "Qwen/Qwen3.5-35B-A3B",
        "MOE_QWEN3_5_SMOKE",
        {"trust_remote_code": True, "device_memory_ratio": 0.75},
    ),
    (
        "gpt_oss",
        "openai/gpt-oss-20b",
        "MOE_GPT_OSS_SMOKE",
        {"trust_remote_code": False, "device_memory_ratio": 0.75},
    ),
    (
        "dbrx",
        "databricks/dbrx-instruct",
        "MOE_DBRX_SMOKE",
        {"trust_remote_code": True, "device_memory_ratio": 0.75},
    ),
    (
        "jamba",
        "ai21labs/Jamba-v0.1",
        "MOE_JAMBA_SMOKE",
        {"trust_remote_code": True, "device_memory_ratio": 0.75},
    ),
    (
        "olmoe",
        "allenai/OLMoE-1B-7B-0924-Instruct",
        "MOE_OLMOE_SMOKE",
        {"trust_remote_code": False, "device_memory_ratio": 0.75},
    ),
]


@pytest.mark.parametrize(
    ("model_id", "checkpoint", "env_var", "options"),
    MODELS,
    ids=[model[0] for model in MODELS],
)
def test_generate_smoke(model_id, checkpoint, env_var, options, tmp_path):
    if os.environ.get(env_var) != "1":
        pytest.skip(
            f"Set {env_var}=1 to run the heavy {model_id} end-to-end smoke."
        )

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from transformers import AutoTokenizer

    from moe_infinity import MoE

    model = MoE(
        checkpoint,
        {
            "offload_path": str(tmp_path / model_id),
            "device_memory_ratio": options["device_memory_ratio"],
        },
    )
    tok = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=options["trust_remote_code"]
    )
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    out = model.generate(ids, max_new_tokens=16)
    text = tok.decode(out[0], skip_special_tokens=True)
    assert out.shape[1] >= ids.shape[1] + 1
    assert len(text) > 0
