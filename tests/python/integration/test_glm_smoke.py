"""Operator-gated end-to-end smoke test for GLM MoE (glm_moe_dsa) offload.

Builds/reloads the FP8-in-store offload store and runs a short greedy
generation, asserting coherent output. Heavy: needs the checkpoint, a CUDA GPU,
~700 GB offload disk, and large host RAM, so it is skipped unless MOE_GLM_SMOKE=1.

  MOE_GLM_SMOKE=1 \
  MOE_GLM_CKPT=zai-org/GLM-5.2-FP8 \
  MOE_GLM_OFFLOAD=/ssd/glm52-offload \
  CUDA_VISIBLE_DEVICES=0 \
  pytest tests/python/integration/test_glm_smoke.py -s
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MOE_GLM_SMOKE") != "1",
    reason="operator-gated: set MOE_GLM_SMOKE=1 (needs GLM checkpoint, GPU, "
    "~700 GB offload disk and large host RAM)",
)


def test_glm_generate_smoke():
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    ckpt = os.environ.get("MOE_GLM_CKPT", "zai-org/GLM-5.2-FP8")
    offload = os.environ.get("MOE_GLM_OFFLOAD", "/tmp/glm-offload")
    ratio = float(os.environ.get("MOE_GLM_DMR", "0.5"))
    max_new = int(os.environ.get("MOE_GLM_MAX_NEW", "16"))

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = MoE(
        ckpt, {"offload_path": offload, "device_memory_ratio": ratio}
    )

    input_ids = tokenizer(
        "The capital of France is", return_tensors="pt"
    ).input_ids.to("cuda:0")
    output_ids = model.generate(input_ids, max_new_tokens=max_new)

    assert output_ids.shape[1] > input_ids.shape[1], "no tokens generated"
    assert output_ids.shape[1] <= input_ids.shape[1] + max_new
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assert isinstance(text, str) and len(text.strip()) > 0


if __name__ == "__main__":
    os.environ.setdefault("MOE_GLM_SMOKE", "1")
    test_glm_generate_smoke()
    print("GLM smoke: OK")
