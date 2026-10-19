import os

import pytest

pytestmark = pytest.mark.gpu

SMOKE_MODEL = os.environ.get("MOE_GLM_SMOKE_MODEL", "zai-org/GLM-5.2-FP8")


@pytest.mark.skipif(
    os.environ.get("MOE_GLM_SMOKE") != "1",
    reason="Set MOE_GLM_SMOKE=1 to run the heavy GLM end-to-end smoke "
    "(MOE_GLM_SMOKE_MODEL selects the checkpoint).",
)
def test_glm_generate_smoke(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from transformers import AutoTokenizer

    from moe_infinity import MoE

    model = MoE(
        SMOKE_MODEL,
        {"offload_path": str(tmp_path / "glm"), "device_memory_ratio": 0.5},
    )
    tok = AutoTokenizer.from_pretrained(SMOKE_MODEL, trust_remote_code=True)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    out = model.generate(ids, max_new_tokens=16)
    text = tok.decode(out[0], skip_special_tokens=True)
    assert len(text) > 0
    assert out.shape[1] >= ids.shape[1] + 1


@pytest.mark.skipif(
    os.environ.get("MOE_GLM_SMOKE") != "1",
    reason="Set MOE_GLM_SMOKE=1 for the heavy GLM 32k-prefill test "
    "(MOE_GLM_SMOKE_MODEL selects the checkpoint).",
)
def test_glm_long_context_prefill(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from moe_infinity import MoE

    model = MoE(
        SMOKE_MODEL,
        {"offload_path": str(tmp_path / "glm"), "device_memory_ratio": 0.4},
    )
    ids = torch.randint(0, 1000, (1, 32768)).cuda()
    out = model.generate(ids, max_new_tokens=4)
    assert out.shape[1] >= ids.shape[1] + 1
