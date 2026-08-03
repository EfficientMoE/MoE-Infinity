import os

import pytest

pytestmark = pytest.mark.gpu


def _host_ram_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        return 0.0
    return 0.0


@pytest.mark.skipif(
    os.environ.get("MOE_GLM_SMOKE") != "1",
    reason="Set MOE_GLM_SMOKE=1 to run the heavy GLM-5.2 end-to-end smoke (dequant-on-load needs ~1.5TB host RAM).",
)
def test_glm_generate_smoke(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if _host_ram_available_gb() < 1600:
        pytest.skip(
            f"insufficient host RAM ({_host_ram_available_gb():.0f}GB free); "
            "dequant-on-load needs ~1.5TB"
        )

    from transformers import AutoTokenizer

    from moe_infinity import MoE

    model = MoE(
        "zai-org/GLM-5.2-FP8",
        {"offload_path": str(tmp_path / "glm"), "device_memory_ratio": 0.5},
    )
    tok = AutoTokenizer.from_pretrained(
        "zai-org/GLM-5.2-FP8", trust_remote_code=True
    )
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    out = model.generate(ids, max_new_tokens=16)
    text = tok.decode(out[0], skip_special_tokens=True)
    assert len(text) > 0
    assert out.shape[1] >= ids.shape[1] + 1
