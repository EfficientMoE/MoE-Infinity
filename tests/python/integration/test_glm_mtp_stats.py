import os
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(os.environ.get("MOE_GLM_TINY") != "1", reason="set MOE_GLM_TINY=1")
def test_mtp_stats_and_parity(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from tests.python.integration._glm_tiny import build_tiny_glm
    from moe_infinity import MoE
    from moe_infinity.spec_decode.glm_mtp import GlmMtpSpeculator

    d = build_tiny_glm(str(tmp_path / "tiny"))
    model = MoE(d, {"offload_path": str(tmp_path / "off"), "device_memory_ratio": 0.8})
    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    greedy = model.generate(ids, max_new_tokens=10)
    spec = GlmMtpSpeculator(model)
    out = spec.generate(ids, max_new_tokens=10, temperature=0.0)
    assert torch.equal(greedy, out)
    st = spec.last_stats
    assert st["steps"] >= 1
    assert st["mean_accept_len"] >= 1.0
    assert isinstance(st["per_step_accepted"], list)
    assert len(st["per_step_accepted"]) == st["steps"]
    assert all(v in (0, 1) for v in st["per_step_accepted"])
    assert st["accepted"] == sum(st["per_step_accepted"])
