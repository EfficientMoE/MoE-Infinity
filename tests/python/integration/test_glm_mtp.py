import os

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("MOE_GLM_TINY") != "1", reason="set MOE_GLM_TINY=1"
)
def test_glm_mtp_lossless(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from moe_infinity import MoE
    from moe_infinity.spec_decode.glm_mtp import GlmMtpSpeculator
    from tests.python.integration._glm_tiny import build_tiny_glm

    d = build_tiny_glm(str(tmp_path / "tiny"))
    model = MoE(
        d, {"offload_path": str(tmp_path / "off"), "device_memory_ratio": 0.8}
    )
    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    greedy = model.generate(ids, max_new_tokens=12)
    spec = GlmMtpSpeculator(model).generate(
        ids, max_new_tokens=12, temperature=0.0
    )
    assert torch.equal(
        greedy, spec
    ), f"MTP spec-decode NOT lossless:\n greedy={greedy}\n spec  ={spec}"
