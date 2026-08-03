import os
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(os.environ.get("MOE_GLM_TINY") != "1", reason="set MOE_GLM_TINY=1 to run tiny-GLM GPU harness")
def test_tiny_glm_generates(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tests.python.integration._glm_tiny import build_tiny_glm
    from moe_infinity import MoE

    d = build_tiny_glm(str(tmp_path / "tiny"))
    model = MoE(d, {"offload_path": str(tmp_path / "off"), "device_memory_ratio": 0.8})
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    out = model.generate(ids, max_new_tokens=8)
    assert out.shape[1] >= ids.shape[1] + 1
