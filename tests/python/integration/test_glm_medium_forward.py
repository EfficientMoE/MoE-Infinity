import os
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.gpu

_WORKER = r"""
import sys, os
os.environ.setdefault("HF_HUB_CACHE", "/mnt/raid0nvme0/public/huggingface/hub")
import torch
from moe_infinity import MoE

ckpt, offload_path = sys.argv[1], sys.argv[2]
m = MoE(ckpt, {"offload_path": offload_path, "device_memory_ratio": 0.5})

last = None
for name, mod in m.model.named_modules():
    if name.endswith("q_a_layernorm") and hasattr(mod, "weight"):
        last = int(mod.weight.shape[-1])
        break
print("QALN_LASTDIM=%s" % last, flush=True)

ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda:0")
out = m.generate(ids, max_new_tokens=4)
assert out.shape[1] >= ids.shape[1] + 1
print("GEN_OK", flush=True)
"""


@pytest.mark.skipif(
    os.environ.get("MOE_GLM_MEDIUM") != "1",
    reason="set MOE_GLM_MEDIUM=1 to run the medium-GLM resident-weight mis-map repro",
)
def test_medium_glm_resident_weight_forward(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from tests.python.integration._glm_medium import build_medium_glm_fp8

    ckpt = build_medium_glm_fp8(str(tmp_path / "med"))
    offload = str(tmp_path / "off")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_WORKER)
        worker = f.name

    env = dict(os.environ)
    env.setdefault("HF_HUB_CACHE", "/mnt/raid0nvme0/public/huggingface/hub")
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    proc = subprocess.run(
        [sys.executable, worker, ckpt, offload],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    os.unlink(worker)

    combined = proc.stdout + "\n" + proc.stderr
    assert "GEN_OK" in proc.stdout, (
        "medium GLM forward did not complete; resident-weight mis-map likely.\n"
        f"returncode={proc.returncode}\n{combined[-3000:]}"
    )
    assert (
        "QALN_LASTDIM=2048" in proc.stdout
    ), f"q_a_layernorm loaded with wrong last dim.\n{combined[-2000:]}"
