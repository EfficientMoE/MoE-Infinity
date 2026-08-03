import json
import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("MOE_GLM_TINY") != "1",
    reason="Set MOE_GLM_TINY=1 to run the GLM tiny serving test.",
)
def test_glm_serving_completions(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from tests.python.integration._glm_tiny import build_tiny_glm

    model_dir = str(tmp_path / "glm_tiny_srv")
    offload_dir = str(tmp_path / "glm_tiny_off")
    build_tiny_glm(model_dir)

    port = 8019
    env = {**os.environ, "MOE_GLM_TINY": "1"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "moe_infinity.entrypoints.openai.api_server_v2",
            "--model",
            model_dir,
            "--offload-dir",
            offload_dir,
            "--device-memory-ratio",
            "0.8",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    import urllib.error
    import urllib.request

    deadline = time.monotonic() + 120
    healthy = False
    while time.monotonic() < deadline:
        time.sleep(2)
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=2
            ) as resp:
                body = json.loads(resp.read())
                if body.get("status") == "healthy":
                    healthy = True
                    break
        except Exception:
            pass

    if not healthy:
        proc.kill()
        proc.wait()
        pytest.fail("GLM serving did not become healthy within 120s")

    payload = json.dumps(
        {"model": model_dir, "prompt": "hello", "max_tokens": 8}
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except Exception as exc:
        proc.kill()
        proc.wait()
        pytest.fail(f"Completion request failed: {exc}")

    proc.kill()
    proc.wait()

    assert "choices" in result, f"No choices in response: {result}"
    assert len(result["choices"]) > 0
    text = result["choices"][0].get("text", "")
    assert isinstance(text, str) and len(text) > 0, (
        f"Expected non-empty text, got: {result}"
    )
