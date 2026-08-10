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

    try:
        payload = json.dumps(
            {"model": model_dir, "prompt": "hello", "max_tokens": 8}
        ).encode()
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())

        assert "choices" in result, f"No choices in response: {result}"
        assert len(result["choices"]) > 0
        text = result["choices"][0].get("text", "")
        assert (
            isinstance(text, str) and len(text) > 0
        ), f"Expected non-empty text, got: {result}"

        chat_payload = json.dumps(
            {
                "model": model_dir,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            }
        ).encode()
        chat_req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=chat_payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(chat_req, timeout=60) as resp:
            chat_result = json.loads(resp.read())

        assert (
            "choices" in chat_result
        ), f"No choices in chat response: {chat_result}"
        assert len(chat_result["choices"]) > 0
        msg = chat_result["choices"][0].get("message", {})
        assert (
            isinstance(msg.get("content"), str) and len(msg["content"]) > 0
        ), f"Expected non-empty message content, got: {chat_result}"
        assert (
            "finish_reason" in chat_result["choices"][0]
        ), f"Missing finish_reason: {chat_result}"

        stream_payload = json.dumps(
            {
                "model": model_dir,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "stream": True,
            }
        ).encode()
        stream_req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=stream_payload,
            headers={"Content-Type": "application/json"},
        )
        data_chunks = []
        with urllib.request.urlopen(stream_req, timeout=60) as resp:
            deadline_stream = time.monotonic() + 30
            for raw_line in resp:
                if time.monotonic() > deadline_stream:
                    break
                line = raw_line.decode("utf-8").strip()
                if line.startswith("data:"):
                    chunk = line[len("data:") :].strip()
                    if chunk and chunk != "[DONE]":
                        data_chunks.append(chunk)

        assert (
            len(data_chunks) > 0
        ), "No SSE data chunks received from streaming endpoint"

    finally:
        proc.kill()
        proc.wait()
