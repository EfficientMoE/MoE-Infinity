#!/usr/bin/env python3
"""
GPT-OSS 20b End-to-End integration test for MoE-Infinity.

Requires: CUDA GPU with >=16GB VRAM, network access for model download.

Run standalone:
    python tests/docker/test_gpt_oss_e2e.py

Run via Docker:
    docker run --rm --gpus '"device=0"' \
        -v $(pwd):/workspace/MoE-Infinity \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -e HF_HOME=/root/.cache/huggingface \
        -w /workspace/MoE-Infinity \
        moe-infinity-test:latest \
        bash -c "pip install --no-build-isolation -e . -q && python tests/docker/test_gpt_oss_e2e.py"

Run via pytest (inside Docker):
    pytest tests/docker/test_gpt_oss_e2e.py -v -m cuda
"""

import os
import sys
import time

import pytest
import torch

CHECKPOINT = os.environ.get("GPT_OSS_CHECKPOINT", "openai/gpt-oss-20b")
OFFLOAD_PATH = os.environ.get(
    "GPT_OSS_OFFLOAD_PATH", "/tmp/gpt-oss-e2e-offload"
)
DEVICE_MEMORY_RATIO = float(
    os.environ.get("GPT_OSS_DEVICE_MEMORY_RATIO", "0.75")
)


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ── Test 1: Config parsing ──────────────────────────────────────────────────


@pytest.mark.cuda
def test_gpt_oss_config_parsing():
    """GPT-OSS config correctly parsed: 24 layers, 32 experts."""
    _skip_if_no_cuda()
    from transformers import AutoConfig

    from moe_infinity.common.constants import MODEL_MAPPING_NAMES
    from moe_infinity.utils.hf_config import parse_moe_param

    config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
    layers, experts, enc = parse_moe_param(config)
    assert layers == 24, f"Expected 24 layers, got {layers}"
    assert experts == 32, f"Expected 32 experts, got {experts}"
    assert enc == 0

    architectures = getattr(config, "architectures", None)
    assert architectures, "Config architectures missing"
    arch_str = architectures[0].lower()
    matched = next((k for k in MODEL_MAPPING_NAMES if k in arch_str), None)
    assert matched == "gptoss", f"Architecture not matched: {matched}"


# ── Test 2: Model loading ───────────────────────────────────────────────────


@pytest.mark.cuda
def test_gpt_oss_model_loading():
    """GPT-OSS 20b loads via MoE() entrypoint without error."""
    _skip_if_no_cuda()
    os.makedirs(OFFLOAD_PATH, exist_ok=True)

    from moe_infinity import MoE

    t0 = time.time()
    model = MoE(
        CHECKPOINT,
        {
            "offload_path": OFFLOAD_PATH,
            "device_memory_ratio": DEVICE_MEMORY_RATIO,
        },
    )
    load_time = time.time() - t0
    print(f"\n  Load time: {load_time:.1f}s")
    assert model is not None


# ── Test 3: Text generation ─────────────────────────────────────────────────


@pytest.mark.cuda
def test_gpt_oss_text_generation():
    """GPT-OSS 20b generates coherent text."""
    _skip_if_no_cuda()
    os.makedirs(OFFLOAD_PATH, exist_ok=True)

    from transformers import AutoTokenizer

    from moe_infinity import MoE

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = MoE(
        CHECKPOINT,
        {
            "offload_path": OFFLOAD_PATH,
            "device_memory_ratio": DEVICE_MEMORY_RATIO,
        },
    )

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=32)
    gen_time = time.time() - t0

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n  Prompt:    '{prompt}'")
    print(f"  Generated: '{output_text}'")
    print(f"  Time:      {gen_time:.1f}s")
    print(f"  Peak GPU:  {peak_mb:.0f} MB")

    assert len(output_text) > len(prompt), "Model should generate new tokens"


# ── Test 4: Memory reduction ───────────────────────────────────────────────


@pytest.mark.cuda
def test_gpt_oss_memory_reduction():
    """Expert offloading should keep GPU memory well below full model size."""
    _skip_if_no_cuda()
    os.makedirs(OFFLOAD_PATH, exist_ok=True)

    from moe_infinity import MoE

    torch.cuda.reset_peak_memory_stats()
    model = MoE(
        CHECKPOINT,
        {
            "offload_path": OFFLOAD_PATH,
            "device_memory_ratio": 0.5,
        },
    )

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

    print(f"\n  Peak GPU memory:  {peak_mb:.0f} MB")
    print(f"  GPU total memory: {gpu_total_mb:.0f} MB")

    # Full model bf16 is ~41GB. With offloading on a 24GB GPU it must fit.
    assert peak_mb < gpu_total_mb * 0.95, (
        f"Peak memory {peak_mb:.0f}MB is near GPU capacity {gpu_total_mb:.0f}MB — "
        "offloading may not be working"
    )


# ── Standalone runner ───────────────────────────────────────────────────────


def main():
    """Run all tests as a standalone script with pretty output."""
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA not available. Run inside a GPU-enabled Docker container."
        )
        return 1

    print("=" * 60)
    print("  GPT-OSS 20b End-to-End Test")
    print("=" * 60)
    print(f"  CUDA:         {torch.cuda.get_device_name(0)}")
    print(
        f"  GPU Memory:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  Checkpoint:   {CHECKPOINT}")
    print(f"  Offload path: {OFFLOAD_PATH}")
    print(f"  Memory ratio: {DEVICE_MEMORY_RATIO}")
    print("=" * 60)

    tests = [
        ("Config parsing", test_gpt_oss_config_parsing),
        ("Model loading", test_gpt_oss_model_loading),
        ("Text generation", test_gpt_oss_text_generation),
        ("Memory reduction", test_gpt_oss_memory_reduction),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n[TEST] {name}...")
        try:
            test_fn()
            print(f"  ✅ {name} PASS")
            results.append((name, True, None))
        except Exception as e:
            print(f"  ❌ {name} FAIL: {e}")
            results.append((name, False, str(e)))

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_pass = True
    for name, passed, err in results:
        status = "PASS" if passed else f"FAIL: {err}"
        print(f"  {'✅' if passed else '❌'} {name}: {status}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
