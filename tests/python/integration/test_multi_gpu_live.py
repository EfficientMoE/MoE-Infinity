"""Live multi-GPU integration test.

Requires >=2 CUDA GPUs. Tests that MoE-Infinity actually distributes expert
computation across multiple GPUs by monitoring GPU memory allocation.

Run inside Docker:
  docker run --rm --gpus all \
    -v $(pwd):/workspace/MoE-Infinity-dev \
    moe-infinity-test \
    python /workspace/MoE-Infinity-dev/tests/python/integration/test_multi_gpu_live.py
"""

import os
import sys
import time

import torch


def bytes_to_mb(b):
    return b / (1024 * 1024)


def get_gpu_memory_used(device_id):
    return torch.cuda.memory_allocated(device_id)


def get_gpu_memory_reserved(device_id):
    return torch.cuda.memory_reserved(device_id)


def test_multi_gpu_expert_distribution():
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"SKIP: need >=2 GPUs, found {num_gpus}")
        return False

    test_gpus = [0, 1]
    print(f"Testing with GPUs: {test_gpus} (out of {num_gpus} available)")

    for gpu_id in test_gpus:
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(gpu_id)

    baseline_mem = {}
    for gpu_id in test_gpus:
        baseline_mem[gpu_id] = get_gpu_memory_reserved(gpu_id)
        print(
            f"  GPU {gpu_id} baseline reserved: {bytes_to_mb(baseline_mem[gpu_id]):.1f} MB"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("SKIP: transformers not installed")
        return False

    sys.path.insert(0, "/workspace/MoE-Infinity-dev")
    from moe_infinity import MoE

    model_name = "google/switch-base-128"
    offload_dir = "/tmp/moe_multi_gpu_test"
    os.makedirs(offload_dir, exist_ok=True)

    config = {
        "offload_path": offload_dir,
        "device_memory_ratio": 0.5,
    }

    print(f"\nLoading model: {model_name}")
    print(f"Offload dir: {offload_dir}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in test_gpus)

    try:
        model = MoE(model_name, config)
    except Exception as e:
        print(f"FAIL: model load failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    post_load_mem = {}
    for gpu_id in test_gpus:
        post_load_mem[gpu_id] = get_gpu_memory_reserved(gpu_id)
        delta = post_load_mem[gpu_id] - baseline_mem[gpu_id]
        print(
            f"  GPU {gpu_id} after load: {bytes_to_mb(post_load_mem[gpu_id]):.1f} MB "
            f"(+{bytes_to_mb(delta):.1f} MB)"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"FAIL: tokenizer load failed: {e}")
        return False

    print("\nRunning inference...")
    input_text = "Translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
        "cuda:0"
    )

    try:
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=20)
    except Exception as e:
        print(f"FAIL: inference failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Input:  {input_text}")
    print(f"  Output: {output_text}")

    post_infer_mem = {}
    for gpu_id in test_gpus:
        post_infer_mem[gpu_id] = get_gpu_memory_reserved(gpu_id)
        delta = post_infer_mem[gpu_id] - baseline_mem[gpu_id]
        print(
            f"  GPU {gpu_id} after inference: {bytes_to_mb(post_infer_mem[gpu_id]):.1f} MB "
            f"(+{bytes_to_mb(delta):.1f} MB)"
        )

    gpu0_delta = post_infer_mem[0] - baseline_mem[0]
    gpu1_delta = post_infer_mem[1] - baseline_mem[1]

    print(f"\n=== Multi-GPU Distribution Results ===")
    print(f"  GPU 0 memory delta: {bytes_to_mb(gpu0_delta):.1f} MB")
    print(f"  GPU 1 memory delta: {bytes_to_mb(gpu1_delta):.1f} MB")

    if gpu1_delta > 1 * 1024 * 1024:
        print(
            "  PASS: GPU 1 has allocated memory — multi-GPU distribution is working"
        )
        return True
    else:
        print(
            "  FAIL: GPU 1 has no allocated memory — experts are NOT distributed"
        )
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MoE-Infinity Multi-GPU Live Integration Test")
    print("=" * 60)

    success = test_multi_gpu_expert_distribution()

    print("\n" + "=" * 60)
    if success:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
    print("=" * 60)

    sys.exit(0 if success else 1)
