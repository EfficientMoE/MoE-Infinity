"""Diagnostic: compare weights loaded via MoE-Infinity vs direct safetensors.

Run inside Docker container with GPU:
  python tests/python/debug/diagnose_weight_integrity.py

This script checks whether the archer_engine correctly loads weights by:
1. Loading a few key tensors directly from safetensors (ground truth)
2. Loading the model through MoE-Infinity
3. Triggering begin/end on specific parameters to load them from offload
4. Comparing the loaded values against ground truth
"""

import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig

MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"


def load_ground_truth_tensors(model_name, target_keys):
    """Load specific tensors directly from safetensors checkpoint."""
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(model_name)
    index_path = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        weight_map = None

    results = {}
    seen_files = set()
    for key in target_keys:
        if weight_map:
            shard_file = weight_map.get(key)
            if shard_file is None:
                print(f"  [WARN] {key} not in weight_map")
                continue
            shard_path = os.path.join(model_path, shard_file)
        else:
            shard_path = os.path.join(model_path, "model.safetensors")

        if shard_path not in seen_files:
            seen_files.add(shard_path)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if key in f.keys():
                results[key] = f.get_tensor(key).clone()
            else:
                print(f"  [WARN] {key} not found in {shard_path}")

    return results


def load_moe_model(model_name, offload_path):
    """Load model through MoE-Infinity."""
    from moe_infinity import MoE

    config = {"offload_path": offload_path, "device_memory_ratio": 0.75}
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = MoE(model_name, config)
    finally:
        torch.set_default_dtype(default_dtype)
    return model


def extract_moe_param(model, param_name):
    """Get a named parameter from the MoE model, triggering archer load."""
    for name, param in model.model.named_parameters():
        if name == param_name:
            return param.data.clone().cpu()
    return None


def main():
    print("=" * 70)
    print("MoE-Infinity Weight Integrity Diagnostic")
    print("=" * 70)

    target_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.down_proj.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
    ]

    print("\n[1/3] Loading ground truth from safetensors...")
    ground_truth = load_ground_truth_tensors(MODEL_NAME, target_keys)
    print(f"  Loaded {len(ground_truth)} tensors")
    for k, v in ground_truth.items():
        print(
            f"    {k}: shape={v.shape}, dtype={v.dtype}, "
            f"norm={v.float().norm():.4f}, "
            f"first5={v.flatten()[:5].tolist()}"
        )

    offload_path = "/tmp/diag_offload"
    print(
        f"\n[2/3] Loading model through MoE-Infinity (offload={offload_path})..."
    )
    model = load_moe_model(MODEL_NAME, offload_path)

    print("\n[3/3] Comparing weights...")
    print("-" * 70)

    dummy_input = torch.zeros(1, 1, dtype=torch.long, device="cuda:0")
    dummy_input[0, 0] = 1

    with torch.no_grad():
        _ = model(dummy_input)

    mismatches = 0
    for key in target_keys:
        if key not in ground_truth:
            print(f"  SKIP {key} (no ground truth)")
            continue

        gt = ground_truth[key]
        param_name = (
            key.replace("model.", "", 1) if key.startswith("model.") else key
        )

        moe_val = extract_moe_param(model, key)
        if moe_val is None:
            moe_val = extract_moe_param(model, param_name)
        if moe_val is None:
            print(f"  SKIP {key} (not found in MoE model)")
            continue

        gt_f = gt.float()
        moe_f = moe_val.float()

        if gt_f.shape != moe_f.shape:
            print(
                f"  FAIL {key}: shape mismatch gt={gt_f.shape} vs moe={moe_f.shape}"
            )
            mismatches += 1
            continue

        max_diff = (gt_f - moe_f).abs().max().item()
        mean_diff = (gt_f - moe_f).abs().mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            gt_f.flatten().unsqueeze(0), moe_f.flatten().unsqueeze(0)
        ).item()

        status = "OK" if max_diff < 1e-3 else "FAIL"
        if status == "FAIL":
            mismatches += 1

        print(
            f"  {status} {key}: max_diff={max_diff:.6f}, "
            f"mean_diff={mean_diff:.6f}, cosine_sim={cosine_sim:.6f}"
        )
        if status == "FAIL":
            print(f"       gt_first5={gt.flatten()[:5].tolist()}")
            print(f"      moe_first5={moe_val.flatten()[:5].tolist()}")

    print("-" * 70)
    if mismatches == 0:
        print("RESULT: All weights match — issue is NOT weight corruption")
        print(
            "        Root cause likely in C++ expert dispatch or accumulation"
        )
    else:
        print(
            f"RESULT: {mismatches} weight mismatches found — WEIGHT CORRUPTION CONFIRMED"
        )

    return 1 if mismatches > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
