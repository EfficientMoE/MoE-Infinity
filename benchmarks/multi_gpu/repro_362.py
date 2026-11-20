# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Repro for GitHub issue #362: multi-GPU expert offload, device-mismatch throw.

Loads an MoE checkpoint through the MoE-Infinity offload path with 1..N visible
GPUs, decodes greedily (batch 1) over a fixed prompt list, and writes the
generated token ids per prompt to a JSON file.  Exits 1 on any exception and
prints its text + traceback.

How layers and experts end up on both GPUs (no ``device_map`` is involved; the
``MoE`` class does not take one):
  * ``--devices 0,1`` sets ``CUDA_VISIBLE_DEVICES`` *before* torch is imported
    (setting it afterwards, as tests/python/integration/test_multi_gpu_live.py
    does, has no effect), so the engine sees ``torch.cuda.device_count() == 2``.
  * The engine builds its dispatch topology from the model (``get_topology``);
    each topology node (one dense block, or one MoE layer's expert group) gets a
    ``default_device`` across the visible GPUs, and the archer forward hooks
    copy the hidden states to that node's device before running it.  The
    resulting per-node device list is printed as ``TOPOLOGY ...`` and stored in
    the JSON ("topology_devices") so you can check that layers are spread.
  * Expert weights live in pinned host memory and are fetched on demand /
    prefetched into a per-GPU expert cache sized by ``device_memory_ratio``
    (small ratios, 0.1-0.2, force constant fetches on both devices, with the
    ``lcnu`` eviction policy by default).  ``resident_experts`` and
    ``expert_h2d_bytes_total`` are printed after decode as evidence that
    experts were fetched while decoding.

Usage (run from outside the repo checkout cwd is not required):
  python benchmarks/multi_gpu/repro_362.py --devices 0,1 --out /tmp/two.json
  python benchmarks/multi_gpu/repro_362.py --devices 0   --out /tmp/one.json
Env knobs of interest (read by the C++ engine): MOE_INLINE_WAIT=0,
MOE_GROUPED_DECODE=0.
"""

import argparse
import json
import os
import sys
import traceback

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue.",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea, there lived",
    "The history of the Roman Empire began when",
    "In machine learning, a mixture-of-experts model works by",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0125-Instruct")
    p.add_argument("--devices", default="0,1", help="CUDA_VISIBLE_DEVICES list")
    # Ratio of *each GPU's* memory used for the expert cache.  The reporter's
    # 0.1-0.2 (0.15 here) was on 16 GB cards.  On 96 GB cards it holds all
    # 13.8 GB of OLMoE experts after warm-up, which makes single-GPU decode
    # bitwise reproducible run to run; 0.035 (~3.4 GB/GPU) keeps the cache
    # under constant eviction/fetch churn but single-GPU output is then not
    # run-to-run reproducible on this host (default env, grouped decode on).
    p.add_argument("--device-memory-ratio", type=float, default=0.15)
    p.add_argument("--cache-policy", default="lcnu")
    p.add_argument("--num-prompts", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--offload-path",
        default=os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "opencode", "issue362", "offload"
        ),
    )
    p.add_argument(
        "--native-engine",
        choices=["default", "0", "1"],
        default="default",
        help="override ArcherConfig use_native_engine (default: engine default)",
    )
    p.add_argument("--out", required=True, help="JSON output path")
    return p.parse_args()


def main():
    args = parse_args()
    # Must happen before torch is imported anywhere.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    assert "torch" not in sys.modules

    import torch
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    prompts = (PROMPTS * ((args.num_prompts // len(PROMPTS)) + 1))[
        : args.num_prompts
    ]
    ndev = torch.cuda.device_count()
    print(
        f"CONFIG model={args.model} CUDA_VISIBLE_DEVICES={args.devices} "
        f"visible={ndev} ratio={args.device_memory_ratio} "
        f"policy={args.cache_policy} prompts={len(prompts)} "
        f"new_tokens={args.max_new_tokens} "
        f"MOE_INLINE_WAIT={os.environ.get('MOE_INLINE_WAIT')} "
        f"MOE_GROUPED_DECODE={os.environ.get('MOE_GROUPED_DECODE')}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs(args.offload_path, exist_ok=True)
    config = {
        "offload_path": args.offload_path,
        "device_memory_ratio": args.device_memory_ratio,
        "expert_eviction_policy": args.cache_policy,
    }
    if args.native_engine != "default":
        config["use_native_engine"] = args.native_engine == "1"
    moe = MoE(args.model, config)
    eng = moe.engine
    handle = getattr(eng, "archer_engine", None)

    topo_devices = None
    try:
        snap = handle.get_topology_snapshot()
        topo_devices = [int(d) for (_c, _sparse, d) in snap]
        sparse_devs = [int(d) for (_c, s, d) in snap if s]
        print(
            f"TOPOLOGY nodes={len(snap)} sparse_nodes={len(sparse_devs)} "
            f"devices_all={sorted(set(topo_devices))} "
            f"sparse_per_device="
            f"{ {d: sparse_devs.count(d) for d in sorted(set(sparse_devs))} } "
            f"sequence={topo_devices}",
            flush=True,
        )
    except Exception as e:  # diagnostics only
        print(f"TOPOLOGY unavailable: {e!r}", flush=True)

    embed_dev = moe.model.get_input_embeddings().weight.device
    print(f"INPUT device (embed_tokens) = {embed_dev}", flush=True)

    def stats(tag):
        out = {}
        for name in (
            "get_resident_expert_count",
            "get_expert_h2d_bytes_total",
            "get_expert_occupancy_bytes",
        ):
            try:
                out[name] = int(getattr(handle, name)())
            except Exception as e:
                out[name] = f"unavailable: {e!r}"
        try:
            out["hit_rate"] = float(handle.get_hit_rate())
        except Exception:
            pass
        mem = {}
        for d in range(ndev):
            free, total = torch.cuda.mem_get_info(d)
            mem[d] = round((total - free) / 2**20)
        out["gpu_used_MiB_visible_idx"] = mem
        print(f"STATS[{tag}] {out}", flush=True)
        return out

    stats_before = stats("before_decode")

    results = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            try:
                ids = tok(prompt, return_tensors="pt").input_ids.to(embed_dev)
                out = moe.generate(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            except Exception:
                print(f"FAILED in prompt {i}", flush=True)
                stats("on_exception")
                raise
            new = out[0, ids.shape[1] :].tolist()
            results.append({"prompt": prompt, "token_ids": new})
            print(
                f"PROMPT {i}: {len(new)} new tokens: "
                f"{tok.decode(new, skip_special_tokens=True)[:80]!r}",
                flush=True,
            )
    stats_after = stats("after_decode")

    with open(args.out, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "env": {
                    k: os.environ.get(k)
                    for k in ("MOE_INLINE_WAIT", "MOE_GROUPED_DECODE")
                },
                "topology_devices": topo_devices,
                "stats_before": stats_before,
                "stats_after": stats_after,
                "results": results,
            },
            f,
            indent=1,
        )
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, SystemExit) and not e.code:
            raise
        print(f"EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)  # skip engine teardown hangs after a failure
