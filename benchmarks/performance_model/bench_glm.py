from __future__ import annotations

import argparse
import csv
import os

import torch


def measure_tiny_glm(tmp_dir: str, gen_len: int = 16) -> dict:
    from tests.python.integration._glm_tiny import build_tiny_glm
    from moe_infinity import MoE
    from moe_infinity.spec_decode.glm_mtp import GlmMtpSpeculator
    from benchmarks.performance_model.model_config import extract_model_params
    from benchmarks.performance_model.roofline import predict_decode
    from benchmarks.performance_model.types import WorkloadPoint

    ckpt_dir = os.path.join(tmp_dir, "tiny_glm_ckpt")
    off_dir = os.path.join(tmp_dir, "tiny_glm_off")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(off_dir, exist_ok=True)

    build_tiny_glm(ckpt_dir)

    model = MoE(ckpt_dir, {"offload_path": off_dir, "device_memory_ratio": 0.8})

    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    batch = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=gen_len)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=gen_len)
    end_evt.record()
    torch.cuda.synchronize()

    elapsed_ms = start_evt.elapsed_time(end_evt)
    decode_tok_s = gen_len / (elapsed_ms / 1000.0)
    peak_mem_bytes = torch.cuda.max_memory_allocated()

    spec = GlmMtpSpeculator(model)
    mtp_input = input_ids.clone()

    mtp_start = torch.cuda.Event(enable_timing=True)
    mtp_end = torch.cuda.Event(enable_timing=True)

    mtp_start.record()
    spec.generate(mtp_input, max_new_tokens=gen_len, temperature=0.0)
    mtp_end.record()
    torch.cuda.synchronize()

    mtp_elapsed_ms = mtp_start.elapsed_time(mtp_end)
    mtp_tok_s = gen_len / (mtp_elapsed_ms / 1000.0)
    mean_accept_len = spec.last_stats.get("mean_accept_len", 1.0)

    params = extract_model_params(ckpt_dir)
    wp = WorkloadPoint(batch=batch, seq_len=seq_len, gen_len=gen_len)
    demand = predict_decode(params, wp)

    return {
        "model": "tiny_glm",
        "batch": batch,
        "seq_len": seq_len,
        "gen_len": gen_len,
        "decode_tok_s": decode_tok_s,
        "mtp_tok_s": mtp_tok_s,
        "mean_accept_len": mean_accept_len,
        "peak_mem_bytes": peak_mem_bytes,
        "pred_flops_per_token": demand.flops_per_token,
        "pred_hbm_bytes_per_token": demand.hbm_bytes_per_token,
        "pred_bound": demand.bound,
    }


def run(out_csv: str, quick: bool = True, gen_len: int = 16) -> None:
    import tempfile

    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        row = measure_tiny_glm(tmp_dir, gen_len=gen_len)

    fieldnames = [
        "model", "batch", "seq_len", "gen_len",
        "decode_tok_s", "mtp_tok_s", "mean_accept_len", "peak_mem_bytes",
        "pred_flops_per_token", "pred_hbm_bytes_per_token", "pred_bound",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/perf_model/glm_bench.csv")
    parser.add_argument("--quick", action="store_true", default=True)
    parser.add_argument("--gen", type=int, default=16)
    args = parser.parse_args()
    run(args.out, quick=args.quick, gen_len=args.gen)
