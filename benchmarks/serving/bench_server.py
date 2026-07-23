"""Persistent benchmark server: load a MoE model once, serve timing requests.

Avoids reloading the (large, offloaded) checkpoint for every sweep experiment.
The model + offload store + warm GPU expert cache stay resident in the daemon;
clients send small JSON requests over a TCP socket and get prefill/decode timings
back. Useful for models whose cold load is minutes (e.g. Qwen3.5-35B-A3B with
expert offloading), where per-experiment reload dominates wall-clock.

Usage:
    # start the server (one-time load)
    CUDA_VISIBLE_DEVICES=0 python -m benchmarks.serving.bench_server serve \
        --model Qwen/Qwen3.5-35B-A3B --offload-dir /ssd/off --device-memory-ratio 0.6

    # from another process, run one experiment (model stays resident)
    python -m benchmarks.serving.bench_server client \
        '{"op": "bench", "batch": 4, "prompt_len": 256, "decode": 16}'

Request ops: {"op": "ping"} | {"op": "bench", batch, prompt_len, decode} | {"op": "shutdown"}
"""

import argparse
import json
import socket
import time
import traceback

import torch
from transformers import AutoTokenizer

_LONG = (
    "Explain mixture-of-experts routing and why expert offloading helps "
    "memory-constrained GPUs. " * 40
)


def run_bench(model, tok, batch, prompt_len, decode):
    ids = (
        tok(_LONG, return_tensors="pt")
        .input_ids[:, :prompt_len]
        .to("cuda:0")
        .repeat(batch, 1)
    )
    plen = ids.shape[1]
    torch.cuda.reset_peak_memory_stats()
    pf = []
    for _ in range(2):  # cold then warm prefill
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            o = model.model(ids, use_cache=True)
        torch.cuda.synchronize()
        pf.append(time.perf_counter() - t)
    kv = o.past_key_values
    cur = o.logits[:, -1, :].argmax(-1)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(decode):
        with torch.no_grad():
            o = model.model(cur.view(-1, 1), past_key_values=kv, use_cache=True)
        kv = o.past_key_values
        cur = o.logits[:, -1, :].argmax(-1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return {
        "batch": batch,
        "prompt_len": plen,
        "prefill_cold_s": round(pf[0], 3),
        "prefill_warm_s": round(pf[1], 3),
        "prefill_tok_s": round(batch * plen / pf[1], 1),
        "decode_itl_ms": round(dt / decode * 1000, 2),
        "decode_tok_s_total": round(batch * decode / dt, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def serve(args):
    from moe_infinity import MoE

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    model = MoE(
        args.model,
        {"offload_path": args.offload_dir, "device_memory_ratio": args.device_memory_ratio},
    )
    load_s = round(time.time() - t0, 1)
    print(f"READY load={load_s}s host={args.host} port={args.port}", flush=True)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(4)
    while True:
        conn, _ = srv.accept()
        req = conn.recv(65536).decode().strip()
        try:
            r = json.loads(req)
            op = r.get("op")
            if op == "ping":
                resp = {"ready": True, "load_s": load_s}
            elif op == "shutdown":
                conn.send(b'{"ok": true}')
                conn.close()
                break
            else:
                resp = run_bench(
                    model, tok, r["batch"], r.get("prompt_len", 256), r.get("decode", 16)
                )
        except Exception as e:  # report, keep serving
            resp = {"error": repr(e), "tb": traceback.format_exc()[-400:]}
        conn.send(json.dumps(resp).encode())
        conn.close()


def client(args):
    s = socket.socket()
    s.settimeout(args.timeout)
    s.connect((args.host, args.port))
    s.send(args.payload.encode())
    print(s.recv(65536).decode(), flush=True)
    s.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["serve", "client"])
    p.add_argument("payload", nargs="?", default='{"op": "ping"}')
    p.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--offload-dir", default="/tmp/moe_bench_off")
    p.add_argument("--device-memory-ratio", type=float, default=0.6)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=51567)
    p.add_argument("--timeout", type=float, default=1800)
    a = p.parse_args()
    serve(a) if a.mode == "serve" else client(a)


if __name__ == "__main__":
    main()
