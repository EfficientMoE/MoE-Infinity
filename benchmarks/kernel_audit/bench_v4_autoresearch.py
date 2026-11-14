"""Time one grouped V4.1-Flash layer. Writes nothing; prints JSON lines."""

import importlib.util
import json
import sys

import torch

SO = sys.argv[1]
H, I, E, TOPK = 5120, 2304, 384, 6
POINTS = [
    ("b1", 1, 6, 8),
    ("b64", 64, 244, 8),
    ("1K", 1024, 384, 8),
    ("9K", 9 * 1024, 384, 8),
    ("12K", 12 * 1024, 384, 8),
    ("16K", 16 * 1024, 384, 8),
    ("32K", 32 * 1024, 384, 5),
    ("64K", 64 * 1024, 384, 5),
    ("128K", 128 * 1024, 384, 5),
    ("200K", 200 * 1024, 384, 5),
]


def load():
    spec = importlib.util.spec_from_file_location("_v4_fp4", SO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def active(tokens: int) -> int:
    return int(round(E * (1.0 - (1.0 - TOPK / E) ** tokens)))


def main():
    v4 = load()
    print("device", torch.cuda.get_device_name(0), flush=True)
    print("alloc weights", flush=True)
    experts = []
    for _ in range(E):
        w = {}
        for name, (n, k) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
            w[name] = torch.randint(0, 255, (n, k // 2), device="cuda", dtype=torch.uint8)
            w["s" + name[1]] = torch.randint(
                118, 124, (n, k // 32), device="cuda", dtype=torch.uint8
            )
        experts.append(w)
    table = torch.tensor(
        [[e[n].data_ptr() for n in ("w1", "s1", "w3", "s3", "w2", "s2")] for e in experts],
        dtype=torch.int64,
        device="cuda",
    )
    for name, tokens, expect, iters in POINTS:
        n_exp = active(tokens)
        if n_exp != expect:
            print("active", name, n_exp, "expected", expect, flush=True)
        ids = (torch.arange(tokens * TOPK, device="cuda") % n_exp).view(tokens, TOPK).to(
            torch.int32
        )
        wts = torch.rand(tokens, TOPK, device="cuda")
        x = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16)

        def run(x=x, ids=ids, wts=wts):
            return v4.v4_moe_forward_sm120_pro6000(x, ids, wts, table, I, 10.0, 32)

        rec = {"point": name, "tokens": tokens, "experts": n_exp, "iters": iters}
        try:
            for _ in range(2):
                y = run()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            samples = []
            for _ in range(iters):
                start.record()
                y = run()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
            samples.sort()
            rec["ms_median"] = samples[len(samples) // 2]
            rec["ms_samples"] = samples
            rec["finite"] = bool(torch.isfinite(y).all().item())
            rec["ok"] = True
        except torch.cuda.OutOfMemoryError as exc:
            rec["ok"] = False
            rec["error"] = f"OOM: {exc}"
            torch.cuda.empty_cache()
        except Exception as exc:
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(rec), flush=True)
        del x, ids, wts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
