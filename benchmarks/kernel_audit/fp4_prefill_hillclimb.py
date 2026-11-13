"""Hill-climb harness for the DeepSeek-V4 native FP4 prefill path (#350).

Sweeps the `bt` (block-token) arm of `v4_moe_forward_sm120_pro6000` on
balanced-routing MoE layer shapes and reports per-point timings, parity vs
the auto-chosen arm, and %-of-roofline (ceilings from `ceilings.py`).

The production binding picks `bt = choose_bt(n, E)` internally and exposes no
override; this tool does NOT modify repo sources. It mirrors
`extensions/kernel/` into a build directory, applies the two-line
`bt_override` patch to the *copied* binding, and builds that copy as a single
module named `v4hc`. Two hard-won rules for anyone extending this:

1. Never load the stock `moe_infinity._v4_fp4` AND an instrumented copy in
   one process: both export the same global symbols, calls interpose between
   the modules and the CUDA context dies later at a random trivial op
   (`invalid argument` in a memset/subtract, no Xid). This tool loads exactly
   one module and uses its own `bt=-1` (auto) as the reference.
2. The sources `#include "../hw_profile.h"`, so the mirror must preserve the
   `kernel/v4_fp4/...` layout, not a flat directory.

Usage (one clean GPU):
  CUDA_VISIBLE_DEVICES=0 python -m benchmarks.kernel_audit.fp4_prefill_hillclimb \
      --build-dir /tmp/fp4hc --out fp4_bt_sweep.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys

import torch

from benchmarks.kernel_audit.common import time_cuda
from benchmarks.kernel_audit.models import MODELS

REPO = pathlib.Path(__file__).resolve().parents[2]

_RUN_V4_SIG = (
    "std::pair<torch::Tensor, torch::Tensor> run_v4_moe(\n"
    "    const torch::Tensor& x, const torch::Tensor& topk_ids,\n"
    "    const torch::Tensor& topk_w, const torch::Tensor& table, int64_t I,\n"
    "    double swiglu_limit, bool combine, int64_t act_block) {"
)
_RUN_V4_PATCH = (
    "std::pair<torch::Tensor, torch::Tensor> run_v4_moe(\n"
    "    const torch::Tensor& x, const torch::Tensor& topk_ids,\n"
    "    const torch::Tensor& topk_w, const torch::Tensor& table, int64_t I,\n"
    "    double swiglu_limit, bool combine, int64_t act_block,\n"
    "    int64_t bt_override = -1) {"
)
_CHOOSE_BT = "int bt = v4_moe_choose_bt(n, (int)E);"
_CHOOSE_BT_PATCH = (
    _CHOOSE_BT + "\n  if (bt_override > 0) bt = (int)bt_override;"
)


def _mirror_and_patch(build_dir: pathlib.Path) -> pathlib.Path:
    """Mirror extensions/kernel into build_dir/src and patch the binding copy."""
    src = build_dir / "src" / "kernel"
    if not (src / "v4_fp4").exists():
        src.mkdir(parents=True, exist_ok=True)
        ker = REPO / "extensions" / "kernel"
        shutil.copytree(ker / "v4_fp4", src / "v4_fp4")
        for h in ker.glob("*.h"):
            shutil.copy(h, src / h.name)
        for h in ker.glob("*.cuh"):
            shutil.copy(h, src / h.name)
    binding = src / "v4_fp4" / "v4_fp4_binding.cpp"
    text = binding.read_text()
    if "bt_override" not in text:
        assert _RUN_V4_SIG in text, "binding drifted: run_v4_moe signature"
        text = text.replace(_RUN_V4_SIG, _RUN_V4_PATCH)
        assert _CHOOSE_BT in text, "binding drifted: choose_bt call"
        text = text.replace(_CHOOSE_BT, _CHOOSE_BT_PATCH)
        # thread the knob through the two public entry points
        entry_old = (
            "int64_t act_block) {\n  return run_v4_moe(x, topk_ids, topk_w, "
            "table, inter, swiglu_limit, true,\n"
            "                    act_block)"
        )
        assert entry_old in text, "binding drifted: moe_forward entry"
        text = text.replace(
            entry_old,
            "int64_t act_block, int64_t bt_override) {\n"
            "  return run_v4_moe(x, topk_ids, topk_w, table, inter, "
            "swiglu_limit, true,\n                    act_block, bt_override)",
        )
        def_old = (
            'py::arg("act_block") = 128,\n        "V4 grouped routed-expert'
        )
        assert def_old in text, "binding drifted: pybind def"
        text = text.replace(
            def_old,
            'py::arg("act_block") = 128,\n        py::arg("bt_override") = -1,'
            '\n        "V4 grouped routed-expert',
        )
        binding.write_text(text)
    return src


def _build(src: pathlib.Path, build_dir: pathlib.Path) -> object:
    from torch.utils.cpp_extension import load

    v4 = src / "v4_fp4"
    return load(
        name="v4hc",
        sources=[
            str(v4 / f)
            for f in (
                "v4_fp4_binding.cpp",
                "v4_fp4_dequant.cu",
                "v4_fp4_moe_sm120_pro6000.cu",
                "mxfp4_dequant.cu",
                "mxfp4_gptoss_moe.cu",
                "gptoss_expert.cpp",
                "fp8_dequant.cu",
            )
        ],
        extra_include_paths=[str(REPO / "extensions"), str(build_dir / "src")],
        extra_cflags=["-O3", "-std=c++20", "-fPIC"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++20",
            "-gencode=arch=compute_120a,code=sm_120a",
        ],
        build_directory=str(build_dir / "build"),
        verbose=False,
    )


def _expert_weights(H: int, I: int) -> dict:
    w = {}
    for name, (n, k) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
        w[name] = torch.randint(
            0, 255, (n, k // 2), device="cuda", dtype=torch.uint8
        )
        w["s" + name[1]] = torch.randint(
            118, 124, (n, k // 32), device="cuda", dtype=torch.uint8
        )
    return w


def _layer_case(T: int, n_exp: int, H: int, I: int, k: int):
    experts = [_expert_weights(H, I) for _ in range(n_exp)]
    table = torch.tensor(
        [
            [e[nm].data_ptr() for nm in ("w1", "s1", "w3", "s3", "w2", "s2")]
            for e in experts
        ],
        dtype=torch.int64,
        device="cuda",
    )
    ids = (
        (torch.arange(T * k, device="cuda") % n_exp).view(T, k).to(torch.int32)
    )
    wts = torch.rand(T, k, device="cuda")
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    return x, ids, wts, table, experts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--sizes", default="9K,12K,50K,200K")
    ap.add_argument("--bts", default="16,32,64,128")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--build-dir", default="/tmp/fp4hc")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = MODELS[args.model]
    n_exp = m.experts
    toks = {"9K": 9216, "12K": 12288, "50K": 51200, "200K": 204800}
    sizes = [toks[s] for s in args.sizes.split(",")]
    bts = [int(b) for b in args.bts.split(",")]

    build_dir = pathlib.Path(args.build_dir)
    (build_dir / "build").mkdir(parents=True, exist_ok=True)
    src = _mirror_and_patch(build_dir)
    hc = _build(src, build_dir)
    print("built v4hc", flush=True)

    torch.manual_seed(0)
    rows = []
    for T in sizes:
        x, ids, wts, table, experts = _layer_case(
            T, n_exp, m.hidden, m.inter, m.top_k
        )
        ref = None
        for bt in [-1, *bts]:
            fn = lambda: hc.v4_moe_forward_sm120_pro6000(  # noqa: E731
                x, ids, wts, table, m.inter, 10.0, 128, bt
            )
            out = fn()
            torch.cuda.synchronize()
            if bt == -1:
                ref = out.clone()
                rel = 0.0
            else:
                rel = (out - ref).abs().max().item() / (
                    ref.abs().max().item() + 1e-9
                )
            t = time_cuda(fn, iters=args.iters, warmup=3)
            rows.append(dict(T=T, bt=bt, us=t.median_us, rel=rel))
            print(
                f"T={T} bt={bt:>4}: {t.median_us:9.1f} us  rel {rel:.2e}",
                flush=True,
            )
        del x, ids, wts, table, experts
        torch.cuda.empty_cache()

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
