"""Print the markdown tables used in REPORT.md from results/summary_*.csv."""

from __future__ import annotations

import pandas as pd

from benchmarks.kernel_audit.analyze import (
    ORDER,
    production_attn_impl,
    production_expert_impl,
)
from benchmarks.kernel_audit.common import RESULTS_DIR
from benchmarks.kernel_audit.models import MODELS

MODEL_ORDER = list(MODELS)


def _load(name):
    p = RESULTS_DIR / f"summary_{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _cell(r, kind="pct"):
    if r is None:
        return "–"
    tag = "C" if r["bound"] == "compute" else "M"
    v = r["roof_pct"]
    return f"{v:.0f}% {tag}" if v >= 1 else f"{v:.2f}% {tag}"


def _fmt_us(us):
    if us >= 1e6:
        return f"{us / 1e6:.2f} s"
    if us >= 1e3:
        return f"{us / 1e3:.1f} ms"
    return f"{us:.0f} µs"


def expert_layer_table(df):
    cols = ORDER
    lines = [
        "| model (impl) | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for m in MODEL_ORDER:
        impl = production_expert_impl(m)
        d = df[(df.model == m) & (df.impl == impl)]
        if d.empty:
            continue
        cells = []
        for c in cols:
            r = d[d["size"] == c]
            cells.append(_cell(r.iloc[0]) if len(r) else "–")
        lines.append(f"| {m} ({impl}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def expert_layer_time_table(df):
    cols = ["1K", "12K", "200K", "b1", "b64"]
    lines = [
        "| model | "
        + " | ".join(f"{c} t / roofline" for c in cols)
        + " | MoE layers |",
        "|---|" + "---|" * (len(cols) + 1),
    ]
    for m in MODEL_ORDER:
        impl = production_expert_impl(m)
        d = df[(df.model == m) & (df.impl == impl)]
        if d.empty:
            continue
        cells = []
        for c in cols:
            r = d[d["size"] == c]
            if len(r):
                r = r.iloc[0]
                cells.append(f"{_fmt_us(r.t_med_us)} / {_fmt_us(r.t_roof_us)}")
            else:
                cells.append("–")
        lines.append(
            f"| {m} | " + " | ".join(cells) + f" | {MODELS[m].moe_layers} |"
        )
    return "\n".join(lines)


def kernel_diag_table(df, sizes=("b1", "12K", "200K")):
    lines = [
        "| model | point | impl | t (µs) | grid (main kernel) | waves | warps active % | tensor pipe % | DRAM BW % | DRAM bytes / ideal | L2 hit % |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in MODEL_ORDER:
        for s in sizes:
            for impl in (production_expert_impl(m), "cublas_ref"):
                r = df[(df.model == m) & (df["size"] == s) & (df.impl == impl)]
                if not len(r):
                    continue
                r = r.iloc[0]
                lines.append(
                    f"| {m} | {s} (M={int(r.m_M)}) | {impl} | {r.t_med_us:.0f} | {r.main_grid:.0f} | {r.main_waves:.2f} | "
                    f"{r.warps_pct:.0f} | {r.tensor_pct:.0f} | {r.dram_pct:.0f} | {r.amp:.2f} | {r.l2_hit:.0f} |"
                )
    return "\n".join(lines)


def attn_prefill_table(df):
    cols = ORDER[:6]
    lines = [
        "| model | path | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for m in MODEL_ORDER:
        d = df[df.model == m]
        if d.empty:
            continue
        for impl in (production_attn_impl(m, "attn_prefill", 0), "sdpa_gqa"):
            dd = d[d.impl == impl]
            if dd.empty and impl != "hf_eager":
                continue
            cells = []
            for c in cols:
                r = dd[dd["size"] == c]
                if len(r):
                    r = r.iloc[0]
                    cells.append(f"{_fmt_us(r.t_med_us)} ({r.roof_pct:.0f}%)")
                else:
                    cells.append("OOM" if impl == "hf_eager" else "–")
            lines.append(f"| {m} | {impl} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def attn_decode_table(df, B=1):
    cols = ORDER[:6]
    lines = [
        f"| model | path (B={B}) | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for m in MODEL_ORDER:
        d = df[(df.model == m) & (df.m_B == B)]
        if d.empty:
            continue
        dqk = int(d.m_dqk.iloc[0])
        for impl in (
            production_attn_impl(m, "attn_decode", dqk),
            "triton_decode",
            "sdpa_flash_decode",
        ):
            dd = d[d.impl == impl]
            if dd.empty:
                continue
            cells = []
            for c in cols:
                r = dd[dd["size"] == f"b{B}@{c}"]
                if len(r):
                    r = r.iloc[0]
                    cells.append(
                        f"{_fmt_us(r.t_med_us)} ({r.roof_pct:.1f}%)"
                        if r.roof_pct < 1
                        else f"{_fmt_us(r.t_med_us)} ({r.roof_pct:.0f}%)"
                    )
                else:
                    cells.append("–")
            lines.append(f"| {m} | {impl} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


STEP_MODELS = [
    "qwen3.5-35b-a3b",
    "qwen3-235b-a22b",
    "gpt-oss-120b",
    "glm-5.3",
    "deepseek-v4-flash",
    "minimax-m3",
]


def _pick(df, model, size, impl):
    if df.empty:
        return None
    r = df[(df.model == model) & (df["size"] == size) & (df.impl == impl)]
    return r.iloc[0] if len(r) else None


def step_table(decode_ctx="9K", prefill="12K"):
    """Per-step time = per-layer op time x layers, production paths."""
    el, ap, ad = (
        _load("expert_layer"),
        _load("attn_prefill"),
        _load("attn_decode"),
    )
    ro, kv, gdn = _load("router"), _load("kv_write"), _load("gdn")
    lines = [
        "| model | phase | routed experts | attention | router | KV write | linear attn | **measured step** | roofline step (same ops) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in STEP_MODELS:
        mm = MODELS[m]
        n_attn = mm.attn.full_layers + mm.attn.swa_layers
        for phase, esize, asize, rsize in (
            ("decode b1 @ " + decode_ctx, "b1", f"b1@{decode_ctx}", "b1"),
            ("prefill " + prefill, prefill, prefill, prefill),
        ):
            parts, roof = {}, 0.0
            ex = _pick(el, m, esize, production_expert_impl(m))
            if ex is not None:
                parts["exp"] = ex.t_med_us * mm.moe_layers
                roof += ex.t_roof_us * mm.moe_layers
            src = ad if phase.startswith("decode") else ap
            dqk = (
                int(src[src.model == m].m_dqk.iloc[0])
                if not src[src.model == m].empty
                else 0
            )
            impl = production_attn_impl(
                m,
                "attn_decode" if phase.startswith("decode") else "attn_prefill",
                dqk,
            )
            at = _pick(src, m, asize, impl)
            if at is not None:
                parts["attn"] = at.t_med_us * n_attn
                roof += at.t_roof_us * n_attn
            elif not src[(src.model == m)].empty:
                parts["attn"] = None  # OOM / not runnable
            r = _pick(
                ro,
                m,
                rsize,
                "router_fp32"
                if m in ("glm-5.3", "deepseek-v4-flash")
                else "router_bf16",
            )
            if r is not None:
                parts["router"] = r.t_med_us * mm.moe_layers
                roof += r.t_roof_us * mm.moe_layers
            if m in ("qwen3-235b-a22b", "qwen3.5-35b-a3b"):
                per_tok = _pick(kv, m, "1K", "kv_write_loop")
                one = _pick(kv, m, "b1", "kv_write_loop")
                if phase.startswith("decode") and one is not None:
                    parts["kv"] = one.t_med_us * mm.attn.full_layers
                elif per_tok is not None:
                    from benchmarks.kernel_audit.models import PREFILL_TOKENS

                    parts["kv"] = (
                        per_tok.t_med_us
                        / 1024
                        * PREFILL_TOKENS[prefill]
                        * mm.attn.full_layers
                    )
            if m == "qwen3.5-35b-a3b" and not phase.startswith("decode"):
                g9 = _pick(gdn, m, "9K", "gdn_torch")
                if g9 is not None:
                    from benchmarks.kernel_audit.models import PREFILL_TOKENS

                    parts["gdn"] = (
                        g9.t_med_us
                        * PREFILL_TOKENS[prefill]
                        / 9216
                        * mm.attn.linear_layers
                    )

            def f(k):
                if k not in parts:
                    return "n/m"
                return "OOM" if parts[k] is None else _fmt_us(parts[k])

            total = sum(v for v in parts.values() if v)
            lines.append(
                f"| {m} | {phase} | {f('exp')} | {f('attn')} | {f('router')} | {f('kv')} | {f('gdn')} | "
                f"**{_fmt_us(total)}**{' + OOM' if parts.get('attn', 0) is None else ''} | {_fmt_us(roof)} |"
            )
    return "\n".join(lines)


def grouped_table(sizes=("b1", "b16", "b64", "1K", "12K", "200K")):
    el, gg, gr = (
        _load("expert_layer"),
        _load("expert_layer_graph"),
        _load("expert_grouped"),
    )
    lines = [
        "| model | point | production path | CUTLASS / expert (graph) | cuBLAS / expert (graph) | all experts, 2 batched GEMMs | roofline (BF16 bytes) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODEL_ORDER:
        for s in sizes:
            p = _pick(el, m, s, production_expert_impl(m))
            if p is None:
                continue
            cells = [f"{_fmt_us(p.t_med_us)} ({p.roof_pct:.0f}%)"]
            for df, impl, reps in (
                (gg, "cutlass_fused", 4),
                (gg, "cublas_ref", 4),
                (gr, "bmm_grouped", 1),
            ):
                r = _pick(df, m, s, impl)
                cells.append(
                    f"{_fmt_us(r.t_med_us / reps)} ({r.roof_pct:.0f}%)"
                    if r is not None
                    else "–"
                )
            b = _pick(gr, m, s, "bmm_grouped")
            if b is None:
                b = _pick(gg, m, s, "cublas_ref")
                roof = _fmt_us(b.t_roof_us / 4) if b is not None else "–"
            else:
                roof = _fmt_us(b.t_roof_us)
            lines.append(f"| {m} | {s} | " + " | ".join(cells) + f" | {roof} |")
    return "\n".join(lines)


def gptoss_loop_table():
    d = _load("gptoss_loop")
    el = _load("expert_layer")
    lines = [
        "| model | point | MoE layer, real wrapper loop | expert kernels only | roofline (MXFP4 bytes) |",
        "|---|---|---|---|---|",
    ]
    for _, r in d.iterrows():
        k = _pick(el, r.model, r["size"], "gptoss_resident")
        lines.append(
            f"| {r.model} | {r['size']} | {_fmt_us(r.t_med_us)} | "
            f"{_fmt_us(k.t_med_us) if k is not None else '–'} | {_fmt_us(r.t_roof_us)} |"
        )
    return "\n".join(lines)


def main():
    print("## step table\n")
    print(step_table())
    print("\n## grouped\n")
    print(grouped_table())
    print("\n## gptoss loop\n")
    print(gptoss_loop_table())
    e = _load("expert")
    el = _load("expert_layer")
    ap = _load("attn_prefill")
    ad = _load("attn_decode")
    print("## expert layer % roofline\n")
    print(expert_layer_table(el))
    print("\n## expert layer times\n")
    print(expert_layer_time_table(el))
    print("\n## kernel diagnostics\n")
    print(kernel_diag_table(e))
    print("\n## attention prefill\n")
    print(attn_prefill_table(ap))
    for B in (1, 16):
        print(f"\n## attention decode B={B}\n")
        print(attn_decode_table(ad, B))
    g = _load("expert_layer_graph")
    if not g.empty:
        print("\n## expert layer graph\n")
        print(expert_layer_table(g.assign(impl=g.impl)))


if __name__ == "__main__":
    main()
