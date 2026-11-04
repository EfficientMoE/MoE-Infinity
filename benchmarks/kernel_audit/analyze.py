"""Join timing + ncu counters with roofline ceilings; emit tables and plots.

python -m benchmarks.kernel_audit.analyze
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from benchmarks.kernel_audit.common import RESULTS_DIR

# RTX PRO 6000 Blackwell Max-Q datasheet values (188 SMs, 2280 MHz boost).
SPEC = {"bf16": 438.9e12, "fp8": 877.8e12, "fp4": 1755.6e12, "dram": 1792e9}

# Compute precision of each impl's MMA (production paths dequantize to BF16).
IMPL_PREC = defaultdict(lambda: "bf16")

SAT = 0.80  # "saturated" threshold (fraction of ceiling)


def load_ceilings() -> dict:
    c = json.loads((RESULTS_DIR / "ceilings.json").read_text())
    peak = lambda k: max(x["tflops"] for x in c[k]) * 1e12  # noqa: E731
    return {
        "bf16": peak("bf16"),
        "fp8": peak("fp8") if isinstance(c.get("fp8"), list) else None,
        "fp4": peak("fp4") if isinstance(c.get("fp4"), list) else None,
        "dram": c["dram_read"]["gbs"] * 1e9,
        "l2": c["l2_read"]["gbs"] * 1e9,
        "launch_us": c["launch"]["eager_launch_us"],
        "gpu": c["gpu"],
        "raw": c,
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def ncu_by_case(tag: str) -> dict[str, dict]:
    recs = load_jsonl(RESULTS_DIR / f"ncu_{tag}.jsonl")
    agg: dict[str, dict] = {}
    for r in recs:
        cid = r.get("case")
        if cid is None:
            continue
        t = float(r.get("gpu__time_duration.sum", 0.0))  # ns
        a = agg.setdefault(cid, defaultdict(float))
        a["k_time_ns"] += t
        a["n_kernels"] += 1
        a["dram_bytes"] += float(r.get("dram__bytes_op_read.sum", 0.0)) + float(
            r.get("dram__bytes_op_write.sum", 0.0)
        )
        a["tensor_ops"] += float(
            r.get("sm__ops_path_tensor_src_bf16_dst_fp32.sum", 0.0)
        )
        for key, name in (
            (
                "tensor_pct",
                "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            ),
            ("dram_pct", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            ("l2_hit", "lts__t_sector_hit_rate.pct"),
            ("warps_pct", "sm__warps_active.avg.pct_of_peak_sustained_active"),
            ("sm_hz", "sm__cycles_elapsed.avg.per_second"),
        ):
            a[key + "_tw"] += float(r.get(name, 0.0)) * t
        grid = float(r.get("launch__grid_size", 0.0))
        waves = float(r.get("launch__waves_per_multiprocessor", 0.0))
        kern = a.setdefault("kernels", [])
        kern.append(
            (r["kernel"][:40], int(grid), round(waves, 2), round(t / 1e3, 1))
        )
        if "max_kernel_us" not in a or t / 1e3 > a["max_kernel_us"]:
            a["max_kernel_us"] = t / 1e3
            a["main_grid"] = grid
            a["main_waves"] = waves
            a["main_kernel"] = r["kernel"][:60]
    out = {}
    for cid, a in agg.items():
        tt = a["k_time_ns"] or 1.0
        out[cid] = {
            "k_time_us": a["k_time_ns"] / 1e3,
            "n_kernels": int(a["n_kernels"]),
            "dram_bytes": a["dram_bytes"],
            "tensor_ops": a["tensor_ops"],
            "tensor_pct": a["tensor_pct_tw"] / tt,
            "dram_pct": a["dram_pct_tw"] / tt,
            "l2_hit": a["l2_hit_tw"] / tt,
            "warps_pct": a["warps_pct_tw"] / tt,
            "sm_mhz": a["sm_hz_tw"] / tt / 1e6,
            "main_grid": a.get("main_grid"),
            "main_waves": a.get("main_waves"),
            "main_kernel": a.get("main_kernel"),
            "kernels": a["kernels"],
        }
    return out


def roofline(df: pd.DataFrame, ceil: dict) -> pd.DataFrame:
    P = df["impl"].map(lambda i: ceil[IMPL_PREC[i]])
    t = df["t_med_us"] * 1e-6
    t_comp = df["flops"] / P
    t_mem = df["bytes_min"] / ceil["dram"]
    df["t_roof_us"] = pd.concat([t_comp, t_mem], axis=1).max(axis=1) * 1e6
    df["bound"] = (t_mem >= t_comp).map({True: "memory", False: "compute"})
    df["ai_alg"] = df["flops"] / df["bytes_min"]
    df["tflops"] = df["flops"] / t / 1e12
    df["useful_gbs"] = df["bytes_min"] / t / 1e9
    df["roof_pct"] = 100 * df["t_roof_us"] / df["t_med_us"]
    df["compute_pct"] = 100 * df["flops"] / t / P
    df["bw_pct"] = 100 * df["bytes_min"] / t / ceil["dram"]
    if "dram_bytes" in df:
        df["amp"] = df["dram_bytes"] / df["bytes_min"]
        df["ai_meas"] = df["flops"] / df["dram_bytes"]
        df["meas_gbs"] = df["dram_bytes"] / (df["k_time_us"] * 1e-6) / 1e9
        df["gap_pct"] = 100 * (1 - df["k_time_us"] / df["t_med_us"]).clip(
            lower=0
        )
        df["mma_waste"] = df["tensor_ops"] / df["flops"]
    return df


def verdict(r) -> str:
    if r["bound"] == "compute":
        ok = r["compute_pct"] >= 100 * SAT
        return f"compute-bound: {'SATURATED' if ok else 'NOT saturated'} ({r['compute_pct']:.0f}% of peak)"
    ok = r["bw_pct"] >= 100 * SAT
    return f"memory-bound: {'SATURATED' if ok else 'NOT saturated'} ({r['bw_pct']:.0f}% of DRAM BW)"


def build_frame(suite: str, ncu_tag: str | None, ceil: dict) -> pd.DataFrame:
    rows = [
        r
        for r in load_jsonl(RESULTS_DIR / f"{suite}_time.jsonl")
        if r.get("ok")
    ]
    if not rows:
        return pd.DataFrame()
    for r in rows:
        for k, v in r.pop("meta").items():
            r[f"m_{k}"] = v
    df = pd.DataFrame(rows).drop_duplicates("cid", keep="last")
    if ncu_tag:
        ncu = ncu_by_case(ncu_tag)
        cols = pd.DataFrame.from_dict(ncu, orient="index")
        if not cols.empty:
            df = df.join(cols, on="cid")
    df = roofline(df, ceil)
    df["verdict"] = df.apply(verdict, axis=1)
    return df


ORDER = [
    "1K",
    "2K",
    "9K",
    "12K",
    "50K",
    "200K",
    "b1",
    "b4",
    "b16",
    "b64",
    "b256",
]


def _size_key(s: str) -> int:
    base = s.split("@")[0]
    return ORDER.index(base) if base in ORDER else 99


def fmt_table(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for key, _, f in cols:
            v = r.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("–")
            elif f:
                cells.append(format(v, f))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


EXPERT_COLS = [
    ("size", "point", ""),
    ("m_M", "M/expert", "d"),
    ("impl", "impl", ""),
    ("t_med_us", "t (µs)", ".1f"),
    ("tflops", "TFLOP/s", ".1f"),
    ("useful_gbs", "GB/s (useful)", ".0f"),
    ("bound", "roofline bound", ""),
    ("roof_pct", "% of roofline", ".0f"),
    ("tensor_pct", "tensor pipe %", ".0f"),
    ("dram_pct", "DRAM util %", ".0f"),
    ("amp", "DRAM bytes / ideal", ".2f"),
    ("l2_hit", "L2 hit %", ".0f"),
    ("main_grid", "grid (main)", ".0f"),
    ("main_waves", "waves", ".2f"),
]


def plot_roofline(
    df: pd.DataFrame, ceil: dict, path: Path, title: str, impls: list[str]
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8.5, 6))
    x = np.logspace(-1, 4.3, 200)
    bw = ceil["dram"]
    for prec, ls in (("bf16", "-"), ("fp8", "--"), ("fp4", ":")):
        if ceil.get(prec):
            ax.plot(
                x,
                np.minimum(ceil[prec], x * bw) / 1e12,
                ls,
                color="k",
                lw=1.2,
                label=f"{prec.upper()} roof ({ceil[prec] / 1e12:.0f} TF, {bw / 1e9:.0f} GB/s)",
            )
    ax.plot(
        x,
        np.minimum(SPEC["bf16"], x * SPEC["dram"]) / 1e12,
        "-",
        color="0.7",
        lw=1,
        label="BF16 datasheet (439 TF @2.28 GHz, 1792 GB/s)",
    )
    ax.plot(
        x,
        np.minimum(ceil["bf16"], x * ceil["l2"]) / 1e12,
        "-.",
        color="tab:gray",
        lw=0.8,
        label=f"L2 roof ({ceil['l2'] / 1e12:.1f} TB/s)",
    )
    markers = "osD^vP*Xh"
    cmap = plt.get_cmap("viridis")
    sizes = [
        s for s in ORDER if s in set(df["size"].map(lambda s: s.split("@")[0]))
    ]
    for mi, impl in enumerate(impls):
        d = df[df["impl"] == impl]
        if d.empty:
            continue
        ai = (
            d["ai_meas"]
            if "ai_meas" in d and d["ai_meas"].notna().any()
            else d["ai_alg"]
        )
        cols = [
            cmap(sizes.index(s.split("@")[0]) / max(1, len(sizes) - 1))
            for s in d["size"]
        ]
        ax.scatter(
            ai,
            d["tflops"],
            c=cols,
            marker=markers[mi % len(markers)],
            s=28,
            edgecolors="k",
            linewidths=0.3,
            label=impl,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte, measured by ncu)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_ylim(0.05, 2500)
    ax.set_xlim(0.1, 2e4)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.set_title(title, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(sizes) - 1))
    cb = fig.colorbar(sm, ax=ax, ticks=range(len(sizes)), pad=0.01)
    cb.ax.set_yticklabels(sizes, fontsize=7)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eff_vs_m(
    df: pd.DataFrame, path: Path, impl: str, ref: str = "cublas_ref"
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cmap = plt.get_cmap("tab20")
    for i, (model, d) in enumerate(df.groupby("model", sort=False)):
        for imp, ls in ((impl, "-"), (ref, ":")):
            dd = d[d["impl"] == imp].sort_values("m_M")
            dd = dd.drop_duplicates("m_M")
            if dd.empty:
                continue
            ax.plot(
                dd["m_M"],
                dd["roof_pct"],
                ls,
                marker="o",
                ms=3,
                color=cmap(i % 20),
                label=model if imp == impl else None,
            )
    ax.axhline(80, color="k", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("tokens per expert M  (decode b1..b256 → prefill 1K..200K)")
    ax.set_ylabel("% of roofline attained")
    ax.set_ylim(0, 110)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.set_title(
        f"Routed-expert FFN, one expert, cold L2:  solid = {impl}, dotted = {ref}",
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


PROD_EXPERT = {
    5: "cutlass_fused",
    4: "cutlass_fused",
}


def production_expert_impl(model: str) -> str:
    from benchmarks.kernel_audit.models import MODELS

    m = MODELS[model]
    if m.expert_type == 6:
        return "gptoss_resident"
    if m.expert_dtype.startswith("fp8"):
        return "fp8_prod"
    if m.expert_dtype.startswith("fp4"):
        return "v4_fp4_prod"
    return "cutlass_fused"


def production_attn_impl(model: str, suite: str, dqk: int) -> str:
    shim = {
        "qwen3-30b-a3b",
        "qwen3-235b-a22b",
        "qwen3.5-35b-a3b",
        "qwen3.5-122b-a10b",
        "qwen3.5-397b-a17b",
    }
    if suite == "attn_prefill":
        return "sdpa_shim" if model in shim else "hf_eager"
    if model in shim:
        return "paged_v1" if dqk <= 128 else "sdpa_fallback"
    return "hf_eager_decode"


def oom_points(suite: str, impl_of) -> set[tuple[str, str]]:
    """(model, size) pairs whose production case was skipped / failed with OOM."""
    out = set()
    for r in load_jsonl(RESULTS_DIR / f"{suite}_time.jsonl"):
        if r.get("ok") or "OOM" not in str(r.get("skip", "")):
            continue
        dqk = r.get("meta", {}).get("dqk", 0)
        if r["impl"] == impl_of(r["model"], dqk):
            size = r["size"].split("@")[1] if "@" in r["size"] else r["size"]
            out.add((r["model"], size, r.get("meta", {}).get("B")))
    return out


def heatmap(
    df: pd.DataFrame,
    path: Path,
    title: str,
    cols: list[str],
    value="roof_pct",
    oom: set | None = None,
    batch=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(dict.fromkeys(df["model"]))
    grid = np.full((len(models), len(cols)), np.nan)
    ann = [["" for _ in cols] for _ in models]
    for _, r in df.iterrows():
        if r["size"] not in cols:
            continue
        i, j = models.index(r["model"]), cols.index(r["size"])
        grid[i, j] = r[value]
        tag = "C" if r["bound"] == "compute" else "M"
        ann[i][j] = (
            f"{r[value]:.0f}{tag}" if r[value] >= 1 else f"{r[value]:.1f}{tag}"
        )
    fig, ax = plt.subplots(
        figsize=(1.0 + 0.78 * len(cols), 0.9 + 0.42 * len(models))
    )
    im = ax.imshow(
        np.clip(grid, 0, 100), cmap="RdYlGn", vmin=0, vmax=100, aspect="auto"
    )
    for i in range(len(models)):
        for j in range(len(cols)):
            if ann[i][j]:
                ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=7)
            elif oom and (models[i], cols[j], batch) in oom:
                ax.text(
                    j,
                    i,
                    "OOM",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="tab:red",
                )
            elif np.isnan(grid[i, j]):
                ax.text(
                    j, i, "–", ha="center", va="center", fontsize=7, color="0.5"
                )
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="% of roofline")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ceil = load_ceilings()
    out_md = []
    frames = {}
    for suite, tag in (
        ("expert", "expert"),
        ("expert_layer", None),
        ("expert_layer_graph", None),
        ("expert_grouped", None),
        ("expert_real", None),
        ("gptoss_loop", None),
        ("attn_prefill", "attn_prefill"),
        ("attn_decode", "attn_decode"),
        ("router", None),
        ("gdn", None),
        ("kv_write", None),
    ):
        df = build_frame(suite, tag, ceil)
        if df.empty:
            continue
        df["_k"] = df["size"].map(_size_key)
        df = df.sort_values(["model", "_k", "impl"], kind="stable")
        frames[suite] = df
        df.drop(columns=["kernels"], errors="ignore").to_csv(
            RESULTS_DIR / f"summary_{suite}.csv", index=False
        )
    plots = RESULTS_DIR / "plots"
    if "expert" in frames:
        d = frames["expert"]
        prod = [
            "cutlass_fused",
            "fp8_prod",
            "gptoss_resident",
            "gptoss_offload",
            "v4_fp4_prod",
        ]
        plot_roofline(
            d,
            ceil,
            plots / "roofline_expert_production.png",
            "Routed-expert FFN, production kernels (RTX PRO 6000 Max-Q, sm_120)",
            prod,
        )
        plot_roofline(
            d,
            ceil,
            plots / "roofline_expert_cublas.png",
            "Routed-expert FFN, cuBLAS reference (same shapes)",
            ["cublas_ref"],
        )
        plot_eff_vs_m(
            d[d["impl"].isin(["cutlass_fused", "cublas_ref"])],
            plots / "expert_eff_vs_M.png",
            "cutlass_fused",
        )
        for model, dm in d.groupby("model", sort=False):
            out_md.append(f"### {model}\n")
            out_md.append(fmt_table(dm, EXPERT_COLS))
            out_md.append("")
    (RESULTS_DIR / "tables_expert.md").write_text("\n".join(out_md))

    cols_e = ORDER
    if "expert_layer" in frames:
        d = frames["expert_layer"]
        prod = d[
            d.apply(
                lambda r: r["impl"] == production_expert_impl(r["model"]),
                axis=1,
            )
        ]
        heatmap(
            prod,
            plots / "heatmap_expert_layer.png",
            "MoE layer (all active experts, production kernel path, 4 exec streams) — "
            "% of roofline; C = compute-bound, M = memory-bound",
            cols_e,
        )
        ref = d[d["impl"] == "cublas_ref"]
        heatmap(
            ref,
            plots / "heatmap_expert_layer_cublas.png",
            "MoE layer with cuBLAS per-expert GEMMs (reference, Python-launched)",
            cols_e,
        )
    if "attn_prefill" in frames:
        d = frames["attn_prefill"]
        prod = d[
            d.apply(
                lambda r: r["impl"]
                == production_attn_impl(r["model"], "attn_prefill", r["m_dqk"]),
                axis=1,
            )
        ]
        heatmap(
            prod,
            plots / "heatmap_attn_prefill.png",
            "Attention prefill, production path (SDPA-FA2 shim or HF eager) — % of roofline",
            ORDER[:6],
            oom=oom_points(
                "attn_prefill",
                lambda m, dqk: production_attn_impl(m, "attn_prefill", dqk),
            ),
        )
        heatmap(
            d[d["impl"] == "sdpa_gqa"],
            plots / "heatmap_attn_prefill_sdpa.png",
            "Attention prefill, SDPA-FA2 native GQA (reference) — % of roofline",
            ORDER[:6],
        )
    if "attn_decode" in frames:
        d = frames["attn_decode"].copy()
        for B in (1, 16, 64):
            sub = d[d["m_B"] == B].copy()
            sub["size"] = sub["size"].map(lambda s: s.split("@")[1])
            prod = sub[
                sub.apply(
                    lambda r: r["impl"]
                    == production_attn_impl(
                        r["model"], "attn_decode", r["m_dqk"]
                    ),
                    axis=1,
                )
            ]
            heatmap(
                prod,
                plots / f"heatmap_attn_decode_b{B}.png",
                f"Decode attention B={B}, production path — % of roofline (context length; "
                "– = not run: paged_v1 / fallback take seconds per call)",
                ORDER[:6],
                oom=oom_points(
                    "attn_decode",
                    lambda m, dqk: production_attn_impl(m, "attn_decode", dqk),
                ),
                batch=B,
            )
            heatmap(
                sub[sub["impl"] == "sdpa_flash_decode"],
                plots / f"heatmap_attn_decode_b{B}_flash.png",
                f"Decode attention B={B}, SDPA flash-decoding reference — % of roofline",
                ORDER[:6],
            )
    if "expert_layer" in frames and "expert_grouped" in frames:
        plot_layer_bars(frames, plots / "expert_layer_decode_bars.png")
        g = frames["expert_grouped"]
        heatmap(
            g,
            plots / "heatmap_expert_layer_grouped.png",
            "MoE layer as 2 batched GEMMs over all active experts (grouped-GEMM proxy, cuBLAS bmm)",
            cols_e,
        )
    for name, df in frames.items():
        print(name, len(df))


def plot_layer_bars(frames, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    el, gr = frames["expert_layer"], frames["expert_grouped"]
    gg = frames.get("expert_layer_graph", pd.DataFrame())
    models = list(dict.fromkeys(el["model"]))
    variants = [
        (
            "production path (eager, 4 streams)",
            lambda m, s: _get(el, m, s, production_expert_impl(m)),
        ),
        (
            "CUTLASS per expert, CUDA graph",
            lambda m, s: _get(gg, m, s, "cutlass_fused"),
        ),
        (
            "cuBLAS per expert, CUDA graph",
            lambda m, s: _get(gg, m, s, "cublas_ref"),
        ),
        (
            "all experts in 2 batched GEMMs",
            lambda m, s: _get(gr, m, s, "bmm_grouped"),
        ),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, size in zip(axes, ("b1", "b64")):
        x = np.arange(len(models))
        w = 0.2
        for i, (lbl, fn) in enumerate(variants):
            vals = [fn(m, size) for m in models]
            ax.bar(
                x + (i - 1.5) * w,
                [v if v is not None else 0 for v in vals],
                w,
                label=lbl,
            )
        ax.axhline(80, color="k", lw=0.6, ls="--")
        ax.set_ylabel(f"decode {size}: % of DRAM roof")
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", lw=0.3)
    axes[-1].set_xticks(
        np.arange(len(models)), models, rotation=30, ha="right", fontsize=8
    )
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title(
        "MoE layer at decode: production vs device-only vs grouped (RTX PRO 6000 Max-Q)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _get(df, model, size, impl):
    if df is None or df.empty:
        return None
    r = df[(df["model"] == model) & (df["size"] == size) & (df["impl"] == impl)]
    return float(r["roof_pct"].iloc[0]) if len(r) else None


if __name__ == "__main__":
    main()
