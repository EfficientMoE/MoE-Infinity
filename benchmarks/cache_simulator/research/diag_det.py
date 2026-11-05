#!/usr/bin/env python3
"""Diagnose exp680 det=False: run both grids, deep-diff every cell/arm/field."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "st", str(Path(__file__).parent / "run_stagger.py")
)
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def diff(a, b, path="", out=None):
    if out is None:
        out = []
    if type(a) is not type(b):
        out.append(
            f"{path}: TYPE {type(a).__name__} vs {type(b).__name__} ({a!r} | {b!r})"
        )
    elif isinstance(a, dict):
        for k in set(a) | set(b):
            diff(a.get(k), b.get(k), f"{path}.{k}", out)
    elif isinstance(a, float) and a != b:
        out.append(f"{path}: {a!r} vs {b!r} (delta {b-a:+.3e})")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


g1 = st.run_all()
g2 = st.run_all()
for i, (x, y) in enumerate(zip(g1, g2)):
    d = diff(x, y, f"g[{i}](W={x['width']})")
    if d:
        print(f"--- W={x['width']} DIFFERS:")
        for ln in d[:20]:
            print("   ", ln)
    else:
        print(f"W={x['width']}: identical")
js1, js2 = json.dumps(g1, sort_keys=True), json.dumps(g2, sort_keys=True)
print("grids identical:", js1 == js2, "| lens:", len(js1), len(js2))
if js1 != js2:
    for i, (c1, c2) in enumerate(zip(js1, js2)):
        if c1 != c2:
            print(
                "first char diff at",
                i,
                ":",
                js1[max(0, i - 80) : i + 40],
                "||",
                js2[max(0, i - 80) : i + 40],
            )
            break
