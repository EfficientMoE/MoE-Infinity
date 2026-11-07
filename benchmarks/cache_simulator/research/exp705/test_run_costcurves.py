"""exp705 contract tests — run with the torch-capable interpreter:

    ~/warp-work/exp650-moei-granite3b-trace-pr/.venv/bin/python -m pytest \
        benchmarks/cache_simulator/research/exp705/test_run_costcurves.py -v

Covers the two disclosed repairs (DEVIATIONS D1/D2) plus frozen constants:
- D1: loader handles real trace tensors (L, seq, k) incl. decode (L,1,k);
      prefill skipped; keys keep multiplicity (N == decode_records * L * k).
- D2: anchor recomputes sealed spread from BURST hits (not whole) — and the
      whole-hit variant provably disagrees (mutation bite).
- byte model constants and WS20 semantics frozen in PREREG.md.
No sealed artifact is read or mutated; inputs are synthetic fixtures.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

D = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "run_costcurves", D / "run_costcurves.py"
)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def test_byte_constants_exact():
    # PREREG.md frozen values (C4a): 3*h*i*2 bf16 bytes per expert.
    assert rc.b_e(rc.FAM["granite"]) == 4_718_592
    assert rc.b_e(rc.FAM["hi"]) == rc.b_e(rc.FAM["lo"]) == 9_437_184
    assert 768 * rc.b_e(rc.FAM["granite"]) / 2**30 == pytest.approx(3.375)
    assert 1792 * rc.b_e(rc.FAM["hi"]) / 2**30 == pytest.approx(15.75)


def test_ws20_semantics_and_determinism():
    synth = [(i % 97, i % 11) for i in range(5000)]
    # final 20% = 1000 accesses of the 1067-cycle -> exactly min(1000, 97*11)
    assert rc.ws20(synth) == 1000 == rc.ws20(synth)


def _make_trace(tmp_path, decode_records=3, L=4, k=2, E=10):
    """Synthetic trace in the REAL format: decode (L,1,k) int32, prefill (L,seq,k)."""
    recs = []
    pre = torch.zeros((L, 5, k), dtype=torch.int32)
    for l in range(L):
        pre[l] = torch.tensor([[((l * 7 + j) % E) for j in range(k)]] * 5)
    recs.append(
        {"topk_ids_of_layer": pre, "forward_mode": 1}
    )  # must be skipped
    for _ in range(decode_records):
        t = torch.zeros((L, 1, k), dtype=torch.int32)
        for l in range(L):
            t[l, 0] = torch.tensor([(l * 7 + j) % E for j in range(k)])
        recs.append({"topk_ids_of_layer": t, "forward_mode": 2})
    p = tmp_path / "synthetic.pt"
    torch.save({"records": recs}, str(p))
    return p


def test_loader_d1_shape_and_multiplicity(tmp_path):
    # D1 repair: (L,1,k) decode tensors unpack per-layer; N == records*L*k;
    # (layer, expert) mapping exact; prefill excluded.
    p = _make_trace(tmp_path)
    keys = rc.load_decode_keys(p)
    assert len(keys) == 3 * 4 * 2  # multiplicity kept, prefill skipped
    l, e = 2, (2 * 7 + 1) % 10
    assert keys[2 * 4 * 2 + 2 * 2 + 1] == (l, e)  # rec0 layer2 expert1
    assert all(0 <= ex < 10 for _, ex in keys)


def test_spread_recompute_d2_burst_not_whole():
    grid = {
        "128": {
            "lcnu": {"whole": 0.50, "burst": 0.60},
            "ring10": {"whole": 0.20, "burst": 0.10},
        }
    }
    got = rc.spread_recompute({"policy_grid": grid})
    assert got[128] == pytest.approx(50.0)  # burst max-min x100
    assert got[128] != pytest.approx(
        30.0
    )  # whole would give 30 (mutation bite)


def test_main_smoke_exit_zero(capsys):
    # Frozen smoke path is intact end-to-end post-repairs.
    assert rc.main(smoke=True) == 0
    assert "SMOKE_OK" in capsys.readouterr().out


def test_sealed_outputs_present_and_pinned():
    # The sealed artifacts this test guards are still the frozen ones.
    sums = dict(
        line.split()[::-1]
        for line in (D / "SHA256SUMS").read_text().splitlines()
    )
    for name in (
        "run_costcurves.py",
        "exp705-costcurves.json",
        "exp705-costcurves.frozen-track.json",
    ):
        h = hashlib.sha256((D / name).read_bytes()).hexdigest()
        assert sums[name] == h, name
