import csv
import io
import math

import pytest

from benchmarks.performance_model.report_glm import summarize

FIXTURE_CSV = """\
model,batch,seq_len,gen_len,decode_tok_s,mtp_tok_s,mean_accept_len,peak_mem_bytes,pred_flops_per_token,pred_hbm_bytes_per_token,pred_bound
modelA,1,4,16,200.0,100.0,1.5,10000000,4000000,2000000,hbm
modelB,2,8,32,400.0,600.0,2.0,20000000,8000000,2000000,compute
"""


def test_summarize_row_count(tmp_path):
    p = tmp_path / "bench.csv"
    p.write_text(FIXTURE_CSV)
    s = summarize(str(p))
    assert s["n_rows"] == 2


def test_summarize_arithmetic_intensity(tmp_path):
    p = tmp_path / "bench.csv"
    p.write_text(FIXTURE_CSV)
    s = summarize(str(p))
    rows = s["rows"]
    assert math.isclose(
        rows[0]["arithmetic_intensity"], 4000000 / 2000000, rel_tol=1e-6
    )
    assert math.isclose(
        rows[1]["arithmetic_intensity"], 8000000 / 2000000, rel_tol=1e-6
    )


def test_summarize_mtp_speedup(tmp_path):
    p = tmp_path / "bench.csv"
    p.write_text(FIXTURE_CSV)
    s = summarize(str(p))
    rows = s["rows"]
    assert math.isclose(rows[0]["mtp_speedup"], 100.0 / 200.0, rel_tol=1e-6)
    assert math.isclose(rows[1]["mtp_speedup"], 600.0 / 400.0, rel_tol=1e-6)


def test_summarize_averages(tmp_path):
    p = tmp_path / "bench.csv"
    p.write_text(FIXTURE_CSV)
    s = summarize(str(p))
    assert math.isclose(
        s["avg_decode_tok_s"], (200.0 + 400.0) / 2, rel_tol=1e-6
    )
    assert math.isclose(s["avg_mtp_tok_s"], (100.0 + 600.0) / 2, rel_tol=1e-6)
    assert math.isclose(s["avg_mean_accept_len"], (1.5 + 2.0) / 2, rel_tol=1e-6)


def test_summarize_bounds(tmp_path):
    p = tmp_path / "bench.csv"
    p.write_text(FIXTURE_CSV)
    s = summarize(str(p))
    assert s["bounds"] == ["hbm", "compute"]


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not available")
    is None,
    reason="matplotlib not available",
)
def test_make_plots_creates_pngs(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    from benchmarks.performance_model.report_glm import make_plots

    p = tmp_path / "bench.csv"
    p.write_text(FIXTURE_CSV)
    out_dir = tmp_path / "plots"
    saved = make_plots(str(p), str(out_dir))
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 2, f"Expected >=2 PNGs, got {pngs}"
