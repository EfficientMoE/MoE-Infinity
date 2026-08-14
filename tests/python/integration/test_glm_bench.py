import os

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("MOE_GLM_TINY") != "1", reason="set MOE_GLM_TINY=1"
)
def test_glm_bench_writes_csv(tmp_path):
    import csv

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from benchmarks.performance_model.bench_glm import run

    out = str(tmp_path / "b.csv")
    run(out, quick=True)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) >= 1
    r = rows[0]
    assert float(r["decode_tok_s"]) > 0
    assert float(r["mean_accept_len"]) >= 1.0
    assert int(r["pred_flops_per_token"]) > 0
    assert r["pred_bound"] in ("compute", "hbm", "pcie")
