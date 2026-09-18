import re
from pathlib import Path


def test_model_offload_delegates_transformers_model_overrides_to_boundary():
    source = (
        Path(__file__).parents[3]
        / "moe_infinity"
        / "runtime"
        / "model_offload.py"
    ).read_text()

    assert re.search(r"^\s*import transformers", source, re.MULTILINE) is None
    assert "transformers.models" not in source
