from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_adaptive_documentation_contract():
    text = "\n".join(
        (ROOT / path).read_text()
        for path in (
            "docs/adaptive-expert-precision.md",
            "docs/configuration.md",
            "docs/model-compatibility.md",
            "docs/benchmarking.md",
        )
    )
    required = (
        "adaptive_expert_precision",
        "adaptive_hbm_budget_bytes",
        "CURRENT",
        "derivative-index.v1.json",
        "quality-attestation.v1.json",
        "ReleasedAdaptiveEntry",
        "manifest_unapproved",
        "ExpertResidencyManager",
        "GPT-OSS MXFP4",
        "GLM-5.2 FP8",
        "DeepSeek-V4-Flash FP4",
        "GPTQ/AWQ",
        "deterministic-v1",
        "peak_accounted_bytes",
        "TTFT",
        "TPOT",
        "release_gate",
        "no speedup is guaranteed",
        '"adaptive_expert_precision": false',
        "restart",
        "Preserve",
    )
    assert not [value for value in required if value not in text]
