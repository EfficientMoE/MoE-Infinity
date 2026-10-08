"""Pin the shipped-behavior VL wording in docs/model-compatibility.md
(issue #221 phase 4): Qwen3.5 residency/MTP and the MiniMax-M3 evidence
boundary.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMPAT = (ROOT / "docs/model-compatibility.md").read_text(encoding="utf-8")


def _contains_all(text: str, values: tuple[str, ...]) -> None:
    missing = [value for value in values if value not in text]
    assert not missing, f"missing documentation contracts: {missing}"


def test_documents_qwen35_vision_residency_and_unused_mtp():
    _contains_all(
        COMPAT,
        (
            "visual.*",
            "MTP",
            "unused",
        ),
    )


def test_documents_minimax_m3_license_and_evidence_boundary():
    _contains_all(
        COMPAT,
        (
            "MiniMax-M3",
            "MiniMax Community License",
            "BF16",
            "eager attention",
            "native engine off",
            "no real-checkpoint evidence",
        ),
    )
