"""Opt-in RTX PRO 6000 gate for the PD-DFlash B0-B3 serving runner.

Task 2 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``.
Collection is side-effect free: the module imports only the CPU-safe runner
scaffolding and the single test is skipped unless ``MOE_DFLASH_SERVING_GPU=1``,
so ``pytest`` never initialises CUDA, loads a checkpoint, hits the network, or
creates offload state when the gate is absent (``1 skipped``).
"""

from __future__ import annotations

import json
import os

import pytest

from benchmarks.dflash.pd_dflash_serving import (
    REQUIRED_CONCURRENCY,
    REQUIRED_DRAFTS,
    REQUIRED_MODELS,
    main,
)

RUN_GPU = os.environ.get("MOE_DFLASH_SERVING_GPU") == "1"
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.skipif(not RUN_GPU, reason="set MOE_DFLASH_SERVING_GPU=1"),
]


def test_dry_run_contract_matches_required_matrix(capsys):
    assert main(["--dry-run-contract"]) == 0
    contract = json.loads(capsys.readouterr().out)

    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct" in contract["models"]
    assert "openai/gpt-oss-20b" in contract["models"]
    assert set(contract["models"]) == set(REQUIRED_MODELS)

    assert (
        contract["drafts"]["Qwen/Qwen3-Coder-30B-A3B-Instruct"]
        == "z-lab/Qwen3-Coder-30B-A3B-DFlash"
    )
    assert (
        contract["drafts"]["openai/gpt-oss-20b"] == "z-lab/gpt-oss-20b-DFlash"
    )
    assert contract["drafts"] == dict(REQUIRED_DRAFTS)

    assert set(contract["block_sizes"]) == {8, 16}
    assert set(contract["concurrency"]) == set(REQUIRED_CONCURRENCY)
    assert set(contract["concurrency"]) == {1, 2, 4, 8, 16, 32}
    assert set(contract["baselines"]) == {"B0", "B1", "B2", "B3"}
