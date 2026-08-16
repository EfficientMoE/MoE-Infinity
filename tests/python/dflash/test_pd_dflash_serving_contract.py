"""CPU-only contract tests for the PD-DFlash serving runner scaffolding.

Task 2 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``.
Exercises the pure, torch-free surface of ``pd_dflash_serving`` -- the §8
contract matrix, device/offload guards, the byte-schema observation row, and the
append-without-overwrite JSON writer -- so the runner logic is regression-locked
without a GPU. No CUDA, checkpoint, or network.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.dflash.pd_dflash_serving import (
    B2_UNBOUNDED_EXPERT_BYTES,
    BLOCKED_STATUS,
    REQUIRED_CONCURRENCY,
    REQUIRED_DRAFTS,
    REQUIRED_MODELS,
    append_observation,
    build_b2_serving_config,
    build_contract_matrix,
    load_observations,
    main,
    make_observation_row,
    parse_args,
    require_offloaded,
    validate_device_identity,
)
from benchmarks.dflash.report import REQUIRED_METRICS


def _full_metrics() -> dict[str, float]:
    return {metric: 1.0 for metric in REQUIRED_METRICS}


# ---------------------------------------------------------------------------
# §8 contract matrix
# ---------------------------------------------------------------------------


def test_contract_matrix_pins_models_drafts_sweeps_and_metrics():
    contract = build_contract_matrix()
    assert set(contract["models"]) == set(REQUIRED_MODELS)
    assert contract["drafts"] == dict(REQUIRED_DRAFTS)
    assert set(contract["block_sizes"]) == {8, 16}
    assert set(contract["concurrency"]) == set(REQUIRED_CONCURRENCY)
    assert set(contract["baselines"]) == {"B0", "B1", "B2", "B3"}
    assert tuple(contract["required_metrics"]) == REQUIRED_METRICS
    assert contract["nvtx_ranges"][0] == "dflash_draft"


def test_dry_run_contract_cli_is_cpu_safe(capsys):
    assert main(["--dry-run-contract"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == build_contract_matrix()


# ---------------------------------------------------------------------------
# device + offload guards
# ---------------------------------------------------------------------------


def test_validate_device_identity_accepts_rtx_pro_6000():
    validate_device_identity("NVIDIA RTX PRO 6000 Blackwell", (12, 0))


def test_validate_device_identity_rejects_wrong_name_or_capability():
    with pytest.raises(RuntimeError, match="RTX PRO 6000"):
        validate_device_identity("NVIDIA H100 PCIe", (9, 0))
    with pytest.raises(RuntimeError, match="capability"):
        validate_device_identity("NVIDIA RTX PRO 6000", (9, 0))


def test_require_offloaded_refuses_resident_b0_b1_b2():
    for baseline in ("B0", "B1", "B2"):
        with pytest.raises(RuntimeError, match="offloaded"):
            require_offloaded(baseline, 0)
        require_offloaded(baseline, 1)
    # B3 is the resident upper bound: zero offloaded experts is legal.
    require_offloaded("B3", 0)


# ---------------------------------------------------------------------------
# observation row schema
# ---------------------------------------------------------------------------


def test_observation_row_requires_full_metric_schema():
    row = make_observation_row(
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        draft="z-lab/Qwen3-Coder-30B-A3B-DFlash",
        baseline="B1",
        block_size=16,
        concurrency=8,
        repeat=0,
        metrics=_full_metrics(),
    )
    for metric in REQUIRED_METRICS:
        assert metric in row
    assert row["baseline"] == "B1" and row["block_size"] == 16


def test_observation_row_rejects_missing_metric():
    incomplete = _full_metrics()
    del incomplete["wasted_prefetch_bytes"]
    with pytest.raises(ValueError, match="wasted_prefetch_bytes"):
        make_observation_row(
            model="m",
            draft="d",
            baseline="B0",
            block_size=8,
            concurrency=1,
            repeat=0,
            metrics=incomplete,
        )


def test_blocked_row_carries_status_and_no_metrics():
    row = make_observation_row(
        model="m",
        draft="d",
        baseline="B2",
        block_size=8,
        concurrency=1,
        repeat=0,
        metrics={},
        status=BLOCKED_STATUS,
    )
    assert row["status"] == BLOCKED_STATUS
    assert "output_tokens_per_second" not in row


def test_b2_serving_config_enables_token_only_verify_scheduler():
    from moe_infinity.serving.scheduler import _resolve_verify_config

    config = build_b2_serving_config(
        block_size=16,
        concurrency=4,
        num_layers=24,
        num_kv_heads=8,
        head_dim=64,
        dtype="bfloat16",
        eos_token_id=7,
        num_kv_blocks=128,
        device_memory_ratio=0.85,
    )
    assert config["verify_token_budget"] == 16
    assert config["verify_token_deficit_cap"] == 16
    assert config["verify_expert_byte_budget"] == B2_UNBOUNDED_EXPERT_BYTES
    assert config["verify_expert_byte_deficit_cap"] == B2_UNBOUNDED_EXPERT_BYTES
    assert config["max_batch_size"] == 4
    assert config["block_size"] == 16

    resolved = _resolve_verify_config(
        token_budget=config["verify_token_budget"],
        expert_byte_budget=config["verify_expert_byte_budget"],
        token_deficit_cap=config["verify_token_deficit_cap"],
        expert_byte_deficit_cap=config["verify_expert_byte_deficit_cap"],
    )
    assert resolved.enabled is True


# ---------------------------------------------------------------------------
# append-without-overwrite JSON writer
# ---------------------------------------------------------------------------


def test_append_observation_appends_distinct_and_refuses_duplicates(tmp_path):
    out = str(tmp_path / "raw.json")
    base = dict(
        model="m",
        draft="d",
        baseline="B0",
        block_size=8,
        concurrency=1,
        repeat=0,
        metrics=_full_metrics(),
    )
    append_observation(out, make_observation_row(**base))
    append_observation(out, make_observation_row(**{**base, "concurrency": 2}))
    assert len(load_observations(out)) == 2

    with pytest.raises(ValueError, match="refusing to overwrite"):
        append_observation(out, make_observation_row(**base))
    assert len(load_observations(out)) == 2


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults_cover_the_full_matrix():
    args = parse_args(
        [
            "--model",
            "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "--draft",
            "z-lab/Qwen3-Coder-30B-A3B-DFlash",
            "--offload-dir",
            "/tmp/offload",
            "--output",
            "/tmp/out.json",
        ]
    )
    assert args.baselines == ("B0", "B1", "B2", "B3")
    assert args.block_sizes == (8, 16)
    assert args.concurrency == (1, 2, 4, 8, 16, 32)
    assert args.seed == 1408
    assert args.device_memory_ratio < 0.9


def test_run_requires_model_draft_offload_output():
    with pytest.raises(SystemExit, match="missing required args"):
        main(["--output", "/tmp/out.json"])
