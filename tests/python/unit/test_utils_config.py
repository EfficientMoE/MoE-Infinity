# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportAny=false, reportUnusedCallResult=false

import os
from pathlib import Path

import pytest

from moe_infinity.utils.config import ArcherConfig


def test_load_from_json_sets_paths_and_threads(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    config = ArcherConfig.load_from_json(
        {
            "offload_path": "/tmp/offload",
            "trace_capacity": 123,
            "prefetch": True,
        }
    )

    assert config.offload_path == "/tmp/offload"
    assert config.trace_capacity == 123
    assert config.prefetch is True
    assert config.device_per_node == 2
    assert config.perfect_cache_file == os.path.join(
        "/tmp/offload", "perfect_cache"
    )


def test_load_from_file_sets_trace_path(tmp_path: Path):
    config_path = tmp_path / "config.json"
    trace_file = tmp_path / "trace.json"
    config_path.write_text(
        '{"offload_path": "/tmp/offload", "trace_path": "%s"}'
        % trace_file.as_posix()
    )

    config = ArcherConfig.load_from_file(config_path)

    assert config.trace_path == os.path.abspath(trace_file)


def test_trace_path_directory_raises(tmp_path: Path):
    trace_dir = tmp_path / "trace_dir"
    trace_dir.mkdir()

    with pytest.raises(ValueError):
        ArcherConfig(offload_path=str(tmp_path), trace_path=trace_dir)


def test_kv_cache_fields_default(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.warns(UserWarning, match="auto-set to 0.15"):
        config = ArcherConfig(offload_path="/tmp")
    assert config.kv_cache_memory_ratio == 0.15
    assert config.use_native_engine is True
    assert config.enable_attention_offload is False
    assert config.enable_kv_cache_offload is False
    assert config.attention_backend == "default"


def test_kv_cache_memory_ratio_validation(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError):
        ArcherConfig(
            offload_path="/tmp",
            device_memory_ratio=0.7,
            kv_cache_memory_ratio=0.5,
        )


def test_backwards_compat_old_config(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(
        offload_path="/tmp",
        device_memory_ratio=0.75,
        use_native_engine=False,
    )
    assert config.kv_cache_memory_ratio == 0.0
    assert config.enable_kv_cache_offload is False


def test_native_engine_autocorrects_kv_cache_ratio(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.warns(UserWarning, match="auto-set to 0.15"):
        config = ArcherConfig(
            offload_path="/tmp",
            use_native_engine=True,
            kv_cache_memory_ratio=0.0,
        )
    assert config.kv_cache_memory_ratio == pytest.approx(0.15)


def test_adaptive_fields_default_disabled(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(
        offload_path="/tmp",
        use_native_engine=False,
    )
    assert config.adaptive_expert_precision is False
    assert config.adaptive_hbm_budget_bytes == 0
    assert config.adaptive_variant_build is False
    assert config.adaptive_derivative_root is None


def test_adaptive_budget_must_be_positive_when_enabled(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(
        ValueError, match="adaptive_hbm_budget_bytes must be positive"
    ):
        ArcherConfig(
            offload_path="/tmp",
            use_native_engine=False,
            adaptive_expert_precision=True,
            adaptive_hbm_budget_bytes=0,
        )


def test_adaptive_threshold_ordering_validated(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError):
        ArcherConfig(
            offload_path="/tmp",
            use_native_engine=False,
            adaptive_expert_precision=True,
            adaptive_hbm_budget_bytes=1024,
            adaptive_promotion_threshold=0.2,
            adaptive_demotion_threshold=0.5,
        )


def test_adaptive_derivative_root_resolves_from_offload(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(
        offload_path="/tmp/offload",
        use_native_engine=False,
        adaptive_expert_precision=True,
        adaptive_hbm_budget_bytes=2048,
    )
    assert config.adaptive_derivative_root == os.path.join(
        "/tmp/offload", "adaptive_derivatives"
    )
