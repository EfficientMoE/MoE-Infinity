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
