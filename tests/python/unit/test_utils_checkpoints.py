import json
from pathlib import Path

import pytest

from moe_infinity.utils.checkpoints import get_checkpoint_paths


def test_get_checkpoint_paths_with_single_file(tmp_path: Path):
    ckpt_file = tmp_path / "pytorch_model.bin"
    ckpt_file.write_bytes(b"weights")

    result = get_checkpoint_paths(str(ckpt_file))

    assert result == [str(ckpt_file)]


def test_get_checkpoint_paths_with_bin_in_directory(tmp_path: Path):
    ckpt_file = tmp_path / "pytorch_model.bin"
    ckpt_file.write_bytes(b"weights")

    result = get_checkpoint_paths(str(tmp_path))

    assert result == [str(ckpt_file)]


def test_get_checkpoint_paths_with_index_json(tmp_path: Path):
    shard_a = tmp_path / "shard_a.bin"
    shard_b = tmp_path / "shard_b.bin"
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")

    index = {
        "weight_map": {
            "layer.0": shard_a.name,
            "layer.1": shard_b.name,
            "layer.2": shard_a.name,
        }
    }
    index_file = tmp_path / "model.index.json"
    index_file.write_text(json.dumps(index))

    result = get_checkpoint_paths(str(index_file))

    assert result == [str(shard_a), str(shard_b)]


def test_get_checkpoint_paths_missing_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        get_checkpoint_paths(str(tmp_path / "missing.bin"))
