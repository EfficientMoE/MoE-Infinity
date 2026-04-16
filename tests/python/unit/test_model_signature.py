import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_infinity.runtime.model_offload import (
    _compute_config_fingerprint,
    _validate_model_signature,
    _write_model_signature,
)


def make_mixtral_config(**overrides):
    defaults = dict(
        model_type="mixtral",
        architectures=["MixtralForCausalLM"],
        num_hidden_layers=32,
        hidden_size=4096,
        vocab_size=32000,
        intermediate_size=14336,
        num_local_experts=8,
        torch_dtype="torch.bfloat16",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_compute_config_fingerprint_deterministic():
    config = make_mixtral_config()

    first = _compute_config_fingerprint(config)
    second = _compute_config_fingerprint(config)

    assert first == second


def test_compute_config_fingerprint_different_configs():
    first = make_mixtral_config(hidden_size=4096)
    second = make_mixtral_config(hidden_size=8192)

    assert _compute_config_fingerprint(first) != _compute_config_fingerprint(
        second
    )


def test_compute_config_fingerprint_handles_missing_fields():
    config = SimpleNamespace(
        model_type="mixtral",
        architectures=["MixtralForCausalLM"],
        num_hidden_layers=32,
        hidden_size=4096,
        vocab_size=32000,
        intermediate_size=14336,
        torch_dtype="torch.bfloat16",
    )

    fingerprint = _compute_config_fingerprint(config)

    assert len(fingerprint) == 64
    int(fingerprint, 16)


def test_write_model_signature_creates_file(tmp_path: Path):
    config = make_mixtral_config()

    _write_model_signature(str(tmp_path), "test-model", config)

    sig_path = tmp_path / "model_signature.json"
    assert sig_path.exists()

    with sig_path.open("r") as f:
        data = json.load(f)

    assert data["model_name"] == "test-model"
    assert len(data["config_fingerprint"]) == 64
    int(data["config_fingerprint"], 16)
    assert data["signature_version"] == 1


def test_write_model_signature_atomic(tmp_path: Path):
    config = make_mixtral_config()

    _write_model_signature(str(tmp_path), "test-model", config)

    with (tmp_path / "model_signature.json").open("r") as f:
        data = json.load(f)

    assert data["model_name"] == "test-model"


def test_validate_signature_passes_on_match(tmp_path: Path):
    config = make_mixtral_config()

    _write_model_signature(str(tmp_path), "test-model", config)

    _validate_model_signature(str(tmp_path), "test-model", config)


def test_validate_signature_raises_on_name_mismatch(tmp_path: Path):
    config = make_mixtral_config()

    _write_model_signature(str(tmp_path), "model-A", config)

    with pytest.raises(ValueError) as exc_info:
        _validate_model_signature(str(tmp_path), "model-B", config)

    message = str(exc_info.value)
    assert "model-A" in message
    assert "model-B" in message


def test_validate_signature_raises_on_config_mismatch(tmp_path: Path):
    _write_model_signature(str(tmp_path), "test-model", make_mixtral_config())

    with pytest.raises(ValueError) as exc_info:
        _validate_model_signature(
            str(tmp_path),
            "test-model",
            make_mixtral_config(hidden_size=8192),
        )

    assert "config mismatch" in str(exc_info.value).lower()


def test_validate_signature_raises_on_name_and_config_mismatch(tmp_path: Path):
    _write_model_signature(str(tmp_path), "model-A", make_mixtral_config())

    with pytest.raises(ValueError) as exc_info:
        _validate_model_signature(
            str(tmp_path),
            "model-B",
            make_mixtral_config(hidden_size=8192),
        )

    assert "model-A" in str(exc_info.value)


def test_validate_signature_legacy_cache_warns_and_stamps(
    tmp_path: Path, caplog
):
    config = make_mixtral_config()

    with caplog.at_level("WARNING"):
        _validate_model_signature(str(tmp_path), "test-model", config)

    sig_path = tmp_path / "model_signature.json"
    assert sig_path.exists()
    assert any(record.levelname == "WARNING" for record in caplog.records)

    with sig_path.open("r") as f:
        data = json.load(f)

    assert data["model_name"] == "test-model"


def test_validate_signature_corrupted_json(tmp_path: Path):
    sig_path = tmp_path / "model_signature.json"
    sig_path.write_bytes(b"{not json")

    with pytest.raises(ValueError) as exc_info:
        _validate_model_signature(
            str(tmp_path), "test-model", make_mixtral_config()
        )

    assert "corrupted" in str(exc_info.value).lower()


def test_validate_signature_missing_keys(tmp_path: Path):
    sig_path = tmp_path / "model_signature.json"
    sig_path.write_text(json.dumps({"foo": "bar"}))

    with pytest.raises(ValueError) as exc_info:
        _validate_model_signature(
            str(tmp_path), "test-model", make_mixtral_config()
        )

    assert "corrupted" in str(exc_info.value).lower()


def test_validate_signature_non_dict_json(tmp_path: Path):
    sig_path = tmp_path / "model_signature.json"
    sig_path.write_text(json.dumps([]))  # valid JSON, but not a dict

    with pytest.raises(ValueError) as exc_info:
        _validate_model_signature(
            str(tmp_path), "test-model", make_mixtral_config()
        )

    assert "corrupted" in str(exc_info.value).lower()
