# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Engine-side store_meta.json merge for synthetic-FP8 stores (#234 Phase 1)."""

import json
from types import SimpleNamespace

import pytest

from moe_infinity.entrypoints.big_modeling import (
    _merge_store_quantization_config,
)

_FP8_QC = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "weight_block_size": [128, 128],
}


def _write_meta(store_dir, quantize_experts="fp8", qc=_FP8_QC):
    meta = {
        "store_format_version": 1,
        "quantize_experts": quantize_experts,
        "quantization_config": qc,
    }
    (store_dir / "store_meta.json").write_text(
        json.dumps(meta, sort_keys=True) + "\n"
    )


def test_absent_metadata_is_noop(tmp_path):
    cfg = SimpleNamespace()
    out = _merge_store_quantization_config(cfg, str(tmp_path))
    assert getattr(out, "quantization_config", None) is None


def test_empty_offload_path_is_noop():
    cfg = SimpleNamespace()
    out = _merge_store_quantization_config(cfg, "")
    assert getattr(out, "quantization_config", None) is None


def test_fp8_meta_merges_into_bf16_config(tmp_path):
    _write_meta(tmp_path)
    cfg = SimpleNamespace()
    out = _merge_store_quantization_config(cfg, str(tmp_path))
    assert out.quantization_config == _FP8_QC


def test_identical_existing_config_is_noop(tmp_path):
    _write_meta(tmp_path)
    cfg = SimpleNamespace(quantization_config=dict(_FP8_QC))
    out = _merge_store_quantization_config(cfg, str(tmp_path))
    assert out.quantization_config == _FP8_QC


def test_conflicting_existing_config_raises(tmp_path):
    _write_meta(tmp_path)
    cfg = SimpleNamespace(quantization_config={"quant_method": "gptq"})
    with pytest.raises(RuntimeError, match="already-quantized"):
        _merge_store_quantization_config(cfg, str(tmp_path))
