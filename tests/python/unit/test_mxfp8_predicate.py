# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from moe_infinity.runtime.model_offload import _has_fp8_blockwise


def _config(quant_method):
    return SimpleNamespace(quantization_config={"quant_method": quant_method})


def test_mxfp8_is_not_blockwise_fp8():
    assert _has_fp8_blockwise(_config("mxfp8")) is False


def test_plain_fp8_is_blockwise_fp8():
    assert _has_fp8_blockwise(_config("fp8")) is True
