# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Optional, Protocol

GPTQ_PACKED_SUFFIXES = (".qweight", ".qzeros")
GPTQ_COMPONENT_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")


class _QuantizationConfigLike(Protocol):
    @property
    def quant_method(self) -> Optional[str]: ...


class _ConfigWithQuantization(Protocol):
    @property
    def quantization_config(self) -> Optional[_QuantizationConfigLike]: ...


def _get_quant_field(quant_config, field: str):
    if isinstance(quant_config, dict):
        return quant_config.get(field)
    return getattr(quant_config, field, None)


def is_gptq_quantized(config: _ConfigWithQuantization) -> bool:
    try:
        quant_config = config.quantization_config
    except AttributeError:
        return False
    if quant_config is None:
        return False
    return _get_quant_field(quant_config, "quant_method") == "gptq"


def get_gptq_bits(config: _ConfigWithQuantization) -> int:
    try:
        quant_config = config.quantization_config
    except AttributeError:
        return 4
    if quant_config is None:
        return 4
    bits = _get_quant_field(quant_config, "bits")
    return bits if bits is not None else 4


def get_gptq_group_size(config: _ConfigWithQuantization) -> int:
    try:
        quant_config = config.quantization_config
    except AttributeError:
        return 128
    if quant_config is None:
        return 128
    gs = _get_quant_field(quant_config, "group_size")
    return gs if gs is not None else 128


def is_gptq_packed_tensor(name: str) -> bool:
    # qweight/qzeros are bit-packed INT32; casting to float destroys the structure
    return any(name.endswith(s) for s in GPTQ_PACKED_SUFFIXES)


def is_gptq_component(name: str) -> bool:
    return any(name.endswith(s) for s in GPTQ_COMPONENT_SUFFIXES)
