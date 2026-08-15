# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Protocol

GPTQ_PACKED_SUFFIXES = (".qweight", ".qzeros")
GPTQ_COMPONENT_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")
MARLIN_PACKED_SUFFIXES = (".qweight", ".scales")
MARLIN_COMPONENT_SUFFIXES = (".qweight", ".scales", ".bits", ".group_size")


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


def _get_scalar(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def detect_marlin_weights(state_dict: dict) -> bool:
    qweight_keys = [name for name in state_dict if name.endswith(".qweight")]
    if not qweight_keys:
        return False
    if not any(name.endswith(".scales") for name in state_dict):
        return False

    for qweight_key in qweight_keys:
        prefix = qweight_key[: -len(".qweight")]
        scales_key = prefix + ".scales"
        if scales_key not in state_dict:
            return False

        qweight = state_dict[qweight_key]
        scales = state_dict[scales_key]
        qshape = getattr(qweight, "shape", None)
        sshape = getattr(scales, "shape", None)
        if (
            qshape is None
            or sshape is None
            or len(qshape) != 2
            or len(sshape) != 2
        ):
            return False
        if str(getattr(qweight, "dtype", None)) != "torch.int32":
            return False
        if qshape[0] % 8 != 0:
            return False
        if qshape[1] != sshape[1] * 2:
            return False

    return True


def is_marlin_compatible(state_dict: dict) -> bool:
    qweight_keys = [name for name in state_dict if name.endswith(".qweight")]
    if not qweight_keys:
        return False
    if not any(name.endswith(".scales") for name in state_dict):
        return False

    for qweight_key in qweight_keys:
        prefix = qweight_key[: -len(".qweight")]
        scales_key = prefix + ".scales"
        if scales_key not in state_dict:
            return False

        bits = _get_scalar(state_dict.get(prefix + ".bits"))
        if bits is not None and bits != 4:
            return False

        group_size = _get_scalar(state_dict.get(prefix + ".group_size"))
        if group_size is not None and group_size not in (-1, 128):
            return False

    return True
