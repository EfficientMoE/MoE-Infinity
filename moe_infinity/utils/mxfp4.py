from collections.abc import Sequence
from typing import Optional, Protocol


class _QuantizationConfigLike(Protocol):
    @property
    def quant_method(self) -> Optional[str]: ...

    @property
    def modules_to_not_convert(self) -> Optional[Sequence[str]]: ...


class _ConfigWithQuantization(Protocol):
    @property
    def quantization_config(self) -> Optional[_QuantizationConfigLike]: ...


def is_mxfp4_quantized(config: _ConfigWithQuantization) -> bool:
    try:
        quant_config = config.quantization_config
    except AttributeError:
        return False
    if quant_config is None:
        return False
    return quant_config.quant_method == "mxfp4"


def get_mxfp4_modules_to_not_convert(
    config: _ConfigWithQuantization,
) -> list[str]:
    try:
        quant_config = config.quantization_config
    except AttributeError:
        return []
    if quant_config is None:
        return []
    modules = quant_config.modules_to_not_convert
    if modules is None:
        return []
    if isinstance(modules, str):
        return []
    return list(modules)


def identify_mxfp4_pairs(weight_names: Sequence[str]) -> list[tuple[str, str]]:
    name_set = set(weight_names)
    pairs: list[tuple[str, str]] = []
    for name in weight_names:
        if name.endswith("_blocks"):
            base = name[: -len("_blocks")]
            scales_name = f"{base}_scales"
            if scales_name in name_set:
                pairs.append((name, scales_name))
    return pairs
