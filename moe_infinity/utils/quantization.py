# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

SUPPORTED_QUANT_METHODS = frozenset({"gptq", "awq"})

# MXFP4 has its own runtime path (utils/mxfp4.py); excluded here so
# validate_quantization_support() does not reject GPT-OSS as "unsupported".
_HANDLED_ELSEWHERE_METHODS = frozenset({"mxfp4"})

_QUANT_TENSOR_SUFFIXES = frozenset(
    {
        "qweight",
        "qzeros",
        "scales",
        "g_idx",
    }
)

_UNSUPPORTED_ERROR_MESSAGES: Dict[str, str] = {
    "hqq": (
        "HQQ quantization is not supported by MoE-Infinity. "
        "Use the full-precision or GPTQ/AWQ variant of the model instead."
    ),
    "bitsandbytes": (
        "bitsandbytes quantization is not supported by MoE-Infinity. "
        "Use the full-precision or GPTQ/AWQ variant of the model instead."
    ),
    "gguf": (
        "GGUF format is not supported by MoE-Infinity. "
        "Use llama.cpp or Ollama for GGUF models, or use the "
        "full-precision/GPTQ/AWQ variant with MoE-Infinity."
    ),
    "exl2": (
        "EXL2 quantization is not supported by MoE-Infinity. "
        "Use ExLlamaV2 for EXL2 models, or use the "
        "full-precision/GPTQ/AWQ variant with MoE-Infinity."
    ),
}


@dataclass
class QuantizationInfo:
    method: str
    supported: bool
    bits: Optional[int] = None
    group_size: Optional[int] = None
    config_dict: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


def _try_read_json(filepath: str) -> Optional[Dict[str, Any]]:
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError, TypeError):
        pass
    return None


def _detect_from_config_attr(config: object) -> Optional[QuantizationInfo]:
    qc = getattr(config, "quantization_config", None)
    if qc is None or not isinstance(qc, dict):
        return None

    method = qc.get("quant_method", "").lower().strip()
    if not method or method in _HANDLED_ELSEWHERE_METHODS:
        return None

    return QuantizationInfo(
        method=method,
        supported=method in SUPPORTED_QUANT_METHODS,
        bits=qc.get("bits", qc.get("w_bit")),
        group_size=qc.get("group_size", qc.get("q_group_size")),
        config_dict=dict(qc),
        source="config.quantization_config",
    )


_QUANT_JSON_FILES = [
    ("quantize_config.json", "gptq"),
    ("quant_config.json", "awq"),
    ("quantization_config.json", None),
]


def _detect_from_checkpoint_files(
    checkpoint_path: str,
) -> Optional[QuantizationInfo]:
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        return None

    for filename, expected_method in _QUANT_JSON_FILES:
        filepath = os.path.join(checkpoint_path, filename)
        data = _try_read_json(filepath)
        if data is None:
            continue

        method = data.get("quant_method", data.get("type", "")).lower().strip()
        if not method and expected_method:
            method = expected_method

        if not method or method in _HANDLED_ELSEWHERE_METHODS:
            continue

        return QuantizationInfo(
            method=method,
            supported=method in SUPPORTED_QUANT_METHODS,
            bits=data.get("bits", data.get("w_bit")),
            group_size=data.get("group_size", data.get("q_group_size")),
            config_dict=dict(data),
            source=filename,
        )

    gguf_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".gguf")]
    if gguf_files:
        return QuantizationInfo(
            method="gguf",
            supported=False,
            config_dict={},
            source="file",
        )

    return None


def detect_quantization(
    config: object, checkpoint_path: str
) -> Optional[QuantizationInfo]:
    result = _detect_from_config_attr(config)
    if result is not None:
        return result

    return _detect_from_checkpoint_files(checkpoint_path)


def validate_quantization_support(
    info: QuantizationInfo, model_name: str = ""
) -> None:
    if info.supported:
        return

    base_msg = _UNSUPPORTED_ERROR_MESSAGES.get(
        info.method,
        f"Quantization method '{info.method}' is not supported by MoE-Infinity. "
        "Use the full-precision or GPTQ/AWQ variant of the model instead.",
    )

    if model_name:
        base_msg = f"Model '{model_name}': {base_msg}"

    raise ValueError(base_msg)


def should_cast_tensor(
    tensor_name: str, quant_info: Optional[QuantizationInfo]
) -> bool:
    if quant_info is None:
        return True

    if quant_info.method not in SUPPORTED_QUANT_METHODS:
        return True

    suffix = (
        tensor_name.rsplit(".", 1)[-1] if "." in tensor_name else tensor_name
    )
    return suffix not in _QUANT_TENSOR_SUFFIXES


def get_quant_dtype_for_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    quant_info: Optional[QuantizationInfo],
) -> Optional[torch.dtype]:
    if quant_info is None:
        return None

    if not should_cast_tensor(tensor_name, quant_info):
        return tensor.dtype

    return None
