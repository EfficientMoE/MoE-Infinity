from __future__ import annotations

from typing import Dict

import torch


def extract_fp8_scales(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    scales = {}
    for k in list(state_dict.keys()):
        if k.endswith("_scale_inv"):
            base = k[: -len("_scale_inv")]
            if base in state_dict:
                scales[base] = state_dict[k]
    return scales


def strip_scale_tensors(state_dict: Dict[str, torch.Tensor]) -> None:
    for k in list(state_dict.keys()):
        if k.endswith("_scale_inv"):
            del state_dict[k]
