import warnings

import torch

from moe_infinity.utils.config import ArcherConfig
from moe_infinity.utils.fp8_store import extract_fp8_scales, strip_scale_tensors


def test_flag_default_off():
    assert getattr(ArcherConfig(), "glm_fp8_in_store", None) is False


def test_extract_fp8_scales():
    sd = {
        "a.weight": torch.zeros(4, 4),
        "a.weight_scale_inv": torch.ones(1, 1),
        "b.norm.weight": torch.zeros(4),
    }
    scales = extract_fp8_scales(sd)
    assert set(scales.keys()) == {"a.weight"}


def test_strip_scale_tensors():
    sd = {"a.weight": torch.zeros(2), "a.weight_scale_inv": torch.ones(1)}
    strip_scale_tensors(sd)
    assert "a.weight_scale_inv" not in sd and "a.weight" in sd
