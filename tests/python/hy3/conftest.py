# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import os

import pytest

DEFAULT_CKPT = (
    "/mnt/raid0nvme0/public/huggingface/hub/"
    "models--tencent--Hy3-FP8/snapshots"
)


def _resolve_ckpt() -> str:
    ckpt = os.environ.get("HY3_FP8_CKPT", "")
    if ckpt:
        return ckpt
    if os.path.isdir(DEFAULT_CKPT):
        snaps = sorted(os.listdir(DEFAULT_CKPT))
        if snaps:
            return os.path.join(DEFAULT_CKPT, snaps[-1])
    return DEFAULT_CKPT


@pytest.fixture(scope="session")
def hy3_ckpt_dir():
    ckpt = _resolve_ckpt()
    if not os.path.exists(os.path.join(ckpt, "model.safetensors.index.json")):
        pytest.skip(f"Hy3-FP8 checkpoint not found at {ckpt}")
    return ckpt
