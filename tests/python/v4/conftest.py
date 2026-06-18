# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import importlib.util
import os
import sys

import pytest

if "flash_attn" in sys.modules and not hasattr(
    sys.modules["flash_attn"], "__spec__"
):
    sys.modules["flash_attn"].__spec__ = importlib.util.spec_from_loader(
        "flash_attn", loader=None
    )

DEFAULT_CKPT = (
    "/mnt/raid0nvme0/public/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V4-Flash/snapshots/"
    "6976c7ff1b30a1b2cb7805021b8ba4684041f136"
)


@pytest.fixture(scope="session")
def v4_ckpt_dir():
    ckpt = os.environ.get("DSV4_FLASH_CKPT", DEFAULT_CKPT)
    if not os.path.exists(os.path.join(ckpt, "model.safetensors.index.json")):
        pytest.skip(f"DeepSeek-V4-Flash checkpoint not found at {ckpt}")
    return ckpt


@pytest.fixture(scope="session")
def indexer(v4_ckpt_dir):
    from moe_infinity.models.deepseek_v4 import DeepSeekV4ExpertTensorIndexer

    return DeepSeekV4ExpertTensorIndexer(v4_ckpt_dir)


def _pick_free_cuda_device(min_free_mib: int = 2048):
    import torch

    if not torch.cuda.is_available():
        return None
    best, best_free = None, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        free_mib = free // (1024 * 1024)
        if free_mib > best_free:
            best, best_free = i, free_mib
    if best is None or best_free < min_free_mib:
        return None
    return f"cuda:{best}"


@pytest.fixture(scope="session")
def device():
    dev = _pick_free_cuda_device()
    return dev if dev is not None else "cpu"
