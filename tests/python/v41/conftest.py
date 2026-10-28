# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import os
import sys

import pytest

CKPT_ENV = "MOE_V41_FLASH_CKPT"
OFFICIAL_ENV = "MOE_V41_OFFICIAL_DIR"


@pytest.fixture(scope="session")
def v41_ckpt_dir() -> str:
    ckpt = os.environ.get(CKPT_ENV, "")
    if not ckpt or not os.path.isdir(ckpt):
        pytest.skip(f"set {CKPT_ENV} to a local DeepSeek-V4.1-Flash checkpoint")
    return ckpt


@pytest.fixture(scope="session")
def v41_official_module(v41_ckpt_dir: str):
    """Import the pinned reference inference/model.py (see OFFICIAL_PIN.md
    in moe_store/wrappers/deepseek_v41 for the revision and review notes)."""
    official = os.environ.get(OFFICIAL_ENV, "")
    if not official or not os.path.isdir(official):
        pytest.skip(f"set {OFFICIAL_ENV} to the pinned official inference dir")
    sys.path.insert(0, official)
    import model as official_model  # the reference inference/model.py

    return official_model
