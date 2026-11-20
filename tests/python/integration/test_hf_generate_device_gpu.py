"""GPU + cached model: MoE.generate on the HF path accepts CPU input_ids.

The HF path (use_native_engine=False) used to hand CPU ``input_ids`` to
transformers, whose rotary embedding then mixed a CPU ``position_ids`` with
CUDA weights. Skipped without a CUDA device or when the checkpoint is not in
the local Hugging Face cache (nothing is downloaded).
"""

from __future__ import annotations

import os
import sys
import warnings
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

CHECKPOINT = "allenai/OLMoE-1B-7B-0125-Instruct"


def _checkpoint_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(CHECKPOINT, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not _checkpoint_cached(), reason=f"{CHECKPOINT} is not in the local cache"
)
def test_hf_path_generate_accepts_cpu_input_ids(tmp_path, monkeypatch) -> None:
    # tests/conftest.py stubs a missing flash_attn with a MagicMock, which makes
    # MoE (and transformers) believe it is installed; probe for the real one.
    if isinstance(sys.modules.get("flash_attn"), MagicMock):
        monkeypatch.delitem(sys.modules, "flash_attn")

    from transformers import AutoTokenizer

    from moe_infinity import MoE

    offload = os.environ.get("MOE_HF_DEVICE_OFFLOAD", str(tmp_path / "offload"))
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = MoE(
        CHECKPOINT,
        {
            "offload_path": offload,
            "device_memory_ratio": 0.75,
            "use_native_engine": False,
        },
    )
    cpu_ids = tokenizer("hi", return_tensors="pt").input_ids
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from_cpu = model.generate(cpu_ids, do_sample=False, max_new_tokens=4)
        from_cuda = model.generate(
            cpu_ids.to("cuda:0"), do_sample=False, max_new_tokens=4
        )

    # The result comes back on the device the ids came from.
    assert from_cpu.device.type == "cpu"
    assert from_cuda.device == torch.device("cuda:0")
    assert from_cpu.shape == from_cuda.shape == (1, cpu_ids.shape[1] + 4)
    assert from_cpu.tolist() == from_cuda.cpu().tolist()
