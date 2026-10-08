"""GPU-gated: factory output matches the deprecated MoE.generate() greedily.

This is the migration-safety anchor for the v0.3.0 removal: while
``MoE.generate()`` still exists, prove the factory path is token-identical on
the supported greedy path. After the removal PR drops ``MoE.generate()``, the
baseline arm is removed and the factory smoke assertions stay.

Gated behind both a CUDA device and ``MOE_FACTORY_GPU`` so it skips cleanly in
CPU CI; run it on the GPU rig and paste the output into the PR.
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.network]

CHECKPOINT = "deepseek-ai/DeepSeek-V2-Lite-Chat"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not os.environ.get("MOE_FACTORY_GPU"),
    reason="MOE_FACTORY_GPU unset (GPU-gated factory harness)",
)
def test_factory_matches_deprecated_generate_greedy() -> None:
    from transformers import AutoTokenizer

    from moe_infinity import MoE
    from moe_infinity.serving.factory import (
        create_serving_engine,
        generate_tokens,
    )

    offload = os.environ.get(
        "MOE_FACTORY_OFFLOAD", "/tmp/opencode/moe-offload/factory"
    )
    os.makedirs(offload, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT, trust_remote_code=True
    )
    model = MoE(
        CHECKPOINT,
        {"offload_path": offload, "device_memory_ratio": 0.75},
    )

    input_ids = tokenizer(
        "The capital of France is", return_tensors="pt"
    ).input_ids.to("cuda:0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        baseline = model.generate(input_ids, do_sample=False, max_new_tokens=16)
    baseline_new = [int(t) for t in baseline[0, input_ids.shape[1] :].tolist()]

    engine = create_serving_engine(model, tokenizer=tokenizer)
    [factory_new] = generate_tokens(
        engine, [input_ids[0].tolist()], max_new_tokens=16
    )
    assert factory_new == baseline_new
