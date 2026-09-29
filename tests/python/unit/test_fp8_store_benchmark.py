# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from benchmarks.eval.perplexity import evaluate_perplexity, resolve_dataset
from benchmarks.fp8_store.bench_store_parity import _json_safe


def test_wikitext_uses_canonical_hugging_face_dataset_id():
    assert resolve_dataset("wikitext", "test") == (
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        "test",
    )


def test_perplexity_disables_grad_without_creating_inference_tensors():
    execution_modes = []

    class Tokenizer:
        def __call__(self, texts, **_kwargs):
            batch = len(texts)
            return {
                "input_ids": torch.tensor([[1, 2, 3]]).repeat(batch, 1),
                "attention_mask": torch.ones((batch, 3), dtype=torch.long),
            }

    class Target:
        def eval(self):
            return self

        def __call__(self, input_ids, **_kwargs):
            execution_modes.append(
                (
                    torch.is_grad_enabled(),
                    torch.is_inference_mode_enabled(),
                )
            )
            batch, length = input_ids.shape
            return SimpleNamespace(
                logits=torch.zeros(batch, length, 4, device=input_ids.device)
            )

    model = SimpleNamespace(model=Target())
    ppl, nll, processed = evaluate_perplexity(
        model,
        Tokenizer(),
        ["one sample"],
        seq_length=8,
        max_samples=1,
        batch_size=1,
    )

    assert execution_modes == [(False, False)]
    assert ppl > 0
    assert nll > 0
    assert processed == 1


def test_json_safe_converts_native_tensor_metrics_recursively():
    payload = {
        "scalar": torch.tensor(7),
        "matrix": torch.tensor([[1, 2], [3, 4]]),
        "nested": (torch.tensor(5.5),),
    }

    assert _json_safe(payload) == {
        "scalar": 7,
        "matrix": [[1, 2], [3, 4]],
        "nested": [5.5],
    }
