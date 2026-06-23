# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile

import pytest
import torch
from transformers import AutoTokenizer

E2E_CACHE_DIR = os.environ.get(
    "MOE_E2E_CACHE_DIR",
    "/mnt/raid0sata2/xly/moe-infinity-e2e-checkpoints/hf_cache",
)

GPTQ_MODEL = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
AWQ_MODEL = "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ"
HQQ_MODEL = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
FP_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 20


def _model_available(model_id: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir(E2E_CACHE_DIR)
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                return True
    except Exception:
        pass
    return False


def _make_offload_dir(tag: str) -> str:
    d = tempfile.mkdtemp(prefix=f"moe_e2e_{tag}_")
    return d


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for e2e"
)


class TestGPTQEndToEnd:
    @requires_gpu
    @pytest.mark.skipif(
        not _model_available(GPTQ_MODEL),
        reason=f"{GPTQ_MODEL} not downloaded to {E2E_CACHE_DIR}",
    )
    def test_gptq_mixtral_loads_and_generates(self):
        os.environ["TRANSFORMERS_CACHE"] = E2E_CACHE_DIR
        from moe_infinity import MoE

        offload_dir = _make_offload_dir("gptq")
        try:
            config = {
                "offload_path": offload_dir,
                "device_memory_ratio": 0.75,
            }
            model = MoE(GPTQ_MODEL, config)

            tokenizer = AutoTokenizer.from_pretrained(
                GPTQ_MODEL, cache_dir=E2E_CACHE_DIR
            )
            input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(
                "cuda:0"
            )
            output_ids = model.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS
            )
            output_text = tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )

            assert len(output_text) > len(
                PROMPT
            ), f"GPTQ model generated no new text. Output: {output_text!r}"
            assert output_ids.shape[1] > input_ids.shape[1], (
                f"GPTQ model did not produce new tokens. "
                f"Input: {input_ids.shape[1]}, Output: {output_ids.shape[1]}"
            )
            print(f"\n[GPTQ] Generated: {output_text}")
        finally:
            shutil.rmtree(offload_dir, ignore_errors=True)


class TestAWQEndToEnd:
    @requires_gpu
    @pytest.mark.skipif(
        not _model_available(AWQ_MODEL),
        reason=f"{AWQ_MODEL} not downloaded to {E2E_CACHE_DIR}",
    )
    def test_awq_mixtral_loads_and_generates(self):
        os.environ["TRANSFORMERS_CACHE"] = E2E_CACHE_DIR
        from moe_infinity import MoE

        offload_dir = _make_offload_dir("awq")
        try:
            config = {
                "offload_path": offload_dir,
                "device_memory_ratio": 0.75,
            }
            model = MoE(AWQ_MODEL, config)

            tokenizer = AutoTokenizer.from_pretrained(
                AWQ_MODEL, cache_dir=E2E_CACHE_DIR
            )
            input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(
                "cuda:0"
            )
            output_ids = model.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS
            )
            output_text = tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )

            assert len(output_text) > len(
                PROMPT
            ), f"AWQ model generated no new text. Output: {output_text!r}"
            assert output_ids.shape[1] > input_ids.shape[1], (
                f"AWQ model did not produce new tokens. "
                f"Input: {input_ids.shape[1]}, Output: {output_ids.shape[1]}"
            )
            print(f"\n[AWQ] Generated: {output_text}")
        finally:
            shutil.rmtree(offload_dir, ignore_errors=True)


class TestHQQFailFastEndToEnd:
    @pytest.mark.skipif(
        not _model_available(HQQ_MODEL),
        reason=f"{HQQ_MODEL} not downloaded to {E2E_CACHE_DIR}",
    )
    def test_hqq_mixtral_raises_clear_error(self):
        os.environ["TRANSFORMERS_CACHE"] = E2E_CACHE_DIR
        from moe_infinity import MoE

        offload_dir = _make_offload_dir("hqq")
        try:
            config = {
                "offload_path": offload_dir,
                "device_memory_ratio": 0.75,
            }
            with pytest.raises(ValueError, match="(?i)hqq"):
                MoE(HQQ_MODEL, config)
        finally:
            shutil.rmtree(offload_dir, ignore_errors=True)


class TestFullPrecisionRegressionEndToEnd:
    @requires_gpu
    @pytest.mark.skipif(
        not _model_available(FP_MODEL),
        reason=f"{FP_MODEL} not downloaded to {E2E_CACHE_DIR}",
    )
    def test_fp_deepseek_v2_lite_loads_and_generates(self):
        os.environ["TRANSFORMERS_CACHE"] = E2E_CACHE_DIR
        from moe_infinity import MoE

        offload_dir = _make_offload_dir("fp")
        try:
            config = {
                "offload_path": offload_dir,
                "device_memory_ratio": 0.75,
            }
            model = MoE(FP_MODEL, config)

            tokenizer = AutoTokenizer.from_pretrained(
                FP_MODEL, cache_dir=E2E_CACHE_DIR, trust_remote_code=True
            )
            input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(
                "cuda:0"
            )
            output_ids = model.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS
            )
            output_text = tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )

            assert len(output_text) > len(
                PROMPT
            ), f"FP model generated no new text. Output: {output_text!r}"
            assert output_ids.shape[1] > input_ids.shape[1], (
                f"FP model did not produce new tokens. "
                f"Input: {input_ids.shape[1]}, Output: {output_ids.shape[1]}"
            )
            print(f"\n[FP16] Generated: {output_text}")
        finally:
            shutil.rmtree(offload_dir, ignore_errors=True)
