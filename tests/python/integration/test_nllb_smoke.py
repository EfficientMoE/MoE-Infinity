import os

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("MOE_NLLB_SMOKE") != "1",
    reason="Set MOE_NLLB_SMOKE=1 to run the heavy NLLB-MoE end-to-end smoke.",
)
def test_nllb_translation_smoke(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from transformers import AutoTokenizer

    from moe_infinity import MoE

    checkpoint = "facebook/nllb-moe-54b"
    model = MoE(
        checkpoint,
        {"offload_path": str(tmp_path / "nllb"), "device_memory_ratio": 0.75},
    )
    tok = AutoTokenizer.from_pretrained(checkpoint, src_lang="eng_Latn")
    ids = tok("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    out = model.generate(
        ids,
        max_new_tokens=16,
        forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    assert out.shape[1] >= ids.shape[1] + 1
    assert len(text) > 0
