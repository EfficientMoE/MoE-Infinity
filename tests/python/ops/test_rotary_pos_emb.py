import importlib
import inspect

import pytest

torch = pytest.importorskip("torch")

from moe_infinity.models.model_utils import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_deepseek,
)
from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    seed_everything,
)


def _build_rope_inputs(seq_len: int, device: str = "cuda"):
    batch_size = 2
    num_heads = 8
    head_dim = 128
    max_position = seq_len + 17

    q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    angles = torch.randn(
        max_position,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    cos = torch.cos(angles).to(torch.bfloat16)
    sin = torch.sin(angles).to(torch.bfloat16)

    position_ids = torch.randint(
        low=0,
        high=max_position,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )

    return q, k, cos, sin, position_ids


def _call_hf_mixtral_apply(
    q,
    k,
    cos,
    sin,
    position_ids,
):
    mixtral_modeling = importlib.import_module(
        "transformers.models.mixtral.modeling_mixtral"
    )
    hf_apply_rope = mixtral_modeling.apply_rotary_pos_emb

    hf_params = inspect.signature(hf_apply_rope).parameters
    use_unsqueeze_dim = "unsqueeze_dim" in hf_params
    hf_cos = cos[position_ids]
    hf_sin = sin[position_ids]

    preindexed_kwargs = {"unsqueeze_dim": 1} if use_unsqueeze_dim else {}
    try:
        return hf_apply_rope(
            q,
            k,
            hf_cos,
            hf_sin,
            **preindexed_kwargs,
        )
    except Exception as preindexed_error:
        if "position_ids" not in hf_params:
            raise

        legacy_kwargs = {"position_ids": position_ids}
        if use_unsqueeze_dim:
            legacy_kwargs["unsqueeze_dim"] = 1

        try:
            return hf_apply_rope(
                q,
                k,
                cos,
                sin,
                **legacy_kwargs,
            )
        except Exception:
            raise preindexed_error


@requires_cuda
@pytest.mark.parametrize("seq_len", [1, 64, 512])
def test_apply_rotary_pos_emb_mixtral_matches_hf(seed_everything, seq_len):
    try:
        importlib.import_module("transformers.models.mixtral.modeling_mixtral")
    except ModuleNotFoundError:
        pytest.skip("transformers Mixtral module not available")

    q, k, cos, sin, position_ids = _build_rope_inputs(seq_len)

    custom_q, custom_k = apply_rotary_pos_emb(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        unsqueeze_dim=1,
    )
    hf_q, hf_k = _call_hf_mixtral_apply(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
    )

    if torch.equal(custom_q, hf_q) and torch.equal(custom_k, hf_k):
        assert torch.equal(custom_q, hf_q), "Local version differs from HF"
        assert torch.equal(custom_k, hf_k), "Local version differs from HF"
        return

    torch.testing.assert_close(custom_q, hf_q, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(custom_k, hf_k, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_cuda
@pytest.mark.parametrize("seq_len", [1, 64, 512])
@pytest.mark.parametrize(
    "variant_name,module_path,fn_name",
    [
        (
            "deepseek_v3",
            "transformers.models.deepseek_v3.modeling_deepseek_v3",
            "apply_rotary_pos_emb",
        ),
    ],
)
def test_apply_rotary_pos_emb_deepseek_matches_variants(
    seed_everything, seq_len, variant_name, module_path, fn_name
):
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        pytest.skip(f"{variant_name} module not available")

    reference_apply = getattr(module, fn_name)
    q, k, cos, sin, position_ids = _build_rope_inputs(seq_len)

    custom_q, custom_k = apply_rotary_pos_emb_deepseek(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        unsqueeze_dim=1,
    )
    ref_q, ref_k = reference_apply(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        unsqueeze_dim=1,
    )

    torch.testing.assert_close(custom_q, ref_q, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(custom_k, ref_k, rtol=BF16_RTOL, atol=BF16_ATOL)
