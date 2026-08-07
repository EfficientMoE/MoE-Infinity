import pytest
import torch

from moe_infinity.utils.fp8 import FP8_BLOCK, dequant_fp8_blockwise


def test_fp8_block_constant():
    assert FP8_BLOCK == 128


def test_dequant_known_4x4_block():
    weight = torch.ones(4, 4, dtype=torch.float32)
    scale = torch.tensor([[2.0]], dtype=torch.float32)
    result = dequant_fp8_blockwise(
        weight, scale, dtype=torch.bfloat16, block_size=4
    )
    expected = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
    assert result.shape == (4, 4)
    assert result.dtype == torch.bfloat16
    assert torch.allclose(result.float(), expected.float(), atol=1e-3)


def test_dequant_non_divisible_block_boundary():
    n, k = 5, 5
    block_size = 4
    weight = torch.ones(n, k, dtype=torch.float32)
    scale = torch.ones(2, 2, dtype=torch.float32) * 3.0
    result = dequant_fp8_blockwise(
        weight, scale, dtype=torch.bfloat16, block_size=block_size
    )
    expected = torch.full((n, k), 3.0, dtype=torch.bfloat16)
    assert result.shape == (n, k)
    assert torch.allclose(result.float(), expected.float(), atol=1e-3)


def test_dequant_per_block_scale_values():
    block_size = 2
    weight = torch.ones(4, 4, dtype=torch.float32)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    result = dequant_fp8_blockwise(
        weight, scale, dtype=torch.float32, block_size=block_size
    )
    expected = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected, atol=1e-3)


def test_back_compat_import_from_fp8_expert():
    from moe_infinity.models.deepseek_v4.fp8_expert import FP8_BLOCK as blk
    from moe_infinity.models.deepseek_v4.fp8_expert import (
        dequant_fp8_blockwise as fn,
    )

    assert blk == 128
    assert callable(fn)


def test_back_compat_same_function():
    from moe_infinity.models.deepseek_v4.fp8_expert import (
        dequant_fp8_blockwise as fn_expert,
    )
    from moe_infinity.utils.fp8 import dequant_fp8_blockwise as fn_utils

    assert fn_utils is fn_expert
