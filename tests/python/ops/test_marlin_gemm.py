import importlib

import pytest

from tests.python.ops.conftest import requires_cuda

torch = importlib.import_module("torch")


def _marlin_available():
    try:
        importlib.import_module("moe_infinity._marlin")
        return True
    except ImportError:
        return False


requires_marlin = pytest.mark.skipif(
    not _marlin_available(), reason="moe_infinity._marlin not compiled"
)


@pytest.fixture
def marlin_module():
    return importlib.import_module("moe_infinity.kernel.marlin_gemm")


class TestMarlinQuantize:
    @requires_cuda
    @pytest.mark.parametrize("groupsize", [-1, 128])
    @pytest.mark.parametrize("K,N", [(4096, 4096), (4096, 11008), (2048, 8192)])
    def test_pack_roundtrip(self, marlin_module, K, N, groupsize):
        weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
        packed, scales = marlin_module.marlin_quantize(weight, groupsize)

        assert packed.dtype == torch.int32
        assert scales.dtype == torch.float16
        assert packed.shape == (K // 16, N * 16 // 8)

    @requires_cuda
    @pytest.mark.parametrize("groupsize", [-1, 128])
    def test_quantize_deterministic(self, marlin_module, groupsize):
        weight = torch.randn(2048, 256, dtype=torch.float16, device="cuda")
        p1, s1 = marlin_module.marlin_quantize(weight, groupsize)
        p2, s2 = marlin_module.marlin_quantize(weight, groupsize)
        assert torch.equal(p1, p2)
        assert torch.equal(s1, s2)


class TestMarlinGemm:
    @requires_cuda
    @requires_marlin
    @pytest.mark.parametrize("groupsize", [-1, 128])
    @pytest.mark.parametrize(
        "M,K,N", [(1, 4096, 4096), (16, 4096, 11008), (32, 2048, 8192)]
    )
    def test_correctness(self, marlin_module, M, K, N, groupsize):
        weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
        packed, scales = marlin_module.marlin_quantize(weight, groupsize)
        workspace = marlin_module.prepare_workspace(N, torch.device("cuda"))
        input_tensor = torch.randn(M, K, dtype=torch.float16, device="cuda")

        output = marlin_module.marlin_gemm(
            input_tensor, packed, scales, workspace
        )
        reference = marlin_module.reference_dequant_gemm(
            input_tensor, packed, scales, K, N, groupsize
        )

        assert output.shape == (M, N)
        assert output.dtype == torch.float16
        assert torch.allclose(
            output.float(), reference.float(), atol=1e-2, rtol=1e-2
        )

    @requires_cuda
    @requires_marlin
    def test_workspace_size(self, marlin_module):
        N = 4096
        workspace = marlin_module.prepare_workspace(N, torch.device("cuda"))
        assert workspace.shape == (N // 128 * 16,)
        assert workspace.dtype == torch.int32


class TestMarlinAvailability:
    def test_is_marlin_available_returns_bool(self, marlin_module):
        result = marlin_module.is_marlin_available()
        assert isinstance(result, bool)
