import pytest

from moe_infinity.models.modeling_arctic.modeling_arctic import ArcticRMSNorm
from moe_infinity.models.modeling_deepseek_v2.modeling_deepseek import (
    DeepseekV2RMSNorm,
)
from moe_infinity.models.modeling_deepseek_v3.modeling_deepseek import (
    DeepseekV3RMSNorm,
)
from moe_infinity.models.modeling_grok.modeling_grok1 import (
    RMSNorm as GrokRMSNorm,
)
from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    reference_rmsnorm,
    requires_cuda,
)

torch = pytest.importorskip("torch")


def reference_rmsnorm_upcast(x, weight, eps: float):
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    return reference_rmsnorm(x_f32, weight, eps).to(input_dtype)


@requires_cuda
@pytest.mark.parametrize("hidden_size", [128, 2048])
class TestGrokRMSNorm:
    def test_hidden_size_128(self, hidden_size):
        if hidden_size != 128:
            pytest.skip("Parameterized test")
        self._test_rmsnorm(hidden_size)

    def test_hidden_size_2048(self, hidden_size):
        if hidden_size != 2048:
            pytest.skip("Parameterized test")
        self._test_rmsnorm(hidden_size)

    def _test_rmsnorm(self, hidden_size: int):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        norm = (
            GrokRMSNorm(hidden_size=hidden_size, eps=1e-5)
            .cuda()
            .bfloat16()
            .eval()
        )
        norm.scale.data.fill_(1.0)

        x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device="cuda")

        with torch.no_grad():
            custom_out = norm(x)
            ref_out = reference_rmsnorm_upcast(
                x, norm.scale, eps=norm.variance_epsilon
            )

        torch.testing.assert_close(
            custom_out, ref_out, rtol=BF16_RTOL, atol=BF16_ATOL
        )

    def test_near_zero_input(self, hidden_size):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        norm = (
            GrokRMSNorm(hidden_size=hidden_size, eps=1e-5)
            .cuda()
            .bfloat16()
            .eval()
        )
        norm.scale.data.fill_(1.0)

        x = torch.full(
            (4, hidden_size), 1e-8, dtype=torch.bfloat16, device="cuda"
        )

        with torch.no_grad():
            out = norm(x)
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"


@requires_cuda
@pytest.mark.parametrize("hidden_size", [128, 2048])
class TestDeepseekV2RMSNorm:
    def test_hidden_size_128(self, hidden_size):
        if hidden_size != 128:
            pytest.skip("Parameterized test")
        self._test_rmsnorm(hidden_size)

    def test_hidden_size_2048(self, hidden_size):
        if hidden_size != 2048:
            pytest.skip("Parameterized test")
        self._test_rmsnorm(hidden_size)

    def _test_rmsnorm(self, hidden_size: int):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        norm = (
            DeepseekV2RMSNorm(hidden_size=hidden_size, eps=1e-6)
            .cuda()
            .bfloat16()
            .eval()
        )

        x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device="cuda")

        with torch.no_grad():
            custom_out = norm(x)
            ref_out = reference_rmsnorm_upcast(
                x, norm.weight, eps=norm.variance_epsilon
            )

        torch.testing.assert_close(
            custom_out, ref_out, rtol=BF16_RTOL, atol=BF16_ATOL
        )

    def test_near_zero_input(self, hidden_size):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        norm = (
            DeepseekV2RMSNorm(hidden_size=hidden_size, eps=1e-6)
            .cuda()
            .bfloat16()
            .eval()
        )

        x = torch.full(
            (4, hidden_size), 1e-8, dtype=torch.bfloat16, device="cuda"
        )

        with torch.no_grad():
            out = norm(x)
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"


@requires_cuda
@pytest.mark.parametrize("hidden_size", [128, 2048])
class TestDeepseekV3RMSNorm:
    def test_hidden_size_128(self, hidden_size):
        if hidden_size != 128:
            pytest.skip("Parameterized test")
        self._test_rmsnorm(hidden_size)

    def test_hidden_size_2048(self, hidden_size):
        if hidden_size != 2048:
            pytest.skip("Parameterized test")
        self._test_rmsnorm(hidden_size)

    def _test_rmsnorm(self, hidden_size: int):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        norm = (
            DeepseekV3RMSNorm(hidden_size=hidden_size, eps=1e-6)
            .cuda()
            .bfloat16()
            .eval()
        )

        x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device="cuda")

        with torch.no_grad():
            custom_out = norm(x)
            ref_out = reference_rmsnorm_upcast(
                x, norm.weight, eps=norm.variance_epsilon
            )

        torch.testing.assert_close(
            custom_out, ref_out, rtol=BF16_RTOL, atol=BF16_ATOL
        )

    def test_near_zero_input(self, hidden_size):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        norm = (
            DeepseekV3RMSNorm(hidden_size=hidden_size, eps=1e-6)
            .cuda()
            .bfloat16()
            .eval()
        )

        x = torch.full(
            (4, hidden_size), 1e-8, dtype=torch.bfloat16, device="cuda"
        )

        with torch.no_grad():
            out = norm(x)
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"


@requires_cuda
@pytest.mark.parametrize("hidden_size", [128, 2048])
class TestArcticRMSNorm:
    def test_hidden_size_128(self, hidden_size):
        if hidden_size != 128:
            pytest.skip("Parameterized test")
        self._test_rmsnorm(hidden_size)

    def test_hidden_size_2048(self, hidden_size):
        if hidden_size != 2048:
            pytest.skip("Parameterized test")
        self._test_rmsnorm(hidden_size)

    def _test_rmsnorm(self, hidden_size: int):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        norm = (
            ArcticRMSNorm(hidden_size=hidden_size, eps=1e-6)
            .cuda()
            .bfloat16()
            .eval()
        )

        x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device="cuda")

        with torch.no_grad():
            custom_out = norm(x)
            ref_out = reference_rmsnorm_upcast(
                x, norm.weight, eps=norm.variance_epsilon
            )

        torch.testing.assert_close(
            custom_out, ref_out, rtol=BF16_RTOL, atol=BF16_ATOL
        )

    def test_near_zero_input(self, hidden_size):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        norm = (
            ArcticRMSNorm(hidden_size=hidden_size, eps=1e-6)
            .cuda()
            .bfloat16()
            .eval()
        )

        x = torch.full(
            (4, hidden_size), 1e-8, dtype=torch.bfloat16, device="cuda"
        )

        with torch.no_grad():
            out = norm(x)
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"
