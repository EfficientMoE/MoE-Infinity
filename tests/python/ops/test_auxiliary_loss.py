"""
Tests for AddAuxiliaryLoss, the custom torch.autograd.Function in DeepSeek-V2.

This module verifies:
1. Forward pass: AddAuxiliaryLoss.apply(x, loss) returns x unchanged (identity)
2. Backward pass: gradients flow correctly to both x and loss
3. Numerical gradient correctness via gradcheck
"""

import pytest

from moe_infinity.models.modeling_deepseek_v2.modeling_deepseek import (
    AddAuxiliaryLoss,
)
from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    seed_everything,
)

torch = pytest.importorskip("torch")


class TestAddAuxiliaryLossForward:
    """Tests for the forward pass of AddAuxiliaryLoss."""

    def test_forward_returns_x_unchanged(self, seed_everything):
        """Forward should return x unchanged (identity pass-through)."""
        x = torch.randn(10, dtype=torch.float32, requires_grad=True)
        loss = torch.tensor(0.5, requires_grad=True)
        out = AddAuxiliaryLoss.apply(x, loss)
        assert torch.equal(out, x), "Forward should return x unchanged"

    def test_forward_returns_x_unchanged_float64(self, seed_everything):
        """Forward should return x unchanged with float64 precision."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        loss = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        out = AddAuxiliaryLoss.apply(x, loss)
        assert torch.equal(out, x), "Forward should return x unchanged"

    def test_forward_returns_x_unchanged_no_grad_loss(self, seed_everything):
        """Forward should return x unchanged when loss does not require grad."""
        x = torch.randn(10, dtype=torch.float32, requires_grad=True)
        loss = torch.tensor(0.5, requires_grad=False)
        out = AddAuxiliaryLoss.apply(x, loss)
        assert torch.equal(out, x), "Forward should return x unchanged"

    @requires_cuda
    def test_forward_returns_x_unchanged_cuda(self, seed_everything):
        """Forward should return x unchanged on CUDA."""
        x = torch.randn(
            10, dtype=torch.float32, requires_grad=True, device="cuda"
        )
        loss = torch.tensor(0.5, requires_grad=True, device="cuda")
        out = AddAuxiliaryLoss.apply(x, loss)
        assert torch.equal(out, x), "Forward should return x unchanged"


class TestAddAuxiliaryLossBackward:
    """Tests for the backward pass of AddAuxiliaryLoss."""

    def test_backward_x_gradient_identity(self, seed_everything):
        """
        x.grad should be all-ones (identity gradient through the forward pass).
        Since forward is identity (y=x), backward passes grad_output directly to x.
        """
        x = torch.randn(10, dtype=torch.float32, requires_grad=True)
        loss = torch.tensor(0.5, requires_grad=True)
        out = AddAuxiliaryLoss.apply(x, loss)
        out.sum().backward()

        # Gradient for x should be identity: grad_x = grad_output = ones
        expected_x_grad = torch.ones_like(x)
        torch.testing.assert_close(
            x.grad, expected_x_grad, rtol=BF16_RTOL, atol=BF16_ATOL
        )

    def test_backward_loss_gradient_ones(self, seed_everything):
        """
        loss.grad should be a ones tensor (not the sum of incoming gradients).

        The AddAuxiliaryLoss.backward returns torch.ones(1, dtype=ctx.dtype)
        as the gradient for loss when loss.requires_grad is True.
        """
        x = torch.randn(10, dtype=torch.float32, requires_grad=True)
        loss = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
        out = AddAuxiliaryLoss.apply(x, loss)
        out.sum().backward()

        # Gradient for loss should be ones(1), not the sum of grad_output
        expected_loss_grad = torch.tensor(1.0, dtype=torch.float32)
        torch.testing.assert_close(
            loss.grad, expected_loss_grad, rtol=BF16_RTOL, atol=BF16_ATOL
        )

    def test_backward_loss_requires_grad_false(self, seed_everything):
        """When loss.requires_grad=False, loss.grad should be None."""
        x = torch.randn(10, dtype=torch.float32, requires_grad=True)
        loss = torch.tensor(0.5, requires_grad=False)
        out = AddAuxiliaryLoss.apply(x, loss)
        out.sum().backward()

        # x should still have gradient
        assert x.grad is not None, "x should have gradient"
        # loss should not have gradient
        assert (
            loss.grad is None
        ), "loss should have no gradient when requires_grad=False"

    @requires_cuda
    def test_backward_cuda(self, seed_everything):
        """Backward pass should work correctly on CUDA."""
        x = torch.randn(
            10, dtype=torch.float32, requires_grad=True, device="cuda"
        )
        loss = torch.tensor(
            0.5, dtype=torch.float32, requires_grad=True, device="cuda"
        )
        out = AddAuxiliaryLoss.apply(x, loss)
        out.sum().backward()

        expected_x_grad = torch.ones_like(x)
        expected_loss_grad = torch.tensor(
            1.0, dtype=torch.float32, device="cuda"
        )

        torch.testing.assert_close(
            x.grad, expected_x_grad, rtol=BF16_RTOL, atol=BF16_ATOL
        )
        torch.testing.assert_close(
            loss.grad, expected_loss_grad, rtol=BF16_RTOL, atol=BF16_ATOL
        )


class TestAddAuxiliaryLossGradcheck:
    """
    Numerical gradient checks using torch.autograd.gradcheck.

    Note: The AddAuxiliaryLoss is a special autograd function where:
    - Forward: y = x (identity, loss is not used)
    - Backward: adds a constant gradient to loss regardless of input

    This is the "auxiliary loss trick" - it allows adding a loss term to the
    computation graph without affecting the forward pass. Standard gradcheck
    may fail because the numerical gradient differs from analytical.
    """

    @pytest.mark.xfail(
        reason="AddAuxiliaryLoss is a non-standard autograd function where "
        "forward doesn't depend on loss but backward adds constant gradient"
    )
    def test_gradcheck_basic(self, seed_everything):
        """Gradcheck may fail due to the auxiliary loss trick behavior."""
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        loss = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            AddAuxiliaryLoss.apply, (x, loss), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_loss_no_grad(self, seed_everything):
        """Gradcheck should pass when loss does not require grad."""
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        loss = torch.tensor(0.5, dtype=torch.float64, requires_grad=False)

        torch.autograd.gradcheck(
            AddAuxiliaryLoss.apply, (x, loss), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    @pytest.mark.xfail(
        reason="AddAuxiliaryLoss is a non-standard autograd function where "
        "forward doesn't depend on loss but backward adds constant gradient"
    )
    def test_gradcheck_larger_input(self, seed_everything):
        """Gradcheck may fail due to the auxiliary loss trick behavior."""
        x = torch.randn(20, dtype=torch.float64, requires_grad=True)
        loss = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            AddAuxiliaryLoss.apply, (x, loss), eps=1e-6, atol=1e-4, rtol=1e-4
        )


class TestAddAuxiliaryLossBF16:
    """Tests for bfloat16 support."""

    @requires_cuda
    def test_backward_bf16(self, seed_everything):
        """Backward pass should work correctly with bfloat16."""
        x = torch.randn(
            10, dtype=torch.bfloat16, requires_grad=True, device="cuda"
        )
        loss = torch.tensor(
            0.5, dtype=torch.bfloat16, requires_grad=True, device="cuda"
        )
        out = AddAuxiliaryLoss.apply(x, loss)
        out.sum().backward()

        expected_x_grad = torch.ones_like(x)
        expected_loss_grad = torch.tensor(
            1.0, dtype=torch.bfloat16, device="cuda"
        )

        torch.testing.assert_close(
            x.grad, expected_x_grad, rtol=BF16_RTOL, atol=BF16_ATOL
        )
        torch.testing.assert_close(
            loss.grad, expected_loss_grad, rtol=BF16_RTOL, atol=BF16_ATOL
        )
