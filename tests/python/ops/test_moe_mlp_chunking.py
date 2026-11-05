"""Native expert path (C++ MoEMLP::forward) above kMaxTokens rows (#309).

MoEMLP stages activations in buffers of kMaxTokens = 8192 rows; larger
batches must be processed in chunks instead of aborting the process.
"""

import pytest

from tests.python.ops.conftest import requires_cuda

torch = pytest.importorskip("torch")
F = torch.nn.functional

K_MAX_TOKENS = 8192
MIXTRAL_HIDDEN = 4096
MIXTRAL_INTER = 14336
DTYPE_BFLOAT16 = 0
MIXTRAL_EXPERT_TYPE = 4  # params ordered [w1 (gate), w2 (down), w3 (up)]


def _store():
    try:
        from moe_infinity import _store
    except ImportError as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"moe_infinity._store unavailable: {exc}")
    if not hasattr(_store, "_moe_mlp_for_test"):
        pytest.skip("moe_infinity._store built without _moe_mlp_for_test")
    return _store


@pytest.fixture(scope="module")
def mixtral_expert():
    store = _store()
    gen = torch.Generator(device="cuda").manual_seed(309)

    def rand(*shape, scale):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).to(
            torch.bfloat16
        )

    H, I = MIXTRAL_HIDDEN, MIXTRAL_INTER
    w1, w3 = rand(I, H, scale=0.02), rand(I, H, scale=0.02)
    w2 = rand(H, I, scale=0.02)
    mlp = store._moe_mlp_for_test(DTYPE_BFLOAT16, MIXTRAL_EXPERT_TYPE)
    return mlp, [w1, w2, w3], rand


def _fp32_reference(x, w1, w2, w3, chunk=K_MAX_TOKENS):
    w1, w2, w3 = w1.float(), w2.float(), w3.float()
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for s in range(0, x.size(0), chunk):
        xs = x[s : s + chunk].float()
        out[s : s + chunk] = (F.silu(xs @ w1.t()) * (xs @ w3.t())) @ w2.t()
    return out


def _assert_bf16_close(out, x, params):
    # BF16 GEMMs over K = 14336 with a BF16 intermediate: compare against an
    # FP32 reference. torch's own BF16 path lands at rel-L2 ~4e-3 here.
    ref = _fp32_reference(x, *params)
    err = out.float() - ref
    assert (err.norm() / ref.norm()).item() < 1e-2
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=1e-1)


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 4096, 8192, 8193, 12800, 51200])
def test_moe_mlp_matches_torch_reference(mixtral_expert, tokens):
    mlp, params, rand = mixtral_expert
    x = rand(tokens, MIXTRAL_HIDDEN, scale=1.0)

    out = mlp.forward(x, params)
    torch.cuda.synchronize()

    assert out.shape == x.shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    _assert_bf16_close(out, x, params)


@requires_cuda
@pytest.mark.parametrize("tokens", [8193, 12800, 51200])
def test_moe_mlp_chunked_output_is_bit_identical_per_chunk(
    mixtral_expert, tokens
):
    mlp, params, rand = mixtral_expert
    x = rand(tokens, MIXTRAL_HIDDEN, scale=1.0)

    out = mlp.forward(x, params)
    pieces = [
        mlp.forward(x[s : s + K_MAX_TOKENS].contiguous(), params)
        for s in range(0, tokens, K_MAX_TOKENS)
    ]
    torch.cuda.synchronize()

    assert torch.equal(out, torch.cat(pieces))


@requires_cuda
def test_moe_mlp_bad_input_raises_and_module_recovers(mixtral_expert):
    mlp, params, rand = mixtral_expert

    with pytest.raises(RuntimeError, match="hidden"):
        mlp.forward(rand(16, MIXTRAL_HIDDEN // 2, scale=1.0), params)
    with pytest.raises(RuntimeError, match="dtype"):
        mlp.forward(rand(16, MIXTRAL_HIDDEN, scale=1.0).float(), params)
    with pytest.raises(RuntimeError, match="batch"):
        mlp.forward(rand(0, MIXTRAL_HIDDEN, scale=1.0), params)

    x = rand(K_MAX_TOKENS + 7, MIXTRAL_HIDDEN, scale=1.0)
    out = mlp.forward(x, params)
    _assert_bf16_close(out, x, params)
