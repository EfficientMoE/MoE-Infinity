"""Order-deterministic accumulation of the expert dispatcher (slot mode).

Without it the fp32 accumulator returned by ``wait_expert`` depends on the
order concurrent exec streams add their experts' outputs (atomics), so the
same inputs give accumulators that differ at rounding level run to run.
In slot mode every expert stores its weighted output in its own (token, rank)
slot and ``wait_expert`` reduces the slots in a fixed order.

``MOE_DETERMINISTIC_ACCUM=0`` in the environment (read when the dispatcher is
built) or ``set_deterministic_accum(False)`` restores the atomic path; running
the stability tests that way is the red case of these tests.
"""

import gc

import pytest
import torch

from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports
from tests.python.ops.conftest import requires_cuda
from tests.python.ops.test_grouped_moe_decode_pro6000 import (  # noqa: F401
    _dispatch,
    _home_device,
    _make_dispatcher,
    _routing,
    native_moe,
)

REPEATS = 50
NUM_EXPERTS = 16
TOP_K = 6
HIDDEN = 1024
INTER = 192

requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA devices"
)
requires_grouped = pytest.mark.skipif(
    not supports(RTX_PRO_6000), reason="grouped decode is RTX PRO 6000 only"
)


@pytest.fixture
def slot_moe(tmp_path):
    """Like native_moe, but also hands out the prefetch engine."""
    made = []

    def make(num_experts, hidden, inter, generator):
        engine, dispatcher, experts = _make_dispatcher(
            tmp_path, num_experts, hidden, inter, generator
        )
        made.append([engine, dispatcher])
        return engine, dispatcher, experts

    yield make
    while made:
        pair = made.pop()
        pair[1] = None
        pair[0] = None


def _problem(native_moe, tokens, seed, num_experts=NUM_EXPERTS, top_k=TOP_K):
    gen = torch.Generator().manual_seed(seed)
    dispatcher, experts = native_moe(num_experts, HIDDEN, INTER, gen)
    x = torch.randn(tokens, HIDDEN, generator=gen).bfloat16().cuda()
    mask, weights = _routing(tokens, num_experts, top_k, gen, "cpu")
    return (
        dispatcher,
        experts,
        x,
        mask.cuda(),
        weights.to(torch.bfloat16).cuda(),
    )


def _run(dispatcher, x, mask, weights):
    return _dispatch(dispatcher, x, mask, weights, native=True)


def _count_different(dispatcher, x, mask, weights, repeats=REPEATS):
    """How many of ``repeats`` dispatches differ bitwise from the first."""
    _run(dispatcher, x, mask, weights)  # cold: experts get fetched
    reference = _run(dispatcher, x, mask, weights)
    assert reference.dtype == torch.float32
    different = 0
    for _ in range(repeats):
        out = _run(dispatcher, x, mask, weights)
        different += int(not torch.equal(out, reference))
    return different


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 7])
def test_accumulator_is_bitwise_stable_on_one_gpu(native_moe, tokens):
    torch.cuda.set_device(0)
    dispatcher, _, x, mask, weights = _problem(native_moe, tokens, seed=21)
    dispatcher.set_grouped_decode(False, TOP_K)  # per-expert path only

    different = _count_different(dispatcher, x, mask, weights)

    assert different == 0, (
        f"{different} of {REPEATS} dispatches gave a different accumulator "
        f"(tokens={tokens})"
    )
    stats = dispatcher.get_routing_stats()
    assert stats["accum_slot_dispatches"] >= REPEATS
    assert stats["accum_atomic_dispatches"] == 0


@requires_cuda
@requires_two_gpus
@pytest.mark.parametrize("tokens", [1, 7])
def test_accumulator_is_bitwise_stable_with_experts_on_two_gpus(
    native_moe, tokens
):
    torch.cuda.set_device(0)
    dispatcher, _, x, mask, weights = _problem(native_moe, tokens, seed=22)
    dispatcher.set_grouped_decode(False, TOP_K)
    homes = {_home_device(e) for e in mask.any(0).nonzero().flatten().tolist()}
    assert homes == {0, 1}, "the routing must use experts of both devices"

    different = _count_different(dispatcher, x, mask, weights)

    assert different == 0, (
        f"{different} of {REPEATS} dispatches gave a different accumulator "
        f"(tokens={tokens}, two GPUs)"
    )


@requires_cuda
@pytest.mark.parametrize(
    "grouped", [False, pytest.param(True, marks=requires_grouped)]
)
@pytest.mark.parametrize("tokens", [1, 7])
def test_slot_mode_matches_atomic_path_within_summation_order(
    native_moe, tokens, grouped
):
    """Slots versus atomics differ only by the order of the fp32 additions.

    Any two orders of adding k addends differ by at most 2 (k - 1) 2^-24
    sum(|addend|) per element (each of the k - 1 additions rounds by at most
    half an ulp of a partial sum bounded by sum(|addend|)); the addends are
    measured by dispatching every expert alone (one addend, exact).
    """
    torch.cuda.set_device(0)
    dispatcher, _, x, mask, weights = _problem(native_moe, tokens, seed=23)
    dispatcher.set_grouped_decode(grouped, TOP_K)
    _run(dispatcher, x, mask, weights)  # make the experts resident

    addends = []
    for e in mask.any(0).nonzero().flatten().tolist():
        only = torch.zeros_like(mask)
        only[:, e] = mask[:, e]
        addends.append(_run(dispatcher, x, only, weights))
    stacked = torch.stack(addends)
    abs_sum = stacked.abs().sum(0)
    count = int(mask.sum(1).max())
    assert count > 1

    dispatcher.set_deterministic_accum(True)
    with_slots = _run(dispatcher, x, mask, weights)
    dispatcher.set_deterministic_accum(False)
    atomic = _run(dispatcher, x, mask, weights)
    if grouped:
        assert dispatcher.get_routing_stats()["grouped_experts"] > 0

    bound = 2 * (count - 1) * 2**-24 * abs_sum
    assert ((with_slots - atomic).abs() <= bound).all()
    # and both are within that bound of the exact sum of the addends
    exact = stacked.double().sum(0)
    assert ((with_slots.double() - exact).abs() <= bound.double()).all()
    largest = stacked.abs().amax(0)
    ulps = (with_slots - atomic).abs() / (largest * 2**-23).clamp_min(1e-30)
    print(
        f"max |slots - atomic| = {float(ulps.max()):.2f} ulp of largest addend"
    )


@requires_cuda
def test_slot_budget_boundary_and_size(native_moe):
    torch.cuda.set_device(0)
    tokens = 7
    dispatcher, _, x, mask, weights = _problem(native_moe, tokens, seed=24)
    dispatcher.set_grouped_decode(False, TOP_K)
    _run(dispatcher, x, mask, weights)
    # R = min(top_k hint, num_experts); with a hint the buffer has one extra
    # (dump) row for ranks >= R: bytes = T * (R + 1) * H * 4
    expected = tokens * (TOP_K + 1) * HIDDEN * 4

    dispatcher.set_deterministic_accum(True, expected)  # exactly fits
    _run(dispatcher, x, mask, weights)
    stats = dispatcher.get_routing_stats()
    assert stats["accum_slot_last_bytes"] == expected
    assert stats["accum_slot_peak_bytes"] >= expected
    slot_before = stats["accum_slot_dispatches"]
    atomic_before = stats["accum_atomic_dispatches"]

    dispatcher.set_deterministic_accum(True, expected - 1)  # one byte short
    reference = _run(dispatcher, x, mask, weights)
    stats = dispatcher.get_routing_stats()
    assert stats["accum_slot_dispatches"] == slot_before
    assert stats["accum_atomic_dispatches"] == atomic_before + 1
    dispatcher.set_deterministic_accum(True, expected)
    again = _run(dispatcher, x, mask, weights)
    assert torch.allclose(again, reference, rtol=1e-5, atol=1e-5)

    # a single token uses the hint too (R = TOP_K, not num_experts)
    one = x[:1].contiguous()
    _run(dispatcher, one, mask[:1].contiguous(), weights[:1].contiguous())
    assert (
        dispatcher.get_routing_stats()["accum_slot_last_bytes"]
        == (TOP_K + 1) * HIDDEN * 4
    )
    # without a hint R = num_experts and there is no dump row
    dispatcher.set_grouped_decode(False, 0)
    _run(dispatcher, one, mask[:1].contiguous(), weights[:1].contiguous())
    assert (
        dispatcher.get_routing_stats()["accum_slot_last_bytes"]
        == NUM_EXPERTS * HIDDEN * 4
    )


@requires_cuda
def test_inf_nan_and_negative_zero_pass_through_like_the_atomic_path(
    native_moe,
):
    """What an expert contributes reaches the accumulator exactly as the atomic
    path delivers it: a +/-inf stays inf (a masked-out overflow term must not
    turn it into NaN), a NaN stays NaN and a -0 stays what the atomic path
    makes of it; no other token is touched.

    Tokens with a single active expert have a single addend, so slot and
    atomic mode must agree bit for bit on them.
    """
    torch.cuda.set_device(0)
    tokens = 7
    dispatcher, _, x, mask, weights = _problem(native_moe, tokens, seed=28)
    dispatcher.set_grouped_decode(False, TOP_K)  # hint: overflow guard is on
    inf_tok, nan_tok, negzero_tok = 3, 4, 5
    mask = mask.clone()
    weights = weights.clone()
    for tok, expert, weight in (
        (inf_tok, 2, 3e38),  # o * 3e38 overflows fp32 -> +/-inf
        (nan_tok, 5, float("nan")),
        (negzero_tok, 7, -0.0),  # o * -0.0 = -0.0 for o > 0
    ):
        mask[tok] = False
        mask[tok, expert] = True
        weights[tok] = 0
        weights[tok, expert] = weight
    _run(dispatcher, x, mask, weights)  # make the experts resident

    dispatcher.set_deterministic_accum(True)
    slots = _run(dispatcher, x, mask, weights)
    dispatcher.set_deterministic_accum(False)
    atomic = _run(dispatcher, x, mask, weights)

    assert torch.isinf(atomic[inf_tok]).any()
    assert torch.isnan(atomic[nan_tok]).all()
    for tok in (inf_tok, nan_tok, negzero_tok):
        got, want = slots[tok], atomic[tok]
        assert torch.equal(torch.isnan(got), torch.isnan(want)), tok
        assert torch.equal(torch.isposinf(got), torch.isposinf(want)), tok
        assert torch.equal(torch.isneginf(got), torch.isneginf(want)), tok
        finite = torch.isfinite(want)
        # bit for bit (also the sign of zero) where finite
        assert torch.equal(
            got[finite].view(torch.int32), want[finite].view(torch.int32)
        ), tok
    # an inf / NaN in one token does not change any other token: the same
    # batch without the non-finite tokens' experts reproduces their rows
    clean = [
        t for t in range(tokens) if t not in (inf_tok, nan_tok, negzero_tok)
    ]
    dispatcher.set_deterministic_accum(True)
    again = _run(dispatcher, x, mask, weights)
    assert torch.equal(again[clean], slots[clean])


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 7])
def test_non_contiguous_hidden_states_work_in_both_modes(native_moe, tokens):
    """A dense but non-contiguous hidden-state tensor (here a transposed view)
    must dispatch in slot mode as it does in atomic mode: the accumulator
    returned to Python is a plain float32 [T, H] tensor either way."""
    torch.cuda.set_device(0)
    dispatcher, _, x, mask, weights = _problem(native_moe, tokens, seed=29)
    dispatcher.set_grouped_decode(False, TOP_K)
    x_nc = x.t().contiguous().t()  # same values, strides (1, T)
    assert x_nc.shape == x.shape and torch.equal(x_nc, x)
    if tokens > 1:
        assert not x_nc.is_contiguous()
    _run(dispatcher, x, mask, weights)  # make the experts resident

    dispatcher.set_deterministic_accum(True)
    slots = _run(dispatcher, x_nc, mask, weights)
    slots_contiguous = _run(dispatcher, x, mask, weights)
    dispatcher.set_deterministic_accum(False)
    atomic = _run(dispatcher, x_nc, mask, weights)

    for out in (slots, atomic):
        assert out.dtype == torch.float32
        assert out.shape == x.shape
    # same dispatch, same bits, whatever the input's strides
    assert torch.equal(slots, slots_contiguous)
    scale = atomic.abs().max()
    assert (slots - atomic).abs().max() <= 2**-20 * scale * TOP_K


@requires_cuda
def test_token_with_more_experts_than_the_top_k_hint_stays_correct(native_moe):
    """The hint bounds R for batches; a larger mask must not corrupt anything."""
    torch.cuda.set_device(0)
    tokens = 7
    dispatcher, _, x, mask, weights = _problem(native_moe, tokens, seed=25)
    dispatcher.set_grouped_decode(False, 2)  # hint 2, the mask has TOP_K=6
    _run(dispatcher, x, mask, weights)
    dispatcher.set_deterministic_accum(False)
    atomic = _run(dispatcher, x, mask, weights)
    dispatcher.set_deterministic_accum(True)
    slots = _run(dispatcher, x, mask, weights)

    assert dispatcher.get_routing_stats()["accum_slot_overflow"] > 0
    scale = atomic.abs().max()
    assert (slots - atomic).abs().max() <= 1e-5 * scale


@requires_cuda
@requires_grouped
def test_slot_mode_forms_grouped_batches(native_moe):
    """Grouped decode runs in slot mode (its kernel writes the slots)."""
    torch.cuda.set_device(0)
    dispatcher, experts, x, mask, weights = _problem(
        native_moe, 3, seed=26, num_experts=8, top_k=2
    )
    dispatcher.set_grouped_decode(True, 2)
    dispatcher.set_deterministic_accum(True)
    _run(dispatcher, x, mask, weights)  # cold: per-expert
    out = _run(dispatcher, x, mask, weights)  # resident: grouped
    stats = dispatcher.get_routing_stats()
    assert stats["accum_slot_dispatches"] >= 2
    assert stats["accum_atomic_dispatches"] == 0
    assert stats["grouped_batches"] >= 1
    # only experts homed on the hidden-state device (cuda:0) can group
    active = mask.any(0).nonzero().flatten().tolist()
    assert stats["grouped_experts"] == sum(
        1 for e in active if _home_device(e) == 0
    )
    assert stats["grouped_experts"] > 0
    # the grouped result is the same value the per-expert path produces
    dispatcher.set_grouped_decode(False, 2)
    per_expert = _run(dispatcher, x, mask, weights)
    scale = per_expert.abs().max()
    assert (out - per_expert).abs().max() <= 4 * 2**-8 * scale


def _masked_routing(tokens, allowed, num_experts, top_k, gen):
    """top_k experts per token drawn from `allowed` only."""
    logits = torch.randn(tokens, num_experts, generator=gen)
    keep = torch.full((num_experts,), float("-inf"))
    keep[list(allowed)] = 0.0
    top = (logits + keep).topk(top_k, dim=1).indices
    mask = torch.zeros(tokens, num_experts, dtype=torch.bool)
    mask.scatter_(1, top, True)
    weights = torch.softmax(logits, dim=1) * mask
    return mask.cuda(), weights.to(torch.bfloat16).cuda()


def _grouped_problem(slot_moe, tokens, seed, allowed=None):
    gen = torch.Generator().manual_seed(seed)
    engine, dispatcher, experts = slot_moe(NUM_EXPERTS, HIDDEN, INTER, gen)
    x = torch.randn(tokens, HIDDEN, generator=gen).bfloat16().cuda()
    allowed = list(range(NUM_EXPERTS)) if allowed is None else allowed
    mask, weights = _masked_routing(tokens, allowed, NUM_EXPERTS, TOP_K, gen)
    dispatcher.set_grouped_decode(True, TOP_K)
    return engine, dispatcher, experts, x, mask, weights


@requires_cuda
@requires_grouped
@pytest.mark.parametrize("tokens", [1, 7])
def test_all_grouped_accumulator_is_bitwise_stable(slot_moe, tokens):
    torch.cuda.set_device(0)
    # Only experts homed on cuda:0 (with two GPUs the others cannot group).
    allowed = [e for e in range(NUM_EXPERTS) if _home_device(e) == 0]
    _, dispatcher, _, x, mask, weights = _grouped_problem(
        slot_moe, tokens, seed=31, allowed=allowed
    )
    different = _count_different(dispatcher, x, mask, weights)
    stats = dispatcher.get_routing_stats()
    assert different == 0, (
        f"{different} of {REPEATS} dispatches gave a different accumulator "
        f"(all grouped, tokens={tokens})"
    )
    assert stats["grouped_experts"] >= (REPEATS + 1) * int(mask.any(0).sum())
    assert stats["accum_slot_dispatches"] > 0
    assert stats["accum_atomic_dispatches"] == 0


@requires_cuda
@requires_grouped
@pytest.mark.parametrize("tokens", [1, 7])
def test_mixed_grouped_and_fetched_accumulator_is_bitwise_stable(
    tmp_path, tokens
):
    """One layer with grouped experts (resident) and per-expert ones (still
    being fetched). Which experts group changes their values (the kernels round
    differently), so the mix has to be the same every time: each repeat builds
    a fresh engine, makes the first half of the active experts resident, then
    dispatches all of them."""
    torch.cuda.set_device(0)

    def one(n):
        gen = torch.Generator().manual_seed(32)
        engine, dispatcher, _ = _make_dispatcher(
            tmp_path / f"run{n}", NUM_EXPERTS, HIDDEN, INTER, gen
        )
        try:
            x = torch.randn(tokens, HIDDEN, generator=gen).bfloat16().cuda()
            allowed = [e for e in range(NUM_EXPERTS) if _home_device(e) == 0]
            mask, weights = _masked_routing(
                tokens, allowed, NUM_EXPERTS, TOP_K, gen
            )
            dispatcher.set_grouped_decode(True, TOP_K)
            active = mask.any(0).nonzero().flatten().tolist()
            assert len(active) >= 4
            half = torch.zeros_like(mask)
            keep = active[: len(active) // 2]
            half[:, keep] = mask[:, keep]
            _run(dispatcher, x, half, weights)  # these become resident
            out = _run(dispatcher, x, mask, weights)  # grouped + fetched
            stats = dispatcher.get_routing_stats()
            return (
                out,
                stats["grouped_experts"],
                len(active),
                stats["accum_atomic_dispatches"],
            )
        finally:
            dispatcher = None
            engine = None
            gc.collect()

    reference, grouped, active, _ = one(0)
    assert 0 < grouped < active, (grouped, active)
    different = 0
    atomic_dispatches = 0
    for n in range(1, REPEATS + 1):
        out, grouped_n, _, atomic = one(n)
        assert grouped_n == grouped
        atomic_dispatches += atomic
        different += int(not torch.equal(out, reference))
    assert different == 0, (
        f"{different} of {REPEATS} dispatches gave a different accumulator "
        f"(mixed, tokens={tokens})"
    )
    assert atomic_dispatches == 0


@requires_cuda
@requires_grouped
@requires_two_gpus
@pytest.mark.parametrize("tokens", [1, 7])
def test_mixed_across_two_gpus_accumulator_is_bitwise_stable(slot_moe, tokens):
    """Experts homed on cuda:0 run grouped, those on cuda:1 per-expert."""
    torch.cuda.set_device(0)
    _, dispatcher, _, x, mask, weights = _grouped_problem(
        slot_moe, tokens, seed=33
    )
    homes = {_home_device(e) for e in mask.any(0).nonzero().flatten().tolist()}
    assert homes == {0, 1}

    different = _count_different(dispatcher, x, mask, weights)
    stats = dispatcher.get_routing_stats()
    assert different == 0, (
        f"{different} of {REPEATS} dispatches gave a different accumulator "
        f"(grouped + other GPU, tokens={tokens})"
    )
    assert stats["grouped_experts"] > 0
    assert stats["accum_atomic_dispatches"] == 0


@requires_cuda
# "output" is left out: with the per-expert path it hits the open straggler race
# of the failure path ("unreaped completion events"; it fails the same way with
# MOE_DETERMINISTIC_ACCUM=0), which is outside this change.
@pytest.mark.parametrize("fault", ["exec", "completion_event"])
def test_failed_dispatch_leaves_no_slot_state_for_the_next(native_moe, fault):
    torch.cuda.set_device(0)
    dispatcher, _, x, mask, weights = _problem(native_moe, 3, seed=27)
    dispatcher.set_grouped_decode(False, TOP_K)
    # Two active experts, as in the grouped failure tests.
    mask = torch.zeros_like(mask)
    mask[:, :2] = True
    weights = (mask.float() * 0.5).to(torch.bfloat16)
    good = _run(dispatcher, x, mask, weights)
    good = _run(dispatcher, x, mask, weights)

    dispatcher._set_dispatch_fault_for_test(fault)
    with pytest.raises(RuntimeError, match=f"injected {fault} failure"):
        _run(dispatcher, x, mask, weights)
    stats = dispatcher.get_routing_stats()
    assert stats["pending"] == 0 and stats["route_pending"] == 0

    after = _run(dispatcher, x, mask, weights)
    assert torch.allclose(after, good, rtol=1e-5, atol=1e-5)
