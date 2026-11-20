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

import pytest
import torch

from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports
from tests.python.ops.conftest import requires_cuda
from tests.python.ops.test_grouped_moe_decode_pro6000 import (  # noqa: F401
    _dispatch,
    _home_device,
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


def _problem(native_moe, tokens, seed, num_experts=NUM_EXPERTS, top_k=TOP_K):
    gen = torch.Generator().manual_seed(seed)
    dispatcher, experts = native_moe(
        num_experts, HIDDEN, INTER, gen, deterministic_accum=None
    )
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
@pytest.mark.parametrize("tokens", [1, 7])
def test_slot_mode_matches_atomic_path_within_summation_order(
    native_moe, tokens
):
    """Slots versus atomics differ only by the order of the fp32 additions.

    Any two orders of adding k addends differ by at most 2 (k - 1) 2^-24
    sum(|addend|) per element (each of the k - 1 additions rounds by at most
    half an ulp of a partial sum bounded by sum(|addend|)); the addends are
    measured by dispatching every expert alone (one addend, exact).
    """
    torch.cuda.set_device(0)
    dispatcher, _, x, mask, weights = _problem(native_moe, tokens, seed=23)
    dispatcher.set_grouped_decode(False, TOP_K)
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
    # batches: R = min(top_k hint, num_experts); bytes = T * R * H * 4
    expected = tokens * TOP_K * HIDDEN * 4

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

    # a single token keeps R = num_experts (every active expert of it fits)
    one = x[:1].contiguous()
    _run(dispatcher, one, mask[:1].contiguous(), weights[:1].contiguous())
    assert (
        dispatcher.get_routing_stats()["accum_slot_last_bytes"]
        == NUM_EXPERTS * HIDDEN * 4
    )


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
@pytest.mark.skipif(
    not supports(RTX_PRO_6000), reason="grouped decode is RTX PRO 6000 only"
)
def test_slot_mode_forms_no_grouped_batches(native_moe):
    """Part 1 rule: in slot mode every expert takes the per-expert path."""
    torch.cuda.set_device(0)
    dispatcher, experts, x, mask, weights = _problem(
        native_moe, 3, seed=26, num_experts=8, top_k=2
    )
    dispatcher.set_grouped_decode(True, 2)
    dispatcher.set_deterministic_accum(True)
    _run(dispatcher, x, mask, weights)
    _run(dispatcher, x, mask, weights)  # resident: would group without slots
    assert dispatcher.get_routing_stats()["grouped_batches"] == 0

    dispatcher.set_deterministic_accum(False)
    _run(dispatcher, x, mask, weights)
    assert dispatcher.get_routing_stats()["grouped_batches"] >= 1


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
