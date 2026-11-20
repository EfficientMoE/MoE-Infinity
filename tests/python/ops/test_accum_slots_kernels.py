"""Kernels of the dispatcher's deterministic accumulation (kernel/accum_slots).

prepare: rank table, per-token counts, overflow counter, zero fill
store:   weighted expert output -> its (token, rank) slot, or the dump row
reduce:  fixed-order sum of the slots into the accumulator
"""

import pytest
import torch

from tests.python.ops.conftest import requires_cuda


def _store():
    try:
        import moe_infinity._store as store
    except Exception as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"moe_infinity._store unavailable: {exc}")
    if not hasattr(store, "accum_slots_prepare"):
        pytest.skip("moe_infinity._store built without accum_slots")
    return store


def _mask(tokens, experts, active, seed):
    gen = torch.Generator().manual_seed(seed)
    scores = torch.rand(tokens, experts, generator=gen)
    mask = torch.zeros(tokens, experts, dtype=torch.bool)
    mask.scatter_(1, scores.topk(active, dim=1).indices, True)
    return mask.cuda()


def _prepare(store, mask, slot_count, rows, hidden):
    tokens, experts = mask.shape
    slots = torch.full((tokens, rows, hidden), 7.0, device="cuda")
    index = torch.full((tokens, experts), -1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.long, device="cuda")
    store.accum_slots_prepare(mask, slots, index, slot_count, counter)
    return slots, index, counter


@requires_cuda
@pytest.mark.parametrize("experts", [8, 64, 300, 600])
@pytest.mark.parametrize("slot_count", [3, 6])
def test_prepare_rank_table_counts_overflow_and_zero_fill(experts, slot_count):
    store = _store()
    tokens, active, hidden = 5, 6, 40
    mask = _mask(tokens, experts, active, seed=experts)
    # strided / uint8 views are accepted too
    wide = torch.zeros(tokens, 2 * experts, dtype=torch.uint8, device="cuda")
    wide[:, ::2] = mask
    for m in (mask, wide[:, ::2]):
        slots, index, counter = _prepare(
            store, m, slot_count, slot_count + 1, hidden
        )
        rank = mask.long().cumsum(1) - 1
        want = rank.clamp(max=slot_count).int()
        assert torch.equal(index[mask], want[mask])
        assert int(counter) == (tokens if active > slot_count else 0)
        assert torch.count_nonzero(slots) == 0  # zero-filled (was 7.0)


@requires_cuda
@pytest.mark.parametrize(
    "out_dtype", [torch.float32, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize(
    "w_dtype", [torch.float32, torch.bfloat16, torch.float16]
)
def test_store_is_float_output_times_weight(out_dtype, w_dtype):
    store = _store()
    tokens, experts, hidden, slot_count = 6, 16, 300, 4
    mask = _mask(tokens, experts, 4, seed=3)
    slots, index, _ = _prepare(store, mask, slot_count, slot_count, hidden)
    weights = torch.rand(tokens, experts, device="cuda").to(w_dtype)
    expert = int(mask.any(0).nonzero()[0])
    token_indices = mask[:, expert].nonzero().flatten()
    output = (torch.randn(len(token_indices), hidden, device="cuda") * 3).to(
        out_dtype
    )
    store.accum_slot_store(
        output, token_indices, expert, weights, index, slots, slot_count
    )
    for j, t in enumerate(token_indices.tolist()):
        want = output[j].float() * weights[t, expert].float()
        got = slots[t, int(index[t, expert])]
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))
    # slots nobody wrote stay zero
    written = len(token_indices)
    assert torch.count_nonzero(slots.abs().sum(2)) == written


@requires_cuda
def test_non_finite_values_and_negative_zero_are_stored_untouched():
    """inf / NaN / -0 reach the slot exactly, and the reduction passes them on
    like a sequential float32 sum would (no masked-out term touches them)."""
    store = _store()
    tokens, experts, hidden, slot_count = 3, 8, 8, 4
    mask = torch.zeros(tokens, experts, dtype=torch.bool, device="cuda")
    mask[:, 1] = True  # one active expert each: a single addend per token
    slots, index, _ = _prepare(store, mask, slot_count, slot_count + 1, hidden)
    special = torch.tensor(
        [float("inf"), -float("inf"), float("nan"), -0.0, 0.0, 1.5, -2.5, 3e38],
        device="cuda",
    )
    output = special.repeat(tokens, 1)
    weights = torch.ones(tokens, experts, device="cuda")
    store.accum_slot_store(
        output,
        torch.arange(tokens, device="cuda"),
        1,
        weights,
        index,
        slots,
        slot_count,
    )
    out = torch.empty(tokens, hidden, device="cuda")
    store.accum_slots_reduce(slots, slot_count, True, out)
    want = (
        torch.zeros(hidden, device="cuda") + special
    )  # 0 + v, as the atomic path
    for t in range(tokens):
        same_nan = torch.equal(torch.isnan(out[t]), torch.isnan(want))
        finite = torch.isfinite(want)
        assert same_nan
        assert torch.equal(
            out[t][finite].view(torch.int32), want[finite].view(torch.int32)
        )
        assert torch.equal(
            out[t][~finite & ~torch.isnan(want)],
            want[~finite & ~torch.isnan(want)],
        )


@requires_cuda
@pytest.mark.parametrize("dump", [False, True])
def test_reduce_adds_dump_row_then_slots_in_rank_order(dump):
    store = _store()
    tokens, hidden, slot_count = 4, 300, 5
    rows = slot_count + 1 if dump else slot_count
    gen = torch.Generator(device="cuda").manual_seed(9)
    slots = (
        torch.randn(tokens, rows, hidden, generator=gen, device="cuda") * 100
    )
    slots[1, 3:slot_count] = 0  # unwritten slots are zero
    slots[2, 1:slot_count] = 0
    slots[3, :slot_count] = 0
    out = torch.full((tokens, hidden), 123.0, device="cuda")  # overwritten
    store.accum_slots_reduce(slots, slot_count, dump, out)
    for t in range(tokens):
        acc = (
            slots[t, slot_count].clone()
            if dump
            else torch.zeros(hidden, device="cuda")
        )
        for r in range(slot_count):
            acc = acc + slots[t, r]  # float32, in rank order
        assert torch.equal(out[t].view(torch.int32), acc.view(torch.int32)), t
