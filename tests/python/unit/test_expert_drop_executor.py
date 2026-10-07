from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from moe_infinity.distributed.expert_executor import DistributedExpertExecutor


class FakeDispatcher:
    def __init__(self):
        self.calls = []
        self.active = []
        self.resident = [1]
        self.resident_calls = []

    def __getattribute__(self, name):
        if name == "resident_on_gpu":
            resident = object.__getattribute__(self, "resident")
            if resident is None:
                raise AttributeError(name)
        return object.__getattribute__(self, name)

    def resident_on_gpu(self, layer):
        self.resident_calls.append(layer)
        if self.resident == "raise":
            raise RuntimeError("residency unavailable")
        return self.resident

    def set_inputs(self, hidden, mask, weights):
        self.calls.append(("set_inputs", hidden, mask, weights))

    def set_expected_queue(self, count):
        self.calls.append(("set_expected_queue", count))

    def enqueue_expert(self, layer, expert, gpu, remote, phase=None):
        self.calls.append(("enqueue_expert", layer, expert, gpu, remote))

    def notify_fetch_start(self):
        self.calls.append(("notify_fetch_start",))

    def wait_expert(self):
        set_inputs_call = next(
            (call for call in self.calls if call[0] == "set_inputs"), None
        )
        hidden = (
            set_inputs_call[1]
            if set_inputs_call is not None
            else torch.zeros(1, 1)
        )
        return torch.zeros_like(hidden, dtype=torch.float32)

    def take_last_active_experts(self):
        return list(self.active)


class FakePrefetcher:
    def __init__(self):
        self.corrected = []
        self.native_route = []

    def correct_prefetch(self, layer, experts, phase=None):
        self.corrected.append((layer, experts))

    def correct_to_native_route(self, layer, experts):
        self.native_route.append((layer, experts))


def make_executor(policy="off", budget=0.0):
    config = SimpleNamespace(
        gpu_only_expert_routing=False,
        speculative_prefetch_overlap=False,
        expert_drop_policy=policy,
        expert_drop_mass_budget=budget,
        expert_drop_min_k=1,
        expert_drop_flatness_floor=0.5,
    )
    executor = DistributedExpertExecutor(config)
    dispatcher = FakeDispatcher()
    executor.set_expert_dispatcher(dispatcher)
    return executor, dispatcher


def test_policy_off_passes_original_tensors():
    mask = torch.tensor([[True, False, True]])
    weights = mask.float()
    executor, dispatcher = make_executor(policy="off")
    hidden = torch.ones(1, 2)
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(3, hidden, mask, weights)
    set_inputs = next(
        call for call in dispatcher.calls if call[0] == "set_inputs"
    )
    assert set_inputs[2] is mask
    assert set_inputs[3] is weights
    assert dispatcher.resident_calls == []
    enqueued = [
        call[2] for call in dispatcher.calls if call[0] == "enqueue_expert"
    ]
    assert enqueued == [0, 2]


def test_on_miss_enqueues_survivors_and_prefetches_original_union():
    executor, dispatcher = make_executor(policy="on_miss", budget=0.25)
    dispatcher.resident = [1, 0, 0, 1]
    prefetcher = FakePrefetcher()
    executor.set_prefetcher(prefetcher)
    hidden = torch.ones(1, 4)
    mask = torch.tensor([[True, True, True, False]])
    weights = torch.tensor([[0.5, 0.25, 0.125, 0.0]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(4, hidden, mask, weights)
        executor.wait_dispatch_local()
    enqueued = [
        call[2] for call in dispatcher.calls if call[0] == "enqueue_expert"
    ]
    assert enqueued == [0, 1]
    assert prefetcher.corrected == [(5, [0, 1, 2])]
    stats = executor.get_expert_drop_stats()
    assert stats["experts_dropped"] == 1
    assert stats["drops_by_layer"] == {4: 1}


def test_budget_zero_does_not_probe():
    executor, dispatcher = make_executor(policy="on_miss", budget=0.0)
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[True, True]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(0, hidden, mask, mask.float())
    assert dispatcher.resident_calls == []
    assert executor.get_expert_drop_stats()["budget_disabled"] == 1


def test_consecutive_dispatches_reprobe_residency():
    executor, dispatcher = make_executor(policy="on_miss", budget=1.0)
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.75, 0.25]])

    dispatcher.resident = [1, 1]
    first_mask, _, _ = executor._apply_expert_drop(0, mask, weights)
    dispatcher.resident = [1, 0]
    second_mask, _, _ = executor._apply_expert_drop(0, mask, weights)

    assert dispatcher.resident_calls == [0, 0]
    assert first_mask is mask
    assert second_mask.tolist() == [[True, False]]


def test_bad_shape_enqueues_everyone():
    executor, dispatcher = make_executor(policy="on_miss", budget=0.25)
    dispatcher.resident = [0, 0, 0]
    hidden = torch.ones(1, 2, 3)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(1, hidden, mask, mask.float())
    assert any(call[0] == "enqueue_expert" for call in dispatcher.calls)
    assert executor.get_expert_drop_stats()["experts_dropped"] == 0
    assert executor.get_expert_drop_stats()["shape_bypasses"] == 1


@pytest.mark.parametrize("resident", [None, "raise", [1]])
def test_unknown_residency_drops_nothing(resident):
    executor, dispatcher = make_executor(policy="on_miss", budget=1.0)
    dispatcher.resident = resident
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.75, 0.25]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(0, hidden, mask, weights)
    enqueued = [
        call[2] for call in dispatcher.calls if call[0] == "enqueue_expert"
    ]
    assert enqueued == [0, 1]
    assert executor.get_expert_drop_stats()["residency_unknown"] == 1
    assert executor.get_expert_drop_stats()["experts_dropped"] == 0


def test_native_wait_splits_executed_ids_from_trace_ids():
    executor, dispatcher = make_executor(policy="off")
    dispatcher.active = [0, 1]
    prefetcher = FakePrefetcher()

    def overlap():
        return True

    prefetcher._overlap_active = overlap
    executor.set_prefetcher(prefetcher)
    executor._last_dispatch_used_native_routing = True
    executor._pending_prefetch = (
        prefetcher,
        7,
        None,
        None,
        None,
        [],
        None,
        [0, 1, 2],
    )
    executor.wait_dispatch_local()
    assert prefetcher.native_route == [(7, [0, 1])]
    assert prefetcher.corrected == [(8, [0, 1, 2])]


def test_reset_clears_drop_stats():
    executor, _dispatcher = make_executor(policy="on_miss", budget=0.0)
    hidden = torch.ones(1, 1)
    mask = torch.tensor([[True]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(0, hidden, mask, mask.float())
    executor.reset_expert_drop_stats()
    stats = executor.get_expert_drop_stats()
    assert stats["budget_disabled"] == 0
    assert stats["drops_by_layer"] == {}
