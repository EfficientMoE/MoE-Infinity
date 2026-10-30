"""Expert prefetch/replacement policies for the offline cache simulator.

Each engine owns an ``expert_states`` matrix of shape ``[n_layer, n_expert]``
where ``1`` marks an expert currently resident on the GPU. On every access the
engine may prefetch extra experts and must evict enough residents to respect
``mem_capacity``. The policies here mirror the caches evaluated in the Archer /
MoE-Infinity expert-prefetch studies:

* ``StreamFetch``    - FIFO eviction with strided look-ahead prefetch.
* ``StreamFetchLRU`` - least-recently-used eviction with strided prefetch.
* ``LFUFetch``       - least-frequently-used eviction and prefetch.
* ``LFUKFetch``      - LFU over a sliding window of the last ``k`` requests.
"""

from __future__ import annotations

import copy
import time

import numpy as np


class FetchEngineBase:
    def __init__(self, n_layer, n_expert, mem_capacity, expert_states):
        self.n_layer = n_layer
        self.n_expert = n_expert
        self.mem_capacity = mem_capacity
        assert (
            len(expert_states) == self.n_layer
            and len(expert_states[0]) == self.n_expert
            and np.sum(expert_states) == self.mem_capacity
        )
        self.expert_states = expert_states

    def _fetch(self, layer_expert_list):
        for layer_expert_id in layer_expert_list:
            self.expert_states[layer_expert_id] = 1

    def _evict(self, layer_expert_list):
        for layer_expert_id in layer_expert_list:
            self.expert_states[layer_expert_id] = 0

    def _prefetch(self, layer_expert_id):
        raise NotImplementedError

    def _apply_replacement_policy(self, n_to_evict, exemption_list):
        raise NotImplementedError

    def fetch(self, layer_id, expert_id):
        # Policies assume fetch-on-miss: fetching an already-resident expert
        # would over-count n_fetch and evict one resident too many.
        assert not self.access(
            layer_id, expert_id
        ), f"fetch() called for resident expert {(layer_id, expert_id)}"
        n_fetch, fetch_list, exemption_list = self._prefetch(
            (layer_id, expert_id)
        )
        assert n_fetch <= self.mem_capacity
        n_to_evict = np.sum(self.expert_states) + n_fetch - self.mem_capacity
        to_evict = self._apply_replacement_policy(n_to_evict, exemption_list)
        assert len(to_evict) == n_to_evict
        self._evict(to_evict)
        self._fetch(fetch_list)
        return n_fetch

    def access(self, layer_id, expert_id):
        layer_expert_id = (layer_id, expert_id)
        return self.expert_states[layer_expert_id] == 1

    def global_update(self, routes):
        raise NotImplementedError


class StreamFetch(FetchEngineBase):
    def __init__(
        self, n_layer, n_expert, mem_capacity, expert_states, stride=1
    ):
        super().__init__(n_layer, n_expert, mem_capacity, expert_states)
        assert 1 <= stride < mem_capacity
        self.stride = stride

        self.expert_queue = []
        for layer_id in range(self.n_layer):
            for expert_id in range(self.n_expert):
                layer_expert_id = (layer_id, expert_id)
                if self.expert_states[layer_expert_id] == 1:
                    self.expert_queue.append(layer_expert_id)

    def _fetch(self, layer_expert_list):
        super()._fetch(layer_expert_list)
        self.expert_queue.extend(layer_expert_list)

    def _evict(self, layer_expert_list):
        super()._evict(layer_expert_list)
        for layer_expert_id in layer_expert_list:
            self.expert_queue.remove(layer_expert_id)

    def _prefetch(self, layer_expert_id):
        n_fetch = 1
        fetch_list = [layer_expert_id]
        exemption_list = [layer_expert_id]
        layer_id, expert_id = layer_expert_id
        for i in range(self.stride):
            # If the next expert is not on the GPU, fetch it too.
            next_expert_id = (expert_id + i + 1) % self.n_expert
            next_layer_id = layer_id + (expert_id + i + 1) // self.n_expert
            if next_layer_id >= self.n_layer:
                break
            layer_expert_id = (next_layer_id, next_expert_id)
            exemption_list.append(layer_expert_id)
            if self.expert_states[layer_expert_id] == 0:
                fetch_list.append(layer_expert_id)
                n_fetch += 1
        return n_fetch, fetch_list, exemption_list

    def _apply_replacement_policy(self, n_to_evict, exemption_list):
        to_evict = []
        if n_to_evict > 0:
            for layer_expert_id in self.expert_queue:
                if layer_expert_id not in exemption_list:
                    to_evict.append(layer_expert_id)
                if len(to_evict) == n_to_evict:
                    break
        return to_evict

    def global_update(self, routes):
        pass


class StreamFetchLRU(FetchEngineBase):
    def __init__(
        self, n_layer, n_expert, mem_capacity, expert_states, stride=1
    ):
        super().__init__(n_layer, n_expert, mem_capacity, expert_states)
        assert 1 <= stride < mem_capacity
        self.stride = stride

        self.expert_ts = {}
        for layer_id in range(self.n_layer):
            for expert_id in range(self.n_expert):
                layer_expert_id = (layer_id, expert_id)
                if self.expert_states[layer_expert_id] == 1:
                    self.expert_ts[layer_expert_id] = time.time_ns()

    def _fetch(self, layer_expert_list):
        super()._fetch(layer_expert_list)
        for layer_expert_id in layer_expert_list:
            self.expert_ts[layer_expert_id] = time.time_ns()

    def _evict(self, layer_expert_list):
        super()._evict(layer_expert_list)
        for layer_expert_id in layer_expert_list:
            self.expert_ts.pop(layer_expert_id)

    def _prefetch(self, layer_expert_id):
        n_fetch = 1
        fetch_list = [layer_expert_id]
        exemption_list = [layer_expert_id]
        layer_id, expert_id = layer_expert_id
        for i in range(self.stride):
            # If the next expert is not on the GPU, fetch it too.
            next_expert_id = (expert_id + i + 1) % self.n_expert
            next_layer_id = layer_id + (expert_id + i + 1) // self.n_expert
            if next_layer_id >= self.n_layer:
                break
            layer_expert_id = (next_layer_id, next_expert_id)
            exemption_list.append(layer_expert_id)
            if self.expert_states[layer_expert_id] == 0:
                fetch_list.append(layer_expert_id)
                n_fetch += 1
        return n_fetch, fetch_list, exemption_list

    def _apply_replacement_policy(self, n_to_evict, exemption_list):
        to_evict = []
        if n_to_evict > 0:
            for layer_expert_id in sorted(
                self.expert_ts, key=self.expert_ts.get
            ):
                if layer_expert_id not in exemption_list:
                    to_evict.append(layer_expert_id)
                if len(to_evict) == n_to_evict:
                    break
        return to_evict

    def global_update(self, routes):
        pass


class LFUFetch(FetchEngineBase):
    def __init__(
        self, n_layer, n_expert, mem_capacity, expert_states, stride=1
    ):
        super().__init__(n_layer, n_expert, mem_capacity, expert_states)
        assert 1 <= stride < mem_capacity
        self.stride = stride

        self.expert_count = {}
        for layer_id in range(self.n_layer):
            for expert_id in range(self.n_expert):
                layer_expert_id = (layer_id, expert_id)
                self.expert_count[layer_expert_id] = 0

    def _prefetch(self, layer_expert_id):
        # Fetch the ``stride`` most frequently used experts (excluding self).
        candidate_expert = [
            (_id, _count)
            for _id, _count in self.expert_count.items()
            if _id != layer_expert_id
        ]
        candidate_expert = sorted(
            candidate_expert, key=lambda x: x[1], reverse=True
        )[: self.stride]
        candidate_expert = [x[0] for x in candidate_expert]
        fetch_list = [layer_expert_id] + [
            _id for _id in candidate_expert if self.expert_states[_id] == 0
        ]
        n_fetch = len(fetch_list)
        exemption_list = [layer_expert_id] + candidate_expert
        return n_fetch, fetch_list, exemption_list

    def _apply_replacement_policy(self, n_to_evict, exemption_list):
        to_evict = []
        if n_to_evict > 0:
            # Find the least frequently used resident experts.
            candidate_expert = [
                (_id, _count)
                for _id, _count in self.expert_count.items()
                if _id not in exemption_list and self.expert_states[_id] == 1
            ]
            candidate_expert = sorted(candidate_expert, key=lambda x: x[1])[
                :n_to_evict
            ]
            to_evict = [x[0] for x in candidate_expert]
        return to_evict

    def global_update(self, routes):
        for layer_id in range(self.n_layer):
            for expert_id in range(self.n_expert):
                layer_expert_id = (layer_id, expert_id)
                if routes[layer_expert_id] > 0:
                    self.expert_count[layer_expert_id] += 1


class LFUKFetch(LFUFetch):
    def __init__(
        self, n_layer, n_expert, mem_capacity, expert_states, stride=1, k=1
    ):
        super().__init__(n_layer, n_expert, mem_capacity, expert_states, stride)
        self.k = k
        self.request_queue = []

    def global_update(self, routes):
        # Append a deep copy of routes to the sliding window.
        self.request_queue.append(copy.deepcopy(routes))
        super().global_update(routes)
        if len(self.request_queue) > self.k:
            expired_routes = self.request_queue.pop(0)
            for layer_id in range(self.n_layer):
                for expert_id in range(self.n_expert):
                    layer_expert_id = (layer_id, expert_id)
                    if expired_routes[layer_expert_id] > 0:
                        self.expert_count[layer_expert_id] -= 1
                        assert self.expert_count[layer_expert_id] >= 0
