from __future__ import annotations

from collections import defaultdict


class ExpertPrefetchCoordinator:
    num_layers: int
    num_experts: int
    _sequence_activations: dict[int, dict[int, set[int]]]
    _total_requested_loads: int
    _unique_loads: int
    _deduplicated_loads: int
    _num_priority_queries: int

    def __init__(self, num_layers: int, num_experts: int):
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")

        self.num_layers = num_layers
        self.num_experts = num_experts
        self._sequence_activations = {}

        self._total_requested_loads = 0
        self._unique_loads = 0
        self._deduplicated_loads = 0
        self._num_priority_queries = 0

    def update_sequence_activations(
        self,
        seq_id: int,
        layer_id: int,
        activated_expert_ids: list[int],
    ) -> None:
        self._validate_layer_id(layer_id)

        sequence_layers = self._sequence_activations.setdefault(seq_id, {})
        layer_activations = sequence_layers.setdefault(layer_id, set())

        for expert_id in activated_expert_ids:
            self._validate_expert_id(expert_id)
            layer_activations.add(expert_id)

    def get_priority_experts(
        self,
        next_layer_id: int,
        max_experts: int = 32,
    ) -> list[tuple[int, int]]:
        self._validate_layer_id(next_layer_id)
        if max_experts <= 0:
            raise ValueError(f"max_experts must be > 0, got {max_experts}")

        support_count: dict[tuple[int, int], int] = defaultdict(int)
        total_requested = 0

        for sequence_layers in self._sequence_activations.values():
            activated = sequence_layers.get(next_layer_id)
            if activated is None:
                continue

            total_requested += len(activated)
            for expert_id in activated:
                support_count[(next_layer_id, expert_id)] += 1

        prioritized = sorted(
            support_count.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
        deduped = [key for key, _ in prioritized[:max_experts]]

        unique_loads = len(support_count)
        saved = max(0, total_requested - unique_loads)

        self._total_requested_loads += total_requested
        self._unique_loads += unique_loads
        self._deduplicated_loads += saved
        self._num_priority_queries += 1

        return deduped

    def clear_sequence(self, seq_id: int) -> None:
        _ = self._sequence_activations.pop(seq_id, None)

    def get_dedup_stats(self) -> dict[str, float | int]:
        ratio = 0.0
        if self._total_requested_loads > 0:
            ratio = self._deduplicated_loads / self._total_requested_loads

        return {
            "tracked_sequences": len(self._sequence_activations),
            "total_requested_loads": self._total_requested_loads,
            "unique_loads": self._unique_loads,
            "deduplicated_loads": self._deduplicated_loads,
            "dedup_ratio": ratio,
            "num_priority_queries": self._num_priority_queries,
        }

    def _validate_layer_id(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(
                f"layer_id must be in [0, {self.num_layers - 1}], got {layer_id}"
            )

    def _validate_expert_id(self, expert_id: int) -> None:
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(
                "expert_id must be in "
                + f"[0, {self.num_experts - 1}], got {expert_id}"
            )
