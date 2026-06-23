from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, cast

import numpy as np

StatValue = Union[int, float]
StatDict = dict[str, StatValue]
TransferStatDict = Mapping[str, Optional[StatValue]]

_percentile = cast(Callable[[Sequence[int], float], float], np.percentile)


@dataclass
class _TransferRecord:
    duration_ns: list[int] = field(default_factory=list)
    bytes_transferred: int = 0


class TimingCollector:
    def __init__(self) -> None:
        self._durations: dict[str, list[int]] = defaultdict(list)
        self._transfers: dict[str, _TransferRecord] = {}

    def record(self, name: str, duration_ns: int) -> None:
        self._durations[name].append(int(duration_ns))

    def record_transfer(
        self, name: str, duration_ns: int, bytes_transferred: int
    ) -> None:
        self.record(name, duration_ns)
        record = self._transfers.setdefault(name, _TransferRecord())
        record.duration_ns.append(int(duration_ns))
        record.bytes_transferred += int(bytes_transferred)

    def _summary(self, durations: list[int]) -> StatDict:
        sorted_values = sorted(durations)
        count = len(sorted_values)
        mean_ns = sum(sorted_values) / count
        return {
            "min_ns": sorted_values[0],
            "max_ns": sorted_values[-1],
            "mean_ns": mean_ns,
            "p50_ns": float(_percentile(sorted_values, 50)),
            "p95_ns": float(_percentile(sorted_values, 95)),
            "p99_ns": float(_percentile(sorted_values, 99)),
            "count": count,
        }

    def get_transfer_stats(self, name: str) -> TransferStatDict:
        record = self._transfers.get(name)
        if record is None or not record.duration_ns:
            return {
                "min_ns": None,
                "max_ns": None,
                "mean_ns": None,
                "p50_ns": None,
                "p95_ns": None,
                "p99_ns": None,
                "count": 0,
                "bytes_transferred": 0,
                "effective_bandwidth_gbps": 0.0,
            }

        stats = self._summary(record.duration_ns)
        total_duration_ns = float(sum(record.duration_ns))
        effective_bandwidth_gbps = (
            0.0
            if total_duration_ns <= 0
            else record.bytes_transferred / total_duration_ns
        )
        stats.update(
            {
                "bytes_transferred": record.bytes_transferred,
                "effective_bandwidth_gbps": effective_bandwidth_gbps,
            }
        )
        return stats

    def to_dict(self) -> dict[str, StatDict]:
        return {
            name: self._summary(values)
            for name, values in self._durations.items()
            if values
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
