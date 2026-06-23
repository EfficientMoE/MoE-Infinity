from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import math
import time
from dataclasses import dataclass, field
from typing import Any


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = (p / 100.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])

    fraction = position - lower
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return float(lower_value + (upper_value - lower_value) * fraction)


def compute_percentiles(
    values: list[float], pcts: list[int] | tuple[int, ...] = (50, 90, 99)
) -> dict[str, float]:
    return {f"p{pct}": _percentile(values, float(pct)) for pct in pcts}


@dataclass
class StopWatch:
    dispatcher: object | None = None
    start_prefilling: float | None = None
    prefilling_time: float | None = None
    start_decoding: float | None = None
    decoding_time: float | None = None
    decoding_iterations: int = 0

    def put(self, _value: Any) -> None:
        now = time.perf_counter()
        if self.start_prefilling is None:
            self.start_prefilling = now
            return

        if self.prefilling_time is None:
            self.prefilling_time = now - self.start_prefilling
            self.start_decoding = now
            self._clear_expert_cache_counts()
        self.decoding_iterations += 1

    def end(self) -> None:
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.perf_counter() - self.start_decoding

    def _clear_expert_cache_counts(self) -> None:
        if self.dispatcher is None:
            return
        clear_member = getattr(
            self.dispatcher, "clear_expert_cache_counts", None
        )
        if callable(clear_member):
            try:
                clear_member()
            except Exception:
                return


@dataclass
class MetricsCollector:
    ttft: list[float] = field(default_factory=list)
    prefill_throughput: list[float] = field(default_factory=list)
    kv_cache_hit_rate: list[float] = field(default_factory=list)
    token_savings_pct: list[float] = field(default_factory=list)
    e2e_latency: list[float] = field(default_factory=list)
    expert_cache_hit_rate: list[float] = field(default_factory=list)

    def add(
        self,
        *,
        ttft: float,
        prefill_throughput: float,
        kv_cache_hit_rate: float,
        token_savings_pct: float,
        e2e_latency: float,
        expert_cache_hit_rate: float,
    ) -> None:
        self.ttft.append(float(ttft))
        self.prefill_throughput.append(float(prefill_throughput))
        self.kv_cache_hit_rate.append(float(kv_cache_hit_rate))
        self.token_savings_pct.append(float(token_savings_pct))
        self.e2e_latency.append(float(e2e_latency))
        self.expert_cache_hit_rate.append(float(expert_cache_hit_rate))

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def summarize_for_baseline(self) -> dict[str, float]:
        ttft_pct = compute_percentiles(self.ttft, pcts=(50, 90, 99))
        e2e_pct = compute_percentiles(self.e2e_latency, pcts=(50, 90, 99))
        return {
            "ttft_p50": ttft_pct["p50"],
            "ttft_p90": ttft_pct["p90"],
            "ttft_p99": ttft_pct["p99"],
            "prefill_throughput": self._mean(self.prefill_throughput),
            "kv_cache_hit_rate": self._mean(self.kv_cache_hit_rate),
            "e2e_latency_p50": e2e_pct["p50"],
            "e2e_latency_p90": e2e_pct["p90"],
            "e2e_latency_p99": e2e_pct["p99"],
            "expert_cache_hit_rate": self._mean(self.expert_cache_hit_rate),
            "token_savings_pct": self._mean(self.token_savings_pct),
        }
