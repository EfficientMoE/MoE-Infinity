"""Offline MoE expert-cache simulator.

Replays recorded expert-activation traces through cache-replacement and
prefetch policies to measure GPU expert-cache hit rate offline, without a GPU
or a live model. See ``README.md`` for trace formats, policies, and where to
obtain the trace data.
"""

from benchmarks.cache_simulator.belady_cache_simulator import (
    belady_simulate,
    build_future_positions,
    extract_access_sequence,
)
from benchmarks.cache_simulator.cache_strategy import (
    AccessSequence,
    ExpertCache,
    ExpertIndex,
    HeapExpertCache,
)
from benchmarks.cache_simulator.fetch_engines import (
    FetchEngineBase,
    LFUFetch,
    LFUKFetch,
    StreamFetch,
    StreamFetchLRU,
)

__all__ = [
    "AccessSequence",
    "ExpertCache",
    "ExpertIndex",
    "FetchEngineBase",
    "HeapExpertCache",
    "LFUFetch",
    "LFUKFetch",
    "StreamFetch",
    "StreamFetchLRU",
    "belady_simulate",
    "build_future_positions",
    "extract_access_sequence",
]
