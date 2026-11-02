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
from benchmarks.cache_simulator.policies import (
    GEOMETRY_POLICIES,
    ORACLE_POLICIES,
    POLICY_FACTORIES,
    ActivationMatrixPolicy,
    ARCPolicy,
    CachePolicy,
    LayerCycleNextUsePolicy,
    LayerLRUPolicy,
    LFUKPolicy,
    LFUPolicy,
    LHUPolicy,
    LIRSPolicy,
    LRUKPolicy,
    LRUPolicy,
    ModelTracingPolicy,
    MultiLayerAheadPolicy,
    S3FIFOPolicy,
    ScorePolicy,
    SIEVEPolicy,
    SingleLayerAheadPolicy,
    StridePrefetchPolicy,
    TwoQPolicy,
    WTinyLFUPolicy,
    build_policy,
    simulate,
)

__all__ = [
    "GEOMETRY_POLICIES",
    "ORACLE_POLICIES",
    "POLICY_FACTORIES",
    "ARCPolicy",
    "AccessSequence",
    "ActivationMatrixPolicy",
    "CachePolicy",
    "ExpertCache",
    "ExpertIndex",
    "FetchEngineBase",
    "HeapExpertCache",
    "LFUFetch",
    "LFUKFetch",
    "LFUKPolicy",
    "LFUPolicy",
    "LHUPolicy",
    "LIRSPolicy",
    "LRUKPolicy",
    "LRUPolicy",
    "LayerCycleNextUsePolicy",
    "LayerLRUPolicy",
    "ModelTracingPolicy",
    "MultiLayerAheadPolicy",
    "S3FIFOPolicy",
    "SIEVEPolicy",
    "ScorePolicy",
    "SingleLayerAheadPolicy",
    "StreamFetch",
    "StreamFetchLRU",
    "StridePrefetchPolicy",
    "TwoQPolicy",
    "WTinyLFUPolicy",
    "belady_simulate",
    "build_policy",
    "build_future_positions",
    "extract_access_sequence",
    "simulate",
]
