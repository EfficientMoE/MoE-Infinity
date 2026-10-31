"""Research tooling for the expert-cache simulator.

Holds the policy benchmark harness (``run_policies.py`` — policy x capacity
hit-rate tables and MRC plots vs Belady), the accelerated miss-ratio-curve
engine (``mrc.py``), speed benchmarks, and the append-only experiment log.
Everything here gates bit-for-bit against the pure-Python reference oracle in
the parent package.
"""

from benchmarks.cache_simulator.research.mrc import (
    encode_accesses,
    lru_mrc,
    lru_mrc_gpu,
    miss_ratio_curve,
    opt_mrc,
    opt_mrc_parallel,
    shards_lru_mrc,
    vectorized_access_sequence,
)

__all__ = [
    "encode_accesses",
    "lru_mrc",
    "lru_mrc_gpu",
    "miss_ratio_curve",
    "opt_mrc",
    "opt_mrc_parallel",
    "shards_lru_mrc",
    "vectorized_access_sequence",
]
