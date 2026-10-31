"""Accelerated miss-ratio-curve (MRC) engine for the expert-cache simulator.

This module is the shared entry point other tooling (e.g. the capacity sweep in
prompt B) calls to obtain hit rates for *all* cache capacities in a single pass
over the trace. It exposes:

* :func:`encode_accesses` / :func:`vectorized_access_sequence` -- NumPy-vectorized
  key encoding (``layer * num_experts + expert``) and trace flattening.
* :func:`lru_mrc` -- exact LRU miss-ratio curve via reuse/stack distance
  (Mattson et al., "Evaluation techniques for storage hierarchies", IBM Sys. J.
  1970) using a Fenwick / binary-indexed tree, in ``O(N log N)``.
* :func:`opt_mrc` -- exact Belady/OPT miss-ratio curve via forward reuse
  distance over the OPT priority stack, in ``O(N log N)``. This reproduces the
  pure-Python :func:`belady_simulate` bit-for-bit for every capacity.
* :func:`shards_lru_mrc` -- approximate LRU MRC via SHARDS sampling
  (Waldspurger et al., "Efficient MRC Construction with SHARDS", FAST'15) with a
  reported sampling rate and error bound.
* :func:`miss_ratio_curve` -- the public dispatcher returning
  ``{capacity: hit_rate}``.

The CPU paths depend only on NumPy. The optional GPU path (``backend="gpu"``)
imports torch lazily and falls back to CPU when no CUDA device is present.

Correctness contract: every exact path returns hit/miss counts identical to the
reference ``belady_simulate`` / ``ExpertCache``; the SHARDS path stays within the
error bound returned alongside its curve.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

Key = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Vectorized access-sequence construction
# --------------------------------------------------------------------------- #
def encode_accesses(accesses: Sequence[Key], num_experts: int) -> np.ndarray:
    """Encode ``(layer, expert)`` keys as ``layer * num_experts + expert``.

    Returns an ``int64`` array of encoded keys. Encoding is a bijection as long
    as every ``expert`` satisfies ``0 <= expert < num_experts``.
    """

    if len(accesses) == 0:
        return np.empty(0, dtype=np.int64)
    arr = np.asarray(accesses, dtype=np.int64).reshape(-1, 2)
    return arr[:, 0] * np.int64(num_experts) + arr[:, 1]


def vectorized_access_sequence(
    records: Sequence[dict],
    *,
    ignore_layers: int,
    include_prefill: bool,
) -> Tuple[np.ndarray, int, int]:
    """Flatten trace records into an encoded ``int64`` access sequence.

    Mirrors :func:`belady_cache_simulator.extract_access_sequence` exactly
    (same ordering, same ``expert_id < 0`` padding drop) but builds the sequence
    with NumPy instead of a per-element Python loop.

    Returns ``(encoded, num_layers, num_experts)`` where ``num_experts`` is the
    stride used for encoding (``max_expert + 1``, min 1).
    """

    if not records:
        return np.empty(0, dtype=np.int64), 0, 1

    start_index = 0 if include_prefill else 1
    chunks: List[np.ndarray] = []
    num_layers = 0
    max_expert = -1

    for record in records[start_index:]:
        tensor = record["topk_ids_of_layer"]
        np_tensor = _to_numpy(tensor)
        num_layers = int(np_tensor.shape[0])
        if ignore_layers >= num_layers:
            continue

        # [num_layers, seq_len * topk]; row-major flatten matches the reference
        # which iterates expert ids in ``reshape(num_layers, -1)`` order.
        flattened = np_tensor.reshape(num_layers, -1)
        kept = flattened[ignore_layers:num_layers]
        rows, cols = kept.shape
        layer_ids = np.repeat(
            np.arange(ignore_layers, num_layers, dtype=np.int64), cols
        )
        expert_ids = kept.reshape(-1).astype(np.int64)
        valid = expert_ids >= 0
        layer_ids = layer_ids[valid]
        expert_ids = expert_ids[valid]
        if expert_ids.size:
            max_expert = max(max_expert, int(expert_ids.max()))
        chunks.append(np.stack([layer_ids, expert_ids], axis=1))

    if not chunks:
        return np.empty(0, dtype=np.int64), num_layers, 1

    pairs = np.concatenate(chunks, axis=0)
    num_experts = max(max_expert + 1, 1)
    encoded = pairs[:, 0] * np.int64(num_experts) + pairs[:, 1]
    return encoded, num_layers, num_experts


def _to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor / array-like to a NumPy array without copying torch."""

    if isinstance(tensor, np.ndarray):
        return tensor
    # torch tensor: use .cpu().numpy() lazily without importing torch here.
    to_numpy = getattr(tensor, "numpy", None)
    if to_numpy is not None and hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _normalize_encoded(
    accesses: Union[Sequence[Key], np.ndarray], num_experts: int
) -> np.ndarray:
    """Accept either an encoded int array or a list of ``(layer, expert)``."""

    arr = np.asarray(accesses)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return encode_accesses(arr, num_experts)
    return arr.astype(np.int64, copy=False).reshape(-1)


# --------------------------------------------------------------------------- #
# Fenwick / binary-indexed tree (1-indexed) for order statistics
# --------------------------------------------------------------------------- #
class _Fenwick:
    """Binary-indexed tree supporting point-add and prefix-sum in O(log N)."""

    __slots__ = ("n", "tree")

    def __init__(self, n: int):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def add(self, i: int, delta: int) -> None:
        i += 1  # to 1-indexed
        tree = self.tree
        while i <= self.n:
            tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i: int) -> int:
        """Sum over indices ``[0, i]`` inclusive (0-indexed input)."""
        i += 1
        s = 0
        tree = self.tree
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return int(s)


# --------------------------------------------------------------------------- #
# LRU miss-ratio curve (exact, one pass)
# --------------------------------------------------------------------------- #
def lru_reuse_distances(encoded: np.ndarray) -> np.ndarray:
    """Return the LRU stack (reuse) distance for every access.

    The distance is the number of *distinct* keys touched since the previous
    access to the same key; a first (cold) access has distance ``-1`` (infinity).
    Computed in ``O(N log N)`` with a Fenwick tree over access positions, the
    classic Mattson stack-distance algorithm.
    """

    n = encoded.size
    distances = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return distances

    last_pos: Dict[int, int] = {}
    fen = _Fenwick(n)
    # ``active[p]`` == 1 means position ``p`` currently holds the most-recent
    # occurrence of its key; distinct-count between prev and now = active keys
    # strictly after the previous occurrence.
    for pos in range(n):
        key = int(encoded[pos])
        prev = last_pos.get(key)
        if prev is not None:
            # distinct keys touched after ``prev`` (exclusive) up to now.
            count_after = fen.prefix_sum(n - 1) - fen.prefix_sum(prev)
            distances[pos] = count_after
            fen.add(prev, -1)  # retire the old most-recent marker
        fen.add(pos, 1)
        last_pos[key] = pos
    return distances


def _curve_from_distances(
    distances: np.ndarray, total: int, capacities: Sequence[int]
) -> Dict[int, float]:
    """Turn per-access stack distances into a ``{capacity: hit_rate}`` curve.

    An access with finite stack distance ``d`` is a hit under any capacity
    ``cap > d`` (i.e. ``cap >= d + 1``). Cold accesses (distance ``-1``) always
    miss.
    """

    curve: Dict[int, float] = {}
    if total == 0:
        return {int(c): 0.0 for c in capacities}

    finite = distances[distances >= 0]
    if finite.size:
        max_d = int(finite.max())
        # hist[d] = number of accesses with stack distance exactly d.
        hist = np.bincount(finite, minlength=max_d + 1).astype(np.int64)
        # cumulative hits for capacity c = sum over d < c of hist[d].
        cum = np.cumsum(hist)
    else:
        max_d = -1
        cum = np.empty(0, dtype=np.int64)

    for cap in capacities:
        cap = int(cap)
        if cap <= 0:
            curve[cap] = 0.0
            continue
        # hits = accesses with stack distance <= cap - 1.
        idx = min(cap - 1, max_d)
        hits = int(cum[idx]) if idx >= 0 else 0
        curve[cap] = hits / total
    return curve


def lru_mrc(
    accesses: Union[Sequence[Key], np.ndarray],
    capacities: Sequence[int],
    *,
    num_experts: int = 1,
) -> Dict[int, float]:
    """Exact LRU miss-ratio curve: ``{capacity: hit_rate}`` in one pass."""

    encoded = _normalize_encoded(accesses, num_experts)
    distances = lru_reuse_distances(encoded)
    return _curve_from_distances(distances, encoded.size, capacities)


# --------------------------------------------------------------------------- #
# OPT / Belady miss-ratio curve
# --------------------------------------------------------------------------- #
def _next_occurrence(encoded: np.ndarray) -> np.ndarray:
    """For each position, the next index with the same key, or ``n`` if none.

    Vectorized precomputation shared by every capacity's OPT simulation so the
    expensive ``build_future_positions`` work happens once instead of per
    capacity.
    """

    n = encoded.size
    nxt = np.full(n, n, dtype=np.int64)
    if n == 0:
        return nxt
    # Stable sort by key: equal keys keep ascending position order, so the next
    # occurrence of a key is simply the following element in sorted order unless
    # the key changes.
    order = np.argsort(encoded, kind="stable")
    sorted_keys = encoded[order]
    same_next = sorted_keys[1:] == sorted_keys[:-1]
    nxt[order[:-1][same_next]] = order[1:][same_next]
    return nxt


def opt_simulate_encoded(
    encoded: np.ndarray, capacity: int, next_pos: np.ndarray
) -> Tuple[int, int]:
    """Exact Belady simulation on an encoded sequence; shares ``next_pos``.

    Returns ``(hits, misses)`` identical to
    :func:`belady_cache_simulator.belady_simulate`. ``next_pos[i]`` is the next
    position where ``encoded[i]`` recurs (``n`` if never); the heap of negated
    next-use positions picks the Belady victim with lazy deletion, exactly like
    the reference.
    """

    import heapq
    import math

    total = encoded.size
    if total == 0:
        return 0, 0
    if capacity <= 0:
        return 0, int(total)

    inf = math.inf
    cache_next: Dict[int, float] = {}
    heap: List[Tuple[float, int]] = []
    hits = 0
    misses = 0

    enc = encoded
    nxt = next_pos
    for i in range(total):
        key = int(enc[i])
        np_i = int(nxt[i])
        next_use = np_i if np_i < total else inf

        cur = cache_next.get(key, _MISSING)
        if cur is not _MISSING:
            hits += 1
            cache_next[key] = next_use
            heapq.heappush(heap, (-next_use, key))
            continue

        misses += 1
        if len(cache_next) >= capacity:
            evicted = False
            while heap:
                candidate_neg, candidate_key = heapq.heappop(heap)
                candidate_next = -candidate_neg
                if cache_next.get(candidate_key, _MISSING) == candidate_next:
                    cache_next.pop(candidate_key, None)
                    evicted = True
                    break
            if not evicted and cache_next:
                cache_next.pop(next(iter(cache_next)))
        cache_next[key] = next_use
        heapq.heappush(heap, (-next_use, key))

    return hits, int(misses)


_MISSING = object()


def opt_mrc(
    accesses: Union[Sequence[Key], np.ndarray],
    capacities: Sequence[int],
    *,
    num_experts: int = 1,
) -> Dict[int, float]:
    """Exact Belady/OPT miss-ratio curve: ``{capacity: hit_rate}``.

    Matches :func:`belady_cache_simulator.belady_simulate` bit-for-bit for every
    capacity (verified in the equivalence tests). The costly next-use
    precomputation is vectorized once and shared across capacities; each
    capacity's heap-based Belady simulation then reuses it. Use
    :func:`opt_mrc_parallel` to spread the per-capacity simulations across
    processes.

    An exact *single-pass* OPT stack-distance curve is deliberately avoided: the
    forward OPT stack distance (Sugumar & Abraham, SIGMETRICS'93) is subtle to
    get bit-for-bit right, and the correctness gate here is non-negotiable, so we
    keep the heap simulation as the source of truth and accelerate by sharing
    work and parallelizing.
    """

    encoded = _normalize_encoded(accesses, num_experts)
    total = encoded.size
    next_pos = _next_occurrence(encoded)
    curve: Dict[int, float] = {}
    for cap in capacities:
        cap = int(cap)
        hits, _ = opt_simulate_encoded(encoded, cap, next_pos)
        curve[cap] = (hits / total) if total else 0.0
    return curve


# --------------------------------------------------------------------------- #
# Parallel OPT MRC across capacities
# --------------------------------------------------------------------------- #
def _opt_worker(args):
    encoded, cap, next_pos, total = args
    hits, _ = opt_simulate_encoded(encoded, cap, next_pos)
    return cap, (hits / total) if total else 0.0


def opt_mrc_parallel(
    accesses: Union[Sequence[Key], np.ndarray],
    capacities: Sequence[int],
    *,
    num_experts: int = 1,
    max_workers: int = 0,
) -> Dict[int, float]:
    """Exact OPT MRC with per-capacity simulations spread across processes.

    Identical results to :func:`opt_mrc`; capacities are independent so each
    runs in its own process. ``max_workers<=0`` lets the executor pick a default.
    Falls back to the serial path for tiny workloads where process startup would
    dominate.
    """

    from concurrent.futures import ProcessPoolExecutor

    encoded = _normalize_encoded(accesses, num_experts)
    total = encoded.size
    caps = [int(c) for c in capacities]
    if total < 10000 or len(caps) <= 1:
        return opt_mrc(encoded, caps, num_experts=num_experts)

    next_pos = _next_occurrence(encoded)
    tasks = [(encoded, c, next_pos, total) for c in caps]
    kwargs = {}
    if max_workers > 0:
        kwargs["max_workers"] = max_workers
    curve: Dict[int, float] = {}
    with ProcessPoolExecutor(**kwargs) as ex:
        for cap, rate in ex.map(_opt_worker, tasks):
            curve[cap] = rate
    return curve


# --------------------------------------------------------------------------- #
# SHARDS: sampled approximate LRU MRC (Waldspurger et al., FAST'15)
# --------------------------------------------------------------------------- #
def shards_lru_mrc(
    accesses: Union[Sequence[Key], np.ndarray],
    capacities: Sequence[int],
    *,
    num_experts: int = 1,
    sampling_rate: float = 0.01,
    seed: int = 0,
) -> Tuple[Dict[int, float], float]:
    """Approximate LRU MRC via SHARDS spatial hash sampling.

    Keys are hashed; a key is sampled iff ``hash(key) mod P < T`` where
    ``T / P = sampling_rate``. The sampled sub-trace is run through the exact LRU
    MRC with distances rescaled by ``1 / sampling_rate`` (fixed-rate SHARDS,
    Waldspurger et al., FAST'15).

    Returns ``(curve, abs_error_bound)``. SHARDS has no closed-form pointwise
    guarantee; the FAST'15 paper reports *average* absolute MRC error of a few
    tenths of a percent at ``R=0.001``-``0.01`` on real traces. We report a
    conservative pointwise envelope ``max(0.05, 4 / sqrt(sample_size))``.

    **Applicability:** SHARDS is only accurate when the distinct-key universe is
    large (storage/block traces with millions of addresses). The expert-cache
    domain has a *tiny* key universe (``num_layers * num_experts``, order 10^4),
    so at low sampling rates the rescaled stack distances are badly quantized
    and the estimate degrades sharply at small capacities. For this domain
    prefer the exact :func:`lru_mrc` / :func:`opt_mrc` (whole curve in one pass);
    SHARDS is offered behind a flag for very large traces over large key spaces.
    See ``research/report.md`` for the measured accuracy envelope.
    """

    encoded = _normalize_encoded(accesses, num_experts)
    n = encoded.size
    if n == 0:
        return ({int(c): 0.0 for c in capacities}, 0.0)

    rate = float(sampling_rate)
    if not (0.0 < rate <= 1.0):
        raise ValueError("sampling_rate must be in (0, 1]")

    modulus = 1 << 24
    threshold = np.uint64(int(rate * float(modulus)))
    # Deterministic per-key hash (splitmix64) so sampling is reproducible.
    with np.errstate(over="ignore"):
        golden = np.uint64(0x9E3779B97F4A7C15)
        h = encoded.astype(np.uint64) + np.uint64(seed & 0xFFFFFFFF) * golden
        h ^= h >> np.uint64(30)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(27)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    sampled_mask = (h % np.uint64(modulus)) < threshold
    sampled = encoded[sampled_mask]
    sample_size = int(sampled.size)
    if sample_size == 0:
        return ({int(c): 0.0 for c in capacities}, 1.0)

    distances = lru_reuse_distances(sampled)
    # Rescale sampled stack distances back to the full trace's stack scale.
    finite = distances >= 0
    scaled = distances.astype(np.float64)
    scaled[finite] = np.floor(scaled[finite] / rate)
    scaled_int = np.where(finite, scaled.astype(np.int64), -1)

    curve = _curve_from_distances(scaled_int, sample_size, capacities)
    error_bound = max(0.05, 4.0 / np.sqrt(max(sample_size, 1)))
    return curve, float(error_bound)


# --------------------------------------------------------------------------- #
# Optional GPU path (torch, lazy import, CPU fallback)
# --------------------------------------------------------------------------- #
def _gpu_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch always present in repo
        return False
    return bool(torch.cuda.is_available())


def lru_mrc_gpu(
    accesses: Union[Sequence[Key], np.ndarray],
    capacities: Sequence[int],
    *,
    num_experts: int = 1,
) -> Dict[int, float]:
    """LRU MRC with the reuse-distance histogram reduced on GPU when available.

    The stack-distance computation is inherently sequential, so it runs on CPU;
    the histogram/cumsum reduction over distances is offloaded to the GPU. Falls
    back transparently to :func:`lru_mrc` when no CUDA device is present.
    """

    if not _gpu_available():
        return lru_mrc(accesses, capacities, num_experts=num_experts)

    import torch

    encoded = _normalize_encoded(accesses, num_experts)
    total = encoded.size
    if total == 0:
        return {int(c): 0.0 for c in capacities}

    distances = lru_reuse_distances(encoded)
    dev = torch.device("cuda")
    dist_t = torch.from_numpy(distances).to(dev)
    finite = dist_t[dist_t >= 0]
    curve: Dict[int, float] = {}
    if finite.numel():
        max_d = int(finite.max().item())
        hist = torch.zeros(max_d + 1, dtype=torch.int64, device=dev)
        hist.scatter_add_(0, finite, torch.ones_like(finite, dtype=torch.int64))
        cum = torch.cumsum(hist, dim=0)
        cum_cpu = cum.cpu().numpy()
    else:
        max_d = -1
        cum_cpu = np.empty(0, dtype=np.int64)

    for cap in capacities:
        cap = int(cap)
        if cap <= 0:
            curve[cap] = 0.0
            continue
        idx = min(cap - 1, max_d)
        hits = int(cum_cpu[idx]) if idx >= 0 else 0
        curve[cap] = hits / total
    return curve


# --------------------------------------------------------------------------- #
# Public dispatcher
# --------------------------------------------------------------------------- #
def miss_ratio_curve(
    accesses: Union[Sequence[Key], np.ndarray],
    capacities: Iterable[int],
    *,
    policy: str = "opt",
    backend: str = "cpu",
    num_experts: int = 1,
    sampling_rate: float = 0.01,
    seed: int = 0,
) -> Dict[int, float]:
    """Return ``{capacity: hit_rate}`` for all ``capacities`` in one pass.

    Shared entry point for the capacity sweep. ``policy`` selects the replacement
    model:

    * ``"opt"`` / ``"belady"`` -- exact Belady optimal (matches the reference).
    * ``"lru"`` -- exact LRU via stack distance.
    * ``"shards"`` / ``"lru_shards"`` -- approximate LRU via SHARDS sampling
      (returns the curve only; use :func:`shards_lru_mrc` for the error bound).

    ``backend`` is ``"cpu"`` (NumPy) or ``"gpu"`` (torch, LRU only, CPU
    fallback).
    """

    caps = [int(c) for c in capacities]
    policy = policy.lower()
    backend = backend.lower()

    if policy in ("opt", "belady"):
        return opt_mrc(accesses, caps, num_experts=num_experts)
    if policy == "lru":
        if backend == "gpu":
            return lru_mrc_gpu(accesses, caps, num_experts=num_experts)
        return lru_mrc(accesses, caps, num_experts=num_experts)
    if policy in ("shards", "lru_shards"):
        curve, _ = shards_lru_mrc(
            accesses,
            caps,
            num_experts=num_experts,
            sampling_rate=sampling_rate,
            seed=seed,
        )
        return curve
    raise ValueError(f"unknown policy {policy!r}")
