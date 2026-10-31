"""Expert-cache admission/eviction/prefetch policies for the simulator.

This module complements the matrix-oriented ``fetch_engines.py`` with a light
``CachePolicy`` ABC that works directly on a flat access sequence of
``(layer_idx, expert_id)`` keys -- the same representation produced by
``belady_cache_simulator.extract_access_sequence``. Every policy is a pure
demand-paging cache (one access => at most one miss) so its hit rate is directly
comparable to ``belady_simulate`` on the identical trace.

The policies here mirror the baselines surveyed for the MoE expert-cache study
(see ``research/policy_survey.md`` / the seed bibliography in
``prompts/01_research_baselines.md``):

Pure eviction / admission (no prefetch):

* ``LRUPolicy``       - least-recently-used eviction (DeepSpeed-Inference,
  Mixtral-Offloading base). Lower-bound-ish baseline.
* ``LFUPolicy``       - least-frequently-used eviction over the whole trace.
* ``LFUKPolicy``      - LFU over a sliding window of the last ``k`` accesses.
* ``ModelTracingPolicy`` - BrainStorm-style global expert counts; evict the
  globally least-used resident. Counts are seeded from one offline pass over
  the trace (profile) so it reflects the model's steady-state popularity.
* ``ScorePolicy``     - score-based eviction (ExpertFlow / Specter style):
  evict the resident with the smallest accumulated, time-decayed activation
  score.
* ``LHUPolicy``       - HOBBIT "least hot unit": frequency with recency-aware
  aging; evict the coldest resident, breaking ties by recency.

Prefetch variants (predictor realised as a *trace oracle* = perfect next-layer
routing lookahead; each is an UPPER BOUND on what the real predictor can do):

* ``SingleLayerAheadPolicy`` - Mixtral-Offloading / Pre-gated MoE: on entering a
  layer, prefetch exactly the experts the *next* layer will activate.
* ``StridePrefetchPolicy``   - ProMoE-style stride prefetch: prefetch the experts
  activated ``stride`` accesses ahead in the trace.
* ``MultiLayerAheadPolicy``  - HOBBIT-style: prefetch the union of experts
  activated across the next ``horizon`` layers.
* ``ActivationMatrixPolicy`` - MoE-Infinity EAM: track per-(prev_expert ->
  next-layer-expert) co-activation counts and prefetch the top co-activated
  experts for the next layer, falling back to the trace oracle's next-layer set.

The on-demand policies need no predictor and are exact. The prefetch policies
consume a precomputed per-position "next-layer activation" oracle derived from
the trace; this is the strongest predictor possible and is labelled as such.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

Key = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Base class + shared harness
# --------------------------------------------------------------------------- #
class CachePolicy(ABC):
    """Demand-paging cache over a flat ``(layer, expert)`` access sequence.

    Subclasses implement ``_select_victim`` (which resident key to drop when
    the cache is full) and may override the prefetch / bookkeeping hooks. The
    base ``on_access`` enforces the invariants the tests rely on: the resident
    set never exceeds ``capacity`` and every access is exactly one hit or one
    miss.
    """

    #: Whether the policy consults future accesses (predictor / oracle). Pure
    #: on-demand policies leave this ``False``.
    uses_oracle: bool = False

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self.resident: "OrderedDict[Key, None]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        # Prefetch accounting. ``hit``/``miss`` above are DEMAND events only:
        # an access counts as a hit iff the expert was resident as a result of
        # an earlier *demand* access. A purely speculative (prefetched) resident
        # does not count as a demand hit on first touch -- prefetch removes a
        # fetch stall, not the compulsory access. This keeps demand hit rate
        # bounded by Belady on every trace. ``prefetch_covered`` records the
        # subset of demand misses whose stall a prefetch would have hidden.
        self._speculative: set = set()
        self.prefetch_covered = 0
        # Position within the access sequence; set by the harness before the
        # first access so oracle policies can look ahead.
        self._accesses: Sequence[Key] = ()
        self._index = 0

    # -- harness plumbing --------------------------------------------------- #
    def bind_trace(self, accesses: Sequence[Key]) -> None:
        """Hand the full trace to the policy before replay (oracle lookahead)."""
        self._accesses = accesses
        self._index = 0

    # -- hooks -------------------------------------------------------------- #
    def _record_access(self, key: Key) -> None:
        """Update per-key statistics (frequency, recency, score, ...)."""

    def _prefetch(self, key: Key) -> List[Key]:
        """Return extra keys to admit alongside ``key`` (default: none)."""
        return []

    @abstractmethod
    def _select_victim(self, protect: Sequence[Key]) -> Optional[Key]:
        """Pick one resident key to evict, never one of ``protect``."""

    # -- core --------------------------------------------------------------- #
    def _admit(self, key: Key, protect: Sequence[Key]) -> None:
        """Insert ``key`` as a demand resident, evicting until within cap."""
        if self.capacity <= 0:
            return
        if key in self.resident:
            self.resident.move_to_end(key)
            self._speculative.discard(key)
            return
        while len(self.resident) >= self.capacity:
            victim = self._select_victim(protect)
            if victim is None:
                return
            del self.resident[victim]
            self._speculative.discard(victim)
        self.resident[key] = None

    def on_access(self, key: Key) -> bool:
        """Replay one access; return ``True`` on a demand cache hit."""
        self._record_access(key)
        resident = key in self.resident
        # A hit requires the key to be resident from an earlier *demand*; a
        # key sitting in cache only because it was prefetched is a (covered)
        # miss on its first demand, then becomes a demand resident.
        demand_hit = resident and key not in self._speculative
        if demand_hit:
            self.hits += 1
            self.resident.move_to_end(key)
        else:
            self.misses += 1
            if resident and key in self._speculative:
                # Prefetch hid this miss's fetch stall.
                self.prefetch_covered += 1
            self._speculative.discard(key)

        if self.capacity > 0:
            # Prefetch first so the requested key is never evicted to make room
            # for its own speculative companions; the requested key and its
            # prefetch set are protected for this step.
            prefetch = [
                k
                for k in self._prefetch(key)
                if k != key and k not in self.resident
            ]
            protect = [key, *prefetch]
            self._admit(key, protect)
            for extra in prefetch:
                if len(self.resident) < self.capacity:
                    self.resident[extra] = None
                    self._speculative.add(extra)
                else:
                    victim = self._select_victim(protect)
                    if victim is None:
                        break
                    del self.resident[victim]
                    self._speculative.discard(victim)
                    self.resident[extra] = None
                    self._speculative.add(extra)

        self._index += 1
        return demand_hit

    @property
    def hit_rate(self) -> float:
        """Demand hit rate; bounded above by Belady on the same trace."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def effective_hit_rate(self) -> float:
        """Stall-free rate: demand hits plus prefetch-covered misses.

        This is the fraction of accesses served without a (visible) fetch
        stall. For prefetch policies it can exceed the demand hit rate -- and
        thus Belady's demand-paging bound -- because prefetch hides the stall
        of a compulsory miss. Report it as an upper bound on the predictor.
        """
        total = self.hits + self.misses
        if not total:
            return 0.0
        return (self.hits + self.prefetch_covered) / total


def simulate(
    policy: CachePolicy, accesses: Sequence[Key]
) -> Tuple[int, int, float]:
    """Replay ``accesses`` through ``policy``; return ``(hits, misses, rate)``.

    This is the single shared harness for every ``CachePolicy``. ``hits`` and
    ``misses`` always sum to ``len(accesses)`` and the policy's resident set is
    guaranteed never to exceed its capacity.
    """
    policy.bind_trace(accesses)
    for key in accesses:
        policy.on_access(key)
        assert len(policy.resident) <= policy.capacity
    total = len(accesses)
    assert policy.hits + policy.misses == total
    hit_rate = policy.hits / total if total else 0.0
    return policy.hits, policy.misses, hit_rate


# --------------------------------------------------------------------------- #
# On-demand eviction / admission policies (no prefetch, exact)
# --------------------------------------------------------------------------- #
class LRUPolicy(CachePolicy):
    """Least-recently-used eviction (DeepSpeed-Inference / Mixtral base)."""

    def _select_victim(self, protect: Sequence[Key]) -> Optional[Key]:
        for key in self.resident:  # insertion/MRU order: oldest first
            if key not in protect:
                return key
        return None


class LFUPolicy(CachePolicy):
    """Least-frequently-used eviction over the running access counts.

    Ties are broken by least-recently-used among the coldest residents so the
    policy is fully deterministic.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.freq: Dict[Key, int] = defaultdict(int)
        self._order = 0
        self._last_used: Dict[Key, int] = {}

    def _record_access(self, key: Key) -> None:
        self.freq[key] += 1
        self._last_used[key] = self._order
        self._order += 1

    def _select_victim(self, protect: Sequence[Key]) -> Optional[Key]:
        best = None
        best_rank = None
        for key in self.resident:
            if key in protect:
                continue
            rank = (self.freq[key], self._last_used.get(key, -1))
            if best_rank is None or rank < best_rank:
                best, best_rank = key, rank
        return best


class LFUKPolicy(LFUPolicy):
    """LFU over a sliding window of the last ``k`` accesses (windowed LFU)."""

    def __init__(self, capacity: int, k: int = 64):
        super().__init__(capacity)
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._window: Deque[Key] = deque()

    def _record_access(self, key: Key) -> None:
        super()._record_access(key)
        self._window.append(key)
        if len(self._window) > self.k:
            expired = self._window.popleft()
            self.freq[expired] -= 1
            if self.freq[expired] <= 0:
                self.freq.pop(expired, None)


class ModelTracingPolicy(CachePolicy):
    """BrainStorm-style model-tracing cache driven by global expert counts.

    A single offline pass over the trace yields each expert's global activation
    count ("tracing the model"). On eviction the globally least-popular resident
    is dropped; ties fall back to LRU. This models a profile-guided static
    popularity prior rather than online frequency.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.global_count: Dict[Key, int] = defaultdict(int)
        self._order = 0
        self._last_used: Dict[Key, int] = {}

    def bind_trace(self, accesses: Sequence[Key]) -> None:
        super().bind_trace(accesses)
        self.global_count = defaultdict(int)
        for key in accesses:
            self.global_count[key] += 1

    def _record_access(self, key: Key) -> None:
        self._last_used[key] = self._order
        self._order += 1

    def _select_victim(self, protect: Sequence[Key]) -> Optional[Key]:
        best = None
        best_rank = None
        for key in self.resident:
            if key in protect:
                continue
            rank = (self.global_count[key], self._last_used.get(key, -1))
            if best_rank is None or rank < best_rank:
                best, best_rank = key, rank
        return best


class ScorePolicy(CachePolicy):
    """Score-based eviction (ExpertFlow / Specter accumulated-activation style).

    Each access adds ``1.0`` to the accessed expert's score; before each access
    all scores decay by ``decay`` (exponential forgetting). The resident with
    the smallest current score is evicted, so recently and repeatedly activated
    experts are retained. ``decay == 1.0`` reduces to plain LFU; ``decay`` near
    ``0`` approaches MRU-like recency.
    """

    def __init__(self, capacity: int, decay: float = 0.9):
        super().__init__(capacity)
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.score: Dict[Key, float] = defaultdict(float)
        self._order = 0
        self._last_used: Dict[Key, int] = {}

    def _record_access(self, key: Key) -> None:
        if self.decay < 1.0:
            for k in list(self.score.keys()):
                self.score[k] *= self.decay
        self.score[key] += 1.0
        self._last_used[key] = self._order
        self._order += 1

    def _select_victim(self, protect: Sequence[Key]) -> Optional[Key]:
        best = None
        best_rank = None
        for key in self.resident:
            if key in protect:
                continue
            rank = (self.score[key], self._last_used.get(key, -1))
            if best_rank is None or rank < best_rank:
                best, best_rank = key, rank
        return best


class LHUPolicy(CachePolicy):
    """HOBBIT "least hot unit" eviction: aged frequency + recency tie-break.

    Each resident keeps a "heat" counter. On every access heat is aged by
    ``alpha`` and the accessed expert gains ``1``. The coldest resident is
    evicted; exact ties fall back to least-recently-used. Unlike ``ScorePolicy``
    this ages only on access of *that* key's layer-step cadence via a global
    clock, which keeps it cheap and matches HOBBIT's precision-aware LFU intent.
    """

    def __init__(self, capacity: int, alpha: float = 0.5):
        super().__init__(capacity)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.heat: Dict[Key, float] = defaultdict(float)
        self._order = 0
        self._last_used: Dict[Key, int] = {}

    def _record_access(self, key: Key) -> None:
        self.heat[key] = self.heat[key] * self.alpha + 1.0
        self._last_used[key] = self._order
        self._order += 1

    def _select_victim(self, protect: Sequence[Key]) -> Optional[Key]:
        best = None
        best_rank = None
        for key in self.resident:
            if key in protect:
                continue
            rank = (self.heat[key], self._last_used.get(key, -1))
            if best_rank is None or rank < best_rank:
                best, best_rank = key, rank
        return best


# --------------------------------------------------------------------------- #
# Trace-oracle lookahead helpers
# --------------------------------------------------------------------------- #
def layer_segments(accesses: Sequence[Key]) -> List[Tuple[int, int, int]]:
    """Split a trace into contiguous same-layer segments.

    Returns ``(layer, start, end)`` tuples (``end`` exclusive). Because the
    simulator flattens a decode/prefill step layer-major, every maximal run of
    equal ``layer`` ids is one layer visit.
    """
    segments: List[Tuple[int, int, int]] = []
    if not accesses:
        return segments
    start = 0
    cur = accesses[0][0]
    for i in range(1, len(accesses)):
        layer = accesses[i][0]
        if layer != cur:
            segments.append((cur, start, i))
            start = i
            cur = layer
    segments.append((cur, start, len(accesses)))
    return segments


def next_layer_oracle(accesses: Sequence[Key]) -> List[List[Key]]:
    """For each position, the set of keys activated in the *next* layer visit.

    This is the perfect next-layer predictor (an upper bound on any real gate /
    hidden-state predictor): position ``i`` maps to the experts of the layer
    segment immediately following the segment containing ``i``.
    """
    oracle: List[List[Key]] = [[] for _ in range(len(accesses))]
    segments = layer_segments(accesses)
    for s in range(len(segments) - 1):
        _, start, end = segments[s]
        _, nstart, nend = segments[s + 1]
        nxt = list(dict.fromkeys(accesses[nstart:nend]))
        for i in range(start, end):
            oracle[i] = nxt
    return oracle


def multi_layer_oracle(
    accesses: Sequence[Key], horizon: int
) -> List[List[Key]]:
    """Union of experts activated across the next ``horizon`` layer visits."""
    oracle: List[List[Key]] = [[] for _ in range(len(accesses))]
    segments = layer_segments(accesses)
    for s in range(len(segments)):
        _, start, end = segments[s]
        future: List[Key] = []
        seen = set()
        for h in range(1, horizon + 1):
            if s + h >= len(segments):
                break
            _, fstart, fend = segments[s + h]
            for key in accesses[fstart:fend]:
                if key not in seen:
                    seen.add(key)
                    future.append(key)
        for i in range(start, end):
            oracle[i] = future
    return oracle


# --------------------------------------------------------------------------- #
# Prefetch policies (trace-oracle predictors => UPPER BOUNDS)
# --------------------------------------------------------------------------- #
class _OracleLRUPolicy(CachePolicy):
    """LRU eviction base shared by the prefetch policies."""

    uses_oracle = True

    def _select_victim(self, protect: Sequence[Key]) -> Optional[Key]:
        for key in self.resident:
            if key not in protect:
                return key
        return None


class SingleLayerAheadPolicy(_OracleLRUPolicy):
    """Prefetch exactly the next layer's experts (Mixtral / Pre-gated MoE).

    Trace-oracle form: the predictor is the recorded next-layer routing, so this
    is an UPPER BOUND on single-layer-ahead speculative prefetch. Eviction is
    LRU over the (demand + prefetched) residents.
    """

    def bind_trace(self, accesses: Sequence[Key]) -> None:
        super().bind_trace(accesses)
        self._oracle = next_layer_oracle(accesses)

    def _prefetch(self, key: Key) -> List[Key]:
        if self._index < len(self._oracle):
            return self._oracle[self._index]
        return []


class StridePrefetchPolicy(_OracleLRUPolicy):
    """ProMoE-style stride prefetch: fetch experts ``stride`` accesses ahead.

    Trace-oracle form of the ProMoE proactive prefetcher: instead of a learned
    stride predictor we read the true access ``stride`` steps in the future,
    i.e. an UPPER BOUND on stride-based prefetch.
    """

    def __init__(self, capacity: int, stride: int = 8):
        super().__init__(capacity)
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = stride

    def _prefetch(self, key: Key) -> List[Key]:
        ahead = self._index + self.stride
        if ahead < len(self._accesses):
            return [self._accesses[ahead]]
        return []


class MultiLayerAheadPolicy(_OracleLRUPolicy):
    """HOBBIT-style multi-layer-ahead prefetch over the next ``horizon`` layers.

    Trace-oracle form: prefetch the union of experts the next ``horizon`` layer
    visits will activate -- an UPPER BOUND on gating-input-similarity
    multi-layer prefetch.
    """

    def __init__(self, capacity: int, horizon: int = 2):
        super().__init__(capacity)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = horizon

    def bind_trace(self, accesses: Sequence[Key]) -> None:
        super().bind_trace(accesses)
        self._oracle = multi_layer_oracle(accesses, self.horizon)

    def _prefetch(self, key: Key) -> List[Key]:
        if self._index < len(self._oracle):
            return self._oracle[self._index]
        return []


class ActivationMatrixPolicy(_OracleLRUPolicy):
    """MoE-Infinity EAM-style activation-matrix prefetch.

    Builds an online co-activation matrix: for the current expert ``e`` in layer
    ``L``, count which experts of layer ``L+1`` co-occur in the same step. On
    access we prefetch the ``top_m`` most co-activated next-layer experts for the
    requested expert. The matrix is learned purely from past steps, but to give
    a clean UPPER BOUND we fall back to the true next-layer oracle set for any
    expert whose matrix row is still empty (cold start). Eviction is LRU.
    """

    def __init__(self, capacity: int, top_m: int = 4):
        super().__init__(capacity)
        if top_m < 1:
            raise ValueError("top_m must be >= 1")
        self.top_m = top_m

    def bind_trace(self, accesses: Sequence[Key]) -> None:
        super().bind_trace(accesses)
        self._oracle = next_layer_oracle(accesses)
        self._segments = layer_segments(accesses)
        # Map each position to the index of its layer segment.
        self._seg_of: List[int] = [0] * len(accesses)
        for s, (_, start, end) in enumerate(self._segments):
            for i in range(start, end):
                self._seg_of[i] = s
        # Co-activation counts: current key -> {next-layer key: count}.
        self._matrix: Dict[Key, Dict[Key, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._learned_upto = -1  # last segment folded into the matrix

    def _learn_through(self, seg_idx: int) -> None:
        """Fold completed (seg, seg+1) co-activations into the matrix."""
        while self._learned_upto + 1 < seg_idx:
            s = self._learned_upto + 1
            if s + 1 >= len(self._segments):
                break
            _, cstart, cend = self._segments[s]
            _, nstart, nend = self._segments[s + 1]
            next_keys = list(dict.fromkeys(self._accesses[nstart:nend]))
            for cur in dict.fromkeys(self._accesses[cstart:cend]):
                row = self._matrix[cur]
                for nxt in next_keys:
                    row[nxt] += 1
            self._learned_upto = s

    def _prefetch(self, key: Key) -> List[Key]:
        if self._index >= len(self._accesses):
            return []
        seg = self._seg_of[self._index]
        # Learn from every fully-observed earlier transition.
        self._learn_through(seg)
        row = self._matrix.get(key)
        if row:
            ranked = sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))
            return [k for k, _ in ranked[: self.top_m]]
        # Cold start: fall back to the perfect next-layer set (upper bound).
        return self._oracle[self._index][: self.top_m]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#: Policies that read future accesses (trace oracle) -> report as upper bounds.
ORACLE_POLICIES = {
    "single_layer_ahead",
    "stride_prefetch",
    "multi_layer_ahead",
    "activation_matrix",
}


def build_policy(name: str, capacity: int, **kwargs) -> CachePolicy:
    """Instantiate a policy by name (see ``POLICY_FACTORIES``)."""
    try:
        factory = POLICY_FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown policy {name!r}; choices: {sorted(POLICY_FACTORIES)}"
        ) from exc
    return factory(capacity, **kwargs)


POLICY_FACTORIES = {
    "lru": lambda cap, **kw: LRUPolicy(cap),
    "lfu": lambda cap, **kw: LFUPolicy(cap),
    "lfu_k": lambda cap, **kw: LFUKPolicy(cap, **kw),
    "model_tracing": lambda cap, **kw: ModelTracingPolicy(cap),
    "score": lambda cap, **kw: ScorePolicy(cap, **kw),
    "lhu": lambda cap, **kw: LHUPolicy(cap, **kw),
    "single_layer_ahead": lambda cap, **kw: SingleLayerAheadPolicy(cap),
    "stride_prefetch": lambda cap, **kw: StridePrefetchPolicy(cap, **kw),
    "multi_layer_ahead": lambda cap, **kw: MultiLayerAheadPolicy(cap, **kw),
    "activation_matrix": lambda cap, **kw: ActivationMatrixPolicy(cap, **kw),
}


# Keep a module-level INF for callers that compare against Belady.
INF = math.inf
