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

Classic recency/frequency hybrids from the caching literature (on-demand,
domain-agnostic, each follows its paper's published algorithm):

* ``ARCPolicy``       - ARC (Megiddo & Modha, FAST'03).
* ``TwoQPolicy``      - full 2Q (Johnson & Shasha, VLDB'94).
* ``LRUKPolicy``      - LRU-K, default K=2 (O'Neil et al., SIGMOD'93).
* ``LIRSPolicy``      - LIRS (Jiang & Zhang, SIGMETRICS'02).
* ``S3FIFOPolicy``    - S3-FIFO (Yang et al., SOSP'23).
* ``SIEVEPolicy``     - SIEVE (Zhang et al., NSDI'24).
* ``WTinyLFUPolicy``  - W-TinyLFU (Einziger et al., ACM ToS 2017).

MoE-structure-aware (on-demand, no lookahead):

* ``LayerLRUPolicy``  - fixed per-layer slot quota with LRU inside each layer
  (the Mixtral-Offloading cache organisation).
* ``LayerCycleNextUsePolicy`` (``lcnu``) - evicts the expert with the farthest
  *expected* next use. Decode executes layers in a fixed cycle, so the time
  until an expert's layer runs again is known exactly; the policy adds the
  expected number of further decode steps until the expert is routed there,
  from an online recency + frequency estimate. An online approximation of
  Belady that does not thrash when the cache is smaller than one decode step's
  working set (where LRU's hit rate collapses to zero).
"""

from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

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
# Self-managed demand-paging policies (own queues, no prefetch)
# --------------------------------------------------------------------------- #
class _SelfManagedPolicy(CachePolicy):
    """Demand-paging policy that keeps its own queues / ghost lists.

    Subclasses implement ``_on_miss`` (admit the missed key, evicting through
    ``_evict`` as needed) and optionally ``_on_hit``. ``self.resident`` mirrors
    the resident set so the shared harness can check the capacity invariant.
    """

    def _select_victim(self, protect: Sequence[Key]) -> Optional[Key]:
        raise NotImplementedError(
            "self-managed policies pick their own victims"
        )

    def _insert(self, key: Key) -> None:
        self.resident[key] = None

    def _evict(self, key: Key) -> None:
        del self.resident[key]

    def _on_hit(self, key: Key) -> None:
        """Update queues on a demand hit (default: nothing)."""

    @abstractmethod
    def _on_miss(self, key: Key) -> None:
        """Admit ``key`` (capacity > 0), evicting to stay within capacity."""

    def on_access(self, key: Key) -> bool:
        self._record_access(key)
        if key in self.resident:
            self.hits += 1
            self._on_hit(key)
            hit = True
        else:
            self.misses += 1
            if self.capacity > 0:
                self._on_miss(key)
            hit = False
        self._index += 1
        return hit


class ARCPolicy(_SelfManagedPolicy):
    """ARC: Adaptive Replacement Cache (Megiddo & Modha, FAST'03).

    ``T1`` holds keys seen once recently, ``T2`` keys seen at least twice;
    ghost lists ``B1`` / ``B2`` remember keys evicted from each. A ghost hit
    moves the adaptive target ``p`` (the size ``T1`` should have) toward the
    list that would have hit.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.t1: "OrderedDict[Key, None]" = OrderedDict()
        self.t2: "OrderedDict[Key, None]" = OrderedDict()
        self.b1: "OrderedDict[Key, None]" = OrderedDict()
        self.b2: "OrderedDict[Key, None]" = OrderedDict()
        self.p = 0.0

    def _replace(self, in_b2: bool) -> None:
        t1 = self.t1
        from_t1 = bool(t1) and (
            (in_b2 and len(t1) == self.p) or len(t1) > self.p
        )
        if not self.t2:
            from_t1 = True
        if from_t1:
            old, _ = t1.popitem(last=False)
            self.b1[old] = None
        else:
            old, _ = self.t2.popitem(last=False)
            self.b2[old] = None
        self._evict(old)

    def _on_hit(self, key: Key) -> None:
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = None
        else:
            self.t2.move_to_end(key)

    def _on_miss(self, key: Key) -> None:
        c = self.capacity
        t1, b1, b2 = self.t1, self.b1, self.b2
        if key in b1:
            self.p = min(float(c), self.p + max(len(b2) / len(b1), 1.0))
            self._replace(False)
            del b1[key]
            self.t2[key] = None
        elif key in b2:
            self.p = max(0.0, self.p - max(len(b1) / len(b2), 1.0))
            self._replace(True)
            del b2[key]
            self.t2[key] = None
        else:
            l1 = len(t1) + len(b1)
            if l1 >= c:
                if len(t1) < c:
                    b1.popitem(last=False)
                    self._replace(False)
                else:
                    old, _ = t1.popitem(last=False)
                    self._evict(old)
            else:
                total = l1 + len(self.t2) + len(b2)
                if total >= c:
                    if total >= 2 * c:
                        b2.popitem(last=False)
                    self._replace(False)
            t1[key] = None
        self._insert(key)


class TwoQPolicy(_SelfManagedPolicy):
    """Full 2Q (Johnson & Shasha, VLDB'94).

    First-time keys enter the FIFO ``A1in`` (target ``kin`` of capacity); keys
    evicted from it are remembered in the ghost FIFO ``A1out`` (``kout`` of
    capacity). A miss on a remembered key admits it to the LRU ``Am``.
    """

    def __init__(self, capacity: int, kin: float = 0.25, kout: float = 0.5):
        super().__init__(capacity)
        if not 0.0 < kin < 1.0 or kout <= 0.0:
            raise ValueError("need 0 < kin < 1 and kout > 0")
        self.kin = max(1, int(kin * capacity))
        self.kout = max(1, int(kout * capacity))
        self.a1in: "OrderedDict[Key, None]" = OrderedDict()
        self.a1out: "OrderedDict[Key, None]" = OrderedDict()
        self.am: "OrderedDict[Key, None]" = OrderedDict()

    def _on_hit(self, key: Key) -> None:
        if key in self.am:
            self.am.move_to_end(key)
        # A hit in A1in is a correlated reference: no reordering.

    def _on_miss(self, key: Key) -> None:
        if len(self.resident) >= self.capacity:
            if len(self.a1in) > self.kin or not self.am:
                old, _ = self.a1in.popitem(last=False)
                self.a1out[old] = None
                if len(self.a1out) > self.kout:
                    self.a1out.popitem(last=False)
            else:
                old, _ = self.am.popitem(last=False)
            self._evict(old)
        if key in self.a1out:
            del self.a1out[key]
            self.am[key] = None
        else:
            self.a1in[key] = None
        self._insert(key)


class LRUKPolicy(_SelfManagedPolicy):
    """LRU-K (O'Neil, O'Neil & Weikum, SIGMOD'93), default K=2.

    Evict the resident with the largest backward K-distance (time since its
    K-th most recent reference). Keys with fewer than K references have
    infinite distance and go first, least-recently-used first. Reference
    history is retained for evicted keys (the key universe is small).
    """

    def __init__(self, capacity: int, k: int = 2):
        super().__init__(capacity)
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._hist: Dict[Key, Deque[int]] = {}
        self._stamp: Dict[Key, int] = {}
        self._heap: List[Tuple[int, int, Key]] = []
        self._clock = 0

    def _record_access(self, key: Key) -> None:
        hist = self._hist.get(key)
        if hist is None:
            hist = self._hist[key] = deque(maxlen=self.k)
        hist.append(self._clock)
        self._clock += 1

    def _push(self, key: Key) -> None:
        hist = self._hist[key]
        kth = hist[0] if len(hist) >= self.k else -1
        stamp = hist[-1]
        self._stamp[key] = stamp
        heap = self._heap
        heapq.heappush(heap, (kth, stamp, key))
        if len(heap) > 8 * self.capacity + 1024:  # drop stale entries
            live = self._stamp
            heap[:] = [e for e in heap if live.get(e[2]) == e[1]]
            heapq.heapify(heap)

    def _on_hit(self, key: Key) -> None:
        self._push(key)

    def _on_miss(self, key: Key) -> None:
        if len(self.resident) >= self.capacity:
            heap, stamp = self._heap, self._stamp
            while True:
                _, when, old = heapq.heappop(heap)
                if old in self.resident and stamp.get(old) == when:
                    break
            del stamp[old]
            self._evict(old)
        self._insert(key)
        self._push(key)


class LIRSPolicy(_SelfManagedPolicy):
    """LIRS (Jiang & Zhang, SIGMETRICS'02).

    Keys with a small inter-reference recency (IRR) form the LIR set
    (``capacity - hir`` slots); the rest of the cache holds resident HIR keys
    in the FIFO ``Q`` (``hir_frac`` of capacity, at least one slot), which is
    where victims come from. The recency stack ``S`` decides promotions: a HIR
    key re-referenced while still in ``S`` has a smaller IRR than the oldest
    LIR key and swaps places with it.
    """

    def __init__(self, capacity: int, hir_frac: float = 0.01):
        super().__init__(capacity)
        if not 0.0 < hir_frac < 1.0:
            raise ValueError("hir_frac must be in (0, 1)")
        self.l_hirs = max(1, int(round(hir_frac * capacity)))
        self.l_lirs = max(0, capacity - self.l_hirs)
        self.stack: "OrderedDict[Key, None]" = OrderedDict()  # last = top
        self.queue: "OrderedDict[Key, None]" = OrderedDict()  # first = victim
        self.lir: Set[Key] = set()

    def _prune(self) -> None:
        stack, lir = self.stack, self.lir
        while stack:
            bottom = next(iter(stack))
            if bottom in lir:
                return
            del stack[bottom]

    def _demote_bottom_lir(self) -> None:
        bottom = next(iter(self.stack))
        self.lir.discard(bottom)
        del self.stack[bottom]
        self.queue[bottom] = None
        self._prune()

    def _on_hit(self, key: Key) -> None:
        stack = self.stack
        if key in self.lir:
            was_bottom = next(iter(stack)) == key
            stack.move_to_end(key)
            if was_bottom:
                self._prune()
        elif key in stack and self.lir:  # resident HIR, small IRR -> LIR
            stack.move_to_end(key)
            del self.queue[key]
            self.lir.add(key)
            self._demote_bottom_lir()
        else:  # resident HIR, stays HIR (always so without an LIR set)
            stack[key] = None
            stack.move_to_end(key)
            self.queue.move_to_end(key)

    def _on_miss(self, key: Key) -> None:
        stack = self.stack
        if len(self.lir) < self.l_lirs:  # warm-up: fill the LIR set first
            self.lir.add(key)
            stack[key] = None
            stack.move_to_end(key)
            self._insert(key)
            return
        if len(self.resident) >= self.capacity:
            old, _ = self.queue.popitem(last=False)
            self._evict(old)
        self._insert(key)
        if key in stack and self.lir:  # non-resident HIR with small IRR
            stack.move_to_end(key)
            self.lir.add(key)
            self._demote_bottom_lir()
        else:
            stack[key] = None
            stack.move_to_end(key)
            self.queue[key] = None


class S3FIFOPolicy(_SelfManagedPolicy):
    """S3-FIFO (Yang et al., SOSP'23, Algorithm 1).

    A small FIFO ``S`` (``small`` of capacity) filters one-hit wonders, a main
    FIFO ``M`` with lazy reinsertion holds the rest, and a ghost FIFO ``G``
    (as many entries as ``M``) remembers keys evicted from ``S``. Each resident
    has a 2-bit access counter; a key leaves ``S`` for ``M`` if it was hit more
    than once (its bits are cleared on the move).
    """

    def __init__(self, capacity: int, small: float = 0.1):
        super().__init__(capacity)
        if not 0.0 < small < 1.0:
            raise ValueError("small must be in (0, 1)")
        self.s_cap = max(1, int(small * capacity))
        self.m_cap = max(0, capacity - self.s_cap)
        self.small: "OrderedDict[Key, None]" = OrderedDict()  # first = tail
        self.main: "OrderedDict[Key, None]" = OrderedDict()
        self.ghost: "OrderedDict[Key, None]" = OrderedDict()
        self.freq: Dict[Key, int] = {}

    def _on_hit(self, key: Key) -> None:
        if self.freq[key] < 3:
            self.freq[key] += 1

    def _evict_main(self) -> None:
        main, freq = self.main, self.freq
        while main:
            old, _ = main.popitem(last=False)
            if freq[old] > 0:
                freq[old] -= 1
                main[old] = None
            else:
                del freq[old]
                self._evict(old)
                return

    def _evict_small(self) -> None:
        small, freq = self.small, self.freq
        while small:
            old, _ = small.popitem(last=False)
            if freq[old] > 1:
                freq[old] = 0
                self.main[old] = None
                if len(self.main) > self.m_cap:
                    self._evict_main()
            else:
                del freq[old]
                self._evict(old)
                self.ghost[old] = None
                if len(self.ghost) > self.m_cap:
                    self.ghost.popitem(last=False)
                return

    def _on_miss(self, key: Key) -> None:
        while len(self.resident) >= self.capacity:
            if len(self.small) >= self.s_cap or not self.main:
                self._evict_small()
            else:
                self._evict_main()
        self.freq[key] = 0
        if key in self.ghost:
            del self.ghost[key]
            self.main[key] = None
        else:
            self.small[key] = None
        self._insert(key)


class SIEVEPolicy(_SelfManagedPolicy):
    """SIEVE (Zhang et al., NSDI'24).

    One FIFO queue plus a visited bit per key. A hand sweeps from the oldest
    toward the newest entry, clearing visited bits, and evicts the first
    unvisited key; it resumes from there on the next eviction.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._newer: Dict[Key, Optional[Key]] = {}
        self._older: Dict[Key, Optional[Key]] = {}
        self.visited: Dict[Key, bool] = {}
        self._head: Optional[Key] = None  # newest
        self._tail: Optional[Key] = None  # oldest
        self._hand: Optional[Key] = None

    def _on_hit(self, key: Key) -> None:
        self.visited[key] = True

    def _unlink(self, key: Key) -> None:
        newer, older = self._newer.pop(key), self._older.pop(key)
        if newer is not None:
            self._older[newer] = older
        else:
            self._head = older
        if older is not None:
            self._newer[older] = newer
        else:
            self._tail = newer

    def _on_miss(self, key: Key) -> None:
        if len(self.resident) >= self.capacity:
            obj = self._hand if self._hand is not None else self._tail
            while self.visited[obj]:
                self.visited[obj] = False
                obj = self._newer[obj]
                if obj is None:
                    obj = self._tail
            self._hand = self._newer[obj]
            self._unlink(obj)
            del self.visited[obj]
            self._evict(obj)
        self._newer[key] = None
        self._older[key] = self._head
        if self._head is not None:
            self._newer[self._head] = key
        self._head = key
        if self._tail is None:
            self._tail = key
        self.visited[key] = False
        self._insert(key)


class WTinyLFUPolicy(_SelfManagedPolicy):
    """W-TinyLFU (Einziger, Friedman & Manes, ACM ToS 2017; Caffeine).

    New keys enter a small LRU window (``window`` of capacity). The window's
    LRU victim competes with the main segmented-LRU's probation victim and is
    admitted only if its frequency is higher. Frequencies are exact counts
    (the key universe is small, so no sketch is needed) halved every
    ``sample * capacity`` accesses, TinyLFU's aging reset.
    """

    def __init__(
        self,
        capacity: int,
        window: float = 0.01,
        protected: float = 0.8,
        sample: int = 10,
    ):
        super().__init__(capacity)
        if not 0.0 < window < 1.0 or not 0.0 <= protected < 1.0:
            raise ValueError("need 0 < window < 1 and 0 <= protected < 1")
        self.w_cap = max(1, int(round(window * capacity)))
        self.m_cap = max(0, capacity - self.w_cap)
        self.p_cap = int(protected * self.m_cap)
        self.window: "OrderedDict[Key, None]" = OrderedDict()
        self.probation: "OrderedDict[Key, None]" = OrderedDict()
        self.protected: "OrderedDict[Key, None]" = OrderedDict()
        self.freq: Dict[Key, int] = defaultdict(int)
        self._reset_every = max(1, sample * capacity)
        self._ops = 0

    def _record_access(self, key: Key) -> None:
        self.freq[key] += 1
        self._ops += 1
        if self._ops >= self._reset_every:
            self._ops = 0
            for k in list(self.freq):
                halved = self.freq[k] >> 1
                if halved:
                    self.freq[k] = halved
                else:
                    del self.freq[k]

    def _on_hit(self, key: Key) -> None:
        if key in self.window:
            self.window.move_to_end(key)
        elif key in self.protected:
            self.protected.move_to_end(key)
        else:
            del self.probation[key]
            self.protected[key] = None
            if len(self.protected) > self.p_cap:
                demoted, _ = self.protected.popitem(last=False)
                self.probation[demoted] = None

    def _on_miss(self, key: Key) -> None:
        self.window[key] = None
        self._insert(key)
        if len(self.window) <= self.w_cap:
            return
        cand, _ = self.window.popitem(last=False)
        if len(self.probation) + len(self.protected) < self.m_cap:
            self.probation[cand] = None
            return
        segment = self.probation if self.probation else self.protected
        if not segment:  # no main region (capacity 1): drop the candidate
            self._evict(cand)
            return
        victim = next(iter(segment))
        if self.freq.get(cand, 0) > self.freq.get(victim, 0):
            del segment[victim]
            self._evict(victim)
            self.probation[cand] = None
        else:
            self._evict(cand)


class LayerLRUPolicy(_SelfManagedPolicy):
    """Per-layer slot quota with LRU inside each layer (Mixtral-Offloading).

    The capacity is split evenly across ``num_layers`` MoE layers (the first
    ``capacity % num_layers`` layers get one extra slot); a miss evicts the
    least-recently-used expert *of the same layer*. ``num_layers`` defaults to
    the largest layer id seen so far plus one.
    """

    def __init__(self, capacity: int, num_layers: Optional[int] = None):
        super().__init__(capacity)
        if num_layers is not None and num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self._max_layer = -1
        self._layers: Dict[int, "OrderedDict[Key, None]"] = defaultdict(
            OrderedDict
        )

    def _quota(self, layer: int) -> int:
        nl = self.num_layers or (self._max_layer + 1)
        base, extra = divmod(self.capacity, nl)
        return base + (1 if layer % nl < extra else 0)

    def _record_access(self, key: Key) -> None:
        if key[0] > self._max_layer:
            self._max_layer = key[0]

    def _on_hit(self, key: Key) -> None:
        self._layers[key[0]].move_to_end(key)

    def _on_miss(self, key: Key) -> None:
        layer = self._layers[key[0]]
        if layer and len(layer) >= max(1, self._quota(key[0])):
            old, _ = layer.popitem(last=False)
            self._evict(old)
        while len(self.resident) >= self.capacity:
            # Only reachable when quotas shift (layer count still being
            # learned) or capacity < layers: drop the oldest entry of the
            # most over-quota layer.
            over = max(
                (lay for lay, od in self._layers.items() if od),
                key=lambda lay: len(self._layers[lay]) - self._quota(lay),
            )
            old, _ = self._layers[over].popitem(last=False)
            self._evict(old)
        layer[key] = None
        self._insert(key)


class LayerCycleNextUsePolicy(_SelfManagedPolicy):
    """LCNU: evict the expert whose expected next use is farthest away.

    Batch-1 decode runs MoE layers in a fixed cycle, one *visit* per layer per
    token, so an online policy knows exactly when each layer runs again. For a
    resident expert ``(L, e)`` the predicted next use, in layer-steps, is::

        d(L) + num_layers * (1 / p - 1)

    ``d(L)`` is the number of layer-steps until layer ``L`` executes again
    (``num_layers`` for experts already used in the current visit of the
    current layer, 0 for current-layer experts not used yet). ``1/p - 1`` is
    the expected number of *further* visits before the expert is routed, with
    ``p`` its estimated per-visit routing probability::

        p = beta * ema + (1 - beta) * (count + 4 * prior) / (visits + 4)

    ``ema`` is an exponential moving average (rate ``alpha``) of "routed on
    this visit" over the visits of the expert's layer (recency / burstiness),
    the second term is its smoothed long-run routing frequency since first
    seen. Each estimator moves at most once per layer visit. Ties go to the
    smaller key, so the policy is deterministic.

    Everything is online: the policy reads only past routing plus the layer
    order (no lookahead). Defaults ``alpha=0.3, beta=0.5`` were selected on
    the first half of a real Qwen3-30B-A3B decode trace. ``prior`` should be
    ``top_k / num_experts``; ``num_layers`` defaults to the largest layer id
    seen so far plus one. ``cycle_aware=False`` drops ``d(L)`` (ablation).

    Victim search is exact but cheap: between two visits of a layer the
    scores of its experts are frozen, so the best candidate per layer is
    cached and only recomputed after that layer runs or loses its candidate.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.3,
        beta: float = 0.5,
        prior: float = 1.0 / 16,
        num_layers: Optional[int] = None,
        cycle_aware: bool = True,
    ):
        super().__init__(capacity)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not 0.0 < prior <= 1.0:
            raise ValueError("prior must be in (0, 1]")
        if num_layers is not None and num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.alpha = alpha
        self.beta = beta
        self.prior = prior
        self.num_layers = num_layers
        self.cycle_aware = cycle_aware
        self._log_keep = math.log1p(-alpha)
        self._last: Dict[Key, int] = {}  # layer-visit index of last routing
        self._ema: Dict[Key, float] = {}
        self._count: Dict[Key, int] = {}
        self._first: Dict[Key, int] = {}
        self._visits: Dict[int, int] = defaultdict(int)
        self._by_layer: Dict[int, Set[Key]] = defaultdict(set)
        self._cand: Dict[int, Tuple[float, Key]] = {}
        self._stale: Set[int] = set()
        self._cur: Optional[int] = None
        self._used_now: Set[Key] = set()
        self._max_layer = -1

    @property
    def cycle(self) -> int:
        """Layer-steps per decode step (``num_layers`` or inferred)."""
        return self.num_layers or (self._max_layer + 1)

    def routing_probability(self, key: Key) -> float:
        """Current per-visit routing-probability estimate for a seen key."""
        return self._probs((key,), self._visits[key[0]])[0]

    def _probs(self, keys, visits: int) -> List[float]:
        """Routing probabilities of ``keys`` (one layer, run ``visits`` times)."""
        beta, prior, log_keep = self.beta, self.prior, self._log_keep
        ema, last, count, first = (
            self._ema,
            self._last,
            self._count,
            self._first,
        )
        exp = math.exp
        out = []
        for key in keys:
            recent = ema[key] * exp(log_keep * (visits - last[key]))
            freq = (count[key] + 4.0 * prior) / (visits - first[key] + 4.0)
            p = beta * recent + (1.0 - beta) * freq
            out.append(p if p > 1e-12 else 1e-12)
        return out

    def _record_access(self, key: Key) -> None:
        layer = key[0]
        if layer > self._max_layer:
            self._max_layer = layer
            if self.num_layers is None:  # inferred cycle grew
                self._cand.clear()
        if layer != self._cur:
            self._cur = layer
            self._visits[layer] += 1
            self._stale.add(layer)
            self._used_now = set()
        visit = self._visits[layer]
        last = self._last.get(key)
        if last is None:
            self._ema[key] = (1.0 - self.alpha) * self.prior + self.alpha
            self._count[key] = 1
            self._first[key] = visit - 1
        elif visit > last:  # one routing event per layer visit
            gap = visit - last
            self._ema[key] = (
                self._ema[key] * math.exp(self._log_keep * gap) + self.alpha
            )
            self._count[key] += 1
        self._last[key] = visit
        self._used_now.add(key)

    def _layer_candidate(self, layer: int, cycle: int) -> Tuple[float, Key]:
        """Max of ``cycle * (1/p - 1)`` over the layer's residents."""
        keys = self._by_layer[layer]
        best_s: Optional[float] = None
        best_k: Optional[Key] = None
        for key, p in zip(keys, self._probs(keys, self._visits[layer])):
            s = cycle * (1.0 / p - 1.0)
            if best_s is None or s > best_s or (s == best_s and key < best_k):
                best_s, best_k = s, key
        return best_s, best_k

    def _distance(self, layer: int, key: Key) -> int:
        """Layer-steps until ``layer`` runs again (the ``d(L)`` term)."""
        if not self.cycle_aware:
            return 0
        cur, cycle = self._cur, self.cycle
        if layer == cur:
            return cycle if key in self._used_now else 0
        return (layer - cur) % cycle

    def _victim(self) -> Tuple[int, Key]:
        cur, cycle = self._cur, self.cycle
        best_s: Optional[float] = None
        best_k: Optional[Key] = None
        best_layer = -1
        for layer, keys in self._by_layer.items():
            if layer == cur or not keys:
                continue
            if layer in self._stale or layer not in self._cand:
                self._cand[layer] = self._layer_candidate(layer, cycle)
                self._stale.discard(layer)
            base, key = self._cand[layer]
            s = self._distance(layer, key) + base
            if best_s is None or s > best_s or (s == best_s and key < best_k):
                best_s, best_k, best_layer = s, key, layer
        keys = self._by_layer.get(cur, ())
        for key, p in zip(keys, self._probs(keys, self._visits[cur])):
            s = self._distance(cur, key) + cycle * (1.0 / p - 1.0)
            if best_s is None or s > best_s or (s == best_s and key < best_k):
                best_s, best_k, best_layer = s, key, cur
        return best_layer, best_k

    def _on_miss(self, key: Key) -> None:
        if len(self.resident) >= self.capacity:
            layer, old = self._victim()
            self._by_layer[layer].discard(old)
            if layer != self._cur:
                self._stale.add(layer)
            self._evict(old)
        self._by_layer[key[0]].add(key)
        self._insert(key)


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
    "arc": lambda cap, **kw: ARCPolicy(cap),
    "2q": lambda cap, **kw: TwoQPolicy(cap, **kw),
    "lru_k": lambda cap, **kw: LRUKPolicy(cap, **kw),
    "lirs": lambda cap, **kw: LIRSPolicy(cap, **kw),
    "s3fifo": lambda cap, **kw: S3FIFOPolicy(cap, **kw),
    "sieve": lambda cap, **kw: SIEVEPolicy(cap),
    "w_tinylfu": lambda cap, **kw: WTinyLFUPolicy(cap, **kw),
    "layer_lru": lambda cap, **kw: LayerLRUPolicy(cap, **kw),
    "lcnu": lambda cap, **kw: LayerCycleNextUsePolicy(cap, **kw),
    "rothead": lambda cap, **kw: _build_rothead(cap, **kw),
}


def _build_rothead(cap, **kw):
    # Lazy import: policies_rothead imports this module (no import cycle at
    # call time because policies.py is fully initialized by then).
    from benchmarks.cache_simulator.research.policies_rothead import (
        RotHeadPolicy,
    )

    return RotHeadPolicy(cap, **kw)

#: Policies that accept the model geometry (``num_layers``; ``lcnu`` also
#: ``prior = top_k / num_experts``) as keyword arguments.
GEOMETRY_POLICIES = {"layer_lru", "lcnu", "rothead"}


# Keep a module-level INF for callers that compare against Belady.
INF = math.inf
