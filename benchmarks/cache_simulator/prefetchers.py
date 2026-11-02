"""Composable expert-prefetch policies for the cache simulator.

Round 2 of the cache-policy study (``auto_research_prefetch_eviction.md``)
makes *prefetch* a policy of its own: a ``PrefetchPolicy`` predicts which
experts the next ``horizon`` layer visits will route to, and
``PrefetchCachePolicy`` composes any such predictor with any on-demand
evictor from ``policies.py`` under an *admission* rule for the prefetched
(speculative) experts. The four trace-oracle prefetch policies in
``policies.py`` stay as they are (hard-wired to LRU).

When prefetches are issued
    Batch-1 decode runs the MoE layers in a fixed cycle, one *visit* per layer
    per token. A layer's routing is known once its gate has run, so the
    prefetcher is consulted **once per layer visit**, after the visit's
    routing is known, and proposes up to ``budget`` experts for each of the
    next ``horizon`` visits. At most ``budget`` prefetches are issued per
    target visit over all the visits that propose it.

Online predictors (CAUSAL: they see only completed layer visits, i.e. the past
plus the current token's routing up to the current layer; unit-tested by
changing the future of a trace):

* ``PreviousTokenPrefetcher`` (``prev_token``) - a layer's experts from its
  previous visit (the previous token). P(an expert is routed again at the
  same layer on the next token) is 0.39-0.43 on Qwen3
  (``research/decode_policy_study.md``).
* ``FrequencyPrefetcher`` (``topk_ema``) - per-layer EMA of "routed on this
  visit"; the hottest experts of each upcoming layer.
* ``CoActivationPrefetcher`` (``coactivation``) - cross-layer co-activation
  learned online, an Expert-Activation-Matrix-style table (MoE-Infinity,
  arXiv:2401.14361; Klotski, arXiv:2502.06888): ``P(e' at layer l+d | e at
  layer l)`` counted over past tokens; the experts just routed vote for each
  upcoming layer.
* ``HybridPrefetcher`` (``hybrid``) - weighted sum of the three scores.
* ``SameIdPrefetcher`` (``same_id``) - the MoE-Infinity engine's current
  ``speculative_prefetch``: the current layer's top experts, same expert ids,
  at the next layer (``moe_infinity/memory/expert_prefetcher.py``).

Oracle bound (``uses_oracle = True``, an UPPER BOUND, not a policy):

* ``OraclePrefetcher`` (``oracle``) - the true routing of the next
  ``horizon`` visits, read from the trace.

Admission modes for prefetched experts (``PrefetchCachePolicy``):

* ``same_queue`` - prefetched experts enter the evictor's own cache and are
  ranked by its rule (today's behaviour: with LRU and the oracle this is the
  legacy ``SingleLayerAheadPolicy``). Non-self-managed evictors (LRU, LFU)
  protect the current prefetch batch; self-managed ones (LIRS, S3-FIFO,
  W-TinyLFU, per-layer LRU, LCNU, ...) insert through their own miss path.
* ``probation`` - prefetched experts wait in a staging area that shares the
  capacity and are first in line for eviction until used (PACMan-M,
  Mixtral-Offloading): a demand miss drops the staged expert with the
  farthest target first.
* ``pin`` - like ``probation`` but staged experts are protected until their
  target visit and the evictor's victim goes first (HOBBIT's mask); at most
  ``pin_fraction`` of the capacity may be staged.
* ``buffer`` - a separate prefetch buffer of ``buffer_slots`` slots, carved out
  of the total (the evictor is built with ``capacity - buffer_slots``);
  prefetches never evict cache residents and demand misses never evict the
  buffer.
* ``do_no_harm`` - integrated prefetching and caching (Cao, Felten, Karlin &
  Li, SIGMETRICS'95, rule 3 "never discard A to prefetch B when A will be
  referenced before B"), online: into a full cache a prefetch is admitted only
  if the candidate's expected next use is sooner than the victim's. Both come
  from LCNU's next-use model (``d(L) + layers * (1 - p) / p``); for the
  candidate the first-visit probability ``p`` is replaced by the prefetcher's
  measured precision at that distance and rank.

In every staging mode a staged expert is *use-it-or-lose-it*: at the gate of
its target visit it is dropped if the visit does not route it (an engine
knows the routing of a layer before fetching its experts; the wrapper reads
the current visit's routing from the bound trace for this, never the future).

Accounting:

* ``hits`` / ``misses`` - DEMAND events only. A prefetched expert's first use
  is a (covered) miss, so the demand hit rate is bounded by Belady on every
  trace.
* ``prefetch_covered`` - demand misses whose transfer a prefetch issued.
* ``prefetch_issued`` - prefetches admitted (already-resident or already
  staged candidates are skipped and cost nothing).
* ``prefetch_wasted`` - prefetched experts evicted or dropped before any
  use. ``issued == covered + wasted + still staged/speculative``.
* A per-visit event log (``visit_fresh``, ``pf_issue``, ``pf_used``,
  ``pf_removed``) feeds ``research/timing_model.py``.

Pure Python over flat ``(layer, expert)`` keys plus numpy; no torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from benchmarks.cache_simulator.policies import (
    CachePolicy,
    LayerCycleNextUsePolicy,
    _SelfManagedPolicy,
    layer_segments,
)

Key = Tuple[int, int]
Candidate = Tuple[Key, float]
#: ``(distance, target layer, candidates)`` for one upcoming layer visit.
Prediction = Tuple[int, int, List[Candidate]]

ADMISSION_MODES = ("same_queue", "probation", "pin", "buffer", "do_no_harm")
_STAGED = frozenset({"probation", "pin", "buffer", "do_no_harm"})
_SHARED = frozenset({"probation", "pin", "do_no_harm"})


# --------------------------------------------------------------------------- #
# Prefetch policies
# --------------------------------------------------------------------------- #
class PrefetchPolicy(ABC):
    """Predictor of the experts the next ``horizon`` layer visits route to.

    ``observe(key)`` is called for every demand access, in order. A layer
    visit is a maximal run of accesses to one layer. ``predict()`` closes the
    open visit and returns, for ``d = 1 .. horizon``, the target layer
    ``(layer + d) % cycle`` and at most ``budget`` ``(key, score)``
    candidates, best first. Online subclasses learn only from closed visits;
    oracle subclasses read the bound trace and set ``uses_oracle``.
    """

    uses_oracle: bool = False
    name: str = "base"

    def __init__(
        self,
        horizon: int = 1,
        budget: int = 8,
        num_layers: Optional[int] = None,
        num_experts: Optional[int] = None,
    ):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if budget < 1:
            raise ValueError("budget must be >= 1")
        if num_layers is not None and num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_experts is not None and num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.horizon = horizon
        self.budget = budget
        self.num_layers = num_layers
        self.num_experts = num_experts or 0
        self._max_layer = -1
        self._layer: Optional[int] = None  # layer of the last (open) visit
        self._keys: List[Key] = []
        self._seen: Set[Key] = set()
        self._open = False
        self.visits = 0  # closed visits

    @property
    def cycle(self) -> int:
        """Layer visits per decode step (``num_layers`` or inferred)."""
        return self.num_layers or (self._max_layer + 1)

    def bind_trace(self, accesses: Sequence[Key]) -> None:
        """Receive the full trace before replay (oracle variants only)."""

    def observe(self, key: Key) -> None:
        """Record one routed access of the current token."""
        layer = key[0]
        if self._open and layer != self._layer:
            self._close()
        if not self._open:
            self._open = True
            self._layer = layer
            self._keys = []
            self._seen = set()
            if layer > self._max_layer:
                self._max_layer = layer
        if key not in self._seen:
            self._seen.add(key)
            self._keys.append(key)
            if key[1] >= self.num_experts:
                self._grow(key[1] + 1)

    def predict(self) -> List[Prediction]:
        """Close the open visit; candidates for the next ``horizon`` visits."""
        if self._open:
            self._close()
        if self._layer is None:
            return []
        cycle = max(self.cycle, 1)
        out: List[Prediction] = []
        for d in range(1, min(self.horizon, cycle) + 1):
            layer = (self._layer + d) % cycle
            cands = self._predict(layer, d)
            out.append((d, layer, cands[: self.budget]))
        return out

    def _close(self) -> None:
        self._open = False
        self._learn(self._layer, self._keys)
        self.visits += 1

    # -- hooks -------------------------------------------------------------- #
    def _grow(self, num_experts: int) -> None:
        """An expert id beyond ``num_experts`` appeared (resize state)."""
        self.num_experts = num_experts

    def _learn(self, layer: int, keys: List[Key]) -> None:
        """A visit of ``layer`` closed with distinct routed ``keys``."""

    @abstractmethod
    def _predict(self, layer: int, d: int) -> List[Candidate]:
        """Ranked candidates for the visit ``d`` steps ahead (of ``layer``)."""


def _top(layer: int, scores: Optional[np.ndarray], m: int) -> List[Candidate]:
    """Best ``m`` experts of ``scores`` (> 0), ties toward the smaller id."""
    if scores is None:
        return []
    nz = np.flatnonzero(scores > 0.0)
    if nz.size == 0:
        return []
    vals = scores[nz]
    if nz.size > m:
        part = np.argpartition(-vals, m - 1)[:m]
        nz, vals = nz[part], vals[part]
    order = np.lexsort((nz, -vals))
    return [((layer, int(nz[i])), float(vals[i])) for i in order]


class PreviousTokenPrefetcher(PrefetchPolicy):
    """A layer's experts from its previous visit (the previous token).

    Candidates keep the routed (gate) order; the score is ``1 - rank /
    (2 * n)`` so a smaller budget keeps the top-weighted experts.
    """

    name = "prev_token"

    def __init__(self, horizon=1, budget=8, num_layers=None, num_experts=None):
        super().__init__(horizon, budget, num_layers, num_experts)
        self._last: Dict[int, List[Key]] = {}

    def _learn(self, layer: int, keys: List[Key]) -> None:
        self._last[layer] = list(keys)

    def scores(self, layer: int) -> Optional[np.ndarray]:
        keys = self._last.get(layer)
        if not keys:
            return None
        out = np.zeros(self.num_experts)
        n = len(keys)
        for rank, (_, expert) in enumerate(keys):
            out[expert] = 1.0 - rank / (2.0 * n)
        return out

    def _predict(self, layer: int, d: int) -> List[Candidate]:
        keys = self._last.get(layer, ())
        n = len(keys)
        return [(k, 1.0 - i / (2.0 * n)) for i, k in enumerate(keys)]


class FrequencyPrefetcher(PrefetchPolicy):
    """Per-layer EMA of routing; the hottest experts of each upcoming layer.

    At every visit of layer ``l``: ``ema_l = (1 - alpha) * ema_l + alpha *
    routed``, so ``ema`` estimates the per-visit routing probability with a
    memory of about ``1 / alpha`` tokens.
    """

    name = "topk_ema"

    def __init__(
        self,
        horizon=1,
        budget=8,
        num_layers=None,
        num_experts=None,
        alpha: float = 0.3,
    ):
        super().__init__(horizon, budget, num_layers, num_experts)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._ema: Dict[int, np.ndarray] = {}
        self._ranked: Dict[int, List[Candidate]] = {}

    def _grow(self, num_experts: int) -> None:
        for layer, vec in self._ema.items():
            self._ema[layer] = np.pad(vec, (0, num_experts - vec.size))
        super()._grow(num_experts)

    def _learn(self, layer: int, keys: List[Key]) -> None:
        vec = self._ema.get(layer)
        if vec is None:
            vec = self._ema[layer] = np.zeros(self.num_experts)
        vec *= 1.0 - self.alpha
        vec[[e for _, e in keys]] += self.alpha
        self._ranked.pop(layer, None)

    def scores(self, layer: int) -> Optional[np.ndarray]:
        return self._ema.get(layer)

    def _predict(self, layer: int, d: int) -> List[Candidate]:
        ranked = self._ranked.get(layer)
        if ranked is None:
            ranked = self._ranked[layer] = _top(
                layer, self._ema.get(layer), self.budget
            )
        return ranked


class CoActivationPrefetcher(PrefetchPolicy):
    """Online cross-layer co-activation (EAM-style) prefetch.

    ``C[l, d][e, e']`` counts how often expert ``e'`` was routed at the visit
    ``d`` steps after a visit of layer ``l`` that routed ``e`` (over the
    continuous layer cycle, so the last layers of a token predict the first
    layers of the next); ``N[l, d][e]`` counts the visits of ``l`` routing
    ``e``. After a visit of ``l`` with routed set ``S``, expert ``e'`` of the
    visit ``d`` steps ahead scores ``mean_{e in S} C[l, d][e, e'] /
    N[l, d][e]``, the average conditional routing probability. Unseen rows
    predict nothing (no oracle fallback).
    """

    name = "coactivation"

    def __init__(self, horizon=1, budget=8, num_layers=None, num_experts=None):
        super().__init__(horizon, budget, num_layers, num_experts)
        self._count: Dict[Tuple[int, int], np.ndarray] = {}
        self._norm: Dict[Tuple[int, int], np.ndarray] = {}
        self._history: Deque[Tuple[int, np.ndarray]] = deque(maxlen=horizon)
        self._cur: Optional[Tuple[int, np.ndarray]] = None
        self._cache: Dict[int, Optional[np.ndarray]] = {}

    def _grow(self, num_experts: int) -> None:
        pad = num_experts - self.num_experts
        for k, mat in self._count.items():
            self._count[k] = np.pad(mat, ((0, pad), (0, pad)))
        for k, vec in self._norm.items():
            self._norm[k] = np.pad(vec, (0, pad))
        super()._grow(num_experts)

    def _learn(self, layer: int, keys: List[Key]) -> None:
        dst = np.fromiter((e for _, e in keys), dtype=np.intp, count=len(keys))
        n = self.num_experts
        for d, (src_layer, src) in enumerate(reversed(self._history), start=1):
            slot = (src_layer, d)
            mat = self._count.get(slot)
            if mat is None:
                mat = self._count[slot] = np.zeros((n, n), dtype=np.float32)
                self._norm[slot] = np.zeros(n, dtype=np.float32)
            mat[np.ix_(src, dst)] += 1.0
            self._norm[slot][src] += 1.0
        self._history.append((layer, dst))
        self._cur = (layer, dst)
        self._cache = {}

    def scores(self, layer: int, d: int) -> Optional[np.ndarray]:
        if d in self._cache:
            return self._cache[d]
        out = None
        if self._cur is not None:
            src_layer, src = self._cur
            mat = self._count.get((src_layer, d))
            if mat is not None and src.size:
                norm = self._norm[(src_layer, d)][src]
                seen = norm > 0
                if seen.any():
                    rows = mat[src[seen]] / norm[seen, None]
                    out = rows.sum(axis=0) / src.size
        self._cache[d] = out
        return out

    def _predict(self, layer: int, d: int) -> List[Candidate]:
        return _top(layer, self.scores(layer, d), self.budget)


class HybridPrefetcher(PrefetchPolicy):
    """Weighted sum of previous-token, EMA-frequency and co-activation scores.

    All three are routing-probability-like scores in ``[0, 1]``:
    ``w_prev * prev_token + w_freq * ema + w_coact * coactivation``.
    """

    name = "hybrid"

    def __init__(
        self,
        horizon=1,
        budget=8,
        num_layers=None,
        num_experts=None,
        weights: Sequence[float] = (1.0, 1.0, 1.0),
        alpha: float = 0.3,
    ):
        super().__init__(horizon, budget, num_layers, num_experts)
        if len(weights) != 3 or any(w < 0 for w in weights) or not any(weights):
            raise ValueError("weights must be 3 non-negative values, not all 0")
        self.weights = tuple(float(w) for w in weights)
        self.prev = PreviousTokenPrefetcher(
            horizon, budget, num_layers, num_experts
        )
        self.freq = FrequencyPrefetcher(
            horizon, budget, num_layers, num_experts, alpha=alpha
        )
        self.coact = CoActivationPrefetcher(
            horizon, budget, num_layers, num_experts
        )
        self._parts = (self.prev, self.freq, self.coact)

    def _grow(self, num_experts: int) -> None:
        for part in self._parts:
            if part.num_experts < num_experts:
                part._grow(num_experts)
        super()._grow(num_experts)

    def _learn(self, layer: int, keys: List[Key]) -> None:
        for part in self._parts:
            part._learn(layer, keys)

    def _predict(self, layer: int, d: int) -> List[Candidate]:
        w_prev, w_freq, w_coact = self.weights
        total = np.zeros(self.num_experts)
        if w_prev:
            s = self.prev.scores(layer)
            if s is not None:
                total += w_prev * s
        if w_freq:
            s = self.freq.scores(layer)
            if s is not None:
                total += w_freq * s
        if w_coact:
            s = self.coact.scores(layer, d)
            if s is not None:
                total += w_coact * s
        return _top(layer, total, self.budget)


class SameIdPrefetcher(PrefetchPolicy):
    """The engine's ``speculative_prefetch``: same expert ids, next layer.

    MoE-Infinity's speculative path takes the top-k experts of layer ``L``'s
    router logits and prefetches the *same expert ids* at layer ``L + 1``
    (``moe_infinity/memory/expert_prefetcher.py``; engine default top-2). The
    routed order of the trace is the gate's top-k order.
    """

    name = "same_id"

    def __init__(self, horizon=1, budget=2, num_layers=None, num_experts=None):
        super().__init__(horizon, budget, num_layers, num_experts)
        self._cur: List[Key] = []

    def _learn(self, layer: int, keys: List[Key]) -> None:
        self._cur = list(keys)

    def _predict(self, layer: int, d: int) -> List[Candidate]:
        n = len(self._cur)
        return [
            ((layer, e), 1.0 - i / (2.0 * n))
            for i, (_, e) in enumerate(self._cur)
        ]


class OraclePrefetcher(PrefetchPolicy):
    """TRUE routing of the next ``horizon`` visits (UPPER BOUND, not online)."""

    uses_oracle = True
    name = "oracle"

    def __init__(self, horizon=1, budget=8, num_layers=None, num_experts=None):
        super().__init__(horizon, budget, num_layers, num_experts)
        self._segs: List[Tuple[int, List[Key]]] = []

    def bind_trace(self, accesses: Sequence[Key]) -> None:
        self._segs = [
            (layer, list(dict.fromkeys(accesses[start:end])))
            for layer, start, end in layer_segments(accesses)
        ]

    def _predict(self, layer: int, d: int) -> List[Candidate]:
        seg = self.visits - 1 + d
        if seg >= len(self._segs) or self._segs[seg][0] != layer:
            return []
        return [(k, 1.0) for k in self._segs[seg][1]]


#: Prefetchers that read future accesses -> report as upper bounds.
ORACLE_PREFETCHERS = frozenset({"oracle"})

PREFETCHER_FACTORIES = {
    "prev_token": PreviousTokenPrefetcher,
    "topk_ema": FrequencyPrefetcher,
    "coactivation": CoActivationPrefetcher,
    "hybrid": HybridPrefetcher,
    "same_id": SameIdPrefetcher,
    "oracle": OraclePrefetcher,
}


def build_prefetcher(name: str, **kwargs) -> PrefetchPolicy:
    """Instantiate a prefetcher by name (see ``PREFETCHER_FACTORIES``)."""
    try:
        factory = PREFETCHER_FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown prefetcher {name!r}; "
            f"choices: {sorted(PREFETCHER_FACTORIES)}"
        ) from exc
    return factory(**kwargs)


# --------------------------------------------------------------------------- #
# Next-use estimates (LCNU's model)
# --------------------------------------------------------------------------- #
def next_use_estimate(estimator: LayerCycleNextUsePolicy, key: Key) -> float:
    """Expected layer-steps until ``key`` is used again (LCNU's score).

    ``estimator`` is an LCNU instance replayed over the demand stream
    (capacity 0 works: it then keeps statistics only). Unseen keys get the
    routing prior.
    """
    cycle = max(estimator.cycle, 1)
    if estimator._cur is None:
        return float(cycle)
    if key in estimator._last:
        p = estimator.routing_probability(key)
    else:
        p = estimator.prior
    return estimator._distance(key[0], key) + cycle * (1.0 / p - 1.0)


# --------------------------------------------------------------------------- #
# Composition: any evictor x any prefetcher x admission mode
# --------------------------------------------------------------------------- #
class _UnionView:
    """Read-only ``len`` / ``in`` / iteration over disjoint key containers."""

    __slots__ = ("_parts",)

    def __init__(self, *parts):
        self._parts = parts

    def __len__(self) -> int:
        return sum(len(part) for part in self._parts)

    def __contains__(self, key) -> bool:
        return any(key in part for part in self._parts)

    def __iter__(self):
        for part in self._parts:
            yield from part


class PrefetchCachePolicy(CachePolicy):
    """Compose an on-demand evictor with a prefetcher under an admission rule.

    ``evictor`` is any pure (non-oracle) ``CachePolicy``; ``prefetcher`` is a
    ``PrefetchPolicy`` or ``None``. With ``prefetcher=None`` every access is
    delegated verbatim to ``evictor.on_access``, so hit/miss counts are
    bit-identical to running the evictor alone. See the module docstring for
    the admission modes and the accounting.

    ``buffer_slots`` (``buffer`` mode) is the prefetch-buffer size; build the
    evictor with ``capacity - buffer_slots`` to compare at equal memory.
    ``estimator`` (``do_no_harm``) is the LCNU next-use model; by default the
    evictor itself if it is LCNU, else a capacity-0 LCNU shadow with
    ``num_layers`` / ``prior``.
    """

    def __init__(
        self,
        evictor: CachePolicy,
        prefetcher: Optional[PrefetchPolicy] = None,
        admission: str = "same_queue",
        buffer_slots: int = 0,
        estimator: Optional[LayerCycleNextUsePolicy] = None,
        pin_fraction: float = 0.5,
        num_layers: Optional[int] = None,
        prior: Optional[float] = None,
    ):
        if admission not in ADMISSION_MODES:
            raise ValueError(
                f"admission must be one of {ADMISSION_MODES}, got {admission!r}"
            )
        if isinstance(evictor, PrefetchCachePolicy):
            raise ValueError("evictor must not itself be a PrefetchCachePolicy")
        if evictor.uses_oracle:
            raise ValueError(
                "wrap a pure evictor; prefetch is the prefetcher's"
            )
        if not 0.0 < pin_fraction <= 1.0:
            raise ValueError("pin_fraction must be in (0, 1]")
        self.evictor = evictor
        self.prefetcher = prefetcher
        self.admission = admission
        self._self_managed = isinstance(evictor, _SelfManagedPolicy)
        if admission == "buffer" and prefetcher is not None:
            if buffer_slots < 1:
                raise ValueError("buffer mode needs buffer_slots >= 1")
        else:
            buffer_slots = 0
        self.buffer_slots = buffer_slots
        self.capacity = evictor.capacity + buffer_slots
        self._pin_cap = max(1, int(pin_fraction * self.capacity))
        self.hits = 0
        self.misses = 0
        self.prefetch_covered = 0
        self.prefetch_issued = 0
        self.prefetch_wasted = 0
        self._accesses: Sequence[Key] = ()
        self._index = 0
        # Staged (prefetched, unused) experts: key -> prefetch id.
        self.staged: Dict[Key, int] = {}
        self._by_target: Dict[int, List[int]] = {}
        self._cur_pids: List[int] = []
        # same_queue: speculative residents inside the evictor -> prefetch id.
        self._spec: Dict[Key, int] = {}
        self._batch: Set[Key] = set()
        # Per-prefetch records (index = prefetch id) and per-visit log.
        self.pf_key: List[Key] = []
        self.pf_issue: List[int] = []
        self.pf_target: List[int] = []
        self.pf_dist: List[int] = []
        self.pf_rank: List[int] = []
        self.pf_used: List[int] = []
        self.pf_removed: List[int] = []
        self.visit_fresh: List[int] = []
        self._visit = -1
        self._cur_layer: Optional[int] = None
        self._issued_for: Dict[int, int] = {}
        self._starts: List[int] = []
        # Prediction quality: predictions per target visit, and per
        # (distance, rank) totals / hits; recall denominators per distance.
        self._preds: Dict[int, List[Tuple[int, List[Key]]]] = {}
        horizon = prefetcher.horizon if prefetcher is not None else 1
        budget = prefetcher.budget if prefetcher is not None else 1
        width = max(1, min(budget, 64))
        self.pred_n = [[0] * width for _ in range(horizon + 1)]
        self.pred_hit = [[0] * width for _ in range(horizon + 1)]
        self.pred_routed = [0] * (horizon + 1)
        self._estimator: Optional[LayerCycleNextUsePolicy] = None
        if admission == "do_no_harm" and prefetcher is not None:
            if estimator is not None:
                self._estimator = estimator
            elif isinstance(evictor, LayerCycleNextUsePolicy):
                self._estimator = evictor
            else:
                kwargs = {}
                if num_layers is not None:
                    kwargs["num_layers"] = num_layers
                if prior is not None:
                    kwargs["prior"] = prior
                self._estimator = LayerCycleNextUsePolicy(0, **kwargs)
        if self._self_managed:
            evictor._evict_callback = self._on_evicted

    # -- plumbing ----------------------------------------------------------- #
    @property
    def uses_oracle(self) -> bool:  # type: ignore[override]
        return self.prefetcher is not None and self.prefetcher.uses_oracle

    @property
    def resident(self):  # type: ignore[override]
        if self.staged:
            return _UnionView(self.evictor.resident, self.staged)
        return self.evictor.resident

    @property
    def visits(self) -> int:
        return len(self.visit_fresh)

    def bind_trace(self, accesses: Sequence[Key]) -> None:
        self._accesses = accesses
        self._index = 0
        self.evictor.bind_trace(accesses)
        if self.prefetcher is not None:
            self.prefetcher.bind_trace(accesses)
            self._starts = [s for _, s, _ in layer_segments(accesses)]
            self._starts.append(len(accesses))
        est = self._estimator
        if est is not None and est is not self.evictor:
            est.bind_trace(accesses)

    def _select_victim(self, protect: Sequence[Key]) -> Optional[Key]:
        raise RuntimeError("PrefetchCachePolicy delegates eviction")

    def precision(self, d: int, rank: int) -> float:
        """Measured precision of predictions ``d`` visits ahead at ``rank``."""
        row_n, row_h = self.pred_n[d], self.pred_hit[d]
        r = min(rank, len(row_n) - 1)
        return (row_h[r] + 1.0) / (row_n[r] + 2.0)

    def prediction_stats(self) -> Dict[int, Tuple[float, float]]:
        """``{distance: (precision, recall)}`` of the prefetcher's candidates."""
        out = {}
        for d in range(1, len(self.pred_n)):
            n, h = sum(self.pred_n[d]), sum(self.pred_hit[d])
            routed = self.pred_routed[d]
            out[d] = (h / n if n else 0.0, h / routed if routed else 0.0)
        return out

    # -- core --------------------------------------------------------------- #
    def on_access(self, key: Key) -> bool:
        ev = self.evictor
        layer = key[0]
        if self.prefetcher is None:
            if layer != self._cur_layer:
                self._cur_layer = layer
                self.visit_fresh.append(0)
            hit = ev.on_access(key)
            if hit:
                self.hits += 1
            else:
                self.misses += 1
                self.visit_fresh[-1] += 1
            self._index += 1
            return hit
        if layer != self._cur_layer:
            if self._cur_layer is not None:
                self._end_visit()
            self._start_visit(layer)
        hit = self._demand(key)
        self.prefetcher.observe(key)
        self._index += 1
        ev._index += 1
        return hit

    # -- visits ------------------------------------------------------------- #
    def _start_visit(self, layer: int) -> None:
        self._visit += 1
        v = self._visit
        self._cur_layer = layer
        self.visit_fresh.append(0)
        self._issued_for.pop(v, None)
        routed: Optional[Set[Key]] = None
        if v + 1 < len(self._starts):
            routed = set(self._accesses[self._starts[v] : self._starts[v + 1]])
        preds = self._preds.pop(v, None)
        if preds and routed is not None:
            seen_d = set()
            for d, keys in preds:
                row_n, row_h = self.pred_n[d], self.pred_hit[d]
                last = len(row_n) - 1
                for rank, k in enumerate(keys):
                    r = rank if rank < last else last
                    row_n[r] += 1
                    if k in routed:
                        row_h[r] += 1
                if d not in seen_d:
                    seen_d.add(d)
                    self.pred_routed[d] += len(routed)
        # Staged experts for this visit: the gate shows whether they are used.
        pids = self._by_target.pop(v, [])
        keep = []
        for pid in pids:
            k = self.pf_key[pid]
            if self.staged.get(k) != pid:
                continue
            if routed is not None and k not in routed:
                self._drop(pid)
            else:
                keep.append(pid)
        self._cur_pids = keep

    def _end_visit(self) -> None:
        for pid in self._cur_pids:  # use it or lose it
            if self.staged.get(self.pf_key[pid]) == pid:
                self._drop(pid)
        self._cur_pids = []
        preds = self.prefetcher.predict()
        if not preds or self.capacity <= 0:
            return
        v = self._visit
        ev = self.evictor
        budget = self.prefetcher.budget
        same_queue = self.admission == "same_queue"
        if same_queue and not self._self_managed:
            self._batch = {k for _, _, cands in preds for k, _ in cands}
        for d, _, cands in preds:
            tv = v + d
            if cands:
                self._preds.setdefault(tv, []).append(
                    (d, [k for k, _ in cands])
                )
            issued = self._issued_for.get(tv, 0)
            for rank, (k, _) in enumerate(cands):
                if issued >= budget:
                    break
                if k in self.staged or k in ev.resident:
                    continue
                if self._admit(k, tv, d, rank):
                    issued += 1
                elif self.admission != "do_no_harm":
                    break  # no room for this target; lower ranks fare no better
            self._issued_for[tv] = issued
        self._batch = set()

    # -- demand path -------------------------------------------------------- #
    def _demand(self, key: Key) -> bool:
        ev = self.evictor
        ev._record_access(key)
        est = self._estimator
        if est is not None and est is not ev:
            est.on_access(key)
        pid = self.staged.pop(key, None)
        if pid is not None:  # a staged prefetch covers this miss
            self.misses += 1
            self.prefetch_covered += 1
            self.pf_used[pid] = self._visit
            self._insert_demand(key, promotion=True)
            return False
        if key in ev.resident:
            spid = self._spec.pop(key, None)
            if spid is not None:  # same_queue speculative resident
                self.misses += 1
                self.prefetch_covered += 1
                self.pf_used[spid] = self._visit
                self._touch(key)
                return False
            self.hits += 1
            self._touch(key)
            return True
        self.misses += 1
        self.visit_fresh[-1] += 1
        self._insert_demand(key)
        return False

    def _touch(self, key: Key) -> None:
        if self._self_managed:
            self.evictor._on_hit(key)
        else:
            self.evictor.resident.move_to_end(key)

    def _insert_demand(self, key: Key, promotion: bool = False) -> None:
        ev = self.evictor
        if ev.capacity <= 0:
            return
        if (
            not promotion
            and self.staged
            and self.admission in _SHARED
            and len(ev.resident) + len(self.staged) >= self.capacity
        ):
            self._room_for_demand()
        if self._self_managed:
            ev._on_miss(key)
            return
        while len(ev.resident) >= ev.capacity:
            victim = ev._select_victim((key,))
            if victim is None:
                return
            del ev.resident[victim]
            self._on_evicted(victim)
        ev.resident[key] = None

    def _room_for_demand(self) -> None:
        """Free one slot of a full shared cache for a demand miss."""
        v = self._visit
        mode = self.admission
        if mode == "probation":
            pid = self._farthest_staged(v)
            if pid is not None:
                self._drop(pid)
                return
            if self._pop_main():
                return
        elif mode == "pin":
            if self._pop_main():
                return
            pid = self._farthest_staged(v)
            if pid is not None:
                self._drop(pid)
                return
        else:  # do_no_harm: evict the farther expected next use
            pid = self._farthest_staged(v)
            hint = self._main_hint()
            if pid is not None and (
                hint is None
                or self._staged_next_use(pid) >= self._next_use(hint)
            ):
                self._drop(pid)
                return
            if hint is not None and self._pop_main():
                return
            if pid is not None:
                self._drop(pid)
                return
        pid = self._farthest_staged(v - 1)  # last resort: a protected entry
        if pid is not None:
            self._drop(pid)

    # -- prefetch admission ------------------------------------------------- #
    def _admit(self, key: Key, tv: int, d: int, rank: int) -> bool:
        mode = self.admission
        ev = self.evictor
        if mode == "same_queue":
            if ev.capacity <= 0:
                return False
            if self._self_managed:
                ev._prepare_speculative(key)
                self._spec[key] = self._new_pid(key, tv, d, rank)
                ev._on_miss(key)
                return True
            if len(ev.resident) >= ev.capacity:
                victim = ev._select_victim(self._batch)
                if victim is None:
                    return False
                del ev.resident[victim]
                self._on_evicted(victim)
            ev.resident[key] = None
            self._spec[key] = self._new_pid(key, tv, d, rank)
            return True
        if mode == "buffer":
            if len(self.staged) >= self.buffer_slots:
                pid = self._farthest_staged(tv)
                if pid is None:
                    return False
                self._drop(pid)
            self._stage(key, tv, d, rank)
            return True
        full = len(ev.resident) + len(self.staged) >= self.capacity
        if mode == "pin":
            if len(self.staged) >= self._pin_cap:
                return False
            if full and not self._pop_main():
                return False
        elif mode == "probation":
            if full:
                pid = self._farthest_staged(tv)
                if pid is not None:
                    self._drop(pid)
                elif not self._pop_main():
                    return False
        elif full:  # do_no_harm
            cand = self._candidate_next_use(key, d, rank)
            pid = self._farthest_staged(self._visit)
            hint = self._main_hint()
            e_s = self._staged_next_use(pid) if pid is not None else -1.0
            e_m = self._next_use(hint) if hint is not None else -1.0
            if max(e_s, e_m) <= cand:
                return False
            if e_s >= e_m:
                self._drop(pid)
            elif not self._pop_main():
                return False
        self._stage(key, tv, d, rank)
        return True

    # -- staging helpers ---------------------------------------------------- #
    def _new_pid(self, key: Key, tv: int, d: int, rank: int) -> int:
        pid = len(self.pf_key)
        self.pf_key.append(key)
        self.pf_issue.append(self._visit)
        self.pf_target.append(tv)
        self.pf_dist.append(d)
        self.pf_rank.append(rank)
        self.pf_used.append(-1)
        self.pf_removed.append(-1)
        self.prefetch_issued += 1
        return pid

    def _stage(self, key: Key, tv: int, d: int, rank: int) -> None:
        pid = self._new_pid(key, tv, d, rank)
        self.staged[key] = pid
        self._by_target.setdefault(tv, []).append(pid)

    def _drop(self, pid: int) -> None:
        del self.staged[self.pf_key[pid]]
        self.pf_removed[pid] = self._visit
        self.prefetch_wasted += 1

    def _on_evicted(self, victim: Key) -> None:
        spid = self._spec.pop(victim, None)
        if spid is not None:
            self.pf_removed[spid] = self._visit
            self.prefetch_wasted += 1

    def _farthest_staged(self, beyond: int) -> Optional[int]:
        """A live staged prefetch with the farthest target ``> beyond``."""
        by_target, staged, keys = self._by_target, self.staged, self.pf_key
        while True:
            best = None
            for t in by_target:
                if t > beyond and (best is None or t > best):
                    best = t
            if best is None:
                if beyond < self._visit:  # protected: current-visit entries
                    for pid in reversed(self._cur_pids):
                        if staged.get(keys[pid]) == pid:
                            return pid
                return None
            pids = by_target[best]
            while pids:
                pid = pids[-1]
                if staged.get(keys[pid]) == pid:
                    return pid
                pids.pop()
            del by_target[best]

    def _main_hint(self) -> Optional[Key]:
        ev = self.evictor
        if not ev.resident:
            return None
        if self._self_managed:
            return ev.victim_hint()
        return ev._select_victim(())

    def _pop_main(self) -> bool:
        ev = self.evictor
        if not ev.resident:
            return False
        if self._self_managed:
            return ev.pop_victim() is not None
        victim = ev._select_victim(())
        if victim is None:
            return False
        del ev.resident[victim]
        self._on_evicted(victim)
        return True

    def _next_use(self, key: Key) -> float:
        return next_use_estimate(self._estimator, key)

    def _expected_next_use(self, key: Key, dist: int, q: float) -> float:
        """``dist + (1 - q) * cycle / p``: LCNU's estimate, first visit at q."""
        est = self._estimator
        cycle = max(est.cycle, 1)
        p = est.routing_probability(key) if key in est._last else est.prior
        return dist + (1.0 - q) * cycle / p

    def _candidate_next_use(self, key: Key, d: int, rank: int) -> float:
        return self._expected_next_use(key, d, self.precision(d, rank))

    def _staged_next_use(self, pid: int) -> float:
        dist = max(self.pf_target[pid] - self._visit, 0)
        q = self.precision(self.pf_dist[pid], self.pf_rank[pid])
        return self._expected_next_use(self.pf_key[pid], dist, q)


def build_composed(
    evictor_name: str,
    capacity: int,
    prefetcher: Optional[PrefetchPolicy] = None,
    admission: str = "same_queue",
    buffer_slots: int = 0,
    num_layers: Optional[int] = None,
    prior: Optional[float] = None,
    pin_fraction: float = 0.5,
) -> PrefetchCachePolicy:
    """``PrefetchCachePolicy`` over ``build_policy(evictor_name, ...)``.

    ``capacity`` is the TOTAL number of expert slots; in ``buffer`` mode the
    evictor gets ``capacity - buffer_slots`` of them.
    """
    from benchmarks.cache_simulator.policies import (
        GEOMETRY_POLICIES,
        build_policy,
    )

    kwargs = {}
    if evictor_name in GEOMETRY_POLICIES and num_layers is not None:
        kwargs["num_layers"] = num_layers
        if evictor_name == "lcnu" and prior is not None:
            kwargs["prior"] = prior
    slots = capacity
    if admission == "buffer" and prefetcher is not None:
        if not 0 < buffer_slots < capacity:
            raise ValueError("need 0 < buffer_slots < capacity")
        slots = capacity - buffer_slots
    evictor = build_policy(evictor_name, slots, **kwargs)
    return PrefetchCachePolicy(
        evictor,
        prefetcher,
        admission=admission,
        buffer_slots=buffer_slots,
        pin_fraction=pin_fraction,
        num_layers=num_layers,
        prior=prior,
    )


__all__ = [
    "ADMISSION_MODES",
    "ORACLE_PREFETCHERS",
    "PREFETCHER_FACTORIES",
    "CoActivationPrefetcher",
    "FrequencyPrefetcher",
    "HybridPrefetcher",
    "OraclePrefetcher",
    "PrefetchCachePolicy",
    "PrefetchPolicy",
    "PreviousTokenPrefetcher",
    "SameIdPrefetcher",
    "build_composed",
    "build_prefetcher",
    "next_use_estimate",
]
