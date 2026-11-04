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
  ``depth`` uses the last few visits as evidence and ``scoring="bayes"``
  combines it naive-Bayes style.
* ``HybridPrefetcher`` (``hybrid``) - weighted sum of the three scores.
* ``LogisticPrefetcher`` (``logistic``) - online logistic regression on the
  routed set, the target layer's previous-token indicator and its EMA.
* ``KNNPrefetcher`` (``knn``) - nearest neighbours over routing trajectories:
  past tokens whose last ``context`` layers routed most like the current
  token vote for the next layer's experts.
* ``SameIdPrefetcher`` (``same_id``) - the MoE-Infinity engine's current
  ``speculative_prefetch``: the current layer's top experts, same expert ids,
  at the next layer (``moe_infinity/memory/expert_prefetcher.py``).

Every predictor accepts ``min_score``: candidates scoring below it are not
proposed (an adaptive budget; scores are routing-probability-like).

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

import math
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
        min_score: float = 0.0,
    ):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if budget < 1:
            raise ValueError("budget must be >= 1")
        if num_layers is not None and num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_experts is not None and num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if min_score < 0:
            raise ValueError("min_score must be >= 0")
        self.horizon = horizon
        self.budget = budget
        #: Adaptive budget: candidates scoring below this are not proposed
        #: (scores are routing-probability-like for every online predictor).
        self.min_score = min_score
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
        floor = self.min_score
        for d in range(1, min(self.horizon, cycle) + 1):
            layer = (self._layer + d) % cycle
            cands = self._predict(layer, d)[: self.budget]
            if floor > 0.0:
                cands = [c for c in cands if c[1] >= floor]
            out.append((d, layer, cands))
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

    def __init__(
        self, horizon=1, budget=8, num_layers=None, num_experts=None, **kw
    ):
        super().__init__(horizon, budget, num_layers, num_experts, **kw)
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
        **kw,
    ):
        super().__init__(horizon, budget, num_layers, num_experts, **kw)
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


#: Co-activation scoring rules.
COACT_SCORING = ("mean", "bayes")


class CoActivationPrefetcher(PrefetchPolicy):
    """Online cross-layer co-activation (EAM-style) prefetch.

    ``C[l, d][e, e']`` counts how often expert ``e'`` was routed at the visit
    ``d`` steps after a visit of layer ``l`` that routed ``e`` (over the
    continuous layer cycle, so the last layers of a token predict the first
    layers of the next); ``N[l, d][e]`` counts the visits of ``l`` routing
    ``e``. Unseen rows predict nothing (no oracle fallback).

    The evidence is the routed sets ``S_j`` of the last ``depth`` visits
    (``j = 0`` is the visit that just closed), weighted ``decay ** j``
    (Klotski uses several preceding layers). ``scoring``:

    * ``mean`` - expert ``e'`` of the visit ``d`` steps ahead scores the
      average conditional routing probability ``mean_{e in S_j}
      C[l_j, d + j][e, e'] / N[l_j, d + j][e]``, averaged over ``j``.
    * ``bayes`` - naive Bayes: ``log pi(e') + sum_j w_j sum_{e in S_j}
      log(P(e' | e) / pi(e'))`` with ``P(e' | e)`` smoothed toward the target
      layer's routing frequency ``pi`` (``smoothing`` pseudo-visits), turned
      into probabilities that sum to the routed-set size.
    """

    name = "coactivation"

    def __init__(
        self,
        horizon=1,
        budget=8,
        num_layers=None,
        num_experts=None,
        depth: int = 1,
        decay: float = 1.0,
        scoring: str = "mean",
        smoothing: float = 1.0,
        **kw,
    ):
        super().__init__(horizon, budget, num_layers, num_experts, **kw)
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if scoring not in COACT_SCORING:
            raise ValueError(f"scoring must be one of {COACT_SCORING}")
        if smoothing <= 0:
            raise ValueError("smoothing must be > 0")
        self.depth = depth
        self.decay = decay
        self.scoring = scoring
        self.smoothing = smoothing
        self._span = horizon + depth - 1
        self._count: Dict[Tuple[int, int], np.ndarray] = {}
        self._norm: Dict[Tuple[int, int], np.ndarray] = {}
        self._history: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self._span)
        self._marg: Dict[int, np.ndarray] = {}
        self._layer_visits: Dict[int, int] = {}
        self._cache: Dict[int, Optional[np.ndarray]] = {}

    def _grow(self, num_experts: int) -> None:
        pad = num_experts - self.num_experts
        for k, mat in self._count.items():
            self._count[k] = np.pad(mat, ((0, pad), (0, pad)))
        for k, vec in self._norm.items():
            self._norm[k] = np.pad(vec, (0, pad))
        for k, vec in self._marg.items():
            self._marg[k] = np.pad(vec, (0, pad))
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
        if self.scoring == "bayes":
            marg = self._marg.get(layer)
            if marg is None:
                marg = self._marg[layer] = np.zeros(n)
            marg[dst] += 1.0
            self._layer_visits[layer] = self._layer_visits.get(layer, 0) + 1
        self._cache = {}

    def _prior(self, layer: int, size: int) -> Optional[np.ndarray]:
        marg = self._marg.get(layer)
        if marg is None:
            return None
        n = self.num_experts
        base = size / n
        return (marg + base) / (self._layer_visits[layer] + 1.0)

    def scores(self, layer: int, d: int) -> Optional[np.ndarray]:
        if d in self._cache:
            return self._cache[d]
        hist = self._history
        total = None
        weight = 0.0
        prior = None
        if self.scoring == "bayes" and hist:
            prior = self._prior(layer, max(hist[-1][1].size, 1))
        for j in range(min(self.depth, len(hist))):
            src_layer, src = hist[-1 - j]
            slot = (src_layer, d + j)
            mat = self._count.get(slot)
            if mat is None or not src.size:
                continue
            norm = self._norm[slot][src]
            w = self.decay**j
            if self.scoring == "mean":
                seen = norm > 0
                if not seen.any():
                    continue
                part = (mat[src[seen]] / norm[seen, None]).sum(
                    axis=0
                ) / src.size
            else:
                if prior is None:
                    continue
                k = self.smoothing
                cond = (mat[src] + k * prior) / (norm[:, None] + k)
                part = (np.log(cond) - np.log(prior)).sum(axis=0)
            total = w * part if total is None else total + w * part
            weight += w
        out = None
        if total is not None:
            if self.scoring == "mean":
                out = total / weight
            else:
                logit = np.log(prior) + total
                prob = np.exp(logit - logit.max())
                out = np.minimum(prob * (hist[-1][1].size / prob.sum()), 1.0)
        self._cache[d] = out
        return out

    def _predict(self, layer: int, d: int) -> List[Candidate]:
        return _top(layer, self.scores(layer, d), self.budget)


class HybridPrefetcher(PrefetchPolicy):
    """Weighted sum of previous-token, EMA-frequency and co-activation scores.

    All three are routing-probability-like scores in ``[0, 1]``:
    ``w_prev * prev_token + w_freq * ema + w_coact * coactivation`` (so a
    ``min_score`` threshold is on that scale, at most ``sum(weights)``).
    ``depth`` / ``decay`` / ``scoring`` configure the co-activation part.
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
        depth: int = 1,
        decay: float = 1.0,
        scoring: str = "mean",
        **kw,
    ):
        super().__init__(horizon, budget, num_layers, num_experts, **kw)
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
            horizon,
            budget,
            num_layers,
            num_experts,
            depth=depth,
            decay=decay,
            scoring=scoring,
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


class LogisticPrefetcher(PrefetchPolicy):
    """Online logistic regression on routing features (learned predictor).

    For the visit ``d`` steps after a visit of layer ``l`` with routed set
    ``S``, expert ``e'`` gets ``P = sigmoid(b[e'] + sum_{e in S} W[e, e'] +
    w_prev * prev(e') + w_ema * ema(e'))``: a learned co-activation table
    ``W`` (one per ``(l, d)``) with a per-expert bias, plus the target layer's
    previous-token indicator and EMA (rate ``alpha``) with learned scalar
    weights. Trained by SGD on the log-loss (rate ``lr``, L2 ``l2`` on the
    touched rows) when the target visit closes; features are those available
    at prediction time. Initialized at the routing prior.
    """

    name = "logistic"

    def __init__(
        self,
        horizon=1,
        budget=8,
        num_layers=None,
        num_experts=None,
        lr: float = 0.05,
        l2: float = 0.0,
        alpha: float = 0.4,
        top_k: int = 8,
        **kw,
    ):
        super().__init__(horizon, budget, num_layers, num_experts, **kw)
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        self.lr = lr
        self.l2 = l2
        self.alpha = alpha
        self.top_k = top_k
        self._W: Dict[Tuple[int, int], np.ndarray] = {}
        self._b: Dict[Tuple[int, int], np.ndarray] = {}
        self._wf: Dict[Tuple[int, int], np.ndarray] = {}  # [w_prev, w_ema]
        self._prev: Dict[int, np.ndarray] = {}
        self._ema: Dict[int, np.ndarray] = {}
        self._history: Deque[Tuple[int, np.ndarray]] = deque(maxlen=horizon)
        self._cache: Dict[int, Optional[np.ndarray]] = {}

    def _grow(self, num_experts: int) -> None:
        pad = num_experts - self.num_experts
        prior = self._prior_logit(num_experts)
        for k, mat in self._W.items():
            self._W[k] = np.pad(mat, ((0, pad), (0, pad)))
        for k, vec in self._b.items():
            self._b[k] = np.concatenate([vec, np.full(pad, prior)])
        for store in (self._prev, self._ema):
            for k, vec in store.items():
                store[k] = np.pad(vec, (0, pad))
        super()._grow(num_experts)

    def _prior_logit(self, n: int) -> float:
        p = min(max(self.top_k / max(n, 1), 1e-3), 0.5)
        return math.log(p / (1.0 - p))

    def _features(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        n = self.num_experts
        prev = self._prev.get(layer)
        ema = self._ema.get(layer)
        return (
            prev if prev is not None else np.zeros(n),
            ema if ema is not None else np.zeros(n),
        )

    def _logit(self, slot, src, layer) -> np.ndarray:
        n = self.num_experts
        W = self._W.get(slot)
        if W is None:
            W = self._W[slot] = np.zeros((n, n))
            self._b[slot] = np.full(n, self._prior_logit(n))
            self._wf[slot] = np.zeros(2)
        prev, ema = self._features(layer)
        wf = self._wf[slot]
        return self._b[slot] + W[src].sum(axis=0) + wf[0] * prev + wf[1] * ema

    def _learn(self, layer: int, keys: List[Key]) -> None:
        dst = np.fromiter((e for _, e in keys), dtype=np.intp, count=len(keys))
        n = self.num_experts
        y = np.zeros(n)
        y[dst] = 1.0
        lr, l2 = self.lr, self.l2
        prev, ema = self._features(layer)
        for d, (src_layer, src) in enumerate(reversed(self._history), start=1):
            slot = (src_layer, d)
            logit = self._logit(slot, src, layer)
            grad = y - 1.0 / (1.0 + np.exp(-logit))
            W = self._W[slot]
            if l2:
                W[src] *= 1.0 - lr * l2
            W[src] += lr * grad
            self._b[slot] += lr * grad
            self._wf[slot] += lr * np.array([grad @ prev / n, grad @ ema / n])
        # Update the target-layer features after learning (causal order).
        p = self._prev.get(layer)
        if p is None:
            p = self._prev[layer] = np.zeros(n)
        p[:] = 0.0
        p[dst] = 1.0
        e = self._ema.get(layer)
        if e is None:
            e = self._ema[layer] = np.zeros(n)
        e *= 1.0 - self.alpha
        e[dst] += self.alpha
        self._history.append((layer, dst))
        self._cache = {}

    def scores(self, layer: int, d: int) -> Optional[np.ndarray]:
        if d in self._cache:
            return self._cache[d]
        out = None
        if self._history:
            src_layer, src = self._history[-1]
            if src.size:
                logit = self._logit((src_layer, d), src, layer)
                out = 1.0 / (1.0 + np.exp(-logit))
        self._cache[d] = out
        return out

    def _predict(self, layer: int, d: int) -> List[Candidate]:
        return _top(layer, self.scores(layer, d), self.budget)


class KNNPrefetcher(PrefetchPolicy):
    """Nearest-neighbour routing trajectories (token-level EAM matching).

    For every layer ``l`` a ring buffer keeps the last ``bank`` tokens'
    routing at layers ``l - context + 1 .. l`` (the signature) and, once
    known, at ``l + 1 .. l + horizon`` (the targets). After a visit of ``l``
    the current token's signature is matched against the bank by overlap
    (shared (layer, expert) pairs); the ``neighbors`` best matches vote, with
    weight ``overlap ** power``, for each expert of the target visit. The
    score is the weighted fraction of neighbours that routed the expert
    (MoE-Infinity's activation-matrix matching, arXiv:2401.14361, at token
    rather than request granularity). Signatures are stored transposed
    (feature x bank) so a query sums ``8 x context`` contiguous rows.
    """

    name = "knn"

    def __init__(
        self,
        horizon=1,
        budget=8,
        num_layers=None,
        num_experts=None,
        bank: int = 4096,
        neighbors: int = 32,
        context: int = 2,
        power: float = 2.0,
        top_k: int = 8,
        **kw,
    ):
        super().__init__(horizon, budget, num_layers, num_experts, **kw)
        if bank < 1 or neighbors < 1 or context < 1:
            raise ValueError("bank, neighbors and context must be >= 1")
        self.bank = bank
        self.neighbors = neighbors
        self.context = context
        self.power = power
        self.width = top_k
        # Per anchor layer: signatures (features x bank), targets
        # (bank x horizon x width), fill count and ring head.
        self._sig: Dict[int, np.ndarray] = {}
        self._tgt: Dict[int, np.ndarray] = {}
        self._fill: Dict[int, int] = {}
        self._head: Dict[int, int] = {}
        self._rows: Dict[int, List[Optional[np.ndarray]]] = {}
        self._recent: Deque[Tuple[int, np.ndarray]] = deque(maxlen=context)
        self._pending: Deque[Tuple[int, int]] = deque(maxlen=horizon)
        self._query: Optional[np.ndarray] = None
        self._query_layer: Optional[int] = None
        self._cache: Dict[int, Optional[np.ndarray]] = {}

    def _grow(self, num_experts: int) -> None:
        old = self.num_experts
        for layer, sig in self._sig.items():
            rows = sig.reshape(self.context, old, sig.shape[1])
            rows = np.pad(rows, ((0, 0), (0, num_experts - old), (0, 0)))
            self._sig[layer] = rows.reshape(-1, sig.shape[1])
        for rows in self._rows.values():
            for i, q in enumerate(rows):
                if q is not None:
                    j, e = np.divmod(q, old)
                    rows[i] = j * num_experts + e
        super()._grow(num_experts)

    def _slot(self, layer: int, query: Optional[np.ndarray]) -> int:
        sig = self._sig.get(layer)
        if sig is None:
            sig = self._sig[layer] = np.zeros(
                (self.context * self.num_experts, self.bank), dtype=np.uint8
            )
            self._tgt[layer] = np.full(
                (self.bank, self.horizon, self.width), -1, dtype=np.int16
            )
            self._fill[layer] = 0
            self._head[layer] = 0
            self._rows[layer] = [None] * self.bank
        head = self._head[layer]
        self._head[layer] = (head + 1) % self.bank
        self._fill[layer] = min(self._fill[layer] + 1, self.bank)
        old = self._rows[layer][head]
        if old is not None:
            sig[old, head] = 0
        if query is not None:
            sig[query, head] = 1
        self._rows[layer][head] = query
        self._tgt[layer][head] = -1
        return head

    def _learn(self, layer: int, keys: List[Key]) -> None:
        experts = np.fromiter(
            (e for _, e in keys), dtype=np.intp, count=len(keys)
        )
        # Fill the targets of the anchors d visits back.
        for d, (anchor_layer, row) in enumerate(reversed(self._pending), 1):
            tgt = self._tgt[anchor_layer][row, d - 1]
            k = min(experts.size, self.width)
            tgt[:k] = experts[:k]
        self._recent.append((layer, experts))
        n = self.num_experts
        query = [ex + j * n for j, (_, ex) in enumerate(reversed(self._recent))]
        self._query = np.concatenate(query) if query else None
        self._query_layer = layer
        row = self._slot(layer, self._query)
        self._pending.append((layer, row))
        self._cache = {}

    def scores(self, layer: int, d: int) -> Optional[np.ndarray]:
        if d in self._cache:
            return self._cache[d]
        out = None
        anchor = self._query_layer
        if anchor is not None and self._query is not None:
            fill = self._fill.get(anchor, 0)
            tgt = self._tgt[anchor][:fill, d - 1]
            known = tgt[:, 0] >= 0
            if known.any():
                sig = self._sig[anchor]
                sim = sig[self._query, :fill].sum(axis=0, dtype=np.int32)
                sim[~known] = -1
                k = min(self.neighbors, int(known.sum()))
                idx = np.argpartition(-sim, k - 1)[:k]
                idx = idx[sim[idx] > 0]
                if idx.size:
                    w = sim[idx].astype(np.float64) ** self.power
                    votes = tgt[idx]
                    mask = votes >= 0
                    out = (
                        np.bincount(
                            votes[mask],
                            weights=np.repeat(w, mask.sum(axis=1)),
                            minlength=self.num_experts,
                        )
                        / w.sum()
                    )
        self._cache[d] = out
        return out

    def _predict(self, layer: int, d: int) -> List[Candidate]:
        return _top(layer, self.scores(layer, d), self.budget)


class SameIdPrefetcher(PrefetchPolicy):
    """The engine's ``speculative_prefetch``: same expert ids, next layer.

    MoE-Infinity's speculative path takes the top-k experts of layer ``L``'s
    router logits and prefetches the *same expert ids* at layer ``L + 1``
    (``moe_infinity/memory/expert_prefetcher.py``; engine default top-2). The
    routed order of the trace is the gate's top-k order.
    """

    name = "same_id"

    def __init__(
        self, horizon=1, budget=2, num_layers=None, num_experts=None, **kw
    ):
        super().__init__(horizon, budget, num_layers, num_experts, **kw)
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

    def __init__(
        self, horizon=1, budget=8, num_layers=None, num_experts=None, **kw
    ):
        super().__init__(horizon, budget, num_layers, num_experts, **kw)
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
    "logistic": LogisticPrefetcher,
    "knn": KNNPrefetcher,
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

    ``issue_budget`` caps the prefetches issued per target visit; it
    defaults to the prefetcher's ``budget`` (the number of top predictions
    to keep resident). A smaller cap over a deeper ranking is a *transfer*
    budget: the best non-resident candidates, as many as the link can move.
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
        issue_budget: Optional[int] = None,
    ):
        if issue_budget is not None and issue_budget < 1:
            raise ValueError("issue_budget must be >= 1")
        self.issue_budget = issue_budget
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
        budget = self.issue_budget or self.prefetcher.budget
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
            if not ev.admits(key):
                return  # the evictor serves it from a transient slot
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
                if key not in ev.resident:  # the evictor declined it
                    del self._spec[key]
                    self._unissue_last()
                    return False
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

    def _unissue_last(self) -> None:
        """Forget the prefetch id just created (it was never admitted)."""
        for arr in (
            self.pf_key,
            self.pf_issue,
            self.pf_target,
            self.pf_dist,
            self.pf_rank,
            self.pf_used,
            self.pf_removed,
        ):
            arr.pop()
        self.prefetch_issued -= 1

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
    top_k: Optional[int] = None,
    evictor_kwargs: Optional[Dict[str, object]] = None,
    issue_budget: Optional[int] = None,
) -> PrefetchCachePolicy:
    """``PrefetchCachePolicy`` over ``build_policy(evictor_name, ...)``.

    ``capacity`` is the TOTAL number of expert slots; in ``buffer`` mode the
    evictor gets ``capacity - buffer_slots`` of them. LCNU variants also get
    ``prior`` and ``decode_width = top_k`` (visits routing more experts are
    prefill passes); ``evictor_kwargs`` override any of these.
    """
    from benchmarks.cache_simulator.policies import (
        GEOMETRY_POLICIES,
        LCNU_POLICIES,
        build_policy,
    )

    kwargs: Dict[str, object] = {}
    if evictor_name in GEOMETRY_POLICIES and num_layers is not None:
        kwargs["num_layers"] = num_layers
    if evictor_name in LCNU_POLICIES:
        if prior is not None:
            kwargs["prior"] = prior
        if top_k is not None:
            kwargs["decode_width"] = top_k
    kwargs.update(evictor_kwargs or {})
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
        issue_budget=issue_budget,
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
