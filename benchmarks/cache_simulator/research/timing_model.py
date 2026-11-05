"""Serial host-to-GPU transfer-timing model for the expert-cache simulator.

Hit rate alone cannot rank prefetchers: a prefetch uses link bandwidth even
when it is useless, and one that arrives late still stalls compute. This
module replays the per-layer-visit event log of a
``prefetchers.PrefetchCachePolicy`` (demand misses, prefetch issues, uses and
removals) against one serial link:

* Batch-1 decode runs one layer *visit* after another. Visit ``v`` starts at
  ``T_v``; its routing (gate) is known at ``T_v``, its demand misses are
  queued then, compute waits until every expert of the visit is resident and
  then takes ``layer_compute_s``: ``T_{v+1} = ready_v + layer_compute_s``.
  ``stall_v = ready_v - T_v``.
* Transfers are served one at a time, ``bytes_per_expert / link_bytes_per_s``
  seconds each, never preempted once started.
* Prefetches decided after visit ``v``'s gate are queued at ``T_v`` behind
  ``v``'s demand misses. With ``demand_priority`` (the default; the engine's
  task pool serves on-demand fetches first) a later demand miss overtakes
  queued prefetches, and queued prefetches are served earliest target visit
  first; with FIFO every transfer is served in issue order.
* A prefetched expert used before its transfer completes stalls only for the
  rest of it (late); one still queued at its use becomes a demand transfer
  (promoted). With ``cancel_unstarted`` a queued prefetch whose expert was
  evicted or dropped before the transfer would start is cancelled and costs
  nothing; one already started is a wasted transfer.

The cache state itself comes from the cache simulation (slots are taken when a
prefetch is issued); the timing replay only decides when transfers finish.

All quantities are parameters (``LinkModel``). The presets document their
arithmetic and sources:

Expert bytes (3 projection matrices ``hidden x moe_intermediate``), from each
model's ``config.json``:

* Qwen3-30B-A3B (https://huggingface.co/Qwen/Qwen3-30B-A3B): hidden 2048,
  moe_intermediate 768, BF16: ``3 * 2048 * 768 * 2 B = 9,437,184 B``.
* Qwen3-235B-A22B-Instruct-2507-FP8
  (https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8): hidden
  4096, moe_intermediate 1536, FP8 e4m3 with 128x128 block scales:
  ``3 * 4096 * 1536 * 1 B = 18,874,368 B`` (+4,608 B of fp32 scales, ignored).

Link: PCIe 5.0 x16 is ``32 GT/s * 16 * 128/130 / 8 = 63.0 GB/s`` per direction
(PCI-SIG PCIe Base Specification 5.0). The default **57.1 GB/s** is measured on
the capture host (RTX PRO 6000 Blackwell, PCIe 5.0 x16, pinned host memory)
by ``research/measure_link.py``: 166 us per 9.44 MB expert, 330 us per
18.87 MB expert, with or without other copies queued.

Per-layer compute: batch-1 decode is memory-bound, so a layer whose experts
are resident takes about ``active_layer_bytes / hbm_bytes_per_s`` times
``compute_scale``. Active bytes are the attention projections (q, k, v, o)
plus the router plus ``top_k`` experts: 113.8 MB (Qwen3-30B-A3B, BF16; 32 q /
4 kv heads of 128) and 223.3 MB (Qwen3-235B-FP8; 64 q / 4 kv heads, BF16
router). ``hbm_bytes_per_s`` defaults to 1.792 TB/s (RTX PRO 6000 Blackwell,
512-bit GDDR7), giving 63.5 us and 124.6 us per layer at ``compute_scale=1``
(the roofline, i.e. the least overlap a prefetcher can get; real kernels at
batch 1 are slower, so ``compute_scale > 1`` is the more realistic setting).

Nothing here imports torch.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Key = Tuple[int, int]

#: Measured on the capture host (``research/measure_link.py``).
MEASURED_PCIE5_BYTES_PER_S = 57.1e9
#: RTX PRO 6000 Blackwell memory bandwidth (512-bit GDDR7, 28 Gbps).
RTX_PRO_6000_HBM_BYTES_PER_S = 1.792e12


def expert_bytes(
    hidden_size: int,
    moe_intermediate_size: int,
    bytes_per_param: float,
    num_matrices: int = 3,
) -> float:
    """Bytes of one routed expert: gate/up/down projection matrices."""
    return num_matrices * hidden_size * moe_intermediate_size * bytes_per_param


def attention_bytes(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_param: float,
) -> float:
    """Bytes of the q, k, v and o projections of one attention layer."""
    q = hidden_size * num_heads * head_dim
    kv = 2 * hidden_size * num_kv_heads * head_dim
    o = num_heads * head_dim * hidden_size
    return (q + kv + o) * bytes_per_param


@dataclass(frozen=True)
class LinkModel:
    """Transfer-timing parameters for one model on one host-to-GPU link."""

    bytes_per_expert: float
    link_bytes_per_s: float
    layer_compute_s: float
    name: str = "custom"

    def __post_init__(self):
        if self.bytes_per_expert < 0:
            raise ValueError("bytes_per_expert must be >= 0")
        if self.link_bytes_per_s <= 0:
            raise ValueError("link_bytes_per_s must be > 0")
        if self.layer_compute_s < 0:
            raise ValueError("layer_compute_s must be >= 0")

    @property
    def expert_transfer_s(self) -> float:
        """Seconds to move one expert over the link (0 if bandwidth is inf)."""
        if math.isinf(self.link_bytes_per_s):
            return 0.0
        return self.bytes_per_expert / self.link_bytes_per_s

    @property
    def overlap_ratio(self) -> float:
        """Experts the link moves during one layer's compute."""
        tx = self.expert_transfer_s
        return math.inf if tx == 0 else self.layer_compute_s / tx


def _preset(
    name: str,
    per_expert: float,
    layer_bytes: float,
    link_bytes_per_s: float,
    layer_compute_s: Optional[float],
    hbm_bytes_per_s: float,
    compute_scale: float,
) -> LinkModel:
    if layer_compute_s is None:
        layer_compute_s = compute_scale * layer_bytes / hbm_bytes_per_s
    return LinkModel(
        bytes_per_expert=per_expert,
        link_bytes_per_s=link_bytes_per_s,
        layer_compute_s=layer_compute_s,
        name=name,
    )


def qwen3_30b_a3b_link(
    link_bytes_per_s: float = MEASURED_PCIE5_BYTES_PER_S,
    layer_compute_s: Optional[float] = None,
    hbm_bytes_per_s: float = RTX_PRO_6000_HBM_BYTES_PER_S,
    compute_scale: float = 1.0,
) -> LinkModel:
    """Qwen3-30B-A3B (BF16) on PCIe 5.0 x16; see the module docstring."""
    per_expert = expert_bytes(2048, 768, 2.0)  # 9,437,184 B
    layer = (
        attention_bytes(2048, 32, 4, 128, 2.0)  # 37,748,736 B
        + 2048 * 128 * 2.0  # router
        + 8 * per_expert
    )  # 113,770,496 B
    return _preset(
        "qwen3-30b-a3b",
        per_expert,
        layer,
        link_bytes_per_s,
        layer_compute_s,
        hbm_bytes_per_s,
        compute_scale,
    )


def qwen3_235b_a22b_fp8_link(
    link_bytes_per_s: float = MEASURED_PCIE5_BYTES_PER_S,
    layer_compute_s: Optional[float] = None,
    hbm_bytes_per_s: float = RTX_PRO_6000_HBM_BYTES_PER_S,
    compute_scale: float = 1.0,
) -> LinkModel:
    """Qwen3-235B-A22B-FP8 on PCIe 5.0 x16; see the module docstring."""
    per_expert = expert_bytes(4096, 1536, 1.0)  # 18,874,368 B
    layer = (
        attention_bytes(4096, 64, 4, 128, 1.0)  # 71,303,168 B
        + 4096 * 128 * 2.0  # BF16 router (not quantized)
        + 8 * per_expert
    )  # 223,346,688 B
    return _preset(
        "qwen3-235b-a22b-fp8",
        per_expert,
        layer,
        link_bytes_per_s,
        layer_compute_s,
        hbm_bytes_per_s,
        compute_scale,
    )


LINK_PRESETS = {
    "qwen3-30b-a3b": qwen3_30b_a3b_link,
    "qwen3-235b-a22b-fp8": qwen3_235b_a22b_fp8_link,
}


@dataclass
class TimingReport:
    """Timing and transfer metrics of one replay."""

    stall_s: float
    tokens: float
    visits: int
    demand_transfers: int
    prefetch_transfers: int
    prefetch_cancelled: int
    prefetch_promoted: int
    covered_on_time: int
    covered_late: int
    compute_s: float

    @property
    def stall_per_token_s(self) -> float:
        return self.stall_s / self.tokens if self.tokens else 0.0

    @property
    def compute_per_token_s(self) -> float:
        return self.compute_s / self.tokens if self.tokens else 0.0

    @property
    def time_per_token_s(self) -> float:
        return self.stall_per_token_s + self.compute_per_token_s

    @property
    def total_transfers(self) -> int:
        return self.demand_transfers + self.prefetch_transfers

    @property
    def transfers_per_token(self) -> float:
        return self.total_transfers / self.tokens if self.tokens else 0.0

    @property
    def wasted_transfers(self) -> int:
        """Prefetch transfers that ran but whose expert was never used."""
        return (
            self.prefetch_transfers - self.covered_on_time - self.covered_late
        )

    def as_dict(self) -> Dict[str, float]:
        out = asdict(self)
        out.update(
            stall_per_token_s=self.stall_per_token_s,
            time_per_token_s=self.time_per_token_s,
            transfers_per_token=self.transfers_per_token,
            wasted_transfers=self.wasted_transfers,
        )
        return out


def replay(
    visit_fresh: Sequence[int],
    pf_issue: Sequence[int],
    pf_used: Sequence[int],
    pf_removed: Sequence[int],
    link: LinkModel,
    *,
    pf_target: Optional[Sequence[int]] = None,
    demand_priority: bool = True,
    cancel_unstarted: bool = True,
    num_layers: Optional[int] = None,
    tokens: Optional[float] = None,
) -> TimingReport:
    """Replay a cache simulation's event log under the serial-link model.

    ``visit_fresh[v]`` is the number of demand misses of visit ``v`` that no
    prefetch covered; prefetch ``i`` was issued after visit ``pf_issue[i]``
    (non-decreasing) for target visit ``pf_target[i]``, first used at visit
    ``pf_used[i]`` (or ``-1``) and removed unused at visit ``pf_removed[i]``
    (or ``-1``). Without ``pf_target`` queued prefetches are served in issue
    order. ``tokens`` defaults to ``visits / num_layers``.
    """
    tx = link.expert_transfer_s
    comp = link.layer_compute_s
    visits = len(visit_fresh)
    count = len(pf_issue)
    used_at: Dict[int, List[int]] = {}
    for pid, when in enumerate(pf_used):
        if when >= 0:
            used_at.setdefault(when, []).append(pid)
    start_of = [0.0] * visits
    issue_time = [0.0] * count
    done = [0.0] * count
    state = bytearray(count)  # 0 queued, 1 started, 2 cancelled, 3 promoted
    edf = demand_priority and pf_target is not None
    queue: deque = deque()
    heap: List[Tuple[int, int]] = []
    link_free = now = stall = 0.0
    nxt = 0
    n_demand = n_pf = n_cancel = n_promoted = n_on_time = n_late = 0

    for v in range(visits):
        start_of[v] = now
        # Serve queued prefetches; with demand priority only those that start
        # before this visit's gate (earliest target first), with FIFO all of
        # them in issue order (they are ahead of this visit's misses).
        while heap or queue:
            if edf:
                pid = heap[0][1]
            else:
                pid = queue[0]
            if state[pid]:
                if edf:
                    heapq.heappop(heap)
                else:
                    queue.popleft()
                continue
            t0 = link_free if link_free > issue_time[pid] else issue_time[pid]
            if demand_priority and t0 >= now:
                break
            if edf:
                heapq.heappop(heap)
            else:
                queue.popleft()
            gone = pf_removed[pid]
            if cancel_unstarted and 0 <= gone <= v and start_of[gone] <= t0:
                state[pid] = 2
                n_cancel += 1
                continue
            state[pid] = 1
            link_free = done[pid] = t0 + tx
            n_pf += 1
        ready = now
        fresh = visit_fresh[v]
        for pid in used_at.get(v, ()):
            if state[pid] == 1:
                if done[pid] > now:
                    n_late += 1
                    if done[pid] > ready:
                        ready = done[pid]
                else:
                    n_on_time += 1
            else:  # still queued at its use: fetch it as a demand miss
                state[pid] = 3
                fresh += 1
                n_promoted += 1
        if fresh:
            t0 = link_free if link_free > now else now
            link_free = t0 + fresh * tx
            if link_free > ready:
                ready = link_free
            n_demand += fresh
        stall += ready - now
        while nxt < count and pf_issue[nxt] == v:
            issue_time[nxt] = now
            if edf:
                heapq.heappush(heap, (pf_target[nxt], nxt))
            else:
                queue.append(nxt)
            nxt += 1
        now = ready + comp

    layers = num_layers or 1
    if tokens is None:
        tokens = visits / layers
    return TimingReport(
        stall_s=stall,
        tokens=tokens,
        visits=visits,
        demand_transfers=n_demand,
        prefetch_transfers=n_pf,
        prefetch_cancelled=n_cancel,
        prefetch_promoted=n_promoted,
        covered_on_time=n_on_time,
        covered_late=n_late,
        compute_s=visits * comp,
    )


def replay_policy(policy, link: LinkModel, **kwargs) -> TimingReport:
    """``replay`` over the log of a simulated ``PrefetchCachePolicy``."""
    return replay(
        policy.visit_fresh,
        policy.pf_issue,
        policy.pf_used,
        policy.pf_removed,
        link,
        pf_target=policy.pf_target,
        **kwargs,
    )


def simulate_timing(
    policy,
    accesses: Sequence[Key],
    link: LinkModel,
    num_layers: Optional[int] = None,
    **kwargs,
) -> TimingReport:
    """Simulate ``policy`` over ``accesses`` and replay it under ``link``.

    A plain ``CachePolicy`` is wrapped in ``PrefetchCachePolicy(policy,
    None)`` (bit-identical hits) to obtain its per-visit miss log.
    """
    from benchmarks.cache_simulator.policies import simulate
    from benchmarks.cache_simulator.prefetchers import PrefetchCachePolicy

    if not isinstance(policy, PrefetchCachePolicy):
        policy = PrefetchCachePolicy(policy, None)
    simulate(policy, accesses)
    if num_layers is None:
        num_layers = max((layer for layer, _ in accesses), default=0) + 1
    return replay_policy(policy, link, num_layers=num_layers, **kwargs)


def stall_floor_s(belady_misses: int, visits: int, link: LinkModel) -> float:
    """Lower bound on total stall for ANY policy at this capacity.

    Belady's MIN minimizes the number of fetches, so every policy moves at
    least ``belady_misses`` experts over the link, and the run lasts at least
    that long; compute fills ``visits * layer_compute_s`` of it.
    """
    return max(
        0.0,
        belady_misses * link.expert_transfer_s - visits * link.layer_compute_s,
    )
