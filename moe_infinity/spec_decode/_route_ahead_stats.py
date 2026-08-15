"""Opt-in coverage/waste metrics for DFlash route-ahead prefetch (Track A5).

Read-only observer for the route-ahead seam (``.sisyphus/plans/
dflash-deferred-tracks-plan.md``, Track A A0 item 4). While the DFlash verify
context is active, the executor route-ahead seam
(``distributed/expert_executor.py`` ``_maybe_route_ahead_prefetch``) reports
each dispatched MoE layer's predicted (pinned/prefetched) expert set and its
routed mask; once the accept rule has fixed the kept prefix,
``DFlashSpeculator.generate`` calls ``commit_step`` to finalize the step:

* coverage: ``sum |P_l ∩ A_l| / sum |A_l|`` -- the fraction of experts the
  verify actually read (``A_l`` = union of the layer's router mask, exactly
  the set ``dispatch_local`` enqueues) that the route-ahead prefetch covered
  (``P_l``). Vacuous layers (``A_l`` empty) contribute nothing; the ratio
  defaults to 1.0 when nothing was ever routed, matching the A1
  ``prefetch_coverage`` empty-actual convention.
* rejected-token waste: ``sum |P_l \\ U_keep_l|`` -- prefetched experts the
  kept (accepted) prefix never routed to, i.e. fetched for draft tokens that
  verification rolled back. Computed with the A1 ``rejected_expert_ids`` on
  the full PREFETCHED union vs. the kept-prefix union; nothing prefetched
  means nothing wasted.

The observer NEVER touches routing, prefetch decisions, or model outputs:
``observe_layer``/``commit_step`` only read and accumulate. Default-off: a
``None`` stats handle short-circuits both ends (executor seam and
speculator), so uninstrumented runs pay exactly zero extra work. Enable via
``DFlashSpeculator.enable_route_ahead_stats()`` and read the counters back
after ``generate()``.
"""

from __future__ import annotations

from typing import (
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from moe_infinity.spec_decode._prefetch_route import (
    rejected_expert_ids,
    union_experts_from_mask,
)


class RouteAheadStepSummary(NamedTuple):
    """One committed verify step's accounting (all in expert-count units)."""

    layers: int  # executor-backed MoE layers observed this step
    predicted: int  # sum |P_l| -- experts the route-ahead prefetch pinned
    actual: int  # sum |A_l| -- experts the verify dispatch routed to
    covered: int  # sum |P_l ∩ A_l|
    kept: int  # sum |U_keep_l| -- union over only the kept prefix rows
    wasted: int  # sum |P_l \ U_keep_l| -- rejected-token prefetch waste
    # Byte-accurate counterparts, scored over the PREFETCHED set only; None
    # when the caller supplied no payload sizes (mock / resident paths).
    predicted_bytes: Optional[int] = None  # stored bytes of P_l
    kept_bytes: Optional[int] = None  # bytes of P_l the kept prefix still used
    wasted_bytes: Optional[int] = None  # bytes of P_l \ U_keep_l (waste)

    @property
    def coverage(self) -> float:
        """Per-step ``sum |P_l ∩ A_l| / sum |A_l|`` (1.0 when nothing routed)."""
        if self.actual == 0:
            return 1.0
        return self.covered / self.actual


class RouteAheadStats:
    """Accumulates per-verify-step route-ahead coverage/waste counters.

    Lifecycle per verify step (driven by ``DFlashSpeculator``):
    ``begin_step`` at verify-forward entry, one ``observe_layer`` per
    executor-backed MoE dispatch (from the route-ahead seam), then
    ``commit_step(kept_rows)`` after the accept rule fixes the kept prefix.
    Records from a step that aborts before commit are dropped by the next
    ``begin_step`` -- they are never committed, so a failed verify cannot
    corrupt the counters. All mutating methods are called only from the
    speculator/executor plumbing; post-``generate()`` this object is a
    plain readout.
    """

    def __init__(self) -> None:
        self.steps: int = 0
        self.layers_observed: int = 0
        self.predicted_experts: int = 0
        self.actual_experts: int = 0
        self.covered_experts: int = 0
        self.kept_experts: int = 0
        self.wasted_experts: int = 0
        self.predicted_prefetch_bytes: int = 0
        self.kept_prefetch_bytes: int = 0
        self.wasted_prefetch_bytes: int = 0
        self._bytes_seen: bool = False
        self._pending: List[
            Tuple[int, List[int], torch.Tensor, Optional[Dict[int, int]]]
        ] = []

    # ------------------------------------------------------------------
    # recorder interface (speculator + executor seam drive these)
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        """Start a verify step; any uncommitted prior records are dropped."""
        self._pending.clear()

    def observe_layer(
        self,
        layer_id: int,
        predicted_ids: Sequence[int],
        router_mask: Union[torch.Tensor, Sequence[Sequence[int]]],
        expert_nbytes: Optional[Mapping[int, int]] = None,
    ) -> None:
        """Record one dispatched layer of the in-flight verify step.

        ``predicted_ids`` is the expert set the route-ahead pin/prefetch
        covered for this layer (``[]`` when the seam did not fire, e.g. no
        prefetcher bound); ``router_mask`` is the layer's OWN dispatch mask
        ``[num_tokens, num_experts]`` whose column-wise any is the actual
        verify-read union. The mask is snapshotted to CPU (a no-op view when
        already on CPU) so the kept-prefix waste can be computed later, once
        the accept length is known. Read-only: the mask is never modified.

        ``expert_nbytes`` optionally maps each prefetched expert id to its
        exact stored payload bytes; when given, ``commit_step`` reports
        byte-accurate predicted/kept/wasted alongside the counts. ``None``
        (mocks, resident-expert runs) keeps every byte field ``None`` -- the
        recorder never fabricates an average expert size.
        """
        mask = (
            router_mask
            if torch.is_tensor(router_mask)
            else torch.tensor(router_mask)
        )
        if mask.dim() != 2:
            raise ValueError(
                "router_mask must be 2-D [num_tokens, num_experts]; "
                f"got shape {tuple(mask.shape)}"
            )
        mask_cpu = mask.detach().to(torch.bool).cpu()
        nbytes = (
            {int(e): int(n) for e, n in expert_nbytes.items()}
            if expert_nbytes is not None
            else None
        )
        self._pending.append(
            (int(layer_id), [int(e) for e in predicted_ids], mask_cpu, nbytes)
        )

    def commit_step(self, kept_rows: int) -> RouteAheadStepSummary:
        """Finalize the in-flight step: coverage + rejected-token waste.

        ``kept_rows`` is the number of leading block rows whose KV survived
        verification (``cache_committed`` in ``dflash.generate`` -- the
        anchor plus the accepted drafts). The kept-prefix union uses exactly
        those rows of each recorded mask; prefetched experts outside it were
        fetched for tokens that got rolled back. Steps with no observed
        layers (bare-HF targets, or the seam never fired) leave the counters
        untouched and return a zero summary.
        """
        pending = self._pending
        self._pending = []
        if not pending:
            return RouteAheadStepSummary(0, 0, 0, 0, 0, 0)

        predicted = actual = covered = kept = wasted = 0
        predicted_b = kept_b = wasted_b = 0
        step_has_bytes = False
        for _layer_id, predicted_ids, mask, nbytes in pending:
            full_union = union_experts_from_mask(mask)
            rows = max(0, min(int(kept_rows), int(mask.shape[0])))
            kept_union = union_experts_from_mask(mask[:rows]) if rows else []
            predicted_set = set(predicted_ids)
            kept_set = set(kept_union)
            predicted += len(predicted_set)
            actual += len(full_union)
            # Same set semantics as the A1 ``prefetch_coverage``; the count
            # form is what the A0 section 4 ratio-of-sums aggregates.
            covered += len(predicted_set & set(full_union))
            kept += len(kept_union)
            wasted += len(rejected_expert_ids(predicted_ids, kept_union))
            if nbytes is not None:
                step_has_bytes = True
                predicted_b += sum(nbytes.get(e, 0) for e in predicted_set)
                kept_b += sum(
                    nbytes.get(e, 0) for e in predicted_set & kept_set
                )
                wasted_b += sum(
                    nbytes.get(e, 0) for e in predicted_set - kept_set
                )

        self.steps += 1
        self.layers_observed += len(pending)
        self.predicted_experts += predicted
        self.actual_experts += actual
        self.covered_experts += covered
        self.kept_experts += kept
        self.wasted_experts += wasted
        if step_has_bytes:
            self._bytes_seen = True
            self.predicted_prefetch_bytes += predicted_b
            self.kept_prefetch_bytes += kept_b
            self.wasted_prefetch_bytes += wasted_b
        return RouteAheadStepSummary(
            len(pending),
            predicted,
            actual,
            covered,
            kept,
            wasted,
            predicted_b if step_has_bytes else None,
            kept_b if step_has_bytes else None,
            wasted_b if step_has_bytes else None,
        )

    # ------------------------------------------------------------------
    # readout
    # ------------------------------------------------------------------

    @property
    def coverage(self) -> float:
        """``sum |P_l ∩ A_l| / sum |A_l|`` over all committed steps (A0 item 4).

        1.0 when nothing was ever routed (nothing to cover), mirroring the
        A1 ``prefetch_coverage`` empty-actual convention.
        """
        if self.actual_experts == 0:
            return 1.0
        return self.covered_experts / self.actual_experts

    @property
    def waste_ratio(self) -> float:
        """``sum |P_l \\ U_keep_l| / sum |P_l|`` -- share of the prefetched
        experts that rejected draft tokens made useless (0.0 when nothing
        was ever prefetched)."""
        if self.predicted_experts == 0:
            return 0.0
        return self.wasted_experts / self.predicted_experts

    def reset(self) -> None:
        """Zero all counters and drop any uncommitted records."""
        self.__init__()

    def as_dict(self) -> Dict[str, Union[int, float, None]]:
        """Flat snapshot of the counters, byte totals, and derived ratios.

        The three ``*_prefetch_bytes`` entries are ``None`` until a step is
        committed with per-expert payload sizes, so uninstrumented and
        resident runs report byte-accurate absence rather than a fake zero.
        """
        return {
            "steps": self.steps,
            "layers_observed": self.layers_observed,
            "predicted_experts": self.predicted_experts,
            "actual_experts": self.actual_experts,
            "covered_experts": self.covered_experts,
            "kept_experts": self.kept_experts,
            "wasted_experts": self.wasted_experts,
            "coverage": self.coverage,
            "waste_ratio": self.waste_ratio,
            "predicted_prefetch_bytes": (
                self.predicted_prefetch_bytes if self._bytes_seen else None
            ),
            "kept_prefetch_bytes": (
                self.kept_prefetch_bytes if self._bytes_seen else None
            ),
            "wasted_prefetch_bytes": (
                self.wasted_prefetch_bytes if self._bytes_seen else None
            ),
        }


__all__ = [
    "RouteAheadStats",
    "RouteAheadStepSummary",
]
