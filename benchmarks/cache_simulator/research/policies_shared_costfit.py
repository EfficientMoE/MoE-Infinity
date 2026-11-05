"""SharedState CostFit / CostFit3: parity-preserving structural metadata sharing.

CostFit-family arbiters run N candidate caches over the SAME access stream, so
the N candidates each keep their own per-key dicts (LCNU: ``_last``/``_ema``/
``_count``/``_first``; S3-FIFO: ``freq``) over the same key universe. Every
dict pays its own hash-table slot per key. Here the per-key state of ALL
candidates lives in ONE ``Key -> record`` table (one ``__slots__`` record per
key, one slot per (cache role, field)), so the table overhead is paid once.

Parity is absolute by construction: the sealed policy classes are SUBCLASSED
and only their per-key dict attributes are swapped for dict-like views over the
shared table. No inner logic is reimplemented; order structures (S3-FIFO
small/main/ghost, ``resident``, ``_by_layer``) stay real containers.

The views implement exactly the operations the sealed inners perform:

* LCNU ``_last/_ema/_count/_first``: ``.get``, ``[]``, ``[]=`` (never delete).
* S3-FIFO ``freq``: ``[]``, ``[]=``, ``del``.

Iteration / ``len`` are deliberately NOT provided (no sealed caller uses them).
Absent state is an UNSET slot (no sentinel object); reading it raises KeyError,
matching ``dict`` semantics. A record is dropped once all its slots are unset.
"""

from typing import Dict, Optional

from ..policies import Key, LayerCycleNextUsePolicy, S3FIFOPolicy
from .policies_costfit import CostFitPolicy
from .policies_costfit3 import CostFit3Policy
from .policies_rothead import RotHeadPolicy


class SharedKeyTable:
    """One ``Key -> record`` dict holding every registered (role, field) slot."""

    def __init__(self) -> None:
        self.table: Dict[Key, object] = {}
        self._names: list = []
        self._cls = None

    def alloc(self, role: str, field: str) -> str:
        """Register a slot; only legal before the first record is created."""
        if self._cls is not None:
            raise RuntimeError("cannot allocate slots after records exist")
        name = f"s{len(self._names)}"  # slot attr; label kept in self.labels
        self._names.append((name, f"{role}.{field}"))
        return name

    @property
    def labels(self) -> list:
        return [label for _, label in self._names]

    def new_record(self):
        if self._cls is None:
            slots = tuple(n for n, _ in self._names) + ("live",)
            self._cls = type("SharedRecord", (), {"__slots__": slots})
        rec = self._cls()
        rec.live = 0
        return rec


class SharedFieldView:
    """dict-like facade over one slot of a ``SharedKeyTable``."""

    __slots__ = ("_t", "_tab", "_name")

    def __init__(self, table: SharedKeyTable, role: str, field: str) -> None:
        self._t = table
        self._tab = table.table
        self._name = table.alloc(role, field)

    def __getitem__(self, key):
        try:
            return getattr(self._tab[key], self._name)
        except (KeyError, AttributeError):
            raise KeyError(key) from None

    def get(self, key, default=None):
        rec = self._tab.get(key)
        if rec is None:
            return default
        return getattr(rec, self._name, default)

    def __contains__(self, key) -> bool:
        rec = self._tab.get(key)
        return rec is not None and hasattr(rec, self._name)

    def __setitem__(self, key, value) -> None:
        rec = self._tab.get(key)
        if rec is None:
            rec = self._tab[key] = self._t.new_record()
        name = self._name
        try:
            getattr(rec, name)
        except AttributeError:
            rec.live += 1
        setattr(rec, name, value)

    def __delitem__(self, key) -> None:
        rec = self._tab.get(key)
        if rec is None or not hasattr(rec, self._name):
            raise KeyError(key)
        delattr(rec, self._name)
        rec.live -= 1
        if rec.live == 0:
            del self._tab[key]


class SharedLCNU(LayerCycleNextUsePolicy):
    def __init__(
        self,
        capacity: int,
        *args,
        table: Optional[SharedKeyTable] = None,
        role: str = "lcnu",
        **kw,
    ):
        super().__init__(capacity, *args, **kw)
        self.shared_table = table if table is not None else SharedKeyTable()
        t = self.shared_table
        self._last = SharedFieldView(t, role, "last")
        self._ema = SharedFieldView(t, role, "ema")
        self._count = SharedFieldView(t, role, "count")
        self._first = SharedFieldView(t, role, "first")


class SharedS3FIFO(S3FIFOPolicy):
    def __init__(
        self,
        capacity: int,
        *args,
        table: Optional[SharedKeyTable] = None,
        role: str = "s3fifo",
        **kw,
    ):
        super().__init__(capacity, *args, **kw)
        self.shared_table = table if table is not None else SharedKeyTable()
        self.freq = SharedFieldView(self.shared_table, role, "freq")


class SharedRotHead(RotHeadPolicy):
    def __init__(
        self,
        capacity: int,
        num_layers: int = 1,
        prior: float = 0.05,
        table: Optional[SharedKeyTable] = None,
        role: str = "rothead",
    ):
        super().__init__(capacity, num_layers=num_layers, prior=prior)
        self.shared_table = table if table is not None else SharedKeyTable()
        t = self.shared_table
        self.lcnu = SharedLCNU(
            capacity,
            num_layers=num_layers,
            prior=prior,
            table=t,
            role=f"{role}.lcnu",
        )
        self.s3fifo = SharedS3FIFO(capacity, table=t, role=f"{role}.s3fifo")


class SharedCostFitPolicy(CostFitPolicy):
    """CostFit with both inners sharing ONE key table."""

    def __init__(
        self,
        capacity: int,
        num_layers: int = 1,
        prior: float = 1.0 / 16,
        table: Optional[SharedKeyTable] = None,
    ):
        super().__init__(capacity, num_layers=num_layers, prior=prior)
        self.shared_table = table if table is not None else SharedKeyTable()
        t = self.shared_table
        self._inners = {
            "lcnu": SharedLCNU(
                capacity,
                num_layers=num_layers,
                prior=prior,
                table=t,
                role="costfit.lcnu",
            ),
            "s3fifo": SharedS3FIFO(capacity, table=t, role="costfit.s3fifo"),
        }


class SharedCostFit3Policy(CostFit3Policy):
    """CostFit3 with all three inners (and RotHead's two) sharing ONE table."""

    def __init__(
        self,
        capacity: int,
        num_layers: int = 1,
        prior: float = 1.0 / 16,
        table: Optional[SharedKeyTable] = None,
    ):
        super().__init__(capacity, num_layers=num_layers, prior=prior)
        self.shared_table = table if table is not None else SharedKeyTable()
        t = self.shared_table
        # insertion order identical to the parent: lcnu, s3fifo, rothead
        self._inners = {
            "lcnu": SharedLCNU(
                capacity,
                num_layers=num_layers,
                prior=prior,
                table=t,
                role="costfit3.lcnu",
            ),
            "s3fifo": SharedS3FIFO(capacity, table=t, role="costfit3.s3fifo"),
            "rothead": SharedRotHead(
                capacity,
                num_layers=num_layers,
                prior=prior,
                table=t,
                role="costfit3.rothead",
            ),
        }
