from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Generic, Optional, TypeVar

from typing_extensions import override

K = TypeVar("K")
V = TypeVar("V")


class CachePolicy(ABC, Generic[K, V]):
    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """Return value if key in cache (moves to MRU), else None."""

    @abstractmethod
    def put(self, key: K, value: V) -> None:
        """Insert key-value; evict LRU if at capacity."""

    @abstractmethod
    def evict(self) -> Optional[tuple[K, V]]:
        """Forcibly evict one item. Returns (key, value) or None."""

    @property
    @abstractmethod
    def capacity(self) -> int: ...

    @abstractmethod
    def __len__(self) -> int: ...


class LRUPolicy(CachePolicy[K, V]):
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity: int = capacity
        self._data: OrderedDict[K, V] = OrderedDict()

    @property
    @override
    def capacity(self) -> int:
        return self._capacity

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def get(self, key: K) -> Optional[V]:
        if key not in self._data:
            return None
        self._data.move_to_end(key, last=True)
        return self._data[key]

    @override
    def put(self, key: K, value: V) -> None:
        if key in self._data:
            self._data[key] = value
            self._data.move_to_end(key, last=True)
            return

        if len(self._data) >= self._capacity:
            _ = self._data.popitem(last=False)
        self._data[key] = value

    @override
    def evict(self) -> Optional[tuple[K, V]]:
        if not self._data:
            return None
        return self._data.popitem(last=False)


class ARCPolicy(CachePolicy[K, V]):
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self._capacity: int = capacity
        self._p: int = 0

        self._t1: OrderedDict[K, V] = OrderedDict()
        self._t2: OrderedDict[K, V] = OrderedDict()
        self._b1: OrderedDict[K, None] = OrderedDict()
        self._b2: OrderedDict[K, None] = OrderedDict()

    @property
    @override
    def capacity(self) -> int:
        return self._capacity

    @property
    def p(self) -> int:
        return self._p

    @property
    def t2_size(self) -> int:
        return len(self._t2)

    @override
    def __len__(self) -> int:
        return len(self._t1) + len(self._t2)

    @override
    def get(self, key: K) -> Optional[V]:
        if key in self._t1:
            value = self._t1.pop(key)
            self._t2[key] = value
            return value

        if key in self._t2:
            value = self._t2.pop(key)
            self._t2[key] = value
            return value

        return None

    @override
    def put(self, key: K, value: V) -> None:
        if key in self._t1:
            _ = self._t1.pop(key)
            self._t2[key] = value
            return

        if key in self._t2:
            _ = self._t2.pop(key)
            self._t2[key] = value
            return

        if key in self._b1:
            delta = max(1, len(self._b2) // max(1, len(self._b1)))
            self._p = min(self._capacity, self._p + delta)
            self._replace(key)
            _ = self._b1.pop(key)
            self._t2[key] = value
            self._trim_ghosts()
            return

        if key in self._b2:
            delta = max(1, len(self._b1) // max(1, len(self._b2)))
            self._p = max(0, self._p - delta)
            self._replace(key)
            _ = self._b2.pop(key)
            self._t2[key] = value
            self._trim_ghosts()
            return

        l1_size = len(self._t1) + len(self._b1)
        if l1_size == self._capacity:
            if len(self._t1) < self._capacity:
                _ = self._b1.popitem(last=False)
                self._replace(key)
            else:
                _ = self._t1.popitem(last=False)
        elif (
            l1_size < self._capacity
            and len(self) + len(self._b1) + len(self._b2) >= self._capacity
        ):
            if (
                len(self) + len(self._b1) + len(self._b2) >= 2 * self._capacity
                and self._b2
            ):
                _ = self._b2.popitem(last=False)
            self._replace(key)

        self._t1[key] = value
        self._trim_ghosts()

    @override
    def evict(self) -> Optional[tuple[K, V]]:
        if self._t1:
            key, value = self._t1.popitem(last=False)
            self._b1[key] = None
            self._trim_ghosts()
            return key, value

        if self._t2:
            key, value = self._t2.popitem(last=False)
            self._b2[key] = None
            self._trim_ghosts()
            return key, value

        return None

    def _replace(self, incoming_key: K) -> None:
        should_replace_t1 = bool(self._t1) and (
            len(self._t1) > self._p
            or (incoming_key in self._b2 and len(self._t1) == self._p)
        )

        if should_replace_t1:
            key, _value = self._t1.popitem(last=False)
            self._b1[key] = None
            return

        if self._t2:
            key, _value = self._t2.popitem(last=False)
            self._b2[key] = None
            return

        if self._t1:
            key, _value = self._t1.popitem(last=False)
            self._b1[key] = None

    def _trim_ghosts(self) -> None:
        while len(self._b1) > self._capacity:
            _ = self._b1.popitem(last=False)

        while len(self._b2) > self._capacity:
            _ = self._b2.popitem(last=False)

        while len(self._b1) + len(self._b2) > self._capacity:
            if len(self._b1) >= len(self._b2) and self._b1:
                _ = self._b1.popitem(last=False)
            elif self._b2:
                _ = self._b2.popitem(last=False)
            else:
                break
