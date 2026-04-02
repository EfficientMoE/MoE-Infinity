from __future__ import annotations

import threading
from typing import Optional, Protocol


class _CPMiddlewareLike(Protocol):
    def on_request_complete(self, request_id: str) -> None: ...


class EvictionSyncAdapter:
    def __init__(self, cp_middleware: Optional[_CPMiddlewareLike] = None):
        """Takes optional CP middleware instance for remove_requests calls."""
        self._cp_middleware: Optional[_CPMiddlewareLike] = cp_middleware
        self._lock: threading.RLock = threading.RLock()
        self._evicted_request_ids: set[str] = set()
        self._evict_incoming: int = 0
        self._evict_removed: int = 0
        self._evict_not_found: int = 0

    def on_request_finished(self, request_id: str) -> None:
        """Terminal completion — CP index should be evicted."""
        self._evict_request(request_id)

    def on_request_aborted(self, request_id: str) -> None:
        """Terminal abort — CP index should be evicted."""
        self._evict_request(request_id)

    def on_kv_blocks_freed(self, request_id: str) -> None:
        """True KV deallocation — CP index should be evicted."""
        self._evict_request(request_id)

    def on_kv_blocks_swapped(self, request_id: str) -> None:
        """Swap-out only — NO CP action (blocks recoverable)."""
        _ = request_id

    def get_counters(self) -> dict[str, int]:
        """Returns: evict_incoming, evict_removed, evict_not_found."""
        with self._lock:
            return {
                "evict_incoming": self._evict_incoming,
                "evict_removed": self._evict_removed,
                "evict_not_found": self._evict_not_found,
            }

    def _evict_request(self, request_id: str) -> None:
        middleware_complete = (
            self._cp_middleware.on_request_complete
            if self._cp_middleware is not None
            else None
        )
        should_notify = False

        with self._lock:
            self._evict_incoming += 1

            if request_id in self._evicted_request_ids:
                self._evict_not_found += 1
                return

            self._evicted_request_ids.add(request_id)

            if callable(middleware_complete):
                self._evict_removed += 1
                should_notify = True

        if should_notify and middleware_complete is not None:
            middleware_complete(request_id)
