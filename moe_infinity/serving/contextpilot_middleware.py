from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Any, Optional, cast

try:
    from contextpilot import ContextPilot
except ImportError:
    ContextPilot = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class ContextPilotMiddleware:
    _enabled: bool
    _reorder_enabled: bool
    _dedup_enabled: bool
    _cp: Any
    _lock: threading.Lock
    token_savings_total: int
    _requests_processed: int
    _savings_pct_total: float
    _reorder_count: int
    _dedup_count: int
    _last_reorder_latency_ms: float
    _last_dedup_latency_ms: float
    _last_tokens_saved: int
    _last_savings_pct: float

    def __init__(
        self,
        use_gpu: bool = False,
        enabled: bool = True,
        dedup_enabled: bool = True,
        reorder_enabled: bool = True,
    ):
        if ContextPilot is None:
            logger.warning(
                "contextpilot package not installed; ContextPilot features disabled. "
                "Install with: pip install 'contextpilot>=0.5.0,<0.6' (requires Python 3.10+)"
            )
            self._enabled = False
            self._reorder_enabled = False
            self._dedup_enabled = False
            self._cp = None
        else:
            self._enabled = bool(enabled)
            self._reorder_enabled = bool(reorder_enabled)
            self._dedup_enabled = bool(dedup_enabled)
            self._cp = ContextPilot(use_gpu=use_gpu)
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.token_savings_total = 0
        self._requests_processed = 0
        self._savings_pct_total = 0.0
        self._reorder_count = 0
        self._dedup_count = 0
        self._last_reorder_latency_ms = 0.0
        self._last_dedup_latency_ms = 0.0
        self._last_tokens_saved = 0
        self._last_savings_pct = 0.0
        self._serving_to_cp_ids: dict[str, set[str]] = {}
        self._serving_before_ids: dict[str, set[str]] = {}

    def process_chat_request(
        self,
        messages: list[dict[str, str]],
        *,
        serving_request_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, str]]:
        if not self._enabled:
            return messages
        if not messages:
            return []

        try:
            reorder_latency_ms = 0.0
            dedup_latency_ms = 0.0
            if self._reorder_enabled:
                reorder_started_at = time.monotonic()
                optimized_messages = self._reorder_messages(
                    messages,
                    serving_request_id=serving_request_id,
                    conversation_id=conversation_id,
                )
                reorder_latency_ms = (
                    time.monotonic() - reorder_started_at
                ) * 1000
            else:
                optimized_messages = [dict(message) for message in messages]

            request_tokens_saved = 0
            request_savings_pct = 0.0
            if self._dedup_enabled:
                dedup_started_at = time.monotonic()
                (
                    optimized_messages,
                    request_tokens_saved,
                    request_savings_pct,
                ) = self._deduplicate_messages(optimized_messages)
                dedup_latency_ms = (time.monotonic() - dedup_started_at) * 1000
                logger.info(
                    "CP dedup: removed ~%d duplicate tokens (%.1f%%)",
                    request_tokens_saved,
                    request_savings_pct,
                )

            with self._stats_lock:
                self.token_savings_total += request_tokens_saved
                self._requests_processed += 1
                self._savings_pct_total += request_savings_pct
                if self._reorder_enabled:
                    self._reorder_count += 1
                if self._dedup_enabled:
                    self._dedup_count += 1
                self._last_reorder_latency_ms = reorder_latency_ms
                self._last_dedup_latency_ms = dedup_latency_ms
                self._last_tokens_saved = request_tokens_saved
                self._last_savings_pct = request_savings_pct
            return optimized_messages
        except Exception as exc:
            logger.warning("ContextPilot optimize failed: %s", exc)
            with self._stats_lock:
                self._requests_processed += 1
                self._last_reorder_latency_ms = 0.0
                self._last_dedup_latency_ms = 0.0
                self._last_tokens_saved = 0
                self._last_savings_pct = 0.0
        return messages

    def process_completion_request(self, prompt: str) -> str:
        if not self._enabled:
            return prompt

        with self._stats_lock:
            self._requests_processed += 1
            self._last_reorder_latency_ms = 0.0
            self._last_dedup_latency_ms = 0.0
            self._last_tokens_saved = 0
            self._last_savings_pct = 0.0
        return prompt

    def on_request_complete(self, request_id: str) -> None:
        if not self._enabled:
            return

        with self._lock:
            request_ids = self._serving_to_cp_ids.pop(request_id, None)
            before_ids = self._serving_before_ids.pop(request_id, None)
            if request_ids is None and before_ids is not None:
                try:
                    request_ids = self._snapshot_request_ids() - before_ids
                except Exception as exc:
                    logger.warning(
                        "ContextPilot request-id snapshot failed for %s: %s",
                        request_id,
                        exc,
                    )
                    return
            if not request_ids:
                return
            try:
                _ = self._cp.remove_requests(set(request_ids))
            except Exception as exc:
                logger.warning(
                    "ContextPilot request cleanup failed for %s: %s",
                    request_id,
                    exc,
                )

    def cp_index_size(self) -> int:
        if self._cp is None:
            return 0
        try:
            return len(self._snapshot_request_ids())
        except Exception:
            return 0

    def _record_new_request_ids(
        self, serving_request_id: str | None, before_ids: set[str]
    ) -> None:
        if not isinstance(serving_request_id, str) or not serving_request_id:
            return
        try:
            new_ids = self._snapshot_request_ids() - before_ids
        except Exception as exc:
            logger.warning(
                "ContextPilot request-id snapshot failed for %s: %s",
                serving_request_id,
                exc,
            )
            self._serving_before_ids[serving_request_id] = set(before_ids)
            return
        if new_ids:
            self._serving_to_cp_ids[serving_request_id] = new_ids

    def _snapshot_request_ids(self) -> set[str]:
        getter = getattr(self._cp, "get_all_request_ids", None)
        if not callable(getter):
            return set()
        found = getter()
        if isinstance(found, (set, list, tuple)):
            return {str(item) for item in found}
        return set()

    def is_enabled(self) -> bool:
        return self._enabled

    def get_token_savings(self) -> dict[str, object]:
        avg_savings_pct = 0.0
        if self._requests_processed > 0:
            avg_savings_pct = self._savings_pct_total / self._requests_processed
        return {
            "total_tokens_saved": int(self.token_savings_total),
            "avg_savings_pct": float(avg_savings_pct),
            "requests_processed": int(self._requests_processed),
        }

    def get_last_request_metrics(self) -> dict[str, float | int]:
        with self._stats_lock:
            return {
                "reorder_latency_ms": float(self._last_reorder_latency_ms),
                "dedup_latency_ms": float(self._last_dedup_latency_ms),
                "tokens_saved": int(self._last_tokens_saved),
                "savings_pct": float(self._last_savings_pct),
                "reorder_count": int(self._reorder_count),
                "dedup_count": int(self._dedup_count),
                "requests_processed": int(self._requests_processed),
            }

    @staticmethod
    def _extract_query(
        messages: list[dict[str, str]],
    ) -> tuple[str, Optional[int]]:
        last_user_index: Optional[int] = None
        query = ""

        for index, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            last_user_index = index
            query = content

        return query, last_user_index

    @staticmethod
    def _copy_messages(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        return [dict(message) for message in messages]

    @staticmethod
    def _conversation_key(
        serving_request_id: str | None,
        conversation_id: str | None,
    ) -> str | None:
        if isinstance(conversation_id, str) and conversation_id:
            return conversation_id
        if isinstance(serving_request_id, str) and serving_request_id:
            return serving_request_id
        return None

    def _reorder_messages(
        self,
        messages: list[dict[str, str]],
        *,
        serving_request_id: str | None,
        conversation_id: str | None,
    ) -> list[dict[str, str]]:
        if any(
            not isinstance(message.get("content"), str) for message in messages
        ):
            return self._copy_messages(messages)

        _query, query_index = self._extract_query(messages)
        doc_slots = [
            index for index in range(len(messages)) if index != query_index
        ]
        if not doc_slots:
            return self._copy_messages(messages)

        first_slot = doc_slots[0]
        last_slot = doc_slots[-1]
        for index in range(first_slot, last_slot + 1):
            if index == query_index:
                continue
            if not isinstance(messages[index].get("content"), str):
                return self._copy_messages(messages)

        docs = [str(messages[index]["content"]) for index in doc_slots]
        conversation_key = self._conversation_key(
            serving_request_id, conversation_id
        )
        working = self._copy_messages(messages)
        with self._lock:
            if isinstance(conversation_id, str) and conversation_id:
                docs = self._apply_cross_turn_dedup(docs, conversation_id)
                for slot, doc in zip(doc_slots, docs):
                    working[slot]["content"] = doc
            before_ids: set[str] = set()
            try:
                before_ids = self._snapshot_request_ids()
                reordered = self._cp.reorder(
                    docs, conversation_id=conversation_key
                )
            except (AttributeError, TypeError, ValueError, IndexError) as exc:
                logger.debug(
                    "ContextPilot.reorder raised %s; preserving original order",
                    exc,
                )
                return self._copy_messages(messages)
            except Exception as exc:
                logger.warning("ContextPilot reorder failed: %s", exc)
                return self._copy_messages(messages)
            self._record_new_request_ids(serving_request_id, before_ids)

        if (
            not isinstance(reordered, tuple)
            or not reordered
            or not isinstance(reordered[0], list)
            or not reordered[0]
            or not isinstance(reordered[0][0], list)
        ):
            return self._copy_messages(messages)
        new_docs = cast(list[object], reordered[0][0])
        if not self._same_string_multiset(docs, new_docs):
            logger.debug(
                "ContextPilot.reorder result is not a permutation; "
                "preserving original order"
            )
            return self._copy_messages(messages)

        buckets: dict[str, list[dict[str, str]]] = {}
        for index in doc_slots:
            content = str(working[index]["content"])
            buckets.setdefault(content, []).append(dict(working[index]))
        rebuilt: list[dict[str, str]] = []
        for doc in new_docs:
            if not isinstance(doc, str):
                return self._copy_messages(messages)
            bucket = buckets.get(doc)
            if not bucket:
                return self._copy_messages(messages)
            rebuilt.append(bucket.pop(0))
        if query_index is not None:
            rebuilt.append(dict(working[query_index]))
        return rebuilt

    def _apply_cross_turn_dedup(
        self, docs: list[str], conversation_id: str
    ) -> list[str]:
        if not self._dedup_enabled:
            return docs
        deduplicate_fn = getattr(self._cp, "deduplicate", None)
        if not callable(deduplicate_fn):
            return docs
        try:
            results = deduplicate_fn([docs], conversation_id=conversation_id)
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            logger.debug(
                "ContextPilot.deduplicate raised %s; keeping documents",
                exc,
            )
            return docs
        except Exception as exc:
            logger.warning("ContextPilot deduplicate failed: %s", exc)
            return docs
        if (
            not isinstance(results, list)
            or not results
            or not isinstance(results[0], dict)
        ):
            return docs
        row = cast(dict[object, object], results[0])
        overlapping = row.get("overlapping_docs")
        hints = row.get("reference_hints")
        if not isinstance(overlapping, list) or not isinstance(hints, list):
            return docs
        hint_for: dict[str, str] = {}
        overlapping_list = cast(list[object], overlapping)
        hint_list = cast(list[object], hints)
        for doc, hint in zip(overlapping_list, hint_list):
            if isinstance(doc, str) and isinstance(hint, str):
                hint_for[doc] = hint
        return [hint_for.get(doc, doc) for doc in docs]

    @staticmethod
    def _same_string_multiset(docs: list[str], new_docs: list[object]) -> bool:
        if any(not isinstance(doc, str) for doc in new_docs):
            return False
        return Counter(docs) == Counter(cast(list[str], new_docs))

    def _deduplicate_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], int, float]:
        return self._fallback_deduplicate(messages)

    @staticmethod
    def _fallback_deduplicate(
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], int, float]:
        seen_content_to_index: dict[str, int] = {}
        deduped_messages: list[dict[str, str]] = []
        saved_tokens = 0
        original_token_estimate = 0

        for index, message in enumerate(messages):
            new_message = dict(message)
            content = new_message.get("content")
            if not isinstance(content, str):
                deduped_messages.append(new_message)
                continue

            estimated_tokens = len(content) // 4
            original_token_estimate += estimated_tokens
            first_seen_index = seen_content_to_index.get(content)
            if first_seen_index is None:
                seen_content_to_index[content] = index
                deduped_messages.append(new_message)
                continue

            saved_tokens += estimated_tokens
            new_message["content"] = (
                f"[Deduplicated content; same as message #{first_seen_index}]"
            )
            deduped_messages.append(new_message)

        savings_pct = 0.0
        if original_token_estimate > 0:
            savings_pct = (saved_tokens / original_token_estimate) * 100.0
        return deduped_messages, saved_tokens, savings_pct


__all__ = ["ContextPilotMiddleware"]
