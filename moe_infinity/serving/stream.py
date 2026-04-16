from __future__ import annotations

import json
import time
import uuid
from _thread import LockType
from collections import deque
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from threading import Condition, Lock
from typing import Optional


def format_sse_event(data: str) -> str:
    return f"data: {data}\n\n"


def format_done_event() -> str:
    return "data: [DONE]\n\n"


@dataclass
class StreamChunk:
    id: str
    model: str
    created: int
    choices: list[dict[str, object]]
    object: str = "chat.completion.chunk"


@dataclass
class _StreamState:
    request_id: str
    stream_id: str
    model: str
    created: int
    queue: deque[str] = field(default_factory=deque)
    finished: bool = False
    condition: Condition = field(default_factory=Condition)


class StreamManager:
    _streams: dict[str, _StreamState]
    _lock: LockType

    def __init__(self) -> None:
        self._streams = {}
        self._lock = Lock()

    def create_stream(
        self, request_id: str, model: str
    ) -> Generator[str, None, None]:
        state = _StreamState(
            request_id=request_id,
            stream_id=f"chatcmpl-{uuid.uuid4()}",
            model=model,
            created=int(time.time()),
        )
        with self._lock:
            if request_id in self._streams:
                raise ValueError(
                    f"stream already exists for request_id '{request_id}'"
                )
            self._streams[request_id] = state
        return self._stream_generator(request_id)

    def _stream_generator(self, request_id: str) -> Generator[str, None, None]:
        state = self._require_stream(request_id)
        try:
            while True:
                with state.condition:
                    while not state.queue:
                        if state.finished:
                            return
                        _ = state.condition.wait()
                    event = state.queue.popleft()
                yield event
        finally:
            with self._lock:
                _ = self._streams.pop(request_id, None)

    def _require_stream(self, request_id: str) -> _StreamState:
        with self._lock:
            state = self._streams.get(request_id)
        if state is None:
            raise KeyError(f"unknown request_id '{request_id}'")
        return state

    def push_token(
        self,
        request_id: str,
        token_text: str,
        finished: bool,
        finish_reason: Optional[str] = None,
    ) -> None:
        state = self._require_stream(request_id)
        if finished and finish_reason is None:
            finish_reason = "stop"
        chunk = self.format_completion_chunk(
            request_id=request_id,
            token_text=token_text,
            model=state.model,
            finish_reason=finish_reason,
        )

        with state.condition:
            state.queue.append(chunk)
            if finished:
                state.queue.append(format_done_event())
                state.finished = True
            state.condition.notify_all()

    def format_completion_chunk(
        self,
        request_id: str,
        token_text: str,
        model: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        with self._lock:
            state = self._streams.get(request_id)

        stream_id = (
            state.stream_id if state is not None else f"chatcmpl-{uuid.uuid4()}"
        )
        created = state.created if state is not None else int(time.time())
        delta: dict[str, str] = {}
        if token_text:
            delta["content"] = token_text

        chunk = StreamChunk(
            id=stream_id,
            model=model,
            created=created,
            choices=[
                {
                    "delta": delta,
                    "index": 0,
                    "finish_reason": finish_reason,
                }
            ],
        )
        payload = json.dumps(
            asdict(chunk), ensure_ascii=False, separators=(",", ":")
        )
        return format_sse_event(payload)
