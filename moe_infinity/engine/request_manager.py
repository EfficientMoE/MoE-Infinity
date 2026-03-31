import threading
import time
import uuid
from typing import Optional

from moe_infinity.engine.types import Request, SamplingParams, SequenceStatus


class RequestManager:
    def __init__(self) -> None:
        self._requests: dict[str, Request] = {}
        self._lock: threading.Lock = threading.Lock()

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> str:
        rid = request_id or str(uuid.uuid4())
        req = Request(
            request_id=rid,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params or SamplingParams(),
            arrival_time=time.time(),
        )
        with self._lock:
            if rid in self._requests:
                raise ValueError(f"Request already exists: {rid}")
            self._requests[rid] = req
        return rid

    def get_request(self, request_id: str) -> Optional[Request]:
        with self._lock:
            return self._requests.get(request_id)

    def abort_request(self, request_id: str) -> None:
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return
            try:
                req.transition_to(SequenceStatus.FINISHED_STOPPED)
            except ValueError:
                return

    def finish_request(
        self, request_id: str, finish_reason: SequenceStatus
    ) -> None:
        if finish_reason not in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
        ):
            raise ValueError(f"Invalid finish reason: {finish_reason}")
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return
            req.transition_to(finish_reason)

    def get_waiting_requests(self) -> list[Request]:
        with self._lock:
            return [
                req
                for req in self._requests.values()
                if req.status == SequenceStatus.WAITING
            ]

    def get_running_requests(self) -> list[Request]:
        with self._lock:
            return [
                req
                for req in self._requests.values()
                if req.status == SequenceStatus.RUNNING
            ]

    def get_swapped_requests(self) -> list[Request]:
        with self._lock:
            return [
                req
                for req in self._requests.values()
                if req.status == SequenceStatus.SWAPPED
            ]

    def transition_request(
        self, request_id: str, new_status: SequenceStatus
    ) -> None:
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return
            req.transition_to(new_status)

    def get_active_count(self) -> int:
        with self._lock:
            return sum(
                1
                for req in self._requests.values()
                if req.status
                in (
                    SequenceStatus.WAITING,
                    SequenceStatus.RUNNING,
                    SequenceStatus.SWAPPED,
                )
            )

    def remove_finished(self) -> int:
        with self._lock:
            finished_request_ids = [
                request_id
                for request_id, req in self._requests.items()
                if req.status
                in (
                    SequenceStatus.FINISHED_STOPPED,
                    SequenceStatus.FINISHED_LENGTH,
                )
            ]
            for request_id in finished_request_ids:
                del self._requests[request_id]
        return len(finished_request_ids)
