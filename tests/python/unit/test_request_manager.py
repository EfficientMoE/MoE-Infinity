import threading

from moe_infinity.engine.request_manager import RequestManager
from moe_infinity.engine.types import SequenceStatus


def test_add_and_get():
    mgr = RequestManager()
    rid = mgr.add_request([1, 2, 3])
    req = mgr.get_request(rid)
    assert req is not None
    assert req.status == SequenceStatus.WAITING


def test_waiting_to_running():
    mgr = RequestManager()
    rid = mgr.add_request([1, 2, 3])
    mgr.transition_request(rid, SequenceStatus.RUNNING)
    assert len(mgr.get_running_requests()) == 1
    assert len(mgr.get_waiting_requests()) == 0


def test_abort_request():
    mgr = RequestManager()
    rid = mgr.add_request([1])
    mgr.abort_request(rid)
    req = mgr.get_request(rid)
    assert req is not None
    assert req.status == SequenceStatus.FINISHED_STOPPED


def test_finish_and_remove():
    mgr = RequestManager()
    r1 = mgr.add_request([1])
    r2 = mgr.add_request([2])
    mgr.transition_request(r1, SequenceStatus.RUNNING)
    mgr.finish_request(r1, SequenceStatus.FINISHED_LENGTH)
    removed = mgr.remove_finished()
    assert removed == 1
    assert mgr.get_request(r1) is None
    assert mgr.get_request(r2) is not None


def test_concurrent_adds():
    mgr = RequestManager()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(10):
                _ = mgr.add_request([1, 2, 3])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert mgr.get_active_count() == 100
    assert len(mgr.get_waiting_requests()) == 100
