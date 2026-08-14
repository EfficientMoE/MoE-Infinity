# pyright: reportAny=false

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")

_ = _load_module(
    "moe_infinity.serving.sequence",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
_ = _load_module(
    "moe_infinity.serving.kv_cache",
    ROOT / "moe_infinity" / "serving" / "kv_cache.py",
)
_ = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)
_SCHEDULER_MODULE = _load_module(
    "moe_infinity.serving.scheduler",
    ROOT / "moe_infinity" / "serving" / "scheduler.py",
)

Deficit2D = _SCHEDULER_MODULE.Deficit2D
VerifyDemand = _SCHEDULER_MODULE.VerifyDemand
admit_verify_demands = _SCHEDULER_MODULE.admit_verify_demands


def test_inflight_is_seated_before_new_rounds() -> None:
    demands = [
        VerifyDemand(1, tokens=16, expert_bytes=80, in_flight=False),
        VerifyDemand(2, tokens=16, expert_bytes=80, in_flight=True),
    ]
    out = admit_verify_demands(
        demands,
        budget=Deficit2D(tokens=16, expert_bytes=80),
        carried=Deficit2D(tokens=0, expert_bytes=0),
        deficit_cap=Deficit2D(tokens=32, expert_bytes=160),
    )
    assert out.seated_ids == (2,)
    assert out.admitted_ids == ()


def test_both_dimensions_must_fit_and_unused_budget_carries() -> None:
    demands = [VerifyDemand(1, tokens=8, expert_bytes=90, in_flight=False)]
    out = admit_verify_demands(
        demands,
        budget=Deficit2D(tokens=16, expert_bytes=80),
        carried=Deficit2D(tokens=0, expert_bytes=0),
        deficit_cap=Deficit2D(tokens=32, expert_bytes=160),
    )
    assert out.admitted_ids == ()
    assert out.carried == Deficit2D(tokens=16, expert_bytes=80)


def test_carried_deficit_is_capped() -> None:
    out = admit_verify_demands(
        [], Deficit2D(16, 80), Deficit2D(30, 150), Deficit2D(32, 160)
    )
    assert out.carried == Deficit2D(32, 160)


def test_inflight_seated_then_remaining_budget_admits_new() -> None:
    demands = [
        VerifyDemand(1, tokens=8, expert_bytes=40, in_flight=True),
        VerifyDemand(2, tokens=8, expert_bytes=40, in_flight=False),
    ]
    out = admit_verify_demands(
        demands, Deficit2D(16, 80), Deficit2D(0, 0), Deficit2D(32, 160)
    )
    assert out.seated_ids == (1,)
    assert out.admitted_ids == (2,)
    assert out.carried == Deficit2D(0, 0)


def test_new_rounds_are_admitted_fcfs_until_pool_exhausts() -> None:
    demands = [
        VerifyDemand(1, tokens=8, expert_bytes=40, in_flight=False),
        VerifyDemand(2, tokens=8, expert_bytes=40, in_flight=False),
        VerifyDemand(3, tokens=8, expert_bytes=40, in_flight=False),
    ]
    out = admit_verify_demands(
        demands, Deficit2D(16, 80), Deficit2D(0, 0), Deficit2D(32, 160)
    )
    assert out.seated_ids == ()
    assert out.admitted_ids == (1, 2)
    assert out.carried == Deficit2D(0, 0)


def test_large_demand_admits_once_carry_accumulates() -> None:
    demand = VerifyDemand(1, tokens=24, expert_bytes=120, in_flight=False)
    budget = Deficit2D(16, 80)
    cap = Deficit2D(32, 160)

    first = admit_verify_demands([demand], budget, Deficit2D(0, 0), cap)
    assert first.admitted_ids == ()
    assert first.carried == Deficit2D(16, 80)

    second = admit_verify_demands([demand], budget, first.carried, cap)
    assert second.admitted_ids == (1,)
    assert second.carried == Deficit2D(8, 40)


def test_seating_beyond_quantum_floors_carry_at_zero() -> None:
    demands = [VerifyDemand(1, tokens=40, expert_bytes=200, in_flight=True)]
    out = admit_verify_demands(
        demands, Deficit2D(16, 80), Deficit2D(0, 0), Deficit2D(64, 320)
    )
    assert out.seated_ids == (1,)
    assert out.admitted_ids == ()
    assert out.carried == Deficit2D(0, 0)


def test_negative_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError):
        admit_verify_demands(
            [], Deficit2D(-1, 0), Deficit2D(0, 0), Deficit2D(32, 160)
        )
    with pytest.raises(ValueError):
        admit_verify_demands(
            [VerifyDemand(1, tokens=-1, expert_bytes=0)],
            Deficit2D(16, 80),
            Deficit2D(0, 0),
            Deficit2D(32, 160),
        )


def test_cap_must_cover_largest_single_demand_in_each_dimension() -> None:
    with pytest.raises(ValueError, match="largest single demand"):
        admit_verify_demands(
            [VerifyDemand(1, tokens=40, expert_bytes=10)],
            Deficit2D(16, 80),
            Deficit2D(0, 0),
            Deficit2D(32, 160),
        )
    with pytest.raises(ValueError, match="largest single demand"):
        admit_verify_demands(
            [VerifyDemand(1, tokens=10, expert_bytes=200)],
            Deficit2D(16, 80),
            Deficit2D(0, 0),
            Deficit2D(32, 160),
        )


def test_admission_is_side_effect_free() -> None:
    demands = [
        VerifyDemand(1, tokens=8, expert_bytes=40, in_flight=False),
        VerifyDemand(2, tokens=8, expert_bytes=40, in_flight=True),
    ]
    budget = Deficit2D(16, 80)
    carried = Deficit2D(0, 0)
    cap = Deficit2D(32, 160)

    first = admit_verify_demands(demands, budget, carried, cap)
    second = admit_verify_demands(demands, budget, carried, cap)

    assert first == second
    assert demands[0] == VerifyDemand(1, tokens=8, expert_bytes=40)
    assert budget == Deficit2D(16, 80)
    assert carried == Deficit2D(0, 0)
