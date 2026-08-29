import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read_literal_list(path: Path, name: str) -> list[str]:
    module = ast.parse(path.read_text())
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            value = ast.literal_eval(statement.value)
            assert isinstance(value, list)
            return value
    raise AssertionError(f"{name} not found in {path}")


def test_store_sources_link_one_residency_authority() -> None:
    sources = read_literal_list(ROOT / "setup.py", "_STORE_SOURCES")
    residency = [p for p in sources if "residency" in p]
    assert residency == ["core/prefetch/expert_residency.cpp"]


def test_cmake_links_one_residency_authority() -> None:
    text = (ROOT / "core/CMakeLists.txt").read_text()
    assert text.count("prefetch/expert_residency.cpp") == 1
    assert text.count("residency.cpp") == 1
