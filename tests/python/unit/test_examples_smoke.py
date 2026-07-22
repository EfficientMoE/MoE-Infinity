import os
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parents[3] / "examples"


def test_interface_example_help():
    """Verify interface_example.py --help exits 0 (all imports OK, argparse OK)."""
    import pytest

    repo_root = str(EXAMPLES_DIR.parent)
    env = {**os.environ, "PYTHONPATH": repo_root}
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "interface_example.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    if result.returncode != 0 and (
        "moe_infinity._store" in result.stderr
        or "No module named 'nvtx'" in result.stderr
    ):
        pytest.skip("moe_infinity compiled extensions not available")
    assert (
        result.returncode == 0
    ), f"--help exited {result.returncode}\n{result.stderr}"


def test_deepseek_v2_chat_example_help():
    """Verify deepseek_v2_chat_example.py --help exits 0 (imports OK, argparse OK)."""
    import pytest

    repo_root = str(EXAMPLES_DIR.parent)
    env = {**os.environ, "PYTHONPATH": repo_root}
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_DIR / "deepseek_v2_chat_example.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    if result.returncode != 0 and (
        "moe_infinity._store" in result.stderr
        or "No module named 'nvtx'" in result.stderr
    ):
        pytest.skip("moe_infinity compiled extensions not available")
    assert (
        result.returncode == 0
    ), f"--help exited {result.returncode}\n{result.stderr}"


def test_example_imports_available():
    """Verify key packages used by examples are importable."""
    import pytest

    required = ["torch", "transformers", "datasets"]
    optional = ["moe_infinity"]
    for pkg in required:
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"Cannot import '{pkg}': {result.stderr.strip()}"
    # moe_infinity is optional - skip test if compiled extensions unavailable
    for pkg in optional:
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(
                f"Optional package '{pkg}' not available: {result.stderr.strip().split(chr(10))[0]}"
            )
