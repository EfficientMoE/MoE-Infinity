import os
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parents[3] / "examples"


def test_interface_example_help():
    """Verify interface_example.py --help exits 0 (all imports OK, argparse OK)."""
    repo_root = str(EXAMPLES_DIR.parent)
    env = {**os.environ, "PYTHONPATH": repo_root}
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "interface_example.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert (
        result.returncode == 0
    ), f"--help exited {result.returncode}\n{result.stderr}"


def test_readme_example_help():
    """Verify readme_example.py --help exits 0 (all imports OK, argparse OK)."""
    repo_root = str(EXAMPLES_DIR.parent)
    env = {**os.environ, "PYTHONPATH": repo_root}
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "readme_example.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert (
        result.returncode == 0
    ), f"--help exited {result.returncode}\n{result.stderr}"


def test_example_imports_available():
    """Verify key packages used by examples are importable."""
    required = ["torch", "transformers", "datasets", "moe_infinity"]
    for pkg in required:
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"Cannot import '{pkg}': {result.stderr.strip()}"
