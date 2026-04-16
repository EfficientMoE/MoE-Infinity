from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires running servers")
    config.addinivalue_line("markers", "gpu: requires GPU hardware")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def sidecar_base_url() -> str:
    return os.getenv("CONTEXTPILOT_SIDECAR_URL", "http://localhost:8765")


@pytest.fixture(scope="session")
def backend_base_url() -> str:
    return os.getenv("CONTEXTPILOT_BACKEND_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def integration_timeout_seconds() -> float:
    return float(os.getenv("CONTEXTPILOT_TEST_TIMEOUT", "5"))
