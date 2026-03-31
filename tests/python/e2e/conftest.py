import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        default="deepseek-ai/DeepSeek-V2-Lite-Chat",
        help="HuggingFace model checkpoint for E2E tests",
    )


@pytest.fixture
def model_name(request):
    return request.config.getoption("--model")
