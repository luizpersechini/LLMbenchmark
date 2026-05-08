"""Root conftest — option flag for integration tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require Ollama to be running.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="Pass --run-integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
