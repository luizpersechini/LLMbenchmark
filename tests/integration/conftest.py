"""Integration test conftest — marks all tests in this package as integration."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.fspath and "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
