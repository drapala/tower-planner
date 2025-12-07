"""Root pytest configuration for all tests.

This conftest adds tests/stubs to sys.path so that unit tests can import
the stub modules (rasterio, affine) without external dependencies.

For integration tests that need real rasterio, use the fixtures in
tests/gis/conftest.py to exclude the stubs from sys.path.
"""

import sys
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Add tests/stubs to sys.path for stub imports.

    This allows tests to import rasterio/affine stubs without the real
    packages being installed. The stubs are minimal implementations that
    support monkeypatching in unit tests.

    Per TD-002: Stubs moved from src/ to tests/stubs/ to avoid conflicts
    with real packages when installed via pip.
    """
    # Resolve stubs path to handle symlinks and relative paths
    stubs_path = (Path(__file__).parent / "stubs").resolve()
    stubs_str = str(stubs_path)

    # Compare resolved paths to detect duplicates robustly
    # (handles symlinks, relative vs absolute, trailing slashes)
    resolved_sys_paths = {str(Path(p).resolve()) for p in sys.path if p}

    # Insert at beginning to take precedence over real packages
    if stubs_str not in resolved_sys_paths:
        sys.path.insert(0, stubs_str)
