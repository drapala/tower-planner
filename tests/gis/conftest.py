"""Pytest configuration for GIS integration tests.

This conftest is for tests/gis/ directory only.

Note on rasterio:
- test_geotiff_adapter.py uses the STUB (src/rasterio/) with monkeypatching
- test_fixtures_sanity.py uses REAL rasterio for integration tests

Path manipulation is centralized here to allow real rasterio to be imported
when available, while still supporting the stub for unit tests.
"""

import sys
from pathlib import Path

# Store original path for restoration
_original_sys_path: list[str] | None = None


def pytest_configure(config):
    """Remove src/ stub path from sys.path during test collection.

    This allows real rasterio (if installed) to be imported instead of
    the stub in src/rasterio/. The stub is useful for unit tests with
    monkeypatching, but integration tests need real rasterio.
    """
    global _original_sys_path
    _original_sys_path = sys.path.copy()

    project_root = Path(__file__).parent.parent.parent
    src_path = str(project_root / "src")

    # Remove src from path to prefer real rasterio from site-packages
    sys.path = [p for p in sys.path if p != src_path]


def pytest_unconfigure(config):
    """Restore original sys.path after tests complete."""
    global _original_sys_path
    if _original_sys_path is not None:
        sys.path = _original_sys_path


def get_fixtures_dir() -> Path:
    """Return path to tests/fixtures/ directory.

    Shared helper for GIS tests that need fixture paths.
    """
    return Path(__file__).parent.parent / "fixtures"
