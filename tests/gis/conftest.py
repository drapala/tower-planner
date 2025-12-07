"""Pytest configuration for GIS integration tests.

This conftest is for tests/gis/ directory only.

Note on rasterio:
- test_geotiff_adapter.py uses the STUB (tests/stubs/rasterio/) with monkeypatching
- test_fixtures_sanity.py uses REAL rasterio for integration tests

Path manipulation uses scoped fixtures to allow real rasterio to be imported
when available, while still supporting the stub for unit tests.

Per TD-002: Stubs moved from src/ to tests/stubs/ to avoid import conflicts.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

from tests.conftest_utils import (
    filter_sys_path_excluding_stubs,
    get_fixtures_dir,
    prepare_real_rasterio_path,
)

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
__all__ = [
    "get_fixtures_dir",
    "real_rasterio_path",
    "exclude_stubs_path",
    "has_real_rasterio",
]


@pytest.fixture(scope="session")
def real_rasterio_path() -> Generator[None, None, None]:
    """Session-scoped fixture that removes stubs from sys.path and sys.modules.

    This allows real rasterio (if installed) to be imported instead of
    the stub in tests/stubs/rasterio/. The stub is useful for unit tests
    with monkeypatching, but integration tests need real rasterio.

    Note: Uses try/finally for exception-safe cleanup since session-scoped
    fixtures cannot use monkeypatch (which is function-scoped).

    Also clears any cached rasterio modules from sys.modules to ensure
    fresh imports use the real package.

    Usage:
        def test_something(real_rasterio_path):
            import rasterio  # Will be real rasterio, not stub
    """
    yield from prepare_real_rasterio_path()


@pytest.fixture(scope="function")
def exclude_stubs_path(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Generator[None, None, None]:
    """Function-scoped fixture to exclude stubs from sys.path and sys.modules.

    Uses monkeypatch for automatic cleanup. Preferred for tests that
    need isolation from the tests/stubs/rasterio/ stub.

    Modifies sys.path IN PLACE to preserve external references to the list.
    Uses request.addfinalizer to restore original contents on cleanup.

    Also removes cached stub modules from sys.modules to ensure fresh
    imports use the real package.

    Usage:
        def test_something(exclude_stubs_path):
            import rasterio  # Will be real rasterio, not stub
    """
    project_root = Path(__file__).parent.parent.parent
    stubs_path = str(project_root / "tests" / "stubs")

    # Use shared helper to filter sys.path (handles symlinks, empty strings, etc.)
    # Returns original path for restoration
    original_path = filter_sys_path_excluding_stubs(stubs_path)

    # Register cleanup to restore original contents via request.addfinalizer
    def restore_path() -> None:
        sys.path.clear()
        sys.path.extend(original_path)

    request.addfinalizer(restore_path)

    # Remove cached stub modules so fresh imports get real package
    for key in list(sys.modules.keys()):
        if key == "rasterio" or key.startswith("rasterio."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    yield  # Allow test to run, then monkeypatch cleanup restores state


# -----------------------------------------------------------------------------
# Helpers for detection (shared by GIS tests)
# -----------------------------------------------------------------------------
def _is_path_under(child: Path, parent: Path) -> bool:
    """Check if child path is under parent path using Path comparison.

    Uses Path.relative_to() which raises ValueError if child is not
    relative to parent. This is more robust than substring matching.
    """
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def has_real_rasterio() -> bool:
    """Return True if real rasterio is importable and not the test stub.

    Uses proper Path comparison instead of substring matching to avoid
    false positives.
    """
    try:
        import rasterio

        if not hasattr(rasterio, "__version__"):
            return False

        rasterio_path = Path(rasterio.__file__).resolve()
        stubs_dir = (Path(__file__).parent.parent / "stubs" / "rasterio").resolve()

        # Consider it real if the module path is NOT under the stubs directory
        return not _is_path_under(rasterio_path, stubs_dir)
    except (ImportError, AttributeError, OSError) as e:
        logger.debug("has_real_rasterio() check failed: %s", e)
        return False
