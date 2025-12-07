"""Pytest configuration for GIS integration tests.

This conftest is for tests/gis/ directory only.

Note on rasterio:
- test_geotiff_adapter.py uses the STUB (tests/stubs/rasterio/) with monkeypatching
- test_fixtures_sanity.py uses REAL rasterio for integration tests

Path manipulation uses scoped fixtures to allow real rasterio to be imported
when available, while still supporting the stub for unit tests.

Per TD-002: Stubs moved from src/ to tests/stubs/ to avoid import conflicts.
"""

import sys
from pathlib import Path

import pytest


def get_fixtures_dir() -> Path:
    """Return path to tests/fixtures/ directory.

    Shared helper for GIS tests that need fixture paths.
    """
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="session")
def real_rasterio_path():
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
    project_root = Path(__file__).parent.parent.parent
    stubs_path = str(project_root / "tests" / "stubs")

    original_path = sys.path.copy()

    # Remove cached rasterio modules so fresh imports get real package
    saved_modules: dict[str, object] = {}
    for key in list(sys.modules.keys()):
        if key == "rasterio" or key.startswith("rasterio."):
            saved_modules[key] = sys.modules.pop(key)

    try:
        # Modify sys.path IN PLACE to preserve external references to the list.
        # Filter out stubs, then clear and extend the same list object.
        # Compare resolved paths to robustly remove stubs directory even with
        # relative paths, symlinks, or trailing slashes.
        resolved_stubs = Path(stubs_path).resolve()
        filtered_path = [
            p for p in sys.path if p and Path(p).resolve() != resolved_stubs
        ]
        sys.path.clear()
        sys.path.extend(filtered_path)
        yield
    finally:
        # Restore original sys.path contents IN PLACE (exception-safe)
        sys.path.clear()
        sys.path.extend(original_path)
        # Restore original sys.modules state
        for key, module in saved_modules.items():
            sys.modules[key] = module


@pytest.fixture(scope="function")
def exclude_stubs_path(monkeypatch):
    """Function-scoped fixture to exclude stubs from sys.path and sys.modules.

    Uses monkeypatch for automatic cleanup. Preferred for tests that
    need isolation from the tests/stubs/rasterio/ stub.

    Modifies sys.path IN PLACE to preserve external references to the list.
    Uses monkeypatch to restore original contents on cleanup.

    Also removes cached stub modules from sys.modules to ensure fresh
    imports use the real package.

    Usage:
        def test_something(exclude_stubs_path):
            import rasterio  # Will be real rasterio, not stub
    """
    project_root = Path(__file__).parent.parent.parent
    stubs_path = str(project_root / "tests" / "stubs")

    # Capture original sys.path contents before modification
    original_path = sys.path.copy()

    # Register cleanup to restore original contents via monkeypatch
    def restore_path():
        sys.path.clear()
        sys.path.extend(original_path)

    monkeypatch.callback(restore_path)

    # Resolve stubs path for robust comparison
    stubs_path_resolved = Path(stubs_path).resolve()

    # Filter out stubs path and modify sys.path IN PLACE
    filtered_path = [
        p for p in sys.path if p and Path(p).resolve() != stubs_path_resolved
    ]
    sys.path.clear()
    sys.path.extend(filtered_path)

    # Remove cached stub modules so fresh imports get real package
    for key in list(sys.modules.keys()):
        if key == "rasterio" or key.startswith("rasterio."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    yield  # Allow test to run, then monkeypatch cleanup restores state


# -----------------------------------------------------------------------------
# Helpers for detection (shared by GIS tests)
# -----------------------------------------------------------------------------
def has_real_rasterio() -> bool:
    """Return True if real rasterio is importable and not the test stub.

    Python-version-safe (does not rely on Path.is_relative_to).
    """
    try:
        import rasterio  # type: ignore

        if not hasattr(rasterio, "__version__"):
            return False
        rasterio_path = str(Path(rasterio.__file__).resolve())
        stubs_path = str(
            (Path(__file__).parent.parent / "stubs" / "rasterio").resolve()
        )
        # Consider it real if the module path does not contain the stubs path
        return stubs_path not in rasterio_path
    except Exception:
        return False
