"""Pytest configuration for terrain domain tests.

This conftest provides fixtures for terrain tests that need real rasterio
to load fixture files for integration testing.

For unit tests of pure domain logic, no fixtures are typically needed as
TerrainGrid can be constructed directly in tests.
"""

import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def real_rasterio_path():
    """Session-scoped fixture that removes stubs from sys.path and sys.modules.

    This allows real rasterio (if installed) to be imported instead of
    the stub in tests/stubs/rasterio/. The stub is useful for unit tests
    with monkeypatching, but integration tests need real rasterio.

    Note: Uses try/finally for exception-safe cleanup since session-scoped
    fixtures cannot use monkeypatch (which is function-scoped).

    Also clears any cached rasterio AND infrastructure modules from sys.modules
    to ensure fresh imports use the real package.

    Usage:
        def test_something(real_rasterio_path):
            import rasterio  # Will be real rasterio, not stub
    """
    project_root = Path(__file__).parent.parent.parent
    stubs_path = str(project_root / "tests" / "stubs")

    original_path = sys.path.copy()

    # Remove cached modules that may have imported stub rasterio
    # Include infrastructure modules that import rasterio at module level
    saved_modules: dict[str, object] = {}
    modules_to_clear = [
        "rasterio",
        "affine",
        "src.infrastructure.terrain",
        "src.infrastructure.terrain.geotiff_adapter",
        "infrastructure.terrain",
        "infrastructure.terrain.geotiff_adapter",
    ]
    for key in list(sys.modules.keys()):
        should_clear = (
            key.startswith("rasterio.")
            or key.startswith("affine.")
            or any(key == m or key.startswith(m + ".") for m in modules_to_clear)
        )
        if key in modules_to_clear or should_clear:
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


def get_fixtures_dir() -> Path:
    """Return path to tests/fixtures/ directory.

    Shared helper for terrain tests that need fixture paths.
    """
    return Path(__file__).parent.parent / "fixtures"
