"""Pytest configuration for terrain domain tests.

This conftest provides fixtures for terrain tests that need real rasterio
to load fixture files for integration testing.

For unit tests of pure domain logic, no fixtures are typically needed as
TerrainGrid can be constructed directly in tests.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from tests.conftest_utils import get_fixtures_dir, prepare_real_rasterio_path

# Re-export for backwards compatibility with existing tests
__all__ = ["get_fixtures_dir", "real_rasterio_path"]


@pytest.fixture(scope="session")
def real_rasterio_path() -> Generator[None, None, None]:
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
    yield from prepare_real_rasterio_path(
        extra_modules=[
            "affine",
            "src.infrastructure.terrain",
            "src.infrastructure.terrain.geotiff_adapter",
            "infrastructure.terrain",
            "infrastructure.terrain.geotiff_adapter",
        ]
    )
