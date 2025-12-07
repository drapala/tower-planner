"""Lightweight local stub for rasterio to enable tests without external deps.

This is NOT a real rasterio implementation. It only provides the symbols
required for our tests where behavior is monkeypatched.

Location: tests/stubs/rasterio/ (moved from src/rasterio/ per TD-002)

Precedence:
    tests/conftest.py inserts tests/stubs/ at sys.path position 0, so these stubs
    take precedence over any installed rasterio in site-packages. This ensures
    unit tests use the stub by default.

    Real rasterio is only used when tests explicitly opt out via:
    - real_rasterio_path fixture (session scope, excludes stubs from sys.path)
    - exclude_stubs_path fixture (function scope, via monkeypatch)
    Both fixtures are defined in tests/gis/conftest.py.

Usage:
    - Most functions raise errors or return minimal values
    - Real behavior is injected via monkeypatch in individual tests
    - Integration tests use the fixtures above to access real rasterio
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any


class RasterioError(Exception):
    """Base exception for rasterio errors."""


class RasterioIOError(RasterioError):
    """I/O related rasterio error."""


# Namespace mimicking rasterio.errors module
errors = SimpleNamespace(RasterioError=RasterioError, RasterioIOError=RasterioIOError)


def open(path: Any) -> Any:
    """Stub for rasterio.open - always raises, must be monkeypatched.

    Tests should monkeypatch this to return a FakeDataset or similar object.
    Example:
        monkeypatch.setattr("rasterio.open", lambda p: FakeDataset(...))
    """
    raise RasterioError("rasterio.open is a stub; monkeypatch in tests")


@contextmanager
def Env(*args: Any, **kwargs: Any) -> Any:
    """Stub for rasterio.Env context manager.

    Provides a minimal no-op context manager. The real rasterio.Env configures
    GDAL/PROJ settings. This stub yields an empty namespace which is sufficient
    for tests since the adapter only uses Env() as a context wrapper.

    Not typically monkeypatched - this default behavior works for most tests.
    """
    yield SimpleNamespace()


def band(dataset: Any, index: int) -> tuple[Any, int]:
    """Stub for rasterio.band - returns (dataset, band_index) tuple.

    Mimics rasterio.band() which is used to create a (dataset, band_number) tuple
    for reprojection operations. The real function creates a special BandReader,
    but for testing purposes, returning the tuple is sufficient since reproject()
    is typically monkeypatched anyway.

    This stub is used directly (not monkeypatched) because the tuple return
    value is compatible with how tests mock the reproject() function.

    Args:
        dataset: The rasterio dataset object (or FakeDataset in tests)
        index: The 1-based band index

    Returns:
        Tuple of (dataset, index) for use with reproject()
    """
    return (dataset, index)
