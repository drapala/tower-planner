"""Lightweight local stub for rasterio to enable tests without external deps.

This is NOT a real rasterio implementation. It only provides the symbols
required for our tests where behavior is monkeypatched.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any


class RasterioError(Exception):
    pass


class RasterioIOError(RasterioError):
    pass


errors = SimpleNamespace(RasterioError=RasterioError, RasterioIOError=RasterioIOError)


def open(path: Any) -> Any:  # monkeypatched in tests
    raise RasterioError("rasterio.open is a stub; monkeypatch in tests")


@contextmanager
def Env(*args: Any, **kwargs: Any) -> Any:  # monkeypatched in tests
    yield SimpleNamespace()


def band(dataset: Any, index: int) -> tuple[Any, int]:
    return (dataset, index)
