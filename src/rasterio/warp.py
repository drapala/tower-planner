"""Stub submodule for rasterio.warp used in tests (monkeypatched)."""

from __future__ import annotations

from typing import Any, Tuple


def calculate_default_transform(*args: Any, **kwargs: Any) -> Tuple[Any, int, int]:
    raise NotImplementedError


def reproject(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError
