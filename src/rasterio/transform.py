"""Stub for rasterio.transform providing array_bounds."""

from __future__ import annotations

from typing import Any, Tuple


def array_bounds(
    height: int, width: int, transform: Any
) -> Tuple[float, float, float, float]:
    # Compute bounds assuming transform maps (col,row) -> (x,y)
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    # Ensure ordering min/max
    minx = min(left, right)
    maxx = max(left, right)
    miny = min(bottom, top)
    maxy = max(bottom, top)
    return (minx, miny, maxx, maxy)
