"""Stub for rasterio.transform providing array_bounds."""

from __future__ import annotations


def array_bounds(
    height: int, width: int, transform: object
) -> tuple[float, float, float, float]:
    """Compute bounding box from raster dimensions and transform.

    Expects a transform object that supports the multiplication operator:
        transform * (col, row) -> (x, y)

    Typically this is an affine.Affine object. The function computes corner
    coordinates and returns them in (minx, miny, maxx, maxy) order.

    Args:
        height: Number of rows in the raster
        width: Number of columns in the raster
        transform: Affine transform supporting (col, row) multiplication

    Returns:
        Tuple of (minx, miny, maxx, maxy) bounds

    Raises:
        TypeError: If transform does not support multiplication with (col, row) tuple
    """
    try:
        left, top = transform * (0, 0)  # type: ignore[operator]
        right, bottom = transform * (width, height)  # type: ignore[operator]
    except TypeError as e:
        raise TypeError(
            f"Transform must support multiplication with (col, row) tuple: {e}"
        ) from e

    # Ensure ordering min/max
    minx = min(left, right)
    maxx = max(left, right)
    miny = min(bottom, top)
    maxy = max(bottom, top)
    return (minx, miny, maxx, maxy)
