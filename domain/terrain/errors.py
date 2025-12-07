"""Terrain Bounded Context - Error Hierarchy.

Custom exceptions for terrain operations.

See spec/features/FEAT-001-load-dem.md for DEM loading errors.
See spec/features/FEAT-002-terrain-profile.md for profile errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.terrain.value_objects import BoundingBox, GeoPoint


class TerrainError(Exception):
    """Base error for terrain operations."""


class InvalidRasterError(TerrainError):
    """File is not a valid raster, wrong format, or corrupted."""


class MissingCRSError(TerrainError):
    """Raster has no CRS defined."""


class InvalidGeotransformError(TerrainError):
    """Raster has invalid or missing geotransform."""


class AllNoDataError(TerrainError):
    """Raster contains 100% NoData pixels - unusable."""


class InvalidBoundsError(TerrainError):
    """Raster bounds are outside valid WGS84 range after reprojection."""


class InsufficientMemoryError(TerrainError):
    """Operation requires more memory than allowed or available."""


# ---------------------------------------------------------------------------
# FEAT-002: TerrainProfile Errors
# ---------------------------------------------------------------------------
class PointOutOfBoundsError(TerrainError):
    """Point is outside the terrain grid bounds.

    Attributes:
        point: The offending GeoPoint
        bounds: The grid's BoundingBox
    """

    def __init__(self, point: "GeoPoint", bounds: "BoundingBox") -> None:
        self.point = point
        self.bounds = bounds
        super().__init__(
            f"Point ({point.latitude:.6f}, {point.longitude:.6f}) outside bounds "
            f"[lat: {bounds.min_y:.6f} to {bounds.max_y:.6f}, "
            f"lon: {bounds.min_x:.6f} to {bounds.max_x:.6f}]"
        )


class InvalidProfileError(TerrainError):
    """Profile parameters are invalid."""

    pass
