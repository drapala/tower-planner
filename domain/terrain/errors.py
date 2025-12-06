"""Terrain Bounded Context - Error Hierarchy.

Custom exceptions for terrain operations.
"""


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
