"""Infrastructure adapters for the terrain bounded context.

This module provides the infrastructure layer implementations for terrain
operations, including loading DEMs from GeoTIFF files.

Per TD-004: Adapter exported for simplified imports.
"""

from .geotiff_adapter import GeoTiffTerrainAdapter

__all__ = ["GeoTiffTerrainAdapter"]
