"""GeoTIFF adapter for TerrainRepository.

Implements loading of DEM rasters from GeoTIFF using rasterio, normalizing to
EPSG:4326 and returning a domain TerrainGrid Value Object.

Lifecycle (to avoid resource leaks):
1) Open dataset with context manager (rasterio.open)
2) Optionally enter rasterio.Env for GDAL/PROJ configuration
3) Read metadata and validate preconditions
4) Reproject if needed using calculate_default_transform
5) Convert nodata -> np.nan; cast to float32; validate at least one valid pixel
6) Build BoundingBox and positive resolution tuple
7) Exit contexts to release GDAL handles
8) Return TerrainGrid
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from domain.terrain.value_objects import BoundingBox, TerrainGrid


class GeoTiffTerrainAdapter:
    """Infrastructure adapter for loading DEMs from GeoTIFF files.

    Parameters
    ----------
    max_bytes: Optional[int]
        Optional memory budget for the resulting float32 grid (height*width*4).
        If specified and exceeded by estimated size, the adapter should raise
        InsufficientMemoryError (as specified in FEAT-001).
    """

    def __init__(self, max_bytes: Optional[int] = None) -> None:
        self.max_bytes = max_bytes

    def load_dem(self, file_path: Path | str) -> TerrainGrid:
        """Load DEM from GeoTIFF and return normalized TerrainGrid.

        Follows FEAT-001 behavior and invariants. This is a skeleton to be
        completed via TDD per CLAUDE.md.
        """
        # NOTE: Implementation will be added following tests.
        raise NotImplementedError("GeoTiffTerrainAdapter.load_dem not implemented yet")

