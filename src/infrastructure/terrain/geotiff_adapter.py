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
from typing import Optional, Tuple

import logging
import math
import os

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject

from domain.terrain.errors import (
    AllNoDataError,
    InsufficientMemoryError,
    InvalidBoundsError,
    InvalidGeotransformError,
    InvalidRasterError,
    MissingCRSError,
)
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
        logger = logging.getLogger(
            "src.infrastructure.terrain.geotiff_adapter"
        )
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(str(path))
        try:
            st = path.stat()
            if st.st_size == 0:
                raise InvalidRasterError("Empty or bandless file")
        except OSError:
            # If stat fails, let rasterio decide later; continue
            pass

        if not os.access(path, os.R_OK):
            raise PermissionError(str(path))

        try:
            with rasterio.Env():
                with rasterio.open(path) as src:
                    if src.count == 0:
                        raise InvalidRasterError("Empty or bandless file")
                    if src.count != 1:
                        raise InvalidRasterError(
                            f"Expected 1 band, got {src.count}"
                        )
                    if src.crs is None:
                        raise MissingCRSError("Raster has no CRS defined")

                    src_crs_str = src.crs.to_string() if src.crs else None

                    # Validate geotransform
                    transform: Affine = src.transform
                    if not isinstance(transform, Affine):
                        raise InvalidGeotransformError("Missing affine transform")
                    if any(
                        math.isnan(v) or math.isinf(v)
                        for v in (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
                    ):
                        raise InvalidGeotransformError("Invalid (NaN/Inf) transform values")
                    if transform.a == 0 or transform.e == 0:
                        raise InvalidGeotransformError("Invalid transform scale (zero)")

                    # Determine if reprojection is needed
                    dst_crs = "EPSG:4326"
                    if str(src.crs).upper() in ("EPSG:4326", "OGC:CRS84"):
                        # Same CRS; read directly
                        data = src.read(1, masked=True, out_dtype="float32").astype(
                            np.float32, copy=False
                        )
                        # Convert mask/nodata -> NaN
                        if hasattr(data, "mask"):
                            data = np.where(data.mask, np.nan, data.data).astype(
                                np.float32, copy=False
                            )

                        height, width = data.shape

                        # Memory budget check
                        if self.max_bytes is not None:
                            est_bytes = int(width) * int(height) * 4
                            if est_bytes > self.max_bytes:
                                raise InsufficientMemoryError(
                                    f"Estimated grid size {est_bytes}B exceeds budget {self.max_bytes}B"
                                )

                        # Bounds and resolution
                        minx, miny, maxx, maxy = array_bounds(height, width, transform)
                        try:
                            bounds = BoundingBox(min_x=minx, min_y=miny, max_x=maxx, max_y=maxy)
                        except ValueError as e:
                            raise InvalidBoundsError(str(e)) from e

                        resolution: Tuple[float, float] = (abs(transform.a), abs(transform.e))

                        # Bit-exact condition implicitly satisfied if source dtype was float32
                        grid = TerrainGrid(
                            data=data,
                            bounds=bounds,
                            crs=dst_crs,
                            resolution=resolution,
                            source_crs=src_crs_str,
                        )

                        # At least one valid pixel enforced by VO; compute NoData % for logging
                        nodata_pct = float(np.isnan(data).mean() * 100.0)
                        if nodata_pct >= 80.0:
                            logger.warning(
                                f"DEM {path}: {nodata_pct:.1f}% NoData pixels detected"
                            )
                        logger.debug(
                            f"DEM {path}: Loaded {width}x{height} grid"
                        )
                        return grid

                    # Reprojection path
                    # Calculate destination transform and shape
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds
                    )

                    # Memory budget check
                    if self.max_bytes is not None:
                        est_bytes = int(dst_width) * int(dst_height) * 4
                        if est_bytes > self.max_bytes:
                            raise InsufficientMemoryError(
                                f"Estimated grid size {est_bytes}B exceeds budget {self.max_bytes}B"
                            )

                    # Prepare destination array filled with NaN
                    dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

                    # Use source nodata if present
                    src_nodata = src.nodata

                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dst,
                        src_transform=transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=src_nodata,
                        dst_nodata=np.nan,
                    )

                    # Validate at least one non-NaN
                    if not np.any(~np.isnan(dst)):
                        raise AllNoDataError("Raster contains 100% NoData pixels - unusable")

                    # Bounds and resolution
                    minx, miny, maxx, maxy = array_bounds(dst.shape[0], dst.shape[1], dst_transform)
                    try:
                        bounds = BoundingBox(min_x=minx, min_y=miny, max_x=maxx, max_y=maxy)
                    except ValueError as e:
                        raise InvalidBoundsError(str(e)) from e

                    resolution = (abs(dst_transform.a), abs(dst_transform.e))

                    logger.info(
                        f"DEM {path}: Reprojected from {src_crs_str} to EPSG:4326"
                    )
                    nodata_pct = float(np.isnan(dst).mean() * 100.0)
                    if nodata_pct >= 80.0:
                        logger.warning(
                            f"DEM {path}: {nodata_pct:.1f}% NoData pixels detected"
                        )
                    logger.debug(
                        f"DEM {path}: Loaded {dst.shape[1]}x{dst.shape[0]} grid"
                    )

                    return TerrainGrid(
                        data=dst,
                        bounds=bounds,
                        crs=dst_crs,
                        resolution=resolution,
                        source_crs=src_crs_str,
                    )

        except (rasterio.errors.RasterioIOError, rasterio.errors.RasterioError) as e:
            raise InvalidRasterError(f"Corrupted or invalid raster: {e}") from e
        except MemoryError as e:
            raise InsufficientMemoryError("Insufficient memory to load raster") from e
