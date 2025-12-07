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

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
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

# Module-level logger (reused across all calls)
logger = logging.getLogger(__name__)

# Target CRS for all output (constructed once for efficient comparison)
_TARGET_CRS = CRS.from_epsg(4326)


def _is_wgs84(crs: Any) -> bool:
    """Check if CRS is WGS84 (EPSG:4326 or equivalent).

    Uses rasterio CRS equality first for robust comparison (handles projjson,
    wkt, etc.), then falls back to string comparison for test mocks.

    Args:
        crs: A rasterio CRS object, string, or test mock.

    Returns:
        True if the CRS represents WGS84/EPSG:4326, False otherwise.
    """
    if crs is None:
        return False
    # Try rasterio CRS equality (handles projjson, wkt, etc.)
    try:
        if crs == _TARGET_CRS:
            return True
    except (TypeError, AttributeError):
        pass
    # Fall back to string comparison for test mocks
    crs_str = str(crs).upper()
    return crs_str in ("EPSG:4326", "OGC:CRS84")


class GeoTiffTerrainAdapter:
    """Infrastructure adapter for loading DEMs from GeoTIFF files.

    Parameters
    ----------
    max_bytes: int | None
        Optional memory budget for the resulting float32 grid (height*width*4).
        If specified and exceeded by estimated size, the adapter should raise
        InsufficientMemoryError (as specified in FEAT-001).
    """

    def __init__(self, max_bytes: int | None = None) -> None:
        self.max_bytes = max_bytes

    def load_dem(self, file_path: Path | str) -> TerrainGrid:
        """Load DEM from GeoTIFF and return normalized TerrainGrid.

        Follows FEAT-001 behavior and invariants. This is a skeleton to be
        completed via TDD per CLAUDE.md.
        """
        path = Path(file_path)

        # Check existence first to ensure missing files surface as FileNotFoundError
        if not path.exists():
            # Raise with full path to aid debugging; callers must sanitize logs per SEC-7
            raise FileNotFoundError(str(path))

        # SEC-10: Extension allowlist - reject non-GeoTIFF for existing files
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise InvalidRasterError(f"Unsupported file extension: {path.suffix}")

        try:
            # SEC-5: Reject symlinks explicitly to avoid traversal
            if path.is_symlink():
                raise InvalidRasterError("Symlinks are not permitted")
            st = path.stat()
            if st.st_size == 0:
                # SPEC TC-010: Use precise message for empty file
                raise InvalidRasterError("Empty file")
            # PERF-7: Pre-flight size check - fail fast before rasterio allocation
            # File size is typically larger than final float32 grid, but if file
            # exceeds 2x budget, it's certainly too large
            if self.max_bytes is not None and st.st_size > self.max_bytes * 2:
                raise InsufficientMemoryError(
                    f"File size {st.st_size}B exceeds 2x memory budget {self.max_bytes}B"
                )
        except OSError as e:
            # Log and re-raise all OS errors (including FileNotFoundError, NotADirectoryError)
            # SEC-7: Log only filename, errno, and strerror to avoid leaking absolute paths
            # Don't silently swallow - let caller handle or rasterio provide better error
            logger.error(
                "Failed to stat %s (errno=%s, strerror=%s)",
                path.name,
                getattr(e, "errno", "unknown"),
                getattr(e, "strerror", "unknown"),
            )
            raise

        try:
            with rasterio.Env():
                with rasterio.open(path) as src:
                    if src.count == 0:
                        raise InvalidRasterError("Empty or bandless file")
                    if src.count != 1:
                        raise InvalidRasterError(f"Expected 1 band, got {src.count}")
                    if src.crs is None:
                        raise MissingCRSError("Raster has no CRS defined")

                    src_crs_str = src.crs.to_string()

                    # Validate geotransform
                    transform: Affine = src.transform
                    if not isinstance(transform, Affine):
                        raise InvalidGeotransformError("Missing affine transform")
                    if any(
                        math.isnan(v) or math.isinf(v)
                        for v in (
                            transform.a,
                            transform.b,
                            transform.c,
                            transform.d,
                            transform.e,
                            transform.f,
                        )
                    ):
                        raise InvalidGeotransformError(
                            "Invalid (NaN/Inf) transform values"
                        )
                    if transform.a == 0 or transform.e == 0:
                        raise InvalidGeotransformError("Invalid transform scale (zero)")

                    # Determine if reprojection is needed
                    if _is_wgs84(src.crs):
                        # Memory budget check BEFORE allocation (PERF-7)
                        if self.max_bytes is not None:
                            est_bytes = src.width * src.height * 4  # float32 = 4 bytes
                            if est_bytes > self.max_bytes:
                                raise InsufficientMemoryError(
                                    f"Estimated grid size {est_bytes}B exceeds budget {self.max_bytes}B"
                                )

                        # Same CRS; read directly (out_dtype="float32" already returns float32)
                        data = src.read(1, masked=True, out_dtype="float32")

                        # Convert nodata -> NaN: handle both masked arrays and explicit nodata
                        # Use np.float32(np.nan) to preserve dtype without upcasting
                        if hasattr(data, "mask") and np.any(data.mask):
                            # Masked array: convert masked values to NaN
                            data = np.where(data.mask, np.float32(np.nan), data.data)
                        elif src.nodata is not None:
                            # Plain array with explicit nodata value: convert to NaN.
                            # Exact equality is intentional: GeoTIFF nodata is stored as an
                            # exact value in file metadata, and rasterio preserves it precisely.
                            # Using tolerance could incorrectly mark valid near-nodata values.
                            data = np.where(
                                data == src.nodata, np.float32(np.nan), data
                            )

                        # All NoData in same-CRS path
                        if not np.any(~np.isnan(data)):
                            raise AllNoDataError(
                                "Raster contains 100% NoData pixels - unusable"
                            )

                        height, width = data.shape

                        # Bounds and resolution
                        minx, miny, maxx, maxy = array_bounds(height, width, transform)
                        try:
                            bounds = BoundingBox(
                                min_x=minx, min_y=miny, max_x=maxx, max_y=maxy
                            )
                        except ValueError as e:
                            raise InvalidBoundsError(str(e)) from e

                        resolution: tuple[float, float] = (
                            abs(transform.a),
                            abs(transform.e),
                        )

                        # Bit-exact condition implicitly satisfied if source dtype was float32
                        grid = TerrainGrid(
                            data=data,
                            bounds=bounds,
                            crs="EPSG:4326",
                            resolution=resolution,
                            source_crs=src_crs_str,
                        )

                        # At least one valid pixel enforced by VO; compute NoData % for logging
                        # SEC-7: Log only filename, not full path (avoid info leakage)
                        nodata_pct = float(np.isnan(data).mean() * 100.0)
                        if nodata_pct > 80.0:
                            logger.warning(
                                "DEM %s: %.1f%% NoData pixels detected",
                                path.name,
                                nodata_pct,
                            )
                        logger.debug(
                            "DEM %s: Loaded %dx%d grid", path.name, width, height
                        )
                        return grid

                    # Reprojection path
                    # Calculate destination transform and shape
                    # Rasterio exposes bounds as an iterable (left, bottom, right, top).
                    # Our tests' FakeDataset provides an object with attributes.
                    sb = src.bounds
                    try:
                        bounds_tuple = (sb.left, sb.bottom, sb.right, sb.top)
                    except (AttributeError, TypeError):
                        # Fallback for bounds returned as tuple instead of object
                        bounds_tuple = tuple(sb)

                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src.crs, _TARGET_CRS, src.width, src.height, *bounds_tuple
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
                        dst_crs=_TARGET_CRS,
                        resampling=Resampling.bilinear,
                        src_nodata=src_nodata,
                        dst_nodata=np.nan,
                    )

                    # Validate at least one non-NaN
                    if not np.any(~np.isnan(dst)):
                        raise AllNoDataError(
                            "Raster contains 100% NoData pixels - unusable"
                        )

                    # Bounds and resolution
                    minx, miny, maxx, maxy = array_bounds(
                        dst.shape[0], dst.shape[1], dst_transform
                    )
                    try:
                        bounds = BoundingBox(
                            min_x=minx, min_y=miny, max_x=maxx, max_y=maxy
                        )
                    except ValueError as e:
                        raise InvalidBoundsError(str(e)) from e

                    resolution = (abs(dst_transform.a), abs(dst_transform.e))

                    # SEC-7: Log only filename, not full path (avoid info leakage)
                    logger.info(
                        "DEM %s: Reprojected from %s to EPSG:4326",
                        path.name,
                        src_crs_str,
                    )
                    nodata_pct = float(np.isnan(dst).mean() * 100.0)
                    if nodata_pct > 80.0:
                        logger.warning(
                            "DEM %s: %.1f%% NoData pixels detected",
                            path.name,
                            nodata_pct,
                        )
                    logger.debug(
                        "DEM %s: Loaded %dx%d grid",
                        path.name,
                        dst.shape[1],
                        dst.shape[0],
                    )

                    return TerrainGrid(
                        data=dst,
                        bounds=bounds,
                        crs="EPSG:4326",
                        resolution=resolution,
                        source_crs=src_crs_str,
                    )

        except PermissionError as e:
            # Re-raise with filename only to avoid leaking full path in logs
            raise PermissionError(path.name) from e
        except (rasterio.errors.RasterioIOError, rasterio.errors.RasterioError) as e:
            raise InvalidRasterError(f"Corrupted or invalid raster: {e}") from e
        except MemoryError as e:
            raise InsufficientMemoryError("Insufficient memory to load raster") from e
