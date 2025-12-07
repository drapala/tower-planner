#!/usr/bin/env python3
"""Generate synthetic GeoTIFF fixtures for FEAT-001 testing.

This script creates all test fixtures required by FEAT-001 (LoadDEM) test cases.
Fixtures are minimal synthetic rasters - not real terrain data.

Usage:
    python scripts/gen_fixtures.py

Requirements:
    pip install rasterio numpy

Output:
    tests/fixtures/*.tif (and .png)

Reference:
    - FEAT-001 v1.9.0 (spec/features/FEAT-001-load-dem.md)
    - PROMPT-001 (PROMPTS/001-create-test-fixtures.md)

Dependencies:
    This script imports from shared/fixtures_expected.py (not tests/) to avoid
    circular dependencies between scripts and tests packages.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import rasterio
from affine import Affine
from rasterio.crs import CRS
from shared.fixtures_expected import EXPECTED_FIXTURE_COUNT, EXPECTED_FIXTURES

# Output directory
FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"

# =============================================================================
# Standard Bounds Configuration
# =============================================================================
# All fixtures use consistent geographic bounds to simplify test case design.
# Tests can use coordinates within these bounds without worrying about fixture
# coverage mismatches.
#
# Standard bounds: Brazil region covering São Paulo to southern Brazil
# - Latitude:  [-25, -15] (10 degrees, ~1110 km north-south)
# - Longitude: [-50, -40] (10 degrees, ~925 km east-west at this latitude)
#
# Extended bounds: For large-scale tests (e.g., 700km paths)
# - Latitude:  [-26, -14] (12 degrees, ~1330 km)
# - Longitude: [-51, -39] (12 degrees, ~1110 km)

# Standard bounds (used by most fixtures)
STD_MIN_LON, STD_MAX_LON = -50.0, -40.0
STD_MIN_LAT, STD_MAX_LAT = -25.0, -15.0

# Extended bounds (for dem_large.tif - covers 700km+ paths)
EXT_MIN_LON, EXT_MAX_LON = -51.0, -39.0
EXT_MIN_LAT, EXT_MAX_LAT = -26.0, -14.0


def ensure_dir() -> None:
    """Ensure fixtures directory exists."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIXTURES_DIR}")


# =============================================================================
# Helper: write_raster
# =============================================================================
def write_raster(
    path: Path,
    data: NDArray[Any],
    transform: Affine,
    crs: CRS | None = None,
    dtype: str | None = None,
    nodata: float | None = None,
    driver: str = "GTiff",
) -> None:
    """Write a raster file using rasterio.

    Handles both single-band (2D array) and multi-band (3D array) writes.
    Reduces duplication across gen_* functions.

    Args:
        path: Output file path
        data: 2D array (single band) or 3D array (multi-band, shape: bands x height x width)
        transform: Affine transform for georeferencing
        crs: Coordinate reference system (None for CRS-less files)
        dtype: Data type string (e.g., "float32", "int16"). Defaults to data.dtype
        nodata: NoData value (optional)
        driver: GDAL driver name (default "GTiff", use "PNG" for PNG files)
    """
    # Determine dimensions
    if data.ndim == 2:
        count = 1
        height, width = data.shape
    elif data.ndim == 3:
        count, height, width = data.shape
    else:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    # Use data's dtype if not specified
    if dtype is None:
        dtype = str(data.dtype)

    # Build kwargs for rasterio.open
    kwargs: dict[str, Any] = {
        "driver": driver,
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "transform": transform,
    }

    # Only add crs if provided (allows CRS-less files)
    if crs is not None:
        kwargs["crs"] = crs

    # Only add nodata if provided
    if nodata is not None:
        kwargs["nodata"] = nodata

    with rasterio.open(path, "w", **kwargs) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            for band_idx in range(count):
                dst.write(data[band_idx], band_idx + 1)


# =============================================================================
# TC-001: Happy Path (EPSG:4326 source)
# =============================================================================
def gen_dem_100x100_4326() -> None:
    """Generate 100x100 float32 DEM in EPSG:4326.

    - Synthetic elevation gradient (0-1000m)
    - No nodata pixels
    - Valid bounds within Brazil (-50 to -40 lon, -25 to -15 lat)
    """
    path = FIXTURES_DIR / "dem_100x100_4326.tif"
    height, width = 100, 100

    # Create gradient elevation data (0-1000m)
    data = np.linspace(0, 1000, height * width, dtype=np.float32).reshape(height, width)

    # Transform: origin at (-50, -15), 0.1 degree resolution
    transform = Affine.translation(-50.0, -15.0) * Affine.scale(0.1, -0.1)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326))
    print(f"  Created: {path.name} (100x100, EPSG:4326, float32)")


# =============================================================================
# TC-002: CRS Reprojection
# =============================================================================
def gen_dem_utm23s() -> None:
    """Generate 50x50 DEM in EPSG:31983 (SIRGAS 2000 / UTM 23S).

    - Synthetic elevation data
    - Requires reprojection to EPSG:4326
    - Bounds roughly in São Paulo region
    """
    path = FIXTURES_DIR / "dem_utm23s.tif"
    height, width = 50, 50

    # Elevation data (100-500m range typical for SP region)
    data = np.linspace(100, 500, height * width, dtype=np.float32).reshape(
        height, width
    )

    # UTM coordinates for São Paulo region
    # Origin at approximately 333000E, 7395000N (near São Paulo)
    transform = Affine.translation(333000.0, 7395000.0) * Affine.scale(30.0, -30.0)

    write_raster(path, data, transform, crs=CRS.from_epsg(31983))
    print(f"  Created: {path.name} (50x50, EPSG:31983, float32)")


# =============================================================================
# TC-004: NoData Conversion
# =============================================================================
def gen_dem_with_nodata() -> None:
    """Generate 100x100 DEM with defined nodata value and masked region.

    - Uses STANDARD bounds (same as dem_100x100_4326.tif)
    - nodata = -9999
    - Diagonal stripe of NoData from top-left to bottom-right (~10% pixels)
    - Ensures any path across the grid will cross NoData
    """
    path = FIXTURES_DIR / "dem_with_nodata.tif"
    height, width = 100, 100
    nodata = -9999.0

    # Create elevation data with gradient
    data = np.linspace(50, 500, height * width, dtype=np.float32).reshape(height, width)

    # Create diagonal NoData stripe (10 pixels wide)
    # This ensures profiles crossing the grid hit NoData
    for i in range(height):
        start_col = max(0, i - 5)
        end_col = min(width, i + 5)
        data[i, start_col:end_col] = nodata

    # Standard bounds: lat [-25, -15], lon [-50, -40]
    # Resolution: 0.1 degrees per pixel
    transform = Affine.translation(STD_MIN_LON, STD_MAX_LAT) * Affine.scale(0.1, -0.1)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326), nodata=nodata)

    nodata_pct = 100 * (np.sum(data == nodata) / data.size)
    print(
        f"  Created: {path.name} (100x100, EPSG:4326, {nodata_pct:.0f}% nodata diagonal)"
    )


# =============================================================================
# TC-005: All NoData Rejected
# =============================================================================
def gen_dem_all_nodata() -> None:
    """Generate 10x10 DEM where 100% pixels are nodata.

    - Should trigger AllNoDataError
    """
    path = FIXTURES_DIR / "dem_all_nodata.tif"
    height, width = 10, 10
    nodata = -9999.0

    # All pixels are nodata
    data = np.full((height, width), nodata, dtype=np.float32)

    transform = Affine.translation(-45.0, -20.0) * Affine.scale(0.01, -0.01)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326), nodata=nodata)
    print(f"  Created: {path.name} (10x10, 100% nodata)")


# =============================================================================
# TC-006: Missing CRS
# =============================================================================
def gen_dem_no_crs() -> None:
    """Generate 10x10 DEM without CRS definition.

    - Should trigger MissingCRSError
    - rasterio supports crs=None to create CRS-less TIFFs
    """
    path = FIXTURES_DIR / "dem_no_crs.tif"
    height, width = 10, 10

    data = np.linspace(0, 100, height * width, dtype=np.float32).reshape(height, width)
    transform = Affine.translation(0.0, 0.0) * Affine.scale(1.0, -1.0)

    write_raster(path, data, transform, crs=None)  # No CRS
    print(f"  Created: {path.name} (10x10, no CRS)")


# =============================================================================
# TC-007: Multi-band Raster
# =============================================================================
def gen_rgb_image() -> None:
    """Generate 10x10 3-band RGB TIFF.

    - Should trigger InvalidRasterError (expected 1 band)
    """
    path = FIXTURES_DIR / "rgb_image.tif"
    height, width = 10, 10

    # 3-band RGB data stacked as (3, height, width)
    red = np.full((height, width), 255, dtype=np.uint8)
    green = np.full((height, width), 128, dtype=np.uint8)
    blue = np.full((height, width), 64, dtype=np.uint8)
    data = np.stack([red, green, blue], axis=0)

    transform = Affine.translation(-45.0, -20.0) * Affine.scale(0.01, -0.01)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326), dtype="uint8")
    print(f"  Created: {path.name} (10x10x3, RGB)")


# =============================================================================
# TC-009: Bit-exact for same CRS
# =============================================================================
def gen_dem_known_values_4326() -> None:
    """Generate 100x100 DEM with known pixel values for bit-exact testing.

    - Uses STANDARD bounds (same as dem_100x100_4326.tif)
    - Known values at specific pixels for interpolation verification
    - float32 dtype
    - Used for FEAT-002 TC-010 (bilinear interpolation)
    """
    path = FIXTURES_DIR / "dem_known_values_4326.tif"
    height, width = 100, 100

    # Create gradient data as base
    data = np.linspace(100, 500, height * width, dtype=np.float32).reshape(
        height, width
    )

    # Add some known values at specific locations
    data[50, 50] = 150.5  # Center
    data[0, 0] = 100.0  # Top-left (north-west)
    data[99, 99] = 500.0  # Bottom-right (south-east)

    # Standard bounds: lat [-25, -15], lon [-50, -40]
    # Resolution: 0.1 degrees per pixel
    transform = Affine.translation(STD_MIN_LON, STD_MAX_LAT) * Affine.scale(0.1, -0.1)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326))
    print(f"  Created: {path.name} (100x100, known values, float32, standard bounds)")


# =============================================================================
# TC-009b: Same CRS, non-float32 source
# =============================================================================
def gen_dem_known_values_4326_int16() -> None:
    """Generate 100x100 DEM with known values in int16 dtype.

    - Uses STANDARD bounds (same as dem_100x100_4326.tif)
    - Known values at specific pixels
    - int16 dtype - tests dtype conversion path
    """
    path = FIXTURES_DIR / "dem_known_values_4326_int16.tif"
    height, width = 100, 100

    # Create gradient data as base
    data = np.linspace(100, 500, height * width, dtype=np.int16).reshape(height, width)

    # Add some known values at specific locations
    data[50, 50] = 150  # Center
    data[0, 0] = 100  # Top-left
    data[99, 99] = 500  # Bottom-right

    # Standard bounds: lat [-25, -15], lon [-50, -40]
    # Resolution: 0.1 degrees per pixel
    transform = Affine.translation(STD_MIN_LON, STD_MAX_LAT) * Affine.scale(0.1, -0.1)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326), dtype="int16")
    print(f"  Created: {path.name} (100x100, known values, int16, standard bounds)")


# =============================================================================
# TC-010: Empty File
# =============================================================================
def gen_empty_tif() -> None:
    """Generate 0-byte empty file.

    - Should trigger InvalidRasterError
    - Special case: does not use write_raster helper
    """
    path = FIXTURES_DIR / "empty.tif"
    path.write_bytes(b"")
    print(f"  Created: {path.name} (0 bytes)")


# =============================================================================
# TC-011: High NoData Warning (>80%)
# =============================================================================
def gen_dem_85pct_nodata() -> None:
    """Generate 20x20 DEM with ~85% nodata pixels.

    - Should log warning but still load
    - 400 total pixels, ~340 nodata, ~60 valid
    """
    path = FIXTURES_DIR / "dem_85pct_nodata.tif"
    height, width = 20, 20
    nodata = -9999.0

    # Start with all nodata
    data = np.full((height, width), nodata, dtype=np.float32)

    # Set bottom-right corner as valid (~15% = 60 pixels -> 8x8 = 64 pixels)
    data[12:20, 12:20] = np.linspace(100, 200, 64, dtype=np.float32).reshape(8, 8)

    transform = Affine.translation(-45.0, -20.0) * Affine.scale(0.01, -0.01)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326), nodata=nodata)

    nodata_pct = 100 * (np.sum(data == nodata) / data.size)
    print(f"  Created: {path.name} (20x20, {nodata_pct:.0f}% nodata)")


# =============================================================================
# TC-012, TC-018: Large File / Memory Budget
# =============================================================================
def gen_dem_large() -> None:
    """Generate 600x600 DEM for memory budget and long path testing.

    - Uses EXTENDED bounds: lat [-26, -14], lon [-51, -39]
    - ~1.4MB uncompressed (600*600*4 bytes = 1,440,000 bytes)
    - Covers ~1330km x 1110km area - sufficient for 700km+ path tests
    - Used for TC-012 (large file), TC-018 (memory budget), FEAT-002 TC-008 (long path)

    Coverage verification:
        FEAT-002 TC-008 uses: start=(-20, -44) to end=(-25, -50)
        This fixture covers: lat [-26, -14], lon [-51, -39]
        Both points are within bounds.
    """
    path = FIXTURES_DIR / "dem_large.tif"
    height, width = 600, 600

    # Create random-ish elevation data using modern Generator API (thread-safe)
    rng = np.random.default_rng(42)
    data = (rng.random((height, width)) * 1000).astype(np.float32)

    # Extended bounds: lat [-26, -14], lon [-51, -39]
    # Resolution: (51-39)/600 = 0.02 degrees lon, (26-14)/600 = 0.02 degrees lat
    lon_res = (EXT_MAX_LON - EXT_MIN_LON) / width  # 0.02
    lat_res = (EXT_MAX_LAT - EXT_MIN_LAT) / height  # 0.02
    transform = Affine.translation(EXT_MIN_LON, EXT_MAX_LAT) * Affine.scale(
        lon_res, -lat_res
    )

    write_raster(path, data, transform, crs=CRS.from_epsg(4326))

    size_kb = path.stat().st_size / 1024
    print(f"  Created: {path.name} (600x600, {size_kb:.0f}KB, extended bounds)")


# =============================================================================
# TC-013: Extreme Elevations Valid
# =============================================================================
def gen_dem_extreme_elevations() -> None:
    """Generate 10x10 DEM with extreme elevation values.

    - Contains -430m (Dead Sea) and +8849m (Everest)
    - Should load without error
    """
    path = FIXTURES_DIR / "dem_extreme_elevations.tif"
    height, width = 10, 10

    data = np.zeros((height, width), dtype=np.float32)
    data[0, 0] = -430.0  # Dead Sea
    data[0, 9] = 8849.0  # Everest
    data[5, 5] = 0.0  # Sea level

    transform = Affine.translation(-45.0, -20.0) * Affine.scale(0.01, -0.01)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326))
    print(f"  Created: {path.name} (10x10, extreme elevations)")


# =============================================================================
# TC-014: Non-GeoTIFF Rejected
# =============================================================================
def gen_image_png() -> None:
    """Generate 10x10 PNG image (not a GeoTIFF).

    - Should trigger InvalidRasterError
    - Uses rasterio's PNG driver via write_raster helper
    """
    path = FIXTURES_DIR / "image.png"
    height, width = 10, 10

    # Simple grayscale image
    data = np.linspace(0, 255, height * width, dtype=np.uint8).reshape(height, width)

    # PNG doesn't use geotransform, but we need a placeholder
    transform = Affine.identity()

    write_raster(path, data, transform, driver="PNG", dtype="uint8")
    print(f"  Created: {path.name} (10x10, PNG)")


# =============================================================================
# TC-015: Corrupted File
# =============================================================================
def gen_dem_corrupted() -> None:
    """Generate corrupted TIFF file.

    - Valid TIFF header but truncated/invalid data
    - Should trigger InvalidRasterError
    - Special case: does not use write_raster helper
    """
    path = FIXTURES_DIR / "dem_corrupted.tif"

    # TIFF header (little-endian) followed by garbage
    # II (little-endian) + 42 (TIFF magic) + offset to IFD
    tiff_header = b"II"  # Little-endian
    tiff_header += struct.pack("<H", 42)  # TIFF magic number
    tiff_header += struct.pack(
        "<I", 8
    )  # Offset to first IFD (immediately after header)
    # Add some garbage that looks like IFD but is invalid
    tiff_header += b"\x00" * 50  # Truncated/invalid IFD

    path.write_bytes(tiff_header)
    print(f"  Created: {path.name} (corrupted TIFF header)")


# =============================================================================
# TC-016: Invalid Bounds After Reprojection
# =============================================================================
def gen_dem_polar_extreme() -> None:
    """Generate DEM with polar coordinates that produce invalid WGS84 bounds.

    - Uses Antarctic Polar Stereographic (EPSG:3031)
    - When reprojected to EPSG:4326, may produce out-of-range coordinates
    """
    path = FIXTURES_DIR / "dem_polar_extreme.tif"
    height, width = 10, 10

    data = np.linspace(0, 100, height * width, dtype=np.float32).reshape(height, width)

    # Antarctic Polar Stereographic coordinates far from pole
    # These should produce extreme lat/lon when reprojected
    transform = Affine.translation(-3000000.0, 3000000.0) * Affine.scale(
        100000.0, -100000.0
    )

    write_raster(path, data, transform, crs=CRS.from_epsg(3031))
    print(f"  Created: {path.name} (10x10, EPSG:3031 polar)")


# =============================================================================
# FEAT-002 TC-014: Partial NoData Region
# =============================================================================
def gen_dem_all_nodata_partial() -> None:
    """Generate 100x100 DEM with valid west half, NoData east half.

    - Uses STANDARD bounds (same as dem_100x100_4326.tif)
    - West 50% has valid elevation data
    - East 50% is NoData (NaN)
    - Used for FEAT-002 TC-014: profile crossing NoData region

    Layout:
        ┌─────────────────┐
        │ valid │  NaN    │
        │ 150m  │  NaN    │
        │ west  │  east   │
        └─────────────────┘
    """
    path = FIXTURES_DIR / "dem_all_nodata_partial.tif"
    height, width = 100, 100

    # Create elevation data - all valid at 150m initially
    data = np.full((height, width), 150.0, dtype=np.float32)

    # East half is NoData (columns 50-99)
    data[:, 50:] = np.nan

    # Standard bounds: lat [-25, -15], lon [-50, -40]
    # Resolution: 0.1 degrees per pixel
    transform = Affine.translation(STD_MIN_LON, STD_MAX_LAT) * Affine.scale(0.1, -0.1)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326), nodata=np.nan)

    nodata_pct = 100 * (np.sum(np.isnan(data)) / data.size)
    print(f"  Created: {path.name} (100x100, {nodata_pct:.0f}% nodata east half)")


# =============================================================================
# TC-019: Invalid Geotransform (NaN/Inf)
# =============================================================================
def gen_dem_valid_for_transform_patch_test() -> None:
    """Generate valid DEM used with monkeypatch for NaN geotransform test.

    Creates a structurally valid GeoTIFF. Tests use monkeypatch to inject
    NaN/Inf transform values at runtime since rasterio cannot write invalid
    transforms directly. The fixture just needs to exist with valid structure.
    """
    path = FIXTURES_DIR / "dem_nan_transform.tif"
    height, width = 10, 10

    data = np.linspace(0, 100, height * width, dtype=np.float32).reshape(height, width)

    # Create valid file first
    transform = Affine.translation(-45.0, -20.0) * Affine.scale(0.01, -0.01)

    write_raster(path, data, transform, crs=CRS.from_epsg(4326))

    # Tests inject NaN/Inf transforms via monkeypatch at runtime
    print(f"  Created: {path.name} (10x10, valid structure for transform patch tests)")


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    """Generate all fixtures.

    Returns:
        0 on success, 1 on failure
    """
    print("=" * 60)
    print("Generating FEAT-001 Test Fixtures")
    print("=" * 60)

    try:
        ensure_dir()
    except OSError as e:
        print(f"ERROR: Cannot create fixtures directory: {e}")
        return 1
    print()

    print("TC-001: Happy Path")
    gen_dem_100x100_4326()

    print("\nTC-002: CRS Reprojection")
    gen_dem_utm23s()

    print("\nTC-004: NoData Conversion")
    gen_dem_with_nodata()

    print("\nTC-005: All NoData")
    gen_dem_all_nodata()

    print("\nTC-006: Missing CRS")
    gen_dem_no_crs()

    print("\nTC-007: Multi-band Raster")
    gen_rgb_image()

    print("\nTC-009: Bit-exact (float32)")
    gen_dem_known_values_4326()

    print("\nTC-009b: Bit-exact (int16)")
    gen_dem_known_values_4326_int16()

    print("\nTC-010: Empty File")
    gen_empty_tif()

    print("\nTC-011: High NoData (>80%)")
    gen_dem_85pct_nodata()

    print("\nTC-012/TC-018: Large File")
    gen_dem_large()

    print("\nTC-013: Extreme Elevations")
    gen_dem_extreme_elevations()

    print("\nTC-014: Non-GeoTIFF")
    gen_image_png()

    print("\nTC-015: Corrupted File")
    gen_dem_corrupted()

    print("\nTC-016: Polar Extreme")
    gen_dem_polar_extreme()

    print("\nTC-019: NaN Transform (valid file for monkeypatch)")
    gen_dem_valid_for_transform_patch_test()

    print("\nFEAT-002 TC-014: Partial NoData")
    gen_dem_all_nodata_partial()

    print()
    print("=" * 60)
    print(f"Done! Generated {EXPECTED_FIXTURE_COUNT} fixtures in {FIXTURES_DIR}")
    print("=" * 60)

    # Verify generated fixtures match expected list exactly
    fixture_files = sorted(
        f.name
        for f in FIXTURES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in (".tif", ".png")
    )
    expected_files = sorted(EXPECTED_FIXTURES)

    found_set = set(fixture_files)
    expected_set = set(expected_files)

    missing = expected_set - found_set
    extra = found_set - expected_set

    # Check count
    if len(fixture_files) != EXPECTED_FIXTURE_COUNT:
        print(
            f"ERROR: Expected {EXPECTED_FIXTURE_COUNT} fixtures, "
            f"found {len(fixture_files)}"
        )
        return 1

    # Check filenames match exactly
    if found_set != expected_set:
        print("ERROR: Fixture filenames do not match expected list!")
        if missing:
            print(f"  Missing (expected but not generated): {sorted(missing)}")
        if extra:
            print(f"  Extra (generated but not expected): {sorted(extra)}")
        print("\nUpdate shared/fixtures_expected.py to match generated fixtures.")
        return 1

    # List all generated files (filter same as validation: .tif, .png only)
    allowed_suffixes = {".tif", ".png"}
    print("\nGenerated files:")
    for f in sorted(FIXTURES_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in allowed_suffixes:
            size = f.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            else:
                size_str = f"{size/1024:.1f}KB"
            print(f"  {f.name:40} {size_str:>10}")

    print(f"\nAll {EXPECTED_FIXTURE_COUNT} fixtures verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
