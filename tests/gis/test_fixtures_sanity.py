"""Sanity tests for FEAT-001 test fixtures.

These tests validate that each fixture:
1. Exists on disk
2. Can be opened by rasterio (where applicable)
3. Has expected shape and CRS (where applicable)

These are NOT behavioral tests - they only verify fixture integrity.
Behavioral tests are in test_geotiff_adapter.py.

Requires: pip install rasterio numpy pytest
Run: pytest tests/gis/test_fixtures_sanity.py -m integration

Note: These tests require REAL rasterio, not the stub in tests/stubs/rasterio/.
      Run with: pytest tests/gis/test_fixtures_sanity.py -m integration
      Or ensure real rasterio is installed and takes precedence.
"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.fixtures_expected import EXPECTED_FIXTURES
from tests.gis.conftest import get_fixtures_dir, has_real_rasterio

if TYPE_CHECKING:
    import rasterio

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

FIXTURES_DIR = get_fixtures_dir()


# Skip tests that need real rasterio if only the stub is available
requires_real_rasterio = pytest.mark.skipif(
    not has_real_rasterio(),
    reason="Requires real rasterio (not stub). Install rasterio: pip install rasterio",
)


# =============================================================================
# Helpers
# =============================================================================
def fixture_path(name: str) -> Path:
    """Get path to a fixture file."""
    return FIXTURES_DIR / name


@contextmanager
def open_fixture(filename: str) -> Generator["rasterio.DatasetReader", None, None]:
    """Context manager to open a fixture file with rasterio.

    Imports rasterio lazily and opens the specified fixture, yielding
    the dataset reader. Ensures proper resource cleanup.

    Args:
        filename: Name of the fixture file (e.g., "dem_100x100_4326.tif")

    Yields:
        rasterio.DatasetReader for the opened fixture
    """
    import rasterio

    path = fixture_path(filename)
    with rasterio.open(path) as src:
        yield src


# =============================================================================
# Existence Tests
# =============================================================================
class TestFixturesExist:
    """Verify all required fixtures exist."""

    @pytest.mark.parametrize("filename", EXPECTED_FIXTURES)
    def test_fixture_exists(self, filename: str) -> None:
        """Verify fixture file exists."""
        path = fixture_path(filename)
        assert path.exists(), f"Missing fixture: {filename}"


# =============================================================================
# Valid GeoTIFF Tests
# =============================================================================
@requires_real_rasterio
class TestValidGeoTiffs:
    """Verify valid GeoTIFFs can be opened and have expected properties."""

    def test_dem_100x100_4326(self) -> None:
        """TC-001: Happy path fixture."""
        with open_fixture("dem_100x100_4326.tif") as src:
            assert src.count == 1
            assert src.width == 100
            assert src.height == 100
            assert src.crs is not None
            assert src.crs.to_epsg() == 4326
            assert src.dtypes[0] == "float32"

    def test_dem_utm23s(self) -> None:
        """TC-002: UTM reprojection fixture."""
        with open_fixture("dem_utm23s.tif") as src:
            assert src.count == 1
            assert src.width == 50
            assert src.height == 50
            assert src.crs is not None
            assert src.crs.to_epsg() == 31983
            assert src.dtypes[0] == "float32"

    def test_dem_with_nodata(self) -> None:
        """TC-004: NoData fixture."""
        with open_fixture("dem_with_nodata.tif") as src:
            assert src.count == 1
            assert src.width == 20
            assert src.height == 20
            assert src.nodata == -9999.0

    def test_dem_all_nodata(self) -> None:
        """TC-005: All nodata fixture."""
        with open_fixture("dem_all_nodata.tif") as src:
            assert src.count == 1
            assert src.width == 10
            assert src.height == 10
            assert src.nodata == -9999.0

    def test_dem_no_crs(self) -> None:
        """TC-006: Missing CRS fixture."""
        with open_fixture("dem_no_crs.tif") as src:
            assert src.count == 1
            assert src.width == 10
            assert src.height == 10
            assert src.crs is None

    def test_rgb_image(self) -> None:
        """TC-007: Multi-band fixture."""
        with open_fixture("rgb_image.tif") as src:
            assert src.count == 3  # RGB
            assert src.width == 10
            assert src.height == 10

    def test_dem_known_values_4326(self) -> None:
        """TC-009: Known values fixture (float32)."""
        import numpy as np

        with open_fixture("dem_known_values_4326.tif") as src:
            assert src.count == 1
            assert src.width == 20
            assert src.height == 20
            assert src.dtypes[0] == "float32"
            data = src.read(1)
            assert data[10, 10] == np.float32(150.5)

    def test_dem_known_values_4326_int16(self) -> None:
        """TC-009b: Known values fixture (int16)."""
        with open_fixture("dem_known_values_4326_int16.tif") as src:
            assert src.count == 1
            assert src.width == 20
            assert src.height == 20
            assert src.dtypes[0] == "int16"
            data = src.read(1)
            assert data[10, 10] == 150

    def test_dem_85pct_nodata(self) -> None:
        """TC-011: High nodata fixture."""
        import numpy as np

        with open_fixture("dem_85pct_nodata.tif") as src:
            assert src.count == 1
            assert src.width == 20
            assert src.height == 20
            assert src.nodata == -9999.0
            data = src.read(1)
            nodata_pct = np.sum(data == src.nodata) / data.size * 100
            # 20x20 = 400 pixels, 8x8 = 64 valid, 336 nodata = 84%
            assert nodata_pct == pytest.approx(84.0, abs=1.0)

    def test_dem_large(self) -> None:
        """TC-012/TC-018: Large file fixture."""
        with open_fixture("dem_large.tif") as src:
            assert src.count == 1
            assert src.width == 500
            assert src.height == 500
            assert src.crs.to_epsg() == 4326

    def test_dem_extreme_elevations(self) -> None:
        """TC-013: Extreme elevations fixture."""
        import numpy as np

        with open_fixture("dem_extreme_elevations.tif") as src:
            assert src.count == 1
            data = src.read(1)
            assert np.min(data) == -430.0  # Dead Sea
            assert np.max(data) == 8849.0  # Everest

    def test_dem_polar_extreme(self) -> None:
        """TC-016: Polar projection fixture."""
        with open_fixture("dem_polar_extreme.tif") as src:
            assert src.count == 1
            assert src.width == 10
            assert src.height == 10
            assert src.crs.to_epsg() == 3031  # Antarctic

    def test_dem_nan_transform(self) -> None:
        """TC-019: NaN transform fixture (structurally valid)."""
        with open_fixture("dem_nan_transform.tif") as src:
            assert src.count == 1
            assert src.width == 10
            assert src.height == 10


# =============================================================================
# Invalid/Special File Tests
# =============================================================================
class TestSpecialFixtures:
    """Test fixtures that are intentionally invalid or special."""

    def test_empty_tif_is_zero_bytes(self) -> None:
        """TC-010: Empty file should be 0 bytes."""
        path = fixture_path("empty.tif")
        assert path.exists()
        assert path.stat().st_size == 0

    def test_image_png_is_png(self) -> None:
        """TC-014: PNG file should have PNG signature."""
        path = fixture_path("image.png")
        assert path.exists()
        with open(path, "rb") as f:
            header = f.read(8)
        # PNG signature
        assert header[:4] == b"\x89PNG"

    def test_dem_corrupted_has_tiff_header(self) -> None:
        """TC-015: Corrupted file should have TIFF header but be invalid."""
        path = fixture_path("dem_corrupted.tif")
        assert path.exists()
        with open(path, "rb") as f:
            header = f.read(4)
        # TIFF little-endian signature
        assert header[:2] == b"II"
        assert header[2:4] == b"\x2a\x00"  # 42 in little-endian

    @requires_real_rasterio
    def test_dem_corrupted_fails_to_open(self) -> None:
        """TC-015: Corrupted file should fail to open."""
        import rasterio

        with pytest.raises(
            (rasterio.errors.RasterioIOError, rasterio.errors.RasterioError)
        ):
            with open_fixture("dem_corrupted.tif"):
                pass


# =============================================================================
# Summary Test
# =============================================================================
# Use set for efficient membership tests
EXPECTED_SET = set(EXPECTED_FIXTURES)


class TestFixturesSummary:
    """Summary tests for fixture collection."""

    def test_total_fixture_count(self) -> None:
        """Verify we have exactly the expected number of fixtures."""
        files = list(FIXTURES_DIR.iterdir())
        # Filter by membership in EXPECTED_FIXTURES (handles any extension)
        fixture_files = [f for f in files if f.name in EXPECTED_SET]
        if len(fixture_files) != len(EXPECTED_FIXTURES):
            actual_names = {f.name for f in fixture_files}
            missing = EXPECTED_SET - actual_names
            extra = actual_names - EXPECTED_SET
            pytest.fail(
                f"Expected {len(EXPECTED_FIXTURES)} fixtures, found {len(fixture_files)}\n"
                f"Missing: {sorted(missing) or 'none'}\n"
                f"Extra: {sorted(extra) or 'none'}"
            )

    def test_total_size_under_2mb(self) -> None:
        """Verify total fixture size is reasonable."""
        total_size = sum(
            f.stat().st_size for f in FIXTURES_DIR.iterdir() if f.is_file()
        )
        # Should be under 2MB
        assert (
            total_size < 2 * 1024 * 1024
        ), f"Total size: {total_size / 1024 / 1024:.2f}MB"

    def test_fixture_filenames_match_expected(self) -> None:
        """Verify actual fixture filenames match EXPECTED_FIXTURES exactly.

        This catches mismatched names even when counts match (e.g., typos,
        renames without updating the expected list).
        """
        # Filter by membership in EXPECTED_FIXTURES (handles multi-part extensions)
        fixture_files = sorted(
            f.name for f in FIXTURES_DIR.iterdir() if f.name in EXPECTED_SET
        )
        expected = sorted(EXPECTED_FIXTURES)

        found_set = set(fixture_files)

        missing = EXPECTED_SET - found_set
        extra = found_set - EXPECTED_SET

        assert found_set == EXPECTED_SET, (
            f"Fixture filenames mismatch.\n"
            f"Missing (expected but not found): {sorted(missing) or 'none'}\n"
            f"Extra (found but not expected): {sorted(extra) or 'none'}"
        )
