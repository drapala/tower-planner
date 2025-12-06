"""Sanity tests for FEAT-001 test fixtures.

These tests validate that each fixture:
1. Exists on disk
2. Can be opened by rasterio (where applicable)
3. Has expected shape and CRS (where applicable)

These are NOT behavioral tests - they only verify fixture integrity.
Behavioral tests are in test_geotiff_adapter.py.

Requires: pip install rasterio numpy pytest
Run: pytest tests/gis/test_fixtures_sanity.py -m integration

Note: These tests require REAL rasterio, not the stub in src/rasterio/.
      Run with: PYTHONPATH= pytest tests/gis/test_fixtures_sanity.py -m integration
      Or ensure real rasterio is installed and takes precedence.
"""

from pathlib import Path

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _is_real_rasterio() -> bool:
    """Check if we have real rasterio (not the stub).

    TEMPORARY WORKAROUND: This detection logic uses __version__ and path
    heuristics because the project has a stub in src/rasterio/ that shadows
    the real package. Once TD-002 is implemented (moving stubs to tests/stubs/),
    this function can be simplified to a plain import check:

        try:
            import rasterio
            return True
        except ImportError:
            return False

    See: spec/tech-debt/FEAT-001-tech-debt.md (TD-002)
    """
    try:
        import rasterio

        # Real rasterio has __version__ attribute; stub does not
        if not hasattr(rasterio, "__version__"):
            return False

        # Check that it's not from our src/ stub directory
        rasterio_path = Path(rasterio.__file__).resolve()
        src_stub_path = (Path(__file__).parent.parent.parent / "src" / "rasterio").resolve()
        return not str(rasterio_path).startswith(str(src_stub_path))
    except ImportError:
        return False


# Skip tests that need real rasterio if we only have the stub
requires_real_rasterio = pytest.mark.skipif(
    not _is_real_rasterio(),
    reason="Requires real rasterio (not stub). Install rasterio: pip install rasterio",
)


# =============================================================================
# Helper
# =============================================================================
def fixture_path(name: str) -> Path:
    """Get path to a fixture file."""
    return FIXTURES_DIR / name


# =============================================================================
# Existence Tests
# =============================================================================
class TestFixturesExist:
    """Verify all required fixtures exist."""

    @pytest.mark.parametrize(
        "filename",
        [
            "dem_100x100_4326.tif",
            "dem_utm23s.tif",
            "dem_with_nodata.tif",
            "dem_all_nodata.tif",
            "dem_no_crs.tif",
            "rgb_image.tif",
            "dem_known_values_4326.tif",
            "dem_known_values_4326_int16.tif",
            "empty.tif",
            "dem_85pct_nodata.tif",
            "dem_large.tif",
            "dem_extreme_elevations.tif",
            "image.png",
            "dem_corrupted.tif",
            "dem_polar_extreme.tif",
            "dem_nan_transform.tif",
        ],
    )
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
        import rasterio

        path = fixture_path("dem_100x100_4326.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 100
            assert src.height == 100
            assert src.crs is not None
            assert src.crs.to_epsg() == 4326
            assert src.dtypes[0] == "float32"

    def test_dem_utm23s(self) -> None:
        """TC-002: UTM reprojection fixture."""
        import rasterio

        path = fixture_path("dem_utm23s.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 50
            assert src.height == 50
            assert src.crs is not None
            assert src.crs.to_epsg() == 31983
            assert src.dtypes[0] == "float32"

    def test_dem_with_nodata(self) -> None:
        """TC-004: NoData fixture."""
        import rasterio

        path = fixture_path("dem_with_nodata.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 20
            assert src.height == 20
            assert src.nodata == -9999.0

    def test_dem_all_nodata(self) -> None:
        """TC-005: All nodata fixture."""
        import rasterio

        path = fixture_path("dem_all_nodata.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 10
            assert src.height == 10
            assert src.nodata == -9999.0

    def test_dem_no_crs(self) -> None:
        """TC-006: Missing CRS fixture."""
        import rasterio

        path = fixture_path("dem_no_crs.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 10
            assert src.height == 10
            assert src.crs is None

    def test_rgb_image(self) -> None:
        """TC-007: Multi-band fixture."""
        import rasterio

        path = fixture_path("rgb_image.tif")
        with rasterio.open(path) as src:
            assert src.count == 3  # RGB
            assert src.width == 10
            assert src.height == 10

    def test_dem_known_values_4326(self) -> None:
        """TC-009: Known values fixture (float32)."""
        import numpy as np
        import rasterio

        path = fixture_path("dem_known_values_4326.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 20
            assert src.height == 20
            assert src.dtypes[0] == "float32"
            data = src.read(1)
            assert data[10, 10] == np.float32(150.5)

    def test_dem_known_values_4326_int16(self) -> None:
        """TC-009b: Known values fixture (int16)."""
        import rasterio

        path = fixture_path("dem_known_values_4326_int16.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 20
            assert src.height == 20
            assert src.dtypes[0] == "int16"
            data = src.read(1)
            assert data[10, 10] == 150

    def test_dem_85pct_nodata(self) -> None:
        """TC-011: High nodata fixture."""
        import numpy as np
        import rasterio

        path = fixture_path("dem_85pct_nodata.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 20
            assert src.height == 20
            assert src.nodata == -9999.0
            data = src.read(1)
            nodata_pct = np.sum(data == src.nodata) / data.size * 100
            assert nodata_pct > 80  # Should be ~84%

    def test_dem_large(self) -> None:
        """TC-012/TC-018: Large file fixture."""
        import rasterio

        path = fixture_path("dem_large.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 500
            assert src.height == 500
            assert src.crs.to_epsg() == 4326

    def test_dem_extreme_elevations(self) -> None:
        """TC-013: Extreme elevations fixture."""
        import numpy as np
        import rasterio

        path = fixture_path("dem_extreme_elevations.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            data = src.read(1)
            assert np.min(data) == -430.0  # Dead Sea
            assert np.max(data) == 8849.0  # Everest

    def test_dem_polar_extreme(self) -> None:
        """TC-016: Polar projection fixture."""
        import rasterio

        path = fixture_path("dem_polar_extreme.tif")
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.width == 10
            assert src.height == 10
            assert src.crs.to_epsg() == 3031  # Antarctic

    def test_dem_nan_transform(self) -> None:
        """TC-019: NaN transform fixture (structurally valid)."""
        import rasterio

        path = fixture_path("dem_nan_transform.tif")
        with rasterio.open(path) as src:
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

        path = fixture_path("dem_corrupted.tif")
        with pytest.raises((rasterio.errors.RasterioIOError, rasterio.errors.RasterioError)):
            with rasterio.open(path):
                pass


# =============================================================================
# Summary Test
# =============================================================================
class TestFixturesSummary:
    """Summary tests for fixture collection."""

    def test_total_fixture_count(self) -> None:
        """Verify we have exactly 16 fixtures."""
        files = list(FIXTURES_DIR.iterdir())
        # Exclude README.md
        fixture_files = [f for f in files if f.suffix in (".tif", ".png")]
        assert len(fixture_files) == 16

    def test_total_size_under_2mb(self) -> None:
        """Verify total fixture size is reasonable."""
        total_size = sum(f.stat().st_size for f in FIXTURES_DIR.iterdir() if f.is_file())
        # Should be under 2MB
        assert total_size < 2 * 1024 * 1024, f"Total size: {total_size / 1024 / 1024:.2f}MB"
