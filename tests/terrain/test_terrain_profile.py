"""Tests for terrain_profile domain service (FEAT-002).

All test coordinates are calibrated to match fixture bounds per FEAT-002 spec v1.3.1.

These tests create TerrainGrids directly with numpy arrays to avoid infrastructure
dependencies (rasterio). This follows DDD principles - domain tests should not
depend on infrastructure adapters.

Fixture Bounds Reference:
- Standard bounds: lat [-25, -15], lon [-50, -40]
- Extended bounds: lat [-26, -14], lon [-51, -39] (for long paths)
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from domain.terrain.value_objects import BoundingBox, GeoPoint, TerrainGrid


# ---------------------------------------------------------------------------
# Test Fixture Helpers - Create TerrainGrids directly (no I/O)
# ---------------------------------------------------------------------------
def create_test_grid_standard() -> TerrainGrid:
    """Create 100x100 test grid with standard bounds.

    Bounds: lat [-25, -15], lon [-50, -40]
    Resolution: 0.1 degrees
    Elevation: gradient 0-1000m
    """
    height, width = 100, 100
    data = np.linspace(0, 1000, height * width, dtype=np.float32).reshape(height, width)

    bounds = BoundingBox(
        min_x=-50.0,  # West
        max_x=-40.0,  # East
        min_y=-25.0,  # South
        max_y=-15.0,  # North
    )

    return TerrainGrid(
        data=data,
        bounds=bounds,
        crs="EPSG:4326",
        resolution=(0.1, 0.1),
        source_crs="EPSG:4326",
    )


def create_test_grid_with_nodata() -> TerrainGrid:
    """Create test grid with diagonal NoData stripe.

    Bounds: lat [-25, -15], lon [-50, -40]
    NoData pattern: diagonal stripe from NW to SE (~10% of pixels)
    """
    height, width = 100, 100
    data = np.linspace(50, 500, height * width, dtype=np.float32).reshape(height, width)

    # Create diagonal NoData stripe (10 pixels wide)
    for i in range(height):
        start_col = max(0, i - 5)
        end_col = min(width, i + 5)
        data[i, start_col:end_col] = np.nan

    bounds = BoundingBox(min_x=-50.0, max_x=-40.0, min_y=-25.0, max_y=-15.0)

    return TerrainGrid(
        data=data,
        bounds=bounds,
        crs="EPSG:4326",
        resolution=(0.1, 0.1),
        source_crs="EPSG:4326",
    )


def create_test_grid_large() -> TerrainGrid:
    """Create 600x600 test grid with extended bounds for long paths.

    Bounds: lat [-26, -14], lon [-51, -39]
    Resolution: 0.02 degrees
    """
    height, width = 600, 600
    rng = np.random.default_rng(42)
    data = (rng.random((height, width)) * 1000).astype(np.float32)

    bounds = BoundingBox(min_x=-51.0, max_x=-39.0, min_y=-26.0, max_y=-14.0)

    return TerrainGrid(
        data=data,
        bounds=bounds,
        crs="EPSG:4326",
        resolution=(0.02, 0.02),
        source_crs="EPSG:4326",
    )


def create_test_grid_known_values() -> TerrainGrid:
    """Create test grid with known values for interpolation verification.

    Bounds: lat [-25, -15], lon [-50, -40]
    """
    height, width = 100, 100
    data = np.linspace(100, 500, height * width, dtype=np.float32).reshape(
        height, width
    )

    # Add known values at specific locations
    data[50, 50] = 150.5
    data[0, 0] = 100.0
    data[99, 99] = 500.0

    bounds = BoundingBox(min_x=-50.0, max_x=-40.0, min_y=-25.0, max_y=-15.0)

    return TerrainGrid(
        data=data,
        bounds=bounds,
        crs="EPSG:4326",
        resolution=(0.1, 0.1),
        source_crs="EPSG:4326",
    )


def create_test_grid_partial_nodata() -> TerrainGrid:
    """Create test grid with east half as NoData.

    Bounds: lat [-25, -15], lon [-50, -40]
    West half: valid elevation 150m
    East half (lon > -45): NoData (NaN)
    """
    height, width = 100, 100
    data = np.full((height, width), 150.0, dtype=np.float32)

    # East half is NoData (columns 50-99)
    data[:, 50:] = np.nan

    bounds = BoundingBox(min_x=-50.0, max_x=-40.0, min_y=-25.0, max_y=-15.0)

    return TerrainGrid(
        data=data,
        bounds=bounds,
        crs="EPSG:4326",
        resolution=(0.1, 0.1),
        source_crs="EPSG:4326",
    )


# ===========================================================================
# TC-001: Happy Path
# ===========================================================================
def test_terrain_profile_basic():
    """TC-001: Basic profile extraction within standard bounds."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    # Bounds: lat [-25, -15], lon [-50, -40]
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.1, longitude=-45.1)

    profile = terrain_profile(grid, start, end)

    assert len(profile.samples) >= 2
    assert profile.samples[0].distance_m == 0
    assert profile.samples[0].point == start
    assert profile.samples[-1].point == end
    assert profile.total_distance_m > 0
    assert profile.step_m > 0
    assert profile.effective_step_m > 0
    assert profile.interpolation == "bilinear"


# ===========================================================================
# TC-002: Distance Accuracy (Geodesic)
# ===========================================================================
def test_terrain_profile_distance_is_geodesic():
    """TC-002: Distance should match pyproj.Geod within 0.001 m (1 mm)."""
    from domain.terrain.services import geodesic_distance, terrain_profile

    grid = create_test_grid_standard()
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.1, longitude=-45.1)

    profile = terrain_profile(grid, start, end)
    expected = geodesic_distance(start, end)

    assert profile.total_distance_m == pytest.approx(expected, abs=0.001)


# ===========================================================================
# TC-003: Start Equals End
# ===========================================================================
def test_terrain_profile_start_equals_end_raises():
    """TC-003: Same start and end should raise InvalidProfileError."""
    from domain.terrain.errors import InvalidProfileError
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    point = GeoPoint(latitude=-20.0, longitude=-45.0)

    with pytest.raises(InvalidProfileError, match="Start equals end"):
        terrain_profile(grid, point, point)


# ===========================================================================
# TC-004: Start Out of Bounds
# ===========================================================================
def test_terrain_profile_start_out_of_bounds():
    """TC-004: Start outside bounds should raise PointOutOfBoundsError."""
    from domain.terrain.errors import PointOutOfBoundsError
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    # Bounds: lat [-25, -15], lon [-50, -40]
    start = GeoPoint(latitude=0.0, longitude=0.0)  # Far outside
    end = GeoPoint(latitude=-20.0, longitude=-45.0)  # Inside

    with pytest.raises(PointOutOfBoundsError) as exc_info:
        terrain_profile(grid, start, end)

    assert exc_info.value.point == start


# ===========================================================================
# TC-005: End Out of Bounds
# ===========================================================================
def test_terrain_profile_end_out_of_bounds():
    """TC-005: End outside bounds should raise PointOutOfBoundsError."""
    from domain.terrain.errors import PointOutOfBoundsError
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    start = GeoPoint(latitude=-20.0, longitude=-45.0)  # Inside
    end = GeoPoint(latitude=0.0, longitude=0.0)  # Far outside

    with pytest.raises(PointOutOfBoundsError) as exc_info:
        terrain_profile(grid, start, end)

    assert exc_info.value.point == end


# ===========================================================================
# TC-006: Path Crosses NoData
# ===========================================================================
def test_terrain_profile_nodata_flagged():
    """TC-006: Profile crossing NoData region should flag affected samples."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_with_nodata()
    # Bounds: lat [-25, -15], lon [-50, -40] (standard bounds)
    # NoData pattern: diagonal stripe from NW to SE (~10% of pixels)
    # Path from NW to SE crosses the diagonal NoData stripe
    start = GeoPoint(latitude=-16.0, longitude=-49.0)  # NW area
    end = GeoPoint(latitude=-24.0, longitude=-41.0)  # SE area

    profile = terrain_profile(grid, start, end)

    # Check explicit flag
    nodata_samples = [s for s in profile.samples if s.is_nodata]
    assert len(nodata_samples) > 0

    # Check consistency
    assert profile.has_nodata is True

    # Check NaN consistency
    for s in nodata_samples:
        assert math.isnan(s.elevation_m)


# ===========================================================================
# TC-007: Very Short Path
# ===========================================================================
def test_terrain_profile_short_path_minimum_samples():
    """TC-007: Very short path should still have minimum 2 samples."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    start = GeoPoint(latitude=-20.000000, longitude=-45.000000)
    end = GeoPoint(latitude=-20.000001, longitude=-45.000001)  # ~0.1m

    profile = terrain_profile(grid, start, end)

    assert len(profile.samples) == 2  # Minimum: start + end


# ===========================================================================
# TC-008: Long Path
# ===========================================================================
@pytest.mark.slow
def test_terrain_profile_long_path():
    """TC-008: Long profile within extended bounds (~450km diagonal)."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_large()
    # Bounds: lat [-26, -14], lon [-51, -39]
    start = GeoPoint(latitude=-15.0, longitude=-50.0)
    end = GeoPoint(latitude=-25.0, longitude=-40.0)  # ~450km diagonal

    profile = terrain_profile(grid, start, end)

    assert len(profile.samples) > 100
    assert profile.total_distance_m > 400_000  # >400km


# ===========================================================================
# TC-009: Samples Strictly Ordered
# ===========================================================================
def test_terrain_profile_samples_strictly_ordered():
    """TC-009: Sample distances must be strictly increasing."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.1, longitude=-45.1)

    profile = terrain_profile(grid, start, end)

    distances = profile.distances()
    for i in range(1, len(distances)):
        assert distances[i] > distances[i - 1]  # Strictly increasing


# ===========================================================================
# TC-010: Bilinear Interpolation
# ===========================================================================
def test_terrain_profile_uses_bilinear():
    """TC-010: Profile should use bilinear interpolation and have valid elevations."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_known_values()
    # Bounds: lat [-25, -15], lon [-50, -40]
    start = GeoPoint(latitude=-20.005, longitude=-45.005)
    end = GeoPoint(latitude=-20.006, longitude=-45.006)

    profile = terrain_profile(grid, start, end)

    assert not math.isnan(profile.samples[0].elevation_m)
    assert profile.interpolation == "bilinear"


# ===========================================================================
# TC-011: Immutability
# ===========================================================================
def test_terrain_profile_is_immutable():
    """TC-011: TerrainProfile should be immutable (frozen Pydantic model)."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.1, longitude=-45.1)

    profile = terrain_profile(grid, start, end)

    with pytest.raises(Exception):  # ValidationError or AttributeError
        profile.total_distance_m = 0


# ===========================================================================
# TC-012: Custom step_m (Oversampling)
# ===========================================================================
def test_terrain_profile_custom_step_oversampling():
    """TC-012: Custom step_m smaller than resolution should work (oversampling)."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.01, longitude=-45.01)  # ~1.5km

    profile = terrain_profile(grid, start, end, step_m=10.0)  # Dense: every 10m

    expected_samples = int(profile.total_distance_m / 10.0) + 1
    assert len(profile.samples) >= expected_samples - 1
    assert profile.step_m == pytest.approx(10.0, rel=0.01)


# ===========================================================================
# TC-013: Custom step_m Larger Than Path
# ===========================================================================
def test_terrain_profile_step_larger_than_path():
    """TC-013: step_m larger than total path should yield minimum 2 samples."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    start = GeoPoint(latitude=-20.000000, longitude=-45.000000)
    end = GeoPoint(latitude=-20.000010, longitude=-45.000010)  # ~1.5m

    profile = terrain_profile(grid, start, end, step_m=1000.0)  # 1km step

    assert len(profile.samples) == 2  # Minimum: start + end


# ===========================================================================
# TC-014: Entire Path is NoData
# ===========================================================================
def test_terrain_profile_all_nodata():
    """TC-014: Profile entirely within NoData region."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_partial_nodata()
    # Bounds: lat [-25, -15], lon [-50, -40]
    # NoData region: eastern half (lon > -45)
    start = GeoPoint(latitude=-20.0, longitude=-44.0)  # In NoData region
    end = GeoPoint(latitude=-20.1, longitude=-43.0)  # In NoData region

    profile = terrain_profile(grid, start, end)

    assert profile.has_nodata is True
    assert profile.nodata_ratio() == 1.0
    assert all(s.is_nodata for s in profile.samples)


# ===========================================================================
# TC-015: nodata_count and nodata_ratio
# ===========================================================================
def test_terrain_profile_nodata_helpers():
    """TC-015: Verify NoData helper methods on a profile crossing NoData stripe."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_with_nodata()
    # Bounds: lat [-25, -15], lon [-50, -40] (standard bounds)
    # NoData pattern: diagonal stripe from NW to SE (~10% coverage)
    # Path crossing the diagonal will hit NoData
    start = GeoPoint(latitude=-16.0, longitude=-49.0)  # NW corner area
    end = GeoPoint(latitude=-24.0, longitude=-41.0)  # SE corner area

    profile = terrain_profile(grid, start, end)

    count = profile.nodata_count()
    ratio = profile.nodata_ratio()

    assert count >= 0
    assert 0.0 <= ratio <= 1.0
    assert count == sum(1 for s in profile.samples if s.is_nodata)
    # This path crosses the diagonal NoData stripe, so some samples should be NoData
    assert profile.has_nodata is True


# ===========================================================================
# TC-016: Point on Grid Corner
# ===========================================================================
def test_terrain_profile_point_on_corner():
    """TC-016: Points exactly on grid boundary should use clamped interpolation."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()

    # Start at exact corner of bounds (southwest corner)
    start = GeoPoint(
        latitude=grid.bounds.min_y, longitude=grid.bounds.min_x  # -25  # -50
    )
    end = GeoPoint(
        latitude=grid.bounds.min_y + 0.01, longitude=grid.bounds.min_x + 0.01
    )

    # Should not raise, should use clamped neighbors
    profile = terrain_profile(grid, start, end)
    assert len(profile.samples) >= 2
    # Corner point degrades to nearest-neighbor (clamped x0==x1, y0==y1)
    assert not math.isnan(profile.samples[0].elevation_m)


# ===========================================================================
# TC-017: High Latitude Step Derivation
# ===========================================================================
def test_terrain_profile_high_latitude_step():
    """TC-017: At high latitudes, step_m should still be derived correctly via geodesic."""
    from domain.terrain.services import MIN_STEP_M, derive_step_m

    # Create mock grid with high-latitude bounds
    mock_grid = MagicMock()
    mock_grid.resolution = (0.001, 0.001)  # ~0.001 degrees
    mock_grid.bounds.min_x = 0
    mock_grid.bounds.max_x = 1
    mock_grid.bounds.min_y = 79.5
    mock_grid.bounds.max_y = 80.5

    start = GeoPoint(latitude=80.0, longitude=0.5)
    end = GeoPoint(latitude=80.0, longitude=0.6)

    step = derive_step_m(mock_grid, start, end)

    # Should be >= MIN_STEP_M even at high latitude
    assert step >= MIN_STEP_M
    # At 80 deg lat, 0.001 deg longitude ~ 19m (not 111m like at equator)
    assert step < 50  # Sanity check


# ===========================================================================
# TC-018: All Grid Boundary Edges
# ===========================================================================
def test_terrain_profile_boundary_edges():
    """TC-018: Profiles along each edge of the grid should work."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    # Bounds: lat [-25, -15], lon [-50, -40]

    # South edge (min_y)
    profile_south = terrain_profile(
        grid,
        GeoPoint(latitude=-25.0, longitude=-48.0),
        GeoPoint(latitude=-25.0, longitude=-42.0),
    )
    assert len(profile_south.samples) >= 2

    # North edge (max_y)
    profile_north = terrain_profile(
        grid,
        GeoPoint(latitude=-15.0, longitude=-48.0),
        GeoPoint(latitude=-15.0, longitude=-42.0),
    )
    assert len(profile_north.samples) >= 2


# ===========================================================================
# TC-019: Effective Step Calculation
# ===========================================================================
def test_terrain_profile_effective_step():
    """TC-019: effective_step_m should equal total_distance / (n_samples - 1)."""
    from domain.terrain.services import terrain_profile

    grid = create_test_grid_standard()
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.1, longitude=-45.1)

    profile = terrain_profile(grid, start, end)

    expected_effective = profile.total_distance_m / (len(profile.samples) - 1)
    assert profile.effective_step_m == pytest.approx(expected_effective, rel=0.01)


# ===========================================================================
# Additional: ProfileSample Invariant Tests (Value Object validation)
# ===========================================================================
class TestProfileSampleInvariants:
    """Tests for ProfileSample Value Object invariants."""

    def test_nodata_true_requires_nan_elevation(self):
        """PS-2: is_nodata=True requires elevation_m=NaN."""
        from domain.terrain.value_objects import GeoPoint, ProfileSample

        with pytest.raises(ValueError, match="is_nodata=True requires elevation_m=NaN"):
            ProfileSample(
                distance_m=0.0,
                elevation_m=100.0,  # Valid elevation
                point=GeoPoint(latitude=0.0, longitude=0.0),
                is_nodata=True,  # Inconsistent
            )

    def test_nodata_false_requires_finite_elevation(self):
        """PS-3: is_nodata=False requires finite elevation_m."""
        from domain.terrain.value_objects import GeoPoint, ProfileSample

        with pytest.raises(
            ValueError, match="is_nodata=False requires finite elevation_m"
        ):
            ProfileSample(
                distance_m=0.0,
                elevation_m=float("nan"),  # NaN elevation
                point=GeoPoint(latitude=0.0, longitude=0.0),
                is_nodata=False,  # Inconsistent
            )

    def test_valid_nodata_sample(self):
        """Valid NoData sample should be created successfully."""
        from domain.terrain.value_objects import GeoPoint, ProfileSample

        sample = ProfileSample(
            distance_m=100.0,
            elevation_m=float("nan"),
            point=GeoPoint(latitude=-20.0, longitude=-45.0),
            is_nodata=True,
        )
        assert sample.is_nodata is True
        assert math.isnan(sample.elevation_m)

    def test_valid_data_sample(self):
        """Valid data sample should be created successfully."""
        from domain.terrain.value_objects import GeoPoint, ProfileSample

        sample = ProfileSample(
            distance_m=100.0,
            elevation_m=500.5,
            point=GeoPoint(latitude=-20.0, longitude=-45.0),
            is_nodata=False,
        )
        assert sample.is_nodata is False
        assert sample.elevation_m == 500.5


# ===========================================================================
# Additional: TerrainProfile Invariant Tests (Value Object validation)
# ===========================================================================
class TestTerrainProfileInvariants:
    """Tests for TerrainProfile Value Object invariants."""

    def test_minimum_two_samples(self):
        """TP-1: Profile must have at least 2 samples."""
        from domain.terrain.value_objects import GeoPoint, ProfileSample, TerrainProfile

        start = GeoPoint(latitude=-20.0, longitude=-45.0)
        end = GeoPoint(latitude=-20.1, longitude=-45.1)

        # Only 1 sample - should fail
        single_sample = ProfileSample(
            distance_m=0.0, elevation_m=100.0, point=start, is_nodata=False
        )

        with pytest.raises(ValueError, match="Profile must have >= 2 samples"):
            TerrainProfile(
                start=start,
                end=end,
                samples=(single_sample,),
                total_distance_m=1000.0,
                step_m=100.0,
                effective_step_m=1000.0,
                has_nodata=False,
            )

    def test_first_sample_at_distance_zero(self):
        """TP-2: First sample must be at distance 0."""
        from domain.terrain.value_objects import GeoPoint, ProfileSample, TerrainProfile

        start = GeoPoint(latitude=-20.0, longitude=-45.0)
        end = GeoPoint(latitude=-20.1, longitude=-45.1)

        sample1 = ProfileSample(
            distance_m=100.0,  # Wrong - should be 0
            elevation_m=100.0,
            point=start,
            is_nodata=False,
        )
        sample2 = ProfileSample(
            distance_m=1000.0, elevation_m=200.0, point=end, is_nodata=False
        )

        with pytest.raises(ValueError, match="First sample must be at distance 0"):
            TerrainProfile(
                start=start,
                end=end,
                samples=(sample1, sample2),
                total_distance_m=1000.0,
                step_m=100.0,
                effective_step_m=1000.0,
                has_nodata=False,
            )

    def test_first_sample_point_equals_start(self):
        """TP-5: First sample point must equal start."""
        from domain.terrain.value_objects import GeoPoint, ProfileSample, TerrainProfile

        start = GeoPoint(latitude=-20.0, longitude=-45.0)
        wrong_start = GeoPoint(latitude=-19.0, longitude=-44.0)
        end = GeoPoint(latitude=-20.1, longitude=-45.1)

        sample1 = ProfileSample(
            distance_m=0.0,
            elevation_m=100.0,
            point=wrong_start,  # Wrong point
            is_nodata=False,
        )
        sample2 = ProfileSample(
            distance_m=1000.0, elevation_m=200.0, point=end, is_nodata=False
        )

        with pytest.raises(ValueError, match="First sample point must equal start"):
            TerrainProfile(
                start=start,
                end=end,
                samples=(sample1, sample2),
                total_distance_m=1000.0,
                step_m=100.0,
                effective_step_m=1000.0,
                has_nodata=False,
            )


# ===========================================================================
# Additional: GeoPoint Invariant Tests (Value Object validation)
# ===========================================================================
class TestGeoPointInvariants:
    """Tests for GeoPoint Value Object invariants."""

    def test_latitude_out_of_range_low(self):
        """GP-1: Latitude below -90 should fail."""
        from domain.terrain.value_objects import GeoPoint

        with pytest.raises(ValueError):
            GeoPoint(latitude=-91.0, longitude=0.0)

    def test_latitude_out_of_range_high(self):
        """GP-1: Latitude above 90 should fail."""
        from domain.terrain.value_objects import GeoPoint

        with pytest.raises(ValueError):
            GeoPoint(latitude=91.0, longitude=0.0)

    def test_longitude_out_of_range_low(self):
        """GP-2: Longitude below -180 should fail."""
        from domain.terrain.value_objects import GeoPoint

        with pytest.raises(ValueError):
            GeoPoint(latitude=0.0, longitude=-181.0)

    def test_longitude_out_of_range_high(self):
        """GP-2: Longitude above 180 should fail."""
        from domain.terrain.value_objects import GeoPoint

        with pytest.raises(ValueError):
            GeoPoint(latitude=0.0, longitude=181.0)

    def test_geopoint_equality_by_value(self):
        """GeoPoints with same values should be equal."""
        from domain.terrain.value_objects import GeoPoint

        p1 = GeoPoint(latitude=-20.0, longitude=-45.0)
        p2 = GeoPoint(latitude=-20.0, longitude=-45.0)

        assert p1 == p2

    def test_geopoint_immutable(self):
        """GeoPoint should be immutable (frozen)."""
        from domain.terrain.value_objects import GeoPoint

        point = GeoPoint(latitude=-20.0, longitude=-45.0)

        with pytest.raises(Exception):  # ValidationError or AttributeError
            point.latitude = 0.0


# ===========================================================================
# Additional: Precondition Tests
# ===========================================================================
class TestTerrainProfilePreconditions:
    """Tests for terrain_profile preconditions."""

    def test_negative_step_m_raises(self):
        """PRE-4: Negative step_m should raise ValueError."""
        from domain.terrain.services import terrain_profile

        grid = create_test_grid_standard()
        start = GeoPoint(latitude=-20.0, longitude=-45.0)
        end = GeoPoint(latitude=-20.1, longitude=-45.1)

        with pytest.raises(ValueError, match="step_m must be positive"):
            terrain_profile(grid, start, end, step_m=-10.0)

    def test_zero_step_m_raises(self):
        """PRE-4: Zero step_m should raise ValueError."""
        from domain.terrain.services import terrain_profile

        grid = create_test_grid_standard()
        start = GeoPoint(latitude=-20.0, longitude=-45.0)
        end = GeoPoint(latitude=-20.1, longitude=-45.1)

        with pytest.raises(ValueError, match="step_m must be positive"):
            terrain_profile(grid, start, end, step_m=0.0)
