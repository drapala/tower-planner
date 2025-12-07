# FEAT-002: TerrainProfile

**Status**: Draft
**BC**: terrain
**Priority**: P0 (Foundation)
**Version**: 1.3.1
**Depends on**: FEAT-001 (LoadDEM)

---

## Summary

Extract an elevation profile between two geographic points over a TerrainGrid, returning a sequence of samples with distance, elevation, coordinates, and NoData flags.

> **Layer**: Domain Service (pure computation, no I/O)
> **Location**: `domain/terrain/services.py`

---

## Function Signature

```python
def terrain_profile(
    grid: TerrainGrid,
    start: GeoPoint,
    end: GeoPoint,
    step_m: float | None = None,
) -> TerrainProfile:
    ...
```

---

## Input

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `grid` | `TerrainGrid` | Yes | Elevation grid (from FEAT-001) |
| `start` | `GeoPoint` | Yes | Starting point (typically Tx location) |
| `end` | `GeoPoint` | Yes | Ending point (typically Rx location) |
| `step_m` | `float \| None` | No | Sample spacing in meters. If None, derived from grid resolution |

### Step Derivation (when `step_m` is None)

```python
from pyproj import Geod

_geod = Geod(ellps="WGS84")

MIN_STEP_M = 1.0  # Floor to avoid zero/tiny steps at extreme latitudes

def derive_step_m(grid: TerrainGrid, start: GeoPoint, end: GeoPoint) -> float:
    """Derive step size from grid resolution using geodesic measurement.
    
    Measures actual ground distance of one grid cell at the path midpoint,
    avoiding the inaccuracy of cos(lat) approximation.
    
    Returns:
        Step size in meters, minimum MIN_STEP_M (1m).
    """
    mid = GeoPoint(
        latitude=(start.latitude + end.latitude) / 2,
        longitude=(start.longitude + end.longitude) / 2,
    )
    
    # Measure x resolution: one grid cell east
    x_neighbor = GeoPoint(
        latitude=mid.latitude,
        longitude=mid.longitude + grid.resolution[0]
    )
    _, _, x_res_m = _geod.inv(
        mid.longitude, mid.latitude,
        x_neighbor.longitude, x_neighbor.latitude
    )
    
    # Measure y resolution: one grid cell south
    y_neighbor = GeoPoint(
        latitude=mid.latitude - grid.resolution[1],
        longitude=mid.longitude
    )
    _, _, y_res_m = _geod.inv(
        mid.longitude, mid.latitude,
        y_neighbor.longitude, y_neighbor.latitude
    )
    
    # Use finer resolution, but never below minimum
    return max(MIN_STEP_M, min(abs(x_res_m), abs(y_res_m)))
```

> **Why geodesic instead of cos(lat)?**
> - `cos(90°) = 0` would produce `step_m = 0` at poles
> - Geodesic measurement is exact, no approximation error
> - pyproj.Geod is already a dependency
> - `MIN_STEP_M` floor prevents degenerate cases

---

## Output

Returns: `TerrainProfile` (see definition below)

---

## Numeric Constants

```python
# Tolerances for floating-point comparisons
DISTANCE_TOLERANCE_M = 0.1      # 10 cm - for distance invariants

MIN_STEP_M = 1.0                # Minimum step size in meters
```

> **Why these values?**
> - `DISTANCE_TOLERANCE_M = 0.1`: Allows for floating-point accumulation over long paths
> - `MIN_STEP_M = 1.0`: Prevents degenerate cases at extreme latitudes
>
> Note on coordinate comparisons: Pydantic frozen models compare nested value
> objects by value; explicit degree tolerances are not required in validators.
> If a future test needs looser comparison for coordinates produced by
> geodesic interpolation, specify the tolerance directly in that test.

---

## GeoPoint Definition

> **Note**: If `GeoPoint` already exists in `domain/terrain/value_objects.py` from FEAT-001,
> **reuse it** instead of redefining. The definition below is for reference.

> **DDD Classification**: Value Object (immutable)
> **Location**: `domain/terrain/value_objects.py`

```python
from pydantic import BaseModel, ConfigDict, Field

class GeoPoint(BaseModel):
    """Geographic coordinate in WGS84 (Value Object)."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    model_config = ConfigDict(frozen=True)
```

### GeoPoint Invariants

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| GP-1 | `latitude` in `[-90, 90]` | Pydantic Field constraint |
| GP-2 | `longitude` in `[-180, 180]` | Pydantic Field constraint |

> **Note on __eq__ and __hash__**: Pydantic frozen models compare by value automatically.
> Verified: `GeoPoint(lat=1, lon=2) == GeoPoint(lat=1, lon=2)` returns `True`.

---

## ProfileSample Definition

> **DDD Classification**: Value Object (immutable)
> **Location**: `domain/terrain/value_objects.py`

```python
import math
from pydantic import BaseModel, ConfigDict, Field, model_validator

class ProfileSample(BaseModel):
    """Single sample point along a terrain profile (Value Object)."""

    distance_m: float = Field(ge=0)   # Cumulative distance from start in meters
    elevation_m: float                 # Elevation at this point (NaN if nodata)
    point: GeoPoint                    # Geographic location of sample
    is_nodata: bool = False            # Explicit flag for NoData regions

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_nodata_consistency(self) -> "ProfileSample":
        # Use math.isnan to avoid numpy dependency in simple VO
        if self.is_nodata and not math.isnan(self.elevation_m):
            raise ValueError("is_nodata=True requires elevation_m=NaN")
        if not self.is_nodata and math.isnan(self.elevation_m):
            raise ValueError("is_nodata=False requires finite elevation_m")
        return self
```

### ProfileSample Invariants

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| PS-1 | `distance_m >= 0` | Pydantic Field constraint |
| PS-2 | If `is_nodata == True`, then `elevation_m` is NaN | Validator (math.isnan) |
| PS-3 | If `is_nodata == False`, then `elevation_m` is finite | Validator (math.isnan) |
| PS-4 | `point` is valid GeoPoint | Nested validation |

> **Why math.isnan instead of np.isnan?**
> ProfileSample is a simple VO. Using `math.isnan` avoids importing numpy
> just for a boolean check, keeping the domain layer lightweight.

---

## TerrainProfile Definition

> **DDD Classification**: Value Object (immutable)
> **Location**: `domain/terrain/value_objects.py`

```python
from pydantic import BaseModel, ConfigDict, Field, model_validator

DISTANCE_TOLERANCE_M = 0.1  # 10 cm

class TerrainProfile(BaseModel):
    """Elevation profile between two points (Value Object)."""

    start: GeoPoint                        # Profile origin
    end: GeoPoint                          # Profile destination
    samples: tuple[ProfileSample, ...]     # Ordered samples from start to end
    total_distance_m: float = Field(gt=0)  # Total path length in meters
    step_m: float = Field(gt=0)            # Requested/derived sample spacing
    effective_step_m: float = Field(gt=0)  # Actual average spacing: total/(n-1)
    has_nodata: bool                       # True if any sample is nodata
    interpolation: str = "bilinear"        # Method used (metadata for audit)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_profile(self) -> "TerrainProfile":
        # TP-1: At least 2 samples (start and end)
        if len(self.samples) < 2:
            raise ValueError(f"Profile must have >= 2 samples, got {len(self.samples)}")
        
        # TP-2: First sample at distance 0
        if self.samples[0].distance_m != 0:
            raise ValueError(f"First sample must be at distance 0, got {self.samples[0].distance_m}")
        
        # TP-3: Samples ordered by distance (strictly increasing)
        for i in range(1, len(self.samples)):
            if self.samples[i].distance_m <= self.samples[i-1].distance_m:
                raise ValueError("Samples must be strictly ordered by distance")
        
        # TP-4: Last sample distance equals total_distance_m (within tolerance)
        if abs(self.samples[-1].distance_m - self.total_distance_m) > DISTANCE_TOLERANCE_M:
            raise ValueError(
                f"Last sample distance ({self.samples[-1].distance_m:.3f}) must equal "
                f"total_distance_m ({self.total_distance_m:.3f}) within {DISTANCE_TOLERANCE_M}m"
            )
        
        # TP-5: First sample point equals start (Pydantic frozen uses value equality)
        if self.samples[0].point != self.start:
            raise ValueError("First sample point must equal start")
        
        # TP-6: Last sample point equals end
        if self.samples[-1].point != self.end:
            raise ValueError("Last sample point must equal end")
        
        # TP-7: has_nodata consistency
        actual_has_nodata = any(s.is_nodata for s in self.samples)
        if self.has_nodata != actual_has_nodata:
            raise ValueError(f"has_nodata={self.has_nodata} but samples say {actual_has_nodata}")
        
        # TP-8: effective_step_m consistency
        expected_effective = self.total_distance_m / (len(self.samples) - 1)
        if abs(self.effective_step_m - expected_effective) > DISTANCE_TOLERANCE_M:
            raise ValueError(
                f"effective_step_m ({self.effective_step_m:.3f}) must equal "
                f"total/(n-1) ({expected_effective:.3f})"
            )
        
        return self

    def elevations(self) -> tuple[float, ...]:
        """Return elevation values (may contain NaN)."""
        return tuple(s.elevation_m for s in self.samples)

    def distances(self) -> tuple[float, ...]:
        """Return cumulative distance values."""
        return tuple(s.distance_m for s in self.samples)
    
    def nodata_count(self) -> int:
        """Return number of NoData samples."""
        return sum(1 for s in self.samples if s.is_nodata)
    
    def nodata_ratio(self) -> float:
        """Return fraction of samples that are NoData (0.0 to 1.0)."""
        return self.nodata_count() / len(self.samples)
```

### TerrainProfile Invariants

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| TP-1 | `len(samples) >= 2` | Validator |
| TP-2 | `samples[0].distance_m == 0` | Validator |
| TP-3 | Samples strictly ordered by `distance_m` | Validator |
| TP-4 | `samples[-1].distance_m ≈ total_distance_m` (±0.1m) | Validator |
| TP-5 | `samples[0].point == start` | Validator (Pydantic value equality) |
| TP-6 | `samples[-1].point == end` | Validator (Pydantic value equality) |
| TP-7 | `has_nodata` matches actual samples | Validator |
| TP-8 | `effective_step_m ≈ total/(n-1)` (±0.1m) | Validator |
| TP-9 | `total_distance_m > 0` | Field constraint |
| TP-10 | `step_m > 0` | Field constraint |
| TP-11 | `effective_step_m > 0` | Field constraint |

> **step_m vs effective_step_m**:
> - `step_m`: The requested or derived spacing (input parameter)
> - `effective_step_m`: The actual average spacing = `total_distance_m / (n_samples - 1)`
> - These may differ slightly due to integer sample count

---

## Preconditions

| ID | Condition | Error |
|----|-----------|-------|
| PRE-1 | `start` is within `grid.bounds` | `PointOutOfBoundsError` |
| PRE-2 | `end` is within `grid.bounds` | `PointOutOfBoundsError` |
| PRE-3 | `start != end` (non-zero distance) | `InvalidProfileError("Start equals end")` |
| PRE-4 | If provided, `step_m > 0` | `ValueError("step_m must be positive")` |

### Bounds Check Implementation

```python
def is_within_bounds(point: GeoPoint, bounds: BoundingBox) -> bool:
    """Check if point is within bounds (inclusive).
    
    Uses BoundingBox from FEAT-001:
    - bounds.min_x = western boundary (longitude)
    - bounds.max_x = eastern boundary (longitude)
    - bounds.min_y = southern boundary (latitude)
    - bounds.max_y = northern boundary (latitude)
    """
    return (
        bounds.min_x <= point.longitude <= bounds.max_x
        and bounds.min_y <= point.latitude <= bounds.max_y
    )
```

---

## Postconditions

| ID | Condition | Validation |
|----|-----------|------------|
| POST-1 | `result.samples[0].point == start` | First sample at start point |
| POST-2 | `result.samples[-1].point == end` | Last sample at end point |
| POST-3 | `result.total_distance_m` is geodesic distance (WGS84) | pyproj.Geod.inv |
| POST-4 | All intermediate points lie on geodesic path | **pyproj.Geod.npts** |
| POST-5 | Sample spacing ≈ `step_m` | Within 1 sample tolerance |
| POST-6 | Elevations use bilinear interpolation | 4-neighbor weighted average |
| POST-7 | NoData regions have `is_nodata=True` and `elevation_m=NaN` | Explicit flagging |
| POST-8 | `result.has_nodata` is True iff any sample is NoData | Consistency |
| POST-9 | No infill/interpolation over NoData gaps | Conservative policy |

> **POST-9 Clarification**: The profile does NOT interpolate over NoData gaps.
> If any of the 4 bilinear neighbors is NaN, the sample is marked `is_nodata=True`.
> FEAT-003 (LoS) will decide how to handle gaps (obstruction vs ignore vs error).

---

## Algorithm Overview

```
1. Validate preconditions (bounds, start != end, step_m)
2. Calculate total geodesic distance using pyproj.Geod.inv()
3. Derive step_m if not provided (geodesic measurement of grid cell)
4. Calculate number of samples: n = max(2, floor(total_distance / step_m) + 1)
5. Calculate effective_step_m = total_distance / (n - 1)
6. Generate intermediate points using pyproj.Geod.npts()
7. For each sample point:
   a. Bilinear interpolate elevation from grid
   b. Check if result is NaN → set is_nodata flag
   c. Calculate cumulative distance from start
8. Build TerrainProfile with all samples and metadata
9. Return immutable TerrainProfile VO
```

### Distance and Path Interpolation (Geodesic — pyproj)

```python
from pyproj import Geod

# WGS84 ellipsoid (same as GPS, EPSG:4326)
_geod = Geod(ellps="WGS84")

def geodesic_distance(start: GeoPoint, end: GeoPoint) -> float:
    """Calculate geodesic distance between two points in meters.
    
    Uses WGS84 ellipsoid for millimeter-level precision.
    """
    _, _, distance = _geod.inv(
        start.longitude, start.latitude,
        end.longitude, end.latitude
    )
    return abs(distance)  # Ensure positive


def interpolate_geodesic_path(
    start: GeoPoint, 
    end: GeoPoint, 
    num_intermediate: int
) -> list[GeoPoint]:
    """Interpolate points along geodesic path.
    
    Args:
        start: Starting point
        end: Ending point
        num_intermediate: Number of points BETWEEN start and end
    
    Returns:
        List of all points: [start, ...intermediate..., end]
        
    Note: Returns list (mutable) during construction; 
          TerrainProfile stores as tuple (immutable).
    """
    if num_intermediate <= 0:
        return [start, end]
    
    # npts returns intermediate points (excludes endpoints)
    intermediate = _geod.npts(
        start.longitude, start.latitude,
        end.longitude, end.latitude,
        num_intermediate
    )
    
    # Build full list with endpoints
    result = [start]
    for lon, lat in intermediate:
        result.append(GeoPoint(latitude=lat, longitude=lon))
    result.append(end)
    
    return result
```

### Bilinear Interpolation

```python
import math
import numpy as np

def bilinear_interpolate(
    grid: TerrainGrid, 
    point: GeoPoint
) -> tuple[float, bool]:
    """Interpolate elevation at arbitrary point using 4 nearest pixels.
    
    Returns (elevation, is_nodata).
    If any of the 4 neighbors is NaN, returns (NaN, True).
    
    Boundary behavior:
        Points exactly on grid boundaries use clamped indices.
        This effectively duplicates edge pixels, causing bilinear
        to degrade to linear (on edges) or nearest (on corners).
        This is intentional — the alternative would require shrinking
        valid bounds by half a pixel, complicating the API.
    """
    # Convert point to fractional pixel coordinates
    # Note: row 0 = north edge (max_y), so y is inverted
    px = (point.longitude - grid.bounds.min_x) / grid.resolution[0]
    py = (grid.bounds.max_y - point.latitude) / grid.resolution[1]

    height, width = grid.data.shape

    # Get integer pixel indices
    x0 = int(math.floor(px))
    y0 = int(math.floor(py))
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp to valid grid indices
    # When x0 == x1 or y0 == y1, bilinear degrades to linear/nearest
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))

    # Get 4 corner values
    q11 = float(grid.data[y0, x0])  # top-left
    q21 = float(grid.data[y0, x1])  # top-right
    q12 = float(grid.data[y1, x0])  # bottom-left
    q22 = float(grid.data[y1, x1])  # bottom-right

    # If any neighbor is NaN, return NaN with nodata flag (no infill)
    if math.isnan(q11) or math.isnan(q21) or math.isnan(q12) or math.isnan(q22):
        return (float('nan'), True)

    # Bilinear weights (fractional part)
    fx = px - math.floor(px)
    fy = py - math.floor(py)

    # Bilinear interpolation formula
    elevation = (
        q11 * (1 - fx) * (1 - fy) +
        q21 * fx * (1 - fy) +
        q12 * (1 - fx) * fy +
        q22 * fx * fy
    )

    return (float(elevation), False)
```

---

## Error Hierarchy

```python
# Extends TerrainError from FEAT-001

class PointOutOfBoundsError(TerrainError):
    """Point is outside the terrain grid bounds.
    
    Attributes:
        point: The offending GeoPoint
        bounds: The grid's BoundingBox
    """
    def __init__(self, point: GeoPoint, bounds: BoundingBox):
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
```

---

## Edge Cases

| Case | Behavior | Test |
|------|----------|------|
| Start == End | Raise `InvalidProfileError` | TC-003 |
| Start outside bounds | Raise `PointOutOfBoundsError` | TC-004 |
| End outside bounds | Raise `PointOutOfBoundsError` | TC-005 |
| Path crosses NoData region | Samples have `is_nodata=True`, `elevation_m=NaN` | TC-006 |
| Very short path (< step_m) | Minimum 2 samples (start + end) | TC-007 |
| Very long path (100+ km) | Works, many samples | TC-008 |
| Path along grid edge | Valid, uses clamped edge pixels | TC-009 |
| Diagonal path | Correct geodesic distance | TC-010 |
| Custom step_m smaller than resolution | Allowed (oversampling) | TC-012 |
| Custom step_m larger than path | Results in 2 samples | TC-013 |
| Entire path is NoData | All samples `is_nodata=True`, `has_nodata=True` | TC-014 |
| Point exactly on grid corner | Uses clamped neighbors (degrades to nearest) | TC-016 |
| High latitude (near poles) | step_m derived correctly via geodesic | TC-017 |
| Grid boundary points | Inclusive, uses clamped interpolation | TC-018 |

---

## Test Cases

> **IMPORTANT**: All test coordinates must be within the bounds of the fixture used.
> See "Appendix: Fixture Bounds Reference" for exact bounds of each fixture.

### TC-001: Happy Path
```python
def test_terrain_profile_basic():
    """Basic profile extraction within dem_100x100_4326 bounds."""
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
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
```

### TC-002: Distance Accuracy (Geodesic)
```python
def test_terrain_profile_distance_is_geodesic():
    """Distance should match pyproj.Geod within 0.1 m (10 cm)."""
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.1, longitude=-45.1)

    profile = terrain_profile(grid, start, end)
    expected = geodesic_distance(start, end)

    assert profile.total_distance_m == pytest.approx(expected, abs=0.001)
```

### TC-003: Start Equals End
```python
def test_terrain_profile_start_equals_end_raises():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    point = GeoPoint(latitude=-20.0, longitude=-45.0)

    with pytest.raises(InvalidProfileError, match="Start equals end"):
        terrain_profile(grid, point, point)
```

### TC-004: Start Out of Bounds
```python
def test_terrain_profile_start_out_of_bounds():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    # Bounds: lat [-25, -15], lon [-50, -40]
    start = GeoPoint(latitude=0, longitude=0)  # Far outside
    end = GeoPoint(latitude=-20.0, longitude=-45.0)  # Inside

    with pytest.raises(PointOutOfBoundsError) as exc_info:
        terrain_profile(grid, start, end)
    
    assert exc_info.value.point == start
```

### TC-005: End Out of Bounds
```python
def test_terrain_profile_end_out_of_bounds():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    start = GeoPoint(latitude=-20.0, longitude=-45.0)  # Inside
    end = GeoPoint(latitude=0, longitude=0)  # Far outside

    with pytest.raises(PointOutOfBoundsError) as exc_info:
        terrain_profile(grid, start, end)
    
    assert exc_info.value.point == end
```

### TC-006: Path Crosses NoData
```python
def test_terrain_profile_nodata_flagged():
    """Profile crossing NoData region should flag affected samples."""
    grid = load_dem("tests/fixtures/dem_with_nodata.tif")
    # Bounds: lat [-25, -15], lon [-50, -40] (standard bounds)
    # NoData pattern: diagonal stripe from NW to SE (~10% of pixels)
    # Path from NW to SE crosses the diagonal NoData stripe
    start = GeoPoint(latitude=-16.0, longitude=-49.0)  # NW area
    end = GeoPoint(latitude=-24.0, longitude=-41.0)    # SE area

    profile = terrain_profile(grid, start, end)

    # Check explicit flag
    nodata_samples = [s for s in profile.samples if s.is_nodata]
    assert len(nodata_samples) > 0

    # Check consistency
    assert profile.has_nodata is True

    # Check NaN consistency
    import math
    for s in nodata_samples:
        assert math.isnan(s.elevation_m)
```

### TC-007: Very Short Path
```python
def test_terrain_profile_short_path_minimum_samples():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    start = GeoPoint(latitude=-20.000000, longitude=-45.000000)
    end = GeoPoint(latitude=-20.000001, longitude=-45.000001)  # ~0.1m

    profile = terrain_profile(grid, start, end)

    assert len(profile.samples) == 2  # Minimum: start + end
```

### TC-008: Long Path
```python
@pytest.mark.slow
def test_terrain_profile_long_path():
    """Long profile within dem_large bounds."""
    grid = load_dem("tests/fixtures/dem_large.tif")
    # Bounds: lat [-15, -10], lon [-50, -45]
    start = GeoPoint(latitude=-11.0, longitude=-49.0)
    end = GeoPoint(latitude=-14.0, longitude=-46.0)  # ~450km diagonal

    profile = terrain_profile(grid, start, end)

    assert len(profile.samples) > 100
    assert profile.total_distance_m > 400_000  # >400km
```

### TC-009: Samples Strictly Ordered
```python
def test_terrain_profile_samples_strictly_ordered():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.1, longitude=-45.1)

    profile = terrain_profile(grid, start, end)

    distances = profile.distances()
    for i in range(1, len(distances)):
        assert distances[i] > distances[i-1]  # Strictly increasing
```

### TC-010: Bilinear Interpolation
```python
def test_terrain_profile_uses_bilinear():
    grid = load_dem("tests/fixtures/dem_known_values_4326.tif")
    # Bounds: lat [-25, -15], lon [-50, -40] (same as dem_100x100)
    start = GeoPoint(latitude=-20.005, longitude=-45.005)
    end = GeoPoint(latitude=-20.006, longitude=-45.006)

    profile = terrain_profile(grid, start, end)

    import math
    assert not math.isnan(profile.samples[0].elevation_m)
    assert profile.interpolation == "bilinear"
```

### TC-011: Immutability
```python
def test_terrain_profile_is_immutable():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.1, longitude=-45.1)

    profile = terrain_profile(grid, start, end)

    with pytest.raises(Exception):  # ValidationError or AttributeError
        profile.total_distance_m = 0
```

### TC-012: Custom step_m (Oversampling)
```python
def test_terrain_profile_custom_step_oversampling():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.01, longitude=-45.01)  # ~1.5km

    profile = terrain_profile(grid, start, end, step_m=10.0)  # Dense: every 10m

    expected_samples = int(profile.total_distance_m / 10.0) + 1
    assert len(profile.samples) >= expected_samples - 1
    assert profile.step_m == pytest.approx(10.0, rel=0.01)
```

### TC-013: Custom step_m Larger Than Path
```python
def test_terrain_profile_step_larger_than_path():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    start = GeoPoint(latitude=-20.000000, longitude=-45.000000)
    end = GeoPoint(latitude=-20.000010, longitude=-45.000010)  # ~1.5m

    profile = terrain_profile(grid, start, end, step_m=1000.0)  # 1km step

    assert len(profile.samples) == 2  # Minimum: start + end
```

### TC-014: Entire Path is NoData
```python
def test_terrain_profile_all_nodata():
    """Profile entirely within NoData region."""
    grid = load_dem("tests/fixtures/dem_all_nodata_partial.tif")
    # Bounds: lat [-25, -15], lon [-50, -40]
    # NoData region: eastern half (lon > -45)
    start = GeoPoint(latitude=-20.0, longitude=-44.0)  # In NoData region
    end = GeoPoint(latitude=-20.1, longitude=-43.0)    # In NoData region

    profile = terrain_profile(grid, start, end)

    assert profile.has_nodata is True
    assert profile.nodata_ratio() == 1.0
    assert all(s.is_nodata for s in profile.samples)
```

### TC-015: nodata_count and nodata_ratio
```python
def test_terrain_profile_nodata_helpers():
    """Verify NoData helper methods on a profile crossing NoData stripe."""
    grid = load_dem("tests/fixtures/dem_with_nodata.tif")
    # Bounds: lat [-25, -15], lon [-50, -40] (standard bounds)
    # NoData pattern: diagonal stripe from NW to SE (~10% coverage)
    # Path crossing the diagonal will hit NoData
    start = GeoPoint(latitude=-16.0, longitude=-49.0)  # NW corner area
    end = GeoPoint(latitude=-24.0, longitude=-41.0)    # SE corner area

    profile = terrain_profile(grid, start, end)

    count = profile.nodata_count()
    ratio = profile.nodata_ratio()

    assert count >= 0
    assert 0.0 <= ratio <= 1.0
    assert count == sum(1 for s in profile.samples if s.is_nodata)
    # This path crosses the diagonal NoData stripe, so some samples should be NoData
    assert profile.has_nodata is True
```

### TC-016: Point on Grid Corner
```python
def test_terrain_profile_point_on_corner():
    """Points exactly on grid boundary should use clamped interpolation."""
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    
    # Start at exact corner of bounds (southwest corner)
    start = GeoPoint(
        latitude=grid.bounds.min_y,  # -25
        longitude=grid.bounds.min_x  # -50
    )
    end = GeoPoint(
        latitude=grid.bounds.min_y + 0.01,
        longitude=grid.bounds.min_x + 0.01
    )
    
    # Should not raise, should use clamped neighbors
    profile = terrain_profile(grid, start, end)
    assert len(profile.samples) >= 2
    # Corner point degrades to nearest-neighbor (clamped x0==x1, y0==y1)
    import math
    assert not math.isnan(profile.samples[0].elevation_m)
```

### TC-017: High Latitude Step Derivation
```python
def test_terrain_profile_high_latitude_step():
    """At high latitudes, step_m should still be derived correctly via geodesic."""
    from domain.terrain.services import derive_step_m, MIN_STEP_M
    from unittest.mock import MagicMock
    
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
    # At 80° lat, 0.001° longitude ≈ 19m (not 111m like at equator)
    assert step < 50  # Sanity check
```

### TC-018: All Grid Boundary Edges
```python
def test_terrain_profile_boundary_edges():
    """Profiles along each edge of the grid should work."""
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    # Bounds: lat [-25, -15], lon [-50, -40]
    
    # South edge (min_y)
    profile_south = terrain_profile(
        grid,
        GeoPoint(latitude=-25.0, longitude=-48.0),
        GeoPoint(latitude=-25.0, longitude=-42.0)
    )
    assert len(profile_south.samples) >= 2
    
    # North edge (max_y)
    profile_north = terrain_profile(
        grid,
        GeoPoint(latitude=-15.0, longitude=-48.0),
        GeoPoint(latitude=-15.0, longitude=-42.0)
    )
    assert len(profile_north.samples) >= 2
```

### TC-019: Effective Step Calculation
```python
def test_terrain_profile_effective_step():
    """effective_step_m should equal total_distance / (n_samples - 1)."""
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    start = GeoPoint(latitude=-20.0, longitude=-45.0)
    end = GeoPoint(latitude=-20.1, longitude=-45.1)

    profile = terrain_profile(grid, start, end)
    
    expected_effective = profile.total_distance_m / (len(profile.samples) - 1)
    assert profile.effective_step_m == pytest.approx(expected_effective, rel=0.01)
```

---

## Non-Functional Requirements

### Performance Baseline

| Metric | Target | Measurement |
|--------|--------|-------------|
| 100-sample profile | < 10 ms | `pytest-benchmark` |
| 1,000-sample profile | < 100 ms | `pytest-benchmark` |
| 10,000-sample profile | < 1 s | `pytest-benchmark` |

> **Memory consideration**: A 50,000-sample profile creates ~50k ProfileSample objects.
> Estimated memory: 10-20 MB. Acceptable for MVP; optimize later if benchmarks fail.

### Numeric Precision

| Operation | Precision | Constant |
|-----------|-----------|----------|
| Distance calculation | float64, mm accuracy | pyproj.Geod |
| Distance tolerance | 0.1 m (10 cm) | `DISTANCE_TOLERANCE_M` |
| Coordinate equality | Pydantic value equality | (exact) |
| Elevation interpolation | float64 intermediate | — |
| Grid data | float32 | TerrainGrid.data |

### Thread Safety

> `terrain_profile()` is **thread-safe** — it only reads from immutable TerrainGrid.
> The internal `pyproj.Geod` instance is thread-safe per pyproj documentation.

---

## Dependencies

- `numpy>=1.24.0` — Array operations, grid access
- `pyproj>=3.4.0` — Geodesic distance and path interpolation (already a rasterio dependency)
- No new external dependencies required

---

## Decisions Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| Input type? | `GeoPoint` VO | Type safety, validated coordinates |
| Interpolation? | **Bilinear** (fixed) | Elevation is continuous; configurability is YAGNI |
| Distance formula? | **pyproj.Geod (WGS84)** | Millimeter precision, handles edge cases |
| Path interpolation? | **pyproj.Geod.npts** | True geodesic, not linear in lat/lon |
| Sample spacing? | **Configurable `step_m`** | Flexibility for dense/sparse profiles |
| Default step derivation? | **Geodesic measurement** | Exact, avoids cos(lat) approximation errors |
| Minimum step? | **1.0 m** | Prevents degenerate cases at extreme latitudes |
| NaN handling? | **Propagate with flag, no infill** | Explicit `is_nodata`, FEAT-003 decides policy |
| NaN check? | **math.isnan** | Avoids numpy dependency in simple VO |
| Distance tolerance? | **0.1 m** | Practical for floating-point accumulation |
| Output type? | **TerrainProfile VO** | Immutable, validated, rich interface |
| Curvature correction? | **No (FEAT-003)** | Profile is raw elevation; LoS applies curvature |
| Min samples? | **2** | Start and end always included |
| Bounds check? | **Error immediately** | No partial profiles for MVP |
| Boundary interpolation? | **Clamp to edge** | Simpler API; degrades gracefully |
| step_m vs effective_step_m? | **Both fields** | Requested vs actual for auditability |

---

## Acceptance Criteria

- [ ] All 19 test cases pass
- [ ] `terrain_profile()` returns validated `TerrainProfile`
- [ ] GeoPoint reused from FEAT-001 (or created if not exists)
- [ ] ProfileSample, TerrainProfile are immutable VOs
- [ ] Bilinear interpolation uses clamped edge handling
- [ ] Geodesic distance matches pyproj reference (within 0.1 m)
- [ ] Geodesic path interpolation uses pyproj.Geod.npts
- [ ] Points outside bounds raise `PointOutOfBoundsError`
- [ ] Start == End raises `InvalidProfileError`
- [ ] NoData regions have `is_nodata=True` AND `elevation_m=NaN`
- [ ] `has_nodata` flag is consistent with samples
- [ ] Custom `step_m` works correctly
- [ ] `effective_step_m` calculated correctly
- [ ] Default `step_m` uses geodesic derivation (not cos approximation)
- [ ] Type hints complete and mypy passes
- [ ] No I/O in domain layer
- [ ] Uses `math.isnan` not `np.isnan` in VOs
- [ ] Docstrings with examples
- [ ] Performance within baseline

---

## Appendix: Dependencies on FEAT-001

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| `TerrainGrid` | `domain/terrain/value_objects.py` | ✅ Exists | Reuse |
| `BoundingBox` | `domain/terrain/value_objects.py` | ✅ Exists | Interface: `min_x`, `min_y`, `max_x`, `max_y` |
| `TerrainError` | `domain/terrain/errors.py` | ✅ Exists | Error hierarchy base |
| `GeoPoint` | `domain/terrain/value_objects.py` | ❌ **Create** | Will be added by FEAT-002 |
| `pyproj` | `pyproject.toml` | ✅ Exists | `pyproj>=3.4.0` already declared |

---

## Appendix: Fixture Bounds Reference

> **CRITICAL**: Test coordinates MUST be within these bounds or tests will fail with `PointOutOfBoundsError`.

| Fixture | Latitude (min_y, max_y) | Longitude (min_x, max_x) | Notes |
|---------|-------------------------|--------------------------|-------|
| `dem_100x100_4326.tif` | [-25, -15] | [-50, -40] | Primary test fixture (standard bounds) |
| `dem_with_nodata.tif` | [-25, -15] | [-50, -40] | Standard bounds, diagonal NoData stripe |
| `dem_large.tif` | [-26, -14] | [-51, -39] | Extended bounds for 700km+ paths |
| `dem_known_values_4326.tif` | [-25, -15] | [-50, -40] | Standard bounds, known values |
| `dem_all_nodata_partial.tif` | [-25, -15] | [-50, -40] | Standard bounds, east half NoData |

### dem_all_nodata_partial.tif Structure

```
Longitude:  -50              -45              -40
            ├────────────────┼────────────────┤
            │     VALID      │     NoData     │
            │   (150.0m)     │     (NaN)      │
            │                │                │
            └────────────────┴────────────────┘
                         ↑
                    lon = -45 boundary
```

---

## Appendix: Test Fixtures Required

| Fixture | Purpose | Status |
|---------|---------|--------|
| `dem_100x100_4326.tif` | Basic happy path | ✅ FEAT-001 |
| `dem_with_nodata.tif` | NoData crossing tests (diagonal stripe) | ✅ Updated |
| `dem_large.tif` | Long path tests (extended bounds) | ✅ Updated |
| `dem_known_values_4326.tif` | Interpolation verification | ✅ Updated |
| `dem_all_nodata.tif` | 100% NoData (FEAT-001 rejects) | ✅ FEAT-001 |
| `dem_all_nodata_partial.tif` | Grid with east-half NoData region | ✅ Created |

### Generator: `dem_all_nodata_partial.tif`

```python
def gen_dem_all_nodata_partial():
    """Grid 100x100 with valid west half, NoData east half.
    
    Bounds match dem_100x100_4326.tif for consistency:
    - lat: [-25, -15]
    - lon: [-50, -40]
    - West half (lon < -45): valid elevation 150.0m
    - East half (lon >= -45): NoData (NaN)
    """
    height, width = 100, 100
    data = np.full((height, width), 150.0, dtype=np.float32)
    
    # East half is NoData (columns 50-99)
    data[:, 50:] = np.nan
    
    # Same bounds as dem_100x100_4326.tif
    transform = from_bounds(-50.0, -25.0, -40.0, -15.0, width, height)
    
    with rasterio.open(
        FIXTURES_DIR / "dem_all_nodata_partial.tif",
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
```

---

## Implementation Order (TDD)

```
1. Generate fixture
   └── Add gen_dem_all_nodata_partial() to scripts/gen_fixtures.py
   └── Run: python scripts/gen_fixtures.py

2. Create tests first (RED)
   └── tests/terrain/test_terrain_profile.py
   └── All 19 TCs from this spec
   └── Tests will fail (functions don't exist yet)

3. Create Value Objects (GREEN - partial)
   └── domain/terrain/value_objects.py
   ├── GeoPoint (new)
   ├── ProfileSample (new)
   └── TerrainProfile (new)

4. Create errors
   └── domain/terrain/errors.py
   ├── PointOutOfBoundsError (new)
   └── InvalidProfileError (new)

5. Implement service (GREEN)
   └── domain/terrain/services.py
   └── terrain_profile()
   └── derive_step_m()
   └── geodesic_distance()
   └── interpolate_geodesic_path()
   └── bilinear_interpolate()

6. Verify
   └── pytest tests/terrain/test_terrain_profile.py
   └── mypy domain/terrain/

7. Update spec/SPEC.md feature index
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | — | Initial spec |
| 1.1.0 | — | Added `step_m`, `is_nodata`, geodesic distance, metadata fields |
| 1.2.0 | — | Fixed POST-4 (geodesic not linear), geodesic step derivation, `math.isnan`, 0.1m tolerance, edge clamping docs |
| 1.2.1 | — | Added verification results: GeoPoint must be created, fixture generator script, implementation order |
| 1.3.0 | — | **CRITICAL FIX**: Corrected all TC coordinates to match fixture bounds. Added `effective_step_m` field. Added Fixture Bounds Reference appendix. Added TC-018, TC-019. Fixed TC-017 to use MagicMock. Updated dem_all_nodata_partial bounds to match dem_100x100. Added POST-9 (no infill). Added numeric constants section. 19 TCs total. |

---

<!-- FEAT-002 v1.3.0 — TerrainProfile with corrected fixture bounds, effective_step_m -->
