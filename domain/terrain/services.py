"""Terrain Bounded Context - Domain Services.

Pure domain logic for terrain calculations.
NO I/O operations - file loading is implemented by infrastructure adapters
under `src/infrastructure/terrain/geotiff_adapter.py` via domain ports.

See spec/features/FEAT-002-terrain-profile.md for TerrainProfile specification.
"""

from __future__ import annotations

import math

from pyproj import Geod

from domain.terrain.errors import InvalidProfileError, PointOutOfBoundsError
from domain.terrain.value_objects import (
    BoundingBox,
    GeoPoint,
    ProfileSample,
    TerrainGrid,
    TerrainProfile,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_STEP_M = 1.0  # Minimum step size in meters (floor to avoid tiny steps)

# WGS84 ellipsoid for geodesic calculations (same as GPS, EPSG:4326)
_geod = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Helper: Bounds Check
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Geodesic Distance
# ---------------------------------------------------------------------------
def geodesic_distance(start: GeoPoint, end: GeoPoint) -> float:
    """Calculate geodesic distance between two points in meters.

    Uses WGS84 ellipsoid for millimeter-level precision.

    Args:
        start: Starting geographic point
        end: Ending geographic point

    Returns:
        Distance in meters (always positive)
    """
    _, _, distance = _geod.inv(
        start.longitude, start.latitude, end.longitude, end.latitude
    )
    return float(abs(distance))  # Ensure positive, explicit float


# ---------------------------------------------------------------------------
# Geodesic Path Interpolation
# ---------------------------------------------------------------------------
def interpolate_geodesic_path(
    start: GeoPoint, end: GeoPoint, num_intermediate: int
) -> list[GeoPoint]:
    """Interpolate points along geodesic path.

    Uses pyproj.Geod.npts for true geodesic interpolation (not linear in lat/lon).

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
        start.longitude, start.latitude, end.longitude, end.latitude, num_intermediate
    )

    # Build full list with endpoints
    result = [start]
    for lon, lat in intermediate:
        result.append(GeoPoint(latitude=lat, longitude=lon))
    result.append(end)

    return result


# ---------------------------------------------------------------------------
# Step Size Derivation
# ---------------------------------------------------------------------------
def derive_step_m(grid: TerrainGrid, start: GeoPoint, end: GeoPoint) -> float:
    """Derive step size from grid resolution using geodesic measurement.

    Measures actual ground distance of one grid cell at the path midpoint,
    avoiding the inaccuracy of cos(lat) approximation.

    Args:
        grid: TerrainGrid with resolution information
        start: Starting point of the profile
        end: Ending point of the profile

    Returns:
        Step size in meters, minimum MIN_STEP_M (1m).
    """
    mid = GeoPoint(
        latitude=(start.latitude + end.latitude) / 2,
        longitude=(start.longitude + end.longitude) / 2,
    )

    # Measure x resolution: one grid cell east
    x_neighbor = GeoPoint(
        latitude=mid.latitude, longitude=mid.longitude + grid.resolution[0]
    )
    _, _, x_res_m = _geod.inv(
        mid.longitude, mid.latitude, x_neighbor.longitude, x_neighbor.latitude
    )

    # Measure y resolution: one grid cell south
    y_neighbor = GeoPoint(
        latitude=mid.latitude - grid.resolution[1], longitude=mid.longitude
    )
    _, _, y_res_m = _geod.inv(
        mid.longitude, mid.latitude, y_neighbor.longitude, y_neighbor.latitude
    )

    # Use finer resolution, but never below minimum
    return float(max(MIN_STEP_M, min(abs(x_res_m), abs(y_res_m))))


# ---------------------------------------------------------------------------
# Bilinear Interpolation
# ---------------------------------------------------------------------------
def bilinear_interpolate(grid: TerrainGrid, point: GeoPoint) -> tuple[float, bool]:
    """Interpolate elevation at arbitrary point using 4 nearest pixels.

    Returns (elevation, is_nodata).
    If any of the 4 neighbors is NaN, returns (NaN, True).

    Boundary behavior:
        Points exactly on grid boundaries use clamped indices.
        This effectively duplicates edge pixels, causing bilinear
        to degrade to linear (on edges) or nearest (on corners).
        This is intentional - the alternative would require shrinking
        valid bounds by half a pixel, complicating the API.

    Args:
        grid: TerrainGrid with elevation data
        point: Geographic point to interpolate

    Returns:
        Tuple of (elevation_m, is_nodata)
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
        return (float("nan"), True)

    # Bilinear weights (fractional part)
    fx = px - math.floor(px)
    fy = py - math.floor(py)

    # Bilinear interpolation formula
    elevation = (
        q11 * (1 - fx) * (1 - fy)
        + q21 * fx * (1 - fy)
        + q12 * (1 - fx) * fy
        + q22 * fx * fy
    )

    return (float(elevation), False)


# ---------------------------------------------------------------------------
# Main Service: terrain_profile
# ---------------------------------------------------------------------------
def terrain_profile(
    grid: TerrainGrid,
    start: GeoPoint,
    end: GeoPoint,
    step_m: float | None = None,
) -> TerrainProfile:
    """Extract elevation profile between two geographic points.

    Creates a sequence of elevation samples along the geodesic path between
    start and end points, using bilinear interpolation from the terrain grid.

    Args:
        grid: Elevation grid (from FEAT-001)
        start: Starting point (typically Tx location)
        end: Ending point (typically Rx location)
        step_m: Sample spacing in meters. If None, derived from grid resolution

    Returns:
        TerrainProfile with all samples and metadata

    Raises:
        PointOutOfBoundsError: If start or end is outside grid bounds
        InvalidProfileError: If start equals end (zero distance)
        ValueError: If step_m is provided but not positive

    Example:
        >>> grid = load_dem("terrain.tif")
        >>> start = GeoPoint(latitude=-20.0, longitude=-45.0)
        >>> end = GeoPoint(latitude=-20.1, longitude=-45.1)
        >>> profile = terrain_profile(grid, start, end)
        >>> print(f"Total distance: {profile.total_distance_m:.0f}m")
        >>> print(f"Samples: {len(profile.samples)}")
    """
    # PRE-1: Start must be within bounds
    if not is_within_bounds(start, grid.bounds):
        raise PointOutOfBoundsError(start, grid.bounds)

    # PRE-2: End must be within bounds
    if not is_within_bounds(end, grid.bounds):
        raise PointOutOfBoundsError(end, grid.bounds)

    # PRE-3: Start != End (non-zero distance)
    if start == end:
        raise InvalidProfileError("Start equals end")

    # PRE-4: step_m must be positive if provided
    if step_m is not None and step_m <= 0:
        raise ValueError("step_m must be positive")

    # Calculate total geodesic distance
    total_distance = geodesic_distance(start, end)

    # Derive step_m if not provided
    if step_m is None:
        step_m = derive_step_m(grid, start, end)

    # Calculate number of samples: n = max(2, floor(total / step) + 1)
    n_samples = max(2, int(math.floor(total_distance / step_m)) + 1)

    # Calculate effective step (actual average spacing)
    effective_step = total_distance / (n_samples - 1)

    # Generate path points: n_samples - 2 intermediate points (excludes start and end)
    num_intermediate = n_samples - 2
    path_points = interpolate_geodesic_path(start, end, num_intermediate)

    # Build samples with elevation interpolation
    samples: list[ProfileSample] = []
    has_nodata = False

    for i, point in enumerate(path_points):
        # Calculate cumulative distance
        if i == 0:
            distance = 0.0
        elif i == len(path_points) - 1:
            distance = total_distance
        else:
            distance = i * effective_step

        # Interpolate elevation
        elevation, is_nodata = bilinear_interpolate(grid, point)

        if is_nodata:
            has_nodata = True

        sample = ProfileSample(
            distance_m=distance, elevation_m=elevation, point=point, is_nodata=is_nodata
        )
        samples.append(sample)

    # Build and return TerrainProfile
    return TerrainProfile(
        start=start,
        end=end,
        samples=tuple(samples),
        total_distance_m=total_distance,
        step_m=step_m,
        effective_step_m=effective_step,
        has_nodata=has_nodata,
        interpolation="bilinear",
    )
