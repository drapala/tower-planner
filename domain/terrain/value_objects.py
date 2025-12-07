"""Terrain Bounded Context - Value Objects.

Immutable data structures representing geographic concepts.
All validation occurs at construction time via Pydantic.

See spec/features/FEAT-001-load-dem.md for complete specifications.
See spec/features/FEAT-002-terrain-profile.md for TerrainProfile specification.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Numeric Constants (FEAT-002)
# ---------------------------------------------------------------------------
# Tolerances for floating-point comparisons
DISTANCE_TOLERANCE_M = 0.1  # 10 cm - for distance invariants


class BoundingBox(BaseModel):
    """Geographic extent in EPSG:4326 (Value Object).

    Invariants are enforced at construction time - invalid BoundingBox
    cannot be instantiated.
    """

    min_x: float  # Western boundary (longitude)
    min_y: float  # Southern boundary (latitude)
    max_x: float  # Eastern boundary (longitude)
    max_y: float  # Northern boundary (latitude)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        # Longitude range
        if not (-180 <= self.min_x <= 180):
            raise ValueError(f"min_x longitude out of range: {self.min_x}")
        if not (-180 <= self.max_x <= 180):
            raise ValueError(f"max_x longitude out of range: {self.max_x}")
        # Latitude range
        if not (-90 <= self.min_y <= 90):
            raise ValueError(f"min_y latitude out of range: {self.min_y}")
        if not (-90 <= self.max_y <= 90):
            raise ValueError(f"max_y latitude out of range: {self.max_y}")
        # Ordering
        if not (self.min_x < self.max_x):
            raise ValueError(
                f"Invalid x ordering: min_x={self.min_x} >= max_x={self.max_x}"
            )
        if not (self.min_y < self.max_y):
            raise ValueError(
                f"Invalid y ordering: min_y={self.min_y} >= max_y={self.max_y}"
            )
        return self

    def is_valid(self) -> bool:  # pragma: no cover - deprecated, always True
        """Check if this BoundingBox is valid.

        .. deprecated:: 1.0
            This method is deprecated and scheduled for removal in v2.0.
            Rely on construction-time validation instead - if you have a BoundingBox
            instance, it is already guaranteed to be valid.

        Always returns True because Pydantic validation at construction time
        guarantees that invalid BoundingBox instances cannot exist.

        Returns:
            True: Always, since construction-time validation enforces invariants.

        Note:
            This method performs no runtime checks. All validation occurs in
            the @model_validator during __init__. Prefer catching ValueError
            during construction rather than calling is_valid() after.

        Migration:
            Remove all calls to `is_valid()`. If you have a BoundingBox instance,
            it is already valid. Handle construction failures with try/except ValueError.
        """
        # TODO(v2.0): Remove this method in next major version.
        # Tracking: See CHANGELOG.md deprecation notice and upgrade guide.
        # When removing: Also delete tests that call is_valid() or update them
        # to verify the deprecation warning is raised.
        warnings.warn(
            "BoundingBox.is_valid() is deprecated and will be removed in v2.0; "
            "rely on construction-time validation",
            DeprecationWarning,
            stacklevel=2,
        )
        return True


class TerrainGrid(BaseModel):
    """Immutable elevation grid with geographic metadata (Value Object).

    The data array is made truly immutable (read-only) at construction time.
    Attempts to modify the array after construction will raise ValueError.
    """

    data: NDArray[np.float32]  # 2D float32 array (height x width), read-only
    bounds: BoundingBox  # Geographic extent in EPSG:4326
    crs: str  # Always "EPSG:4326" (system CRS)
    resolution: tuple[float, float]  # (x_res, y_res) absolute values in degrees
    source_crs: str | None = None  # Original CRS before normalization

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_grid(self) -> "TerrainGrid":
        # 2D array
        if self.data.ndim != 2:
            raise ValueError(f"Data must be 2D, got {self.data.ndim}D")
        # Non-empty
        if self.data.shape[0] == 0 or self.data.shape[1] == 0:
            raise ValueError(f"Data cannot be empty: {self.data.shape}")
        # dtype
        if self.data.dtype != np.float32:
            raise ValueError(f"Data must be float32, got {self.data.dtype}")
        # CRS
        if self.crs != "EPSG:4326":
            raise ValueError(f"CRS must be EPSG:4326, got {self.crs}")
        # Resolution positive (absolute)
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValueError(f"Resolution must be positive: {self.resolution}")
        # At least one valid pixel - .all() short-circuits and avoids extra negation allocation
        if np.isnan(self.data).all():
            raise ValueError("Grid contains 100% NoData")

        # Make array truly immutable without mutating external arrays:
        # Always create an OWNED, contiguous float32 copy and freeze it. This
        # guarantees that the Value Object never changes caller-provided arrays
        # (no in-place flag flips) while enforcing dtype and contiguity.
        immutable = np.array(self.data, dtype=np.float32, copy=True, order="C")
        immutable.flags.writeable = False
        object.__setattr__(self, "data", immutable)

        return self


# ---------------------------------------------------------------------------
# GeoPoint (FEAT-002)
# ---------------------------------------------------------------------------
class GeoPoint(BaseModel):
    """Geographic coordinate in WGS84 (Value Object).

    Represents a single point on the Earth's surface using latitude and longitude
    in the WGS84 coordinate reference system.

    Invariants:
        GP-1: latitude in [-90, 90]
        GP-2: longitude in [-180, 180]

    Note on __eq__ and __hash__: Pydantic frozen models compare by value automatically.
    Verified: GeoPoint(lat=1, lon=2) == GeoPoint(lat=1, lon=2) returns True.
    """

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# ProfileSample (FEAT-002)
# ---------------------------------------------------------------------------
class ProfileSample(BaseModel):
    """Single sample point along a terrain profile (Value Object).

    Represents an elevation sample at a specific distance along a profile path,
    including geographic location and NoData flagging.

    Invariants:
        PS-1: distance_m >= 0
        PS-2: If is_nodata == True, then elevation_m is NaN
        PS-3: If is_nodata == False, then elevation_m is finite
        PS-4: point is valid GeoPoint

    Note: Uses math.isnan instead of np.isnan to avoid numpy dependency in simple VO.
    """

    distance_m: float = Field(ge=0)  # Cumulative distance from start in meters
    elevation_m: float  # Elevation at this point (NaN if nodata)
    point: GeoPoint  # Geographic location of sample
    is_nodata: bool = False  # Explicit flag for NoData regions

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_nodata_consistency(self) -> "ProfileSample":
        """Validate consistency between is_nodata flag and elevation_m."""
        # Use math.isnan to avoid numpy dependency in simple VO
        if self.is_nodata and not math.isnan(self.elevation_m):
            raise ValueError("is_nodata=True requires elevation_m=NaN")
        if not self.is_nodata and math.isnan(self.elevation_m):
            raise ValueError("is_nodata=False requires finite elevation_m")
        return self


# ---------------------------------------------------------------------------
# TerrainProfile (FEAT-002)
# ---------------------------------------------------------------------------
class TerrainProfile(BaseModel):
    """Elevation profile between two points (Value Object).

    Represents an ordered sequence of elevation samples along a geodesic path
    between a start and end point, with metadata about spacing and NoData coverage.

    Invariants:
        TP-1: len(samples) >= 2
        TP-2: samples[0].distance_m == 0
        TP-3: Samples strictly ordered by distance_m
        TP-4: samples[-1].distance_m == total_distance_m (within tolerance)
        TP-5: samples[0].point == start
        TP-6: samples[-1].point == end
        TP-7: has_nodata matches actual samples
        TP-8: effective_step_m == total/(n-1) (within tolerance)
        TP-9: total_distance_m > 0
        TP-10: step_m > 0
        TP-11: effective_step_m > 0

    Fields:
        step_m: The requested or derived spacing (input parameter)
        effective_step_m: The actual average spacing = total_distance_m / (n_samples - 1)
        These may differ slightly due to integer sample count.
    """

    start: GeoPoint  # Profile origin
    end: GeoPoint  # Profile destination
    samples: tuple[ProfileSample, ...]  # Ordered samples from start to end
    total_distance_m: float = Field(gt=0)  # Total path length in meters
    step_m: float = Field(gt=0)  # Requested/derived sample spacing
    effective_step_m: float = Field(gt=0)  # Actual average spacing: total/(n-1)
    has_nodata: bool  # True if any sample is nodata
    interpolation: str = "bilinear"  # Method used (metadata for audit)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_profile(self) -> "TerrainProfile":
        """Validate all TerrainProfile invariants."""
        # TP-1: At least 2 samples (start and end)
        if len(self.samples) < 2:
            raise ValueError(f"Profile must have >= 2 samples, got {len(self.samples)}")

        # TP-2: First sample at distance 0
        if self.samples[0].distance_m != 0:
            raise ValueError(
                f"First sample must be at distance 0, got {self.samples[0].distance_m}"
            )

        # TP-3: Samples ordered by distance (strictly increasing)
        for i in range(1, len(self.samples)):
            if self.samples[i].distance_m <= self.samples[i - 1].distance_m:
                raise ValueError("Samples must be strictly ordered by distance")

        # TP-4: Last sample distance equals total_distance_m (within tolerance)
        if (
            abs(self.samples[-1].distance_m - self.total_distance_m)
            > DISTANCE_TOLERANCE_M
        ):
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
            raise ValueError(
                f"has_nodata={self.has_nodata} but samples say {actual_has_nodata}"
            )

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
