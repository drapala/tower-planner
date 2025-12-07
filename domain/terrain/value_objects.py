"""Terrain Bounded Context - Value Objects.

Immutable data structures representing geographic concepts.
All validation occurs at construction time via Pydantic.

See spec/features/FEAT-001-load-dem.md for complete specifications.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator


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
