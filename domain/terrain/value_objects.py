"""Terrain Bounded Context - Value Objects.

Immutable data structures representing geographic concepts.
All validation occurs at construction time via Pydantic.

See spec/features/FEAT-001-load-dem.md for complete specifications.
"""

from __future__ import annotations

from typing import Tuple

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

    def is_valid(self) -> bool:  # pragma: no cover - trivial method
        """Return True because construction enforces validity."""
        return True


class TerrainGrid(BaseModel):
    """Immutable elevation grid with geographic metadata (Value Object)."""

    data: NDArray[np.float32]  # 2D float32 array (height x width)
    bounds: BoundingBox  # Geographic extent in EPSG:4326
    crs: str  # Always "EPSG:4326" (system CRS)
    resolution: Tuple[float, float]  # (x_res, y_res) absolute values in degrees
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
        # At least one valid pixel
        if not np.any(~np.isnan(self.data)):
            raise ValueError("Grid contains 100% NoData")
        return self
