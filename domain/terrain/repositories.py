"""Domain Port(s) for Terrain I/O.

Defines interfaces (Protocols) that infrastructure adapters must implement.
No concrete I/O here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .value_objects import TerrainGrid


class TerrainRepository(Protocol):
    """Port for obtaining terrain grids from external sources.

    Implementations live in infrastructure (e.g., GeoTIFF adapter).
    """

    def load_dem(self, file_path: Path | str) -> TerrainGrid:
        """Load a DEM and return a normalized TerrainGrid in EPSG:4326."""
        ...
