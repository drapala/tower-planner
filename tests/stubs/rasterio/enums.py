"""Stub for rasterio.enums with minimal Resampling symbols."""

from __future__ import annotations

from typing import Final

__all__ = ["Resampling"]


class Resampling:
    """Enumeration of resampling methods used by the rasterio stub.

    This stub provides only the symbols needed for tests. Real rasterio
    uses an IntEnum; this stub uses simple class attributes for simplicity.
    """

    bilinear: Final[int] = 1
