"""Stub for rasterio.crs providing CRS class."""

from __future__ import annotations

from typing import Any


class CRS:
    """Stub CRS class for testing.

    Provides minimal functionality to support CRS comparisons in tests.
    Real behavior should be monkeypatched when needed.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = data or {}
        self._epsg: int | None = None

    @classmethod
    def from_epsg(cls, code: int) -> "CRS":
        """Create CRS from EPSG code."""
        crs = cls({"init": f"epsg:{code}"})
        crs._epsg = code
        return crs

    def to_epsg(self) -> int | None:
        """Return EPSG code if available."""
        return self._epsg

    def to_string(self) -> str:
        """Return string representation."""
        if self._epsg is not None:
            return f"EPSG:{self._epsg}"
        return str(self._data)

    def __eq__(self, other: object) -> bool:
        """Compare CRS objects for equality.

        Comparison strategy (in order):
        1. If both have EPSG codes, compare by EPSG code only
        2. If exactly one has EPSG and the other doesn't, return False
           (asymmetric: no attempt to resolve EPSG from data)
        3. If neither has EPSG, fall back to _data dict comparison

        Limitation: A CRS created with from_epsg(4326) will NOT equal a CRS
        created with data={'proj': 'longlat', ...} even if semantically
        equivalent, because one has _epsg=4326 and the other has _epsg=None.
        This stub prioritizes simplicity over semantic equivalence.
        """
        if not isinstance(other, CRS):
            return NotImplemented
        # If both have EPSG codes, compare those
        if self._epsg is not None and other._epsg is not None:
            return self._epsg == other._epsg
        # If exactly one has EPSG and the other doesn't, they're not equal
        if (self._epsg is None) != (other._epsg is None):
            return False
        # Both have None EPSG, compare underlying data
        return self._data == other._data

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"CRS({self._data!r})"

    def __hash__(self) -> int:
        """Hash consistent with equality semantics.

        - If EPSG is defined, hash by EPSG code (objects with same EPSG compare equal).
        - Otherwise, hash by the underlying immutable representation of _data.
        """
        if self._epsg is not None:
            return hash(("epsg", self._epsg))
        # Fallback: hash a frozenset of items for stability
        try:
            return hash(("data", frozenset(self._data.items())))
        except TypeError:
            return hash(("data_str", str(self._data)))
