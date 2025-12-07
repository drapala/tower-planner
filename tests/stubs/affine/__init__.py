"""Minimal Affine stub to support tests without external dependency.

Location: tests/stubs/affine/ (moved from src/affine/ per TD-002)
"""

from __future__ import annotations

from types import NotImplementedType
from typing import Iterator


class Affine:
    """2D affine transformation matrix for georeferencing.

    Represents a 3x3 augmented matrix:
        | a  b  c |
        | d  e  f |
        | 0  0  1 |

    Where (a, b, c, d, e, f) are the six parameters:
        - a: x-scale (pixel width)
        - b: x-shear (rotation component)
        - c: x-translation (upper-left x coordinate)
        - d: y-shear (rotation component)
        - e: y-scale (pixel height, typically negative for north-up)
        - f: y-translation (upper-left y coordinate)
    """

    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float):
        """Initialize an affine transformation.

        Args:
            a: x-scale (pixel width in CRS units)
            b: x-shear (rotation component, 0 for north-up grids)
            c: x-translation (upper-left corner x coordinate)
            d: y-shear (rotation component, 0 for north-up grids)
            e: y-scale (pixel height, negative for north-up grids)
            f: y-translation (upper-left corner y coordinate)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    # Factory methods
    @staticmethod
    def identity() -> "Affine":
        return Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @staticmethod
    def translation(tx: float, ty: float) -> "Affine":
        return Affine(1.0, 0.0, tx, 0.0, 1.0, ty)

    @staticmethod
    def scale(sx: float, sy: float) -> "Affine":
        return Affine(sx, 0.0, 0.0, 0.0, sy, 0.0)

    def __mul__(
        self, other: Affine | tuple[float, float]
    ) -> Affine | tuple[float, float] | NotImplementedType:
        """Compose with another Affine or apply to a (col, row) point.

        Args:
            other: Another Affine for composition, or (col, row) tuple to transform.

        Returns:
            Affine: When composing with another Affine.
            tuple[float, float]: When transforming a point (x, y).
            NotImplemented: When other is not a supported type.
        """
        # Composition with another Affine
        if isinstance(other, Affine):
            a = self.a * other.a + self.b * other.d
            b = self.a * other.b + self.b * other.e
            c = self.a * other.c + self.b * other.f + self.c
            d = self.d * other.a + self.e * other.d
            e = self.d * other.b + self.e * other.e
            f = self.d * other.c + self.e * other.f + self.f
            return Affine(a, b, c, d, e, f)
        # Apply to (col,row)
        if isinstance(other, tuple) and len(other) == 2:
            col, row = other
            x = self.a * col + self.b * row + self.c
            y = self.d * col + self.e * row + self.f
            return (x, y)
        return NotImplemented

    # Enable unpacking like a,b,c,d,e,f = transform
    def __iter__(self) -> Iterator[float]:
        yield from (self.a, self.b, self.c, self.d, self.e, self.f)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Affine):
            return NotImplemented
        return (self.a, self.b, self.c, self.d, self.e, self.f) == (
            other.a,
            other.b,
            other.c,
            other.d,
            other.e,
            other.f,
        )

    def __hash__(self) -> int:
        """Return hash based on all six transform parameters.

        Consistent with __eq__ so Affine can be used in sets and as dict keys.
        """
        return hash((self.a, self.b, self.c, self.d, self.e, self.f))

    def __repr__(self) -> str:
        return f"Affine({self.a}, {self.b}, {self.c}, {self.d}, {self.e}, {self.f})"
