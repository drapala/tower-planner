"""Stub submodule for rasterio.warp used in tests (monkeypatched)."""

from __future__ import annotations

from typing import Any, NoReturn

_STUB_MSG = "function is a stub; monkeypatch in tests"


def calculate_default_transform(*args: Any, **kwargs: Any) -> NoReturn:
    """Stub that always raises - must be monkeypatched in tests."""
    raise NotImplementedError(_STUB_MSG)


def reproject(*args: Any, **kwargs: Any) -> NoReturn:
    """Stub that always raises - must be monkeypatched in tests."""
    raise NotImplementedError(_STUB_MSG)
