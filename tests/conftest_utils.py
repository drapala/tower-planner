"""Shared pytest configuration utilities.

This module provides reusable helpers for test configuration, particularly
for managing sys.path and sys.modules when switching between stubs and
real packages.

These utilities are used by:
- tests/gis/conftest.py
- tests/terrain/conftest.py
"""

from __future__ import annotations

import sys
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any


def prepare_real_rasterio_path(
    extra_modules: Iterable[str] = (),
) -> Generator[None, None, None]:
    """Generator that removes stubs from sys.path and clears cached modules.

    This allows real rasterio (if installed) to be imported instead of
    the stub in tests/stubs/rasterio/. The stub is useful for unit tests
    with monkeypatching, but integration tests need real rasterio.

    Uses try/finally for exception-safe cleanup since session-scoped
    fixtures cannot use monkeypatch (which is function-scoped).

    Args:
        extra_modules: Additional module prefixes to clear from sys.modules
            beyond rasterio. For example, infrastructure modules that import
            rasterio at module level.

    Yields:
        None - control returns to the test

    Example:
        @pytest.fixture(scope="session")
        def real_rasterio_path():
            yield from prepare_real_rasterio_path(extra_modules=[
                "affine",
                "src.infrastructure.terrain",
            ])
    """
    # Determine project root and stubs path
    # This assumes the file is at tests/conftest_utils.py
    project_root = Path(__file__).parent.parent
    stubs_path = str(project_root / "tests" / "stubs")

    original_path = sys.path.copy()

    # Build list of module prefixes to clear
    # Always include rasterio; extend with caller-specified modules
    modules_to_clear = ["rasterio", *extra_modules]

    # Remove cached modules that may have imported stub rasterio
    saved_modules: dict[str, Any] = {}
    for key in list(sys.modules.keys()):
        should_clear = any(
            key == m or key.startswith(m + ".") for m in modules_to_clear
        )
        if should_clear:
            saved_modules[key] = sys.modules.pop(key)

    try:
        # Reuse shared helper for sys.path filtering (consolidated from duplicated logic)
        filter_sys_path_excluding_stubs(stubs_path)
        yield
    finally:
        # Restore original sys.path contents IN PLACE (exception-safe)
        sys.path.clear()
        sys.path.extend(original_path)
        # Restore original sys.modules state
        for key, module in saved_modules.items():
            sys.modules[key] = module


def get_fixtures_dir() -> Path:
    """Return path to tests/fixtures/ directory.

    Shared helper for tests that need fixture paths.
    Centralizes fixture directory resolution to avoid duplication.
    """
    return Path(__file__).parent / "fixtures"


def filter_sys_path_excluding_stubs(stubs_path: str | Path) -> list[str]:
    """Filter sys.path to exclude a stubs directory.

    Handles broken symlinks, permission errors, and preserves empty strings (CWD).
    Modifies sys.path IN PLACE and returns the original contents for cleanup.

    Args:
        stubs_path: Path to the stubs directory to exclude

    Returns:
        Original sys.path contents (copy) for later restoration
    """
    import logging

    original_path = sys.path.copy()
    resolved_stubs = Path(stubs_path).resolve()

    filtered_path: list[str] = []
    for p in sys.path:
        # Empty string means CWD - preserve it
        if p == "":
            filtered_path.append(p)
            continue
        try:
            resolved = Path(p).resolve()
            if resolved != resolved_stubs:
                filtered_path.append(p)
        except (OSError, ValueError):
            # Keep paths that can't be resolved (may still be valid for imports)
            logging.getLogger(__name__).debug("Could not resolve path: %s", p)
            filtered_path.append(p)

    sys.path.clear()
    sys.path.extend(filtered_path)
    return original_path
