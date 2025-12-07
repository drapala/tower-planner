"""Single source of truth for expected FEAT-001 test fixtures.

This module defines the list of expected fixture filenames used by both:
- scripts/gen_fixtures.py (generation verification)
- tests/gis/test_fixtures_sanity.py (existence verification)

Location: shared/ (not tests/) to avoid scripts->tests dependency.

When adding/removing fixtures, update ONLY this list.
"""

from __future__ import annotations

# Expected fixtures - SINGLE SOURCE OF TRUTH
# Both scripts/gen_fixtures.py and tests/gis/test_fixtures_sanity.py import from here.
# When adding/removing fixtures, update ONLY this list - no other files need changes.
# Sorted alphabetically for deterministic comparison.
EXPECTED_FIXTURES: list[str] = sorted(
    [
        "dem_100x100_4326.tif",  # TC-001: Happy path
        "dem_85pct_nodata.tif",  # TC-011: High nodata warning
        "dem_all_nodata.tif",  # TC-005: All nodata rejection
        "dem_all_nodata_partial.tif",  # FEAT-002 TC-014: Partial NoData region
        "dem_corrupted.tif",  # TC-015: Corrupted file rejection
        "dem_extreme_elevations.tif",  # TC-013: Extreme elevation values
        "dem_known_values_4326.tif",  # TC-009: Known values (float32)
        "dem_known_values_4326_int16.tif",  # TC-009b: Known values (int16)
        "dem_large.tif",  # TC-012/TC-018: Large file / memory budget / FEAT-002 TC-008
        "dem_nan_transform.tif",  # TC-019: NaN transform (monkeypatch target)
        "dem_no_crs.tif",  # TC-006: Missing CRS rejection
        "dem_polar_extreme.tif",  # TC-016: Polar projection reprojection
        "dem_utm23s.tif",  # TC-002: UTM reprojection
        "dem_with_nodata.tif",  # TC-004: NoData conversion / FEAT-002 TC-006
        "empty.tif",  # TC-010: Empty file rejection
        "image.png",  # TC-014: Non-GeoTIFF rejection (SEC-10)
        "rgb_image.tif",  # TC-007: Multi-band rejection
    ]
)

# Count derived from list for verification
EXPECTED_FIXTURE_COUNT: int = len(EXPECTED_FIXTURES)
