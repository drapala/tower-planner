"""Pytest configuration for GIS integration tests.

This conftest is for tests/gis/ directory only.

Note on rasterio:
- test_geotiff_adapter.py uses the STUB (src/rasterio/) with monkeypatching
- test_fixtures_sanity.py uses REAL rasterio for integration tests

The test_fixtures_sanity.py module handles the path manipulation internally
via the @requires_real_rasterio decorator and must be run with real rasterio
installed. See that module's docstring for details.
"""
