# Test Fixtures for FEAT-001 (LoadDEM)

Synthetic GeoTIFF fixtures for testing the `GeoTiffTerrainAdapter.load_dem()` functionality.

## Generation

Fixtures are generated programmatically using `scripts/gen_fixtures.py`:

```bash
# Requires rasterio, numpy
pip install rasterio numpy

# Generate all fixtures
python scripts/gen_fixtures.py
```

## Fixture Reference

| Fixture | Size | CRS | dtype | nodata | TC | Purpose |
|---------|------|-----|-------|--------|-----|---------|
| `dem_100x100_4326.tif` | 100×100 | EPSG:4326 | float32 | - | TC-001 | Happy path |
| `dem_utm23s.tif` | 50×50 | EPSG:31983 | float32 | - | TC-002 | Reprojection test |
| `dem_with_nodata.tif` | 20×20 | EPSG:4326 | float32 | -9999 | TC-004 | NoData conversion |
| `dem_all_nodata.tif` | 10×10 | EPSG:4326 | float32 | -9999 | TC-005 | 100% nodata → error |
| `dem_no_crs.tif` | 10×10 | None | float32 | - | TC-006 | Missing CRS → error |
| `rgb_image.tif` | 10×10×3 | EPSG:4326 | uint8 | - | TC-007 | Multi-band → error |
| `dem_known_values_4326.tif` | 20×20 | EPSG:4326 | float32 | - | TC-009 | Bit-exact test |
| `dem_known_values_4326_int16.tif` | 20×20 | EPSG:4326 | int16 | - | TC-009b | Dtype conversion |
| `empty.tif` | 0 bytes | - | - | - | TC-010 | Empty file → error |
| `dem_85pct_nodata.tif` | 20×20 | EPSG:4326 | float32 | -9999 | TC-011 | High nodata warning |
| `dem_large.tif` | 500×500 | EPSG:4326 | float32 | - | TC-012,18 | Memory budget test |
| `dem_extreme_elevations.tif` | 10×10 | EPSG:4326 | float32 | - | TC-013 | -430m to +8849m |
| `image.png` | 10×10 | - | uint8 | - | TC-014 | Non-GeoTIFF → error |
| `dem_corrupted.tif` | ~58 bytes | - | - | - | TC-015 | Invalid header → error |
| `dem_polar_extreme.tif` | 10×10 | EPSG:3031 | float32 | - | TC-016 | Invalid bounds after reproject |
| `dem_nan_transform.tif` | 10×10 | EPSG:4326 | float32 | - | TC-019 | NaN geotransform → error |

## Fixture Details

### TC-001: `dem_100x100_4326.tif`
- **Purpose**: Happy path test with valid EPSG:4326 raster
- **Bounds**: Approximately -50 to -40 lon, -25 to -15 lat (Brazil region)
- **Data**: Gradient elevation 0-1000m
- **Expected**: Loads successfully, no reprojection needed

### TC-002: `dem_utm23s.tif`
- **Purpose**: Test CRS reprojection from UTM to WGS84
- **CRS**: EPSG:31983 (SIRGAS 2000 / UTM Zone 23S)
- **Bounds**: São Paulo region in UTM coordinates
- **Expected**: Reprojected to EPSG:4326, `source_crs` preserved

### TC-004: `dem_with_nodata.tif`
- **Purpose**: Test nodata value conversion to NaN
- **NoData**: -9999.0
- **Masked pixels**: Top-left 2×2 corner
- **Expected**: Nodata pixels become `np.nan`

### TC-005: `dem_all_nodata.tif`
- **Purpose**: Test rejection of 100% nodata rasters
- **Data**: All pixels = -9999.0 (nodata)
- **Expected**: Raises `AllNoDataError`

### TC-006: `dem_no_crs.tif`
- **Purpose**: Test rejection of rasters without CRS
- **CRS**: None (stripped)
- **Expected**: Raises `MissingCRSError`

### TC-007: `rgb_image.tif`
- **Purpose**: Test rejection of multi-band rasters
- **Bands**: 3 (RGB)
- **Expected**: Raises `InvalidRasterError("Expected 1 band, got 3")`

### TC-009: `dem_known_values_4326.tif`
- **Purpose**: Bit-exact value preservation test
- **Key values**: `pixel[10,10] = 150.5`, `pixel[0,0] = 100.0`, `pixel[19,19] = 200.0`
- **Expected**: Values preserved exactly (no interpolation for same-CRS float32)

### TC-009b: `dem_known_values_4326_int16.tif`
- **Purpose**: Test dtype conversion (int16 → float32)
- **Key values**: Same positions as TC-009, but stored as int16
- **Expected**: Values preserved with `pytest.approx(rel=1e-6)`

### TC-010: `empty.tif`
- **Purpose**: Test rejection of empty files
- **Size**: 0 bytes
- **Expected**: Raises `InvalidRasterError("Empty or bandless file")`

### TC-011: `dem_85pct_nodata.tif`
- **Purpose**: Test warning for high nodata percentage
- **NoData coverage**: ~84% (336/400 pixels)
- **Valid region**: Bottom-right 8×8 corner
- **Expected**: Loads successfully, logs WARNING

### TC-012/TC-018: `dem_large.tif`
- **Purpose**: Memory budget and large file tests
- **Size**: 500×500 pixels (~1MB)
- **Data**: Random elevation values (seeded for reproducibility)
- **Expected**: Loads successfully; exceeds small memory budgets

### TC-013: `dem_extreme_elevations.tif`
- **Purpose**: Test extreme but valid elevation values
- **Values**: -430m (Dead Sea), +8849m (Everest), 0m (sea level)
- **Expected**: Loads successfully, preserves extreme values

### TC-014: `image.png`
- **Purpose**: Test rejection of non-GeoTIFF formats
- **Format**: PNG (not a valid DEM format)
- **Expected**: Raises `InvalidRasterError`

### TC-015: `dem_corrupted.tif`
- **Purpose**: Test handling of corrupted files
- **Content**: Valid TIFF header followed by truncated/invalid IFD
- **Expected**: Raises `InvalidRasterError("Corrupted...")`

### TC-016: `dem_polar_extreme.tif`
- **Purpose**: Test rejection of out-of-bounds coordinates after reprojection
- **CRS**: EPSG:3031 (Antarctic Polar Stereographic)
- **Bounds**: Far from pole, produces invalid WGS84 bounds when reprojected
- **Expected**: Raises `InvalidBoundsError`

### TC-019: `dem_nan_transform.tif`
- **Purpose**: Test rejection of invalid geotransform
- **Note**: File is structurally valid; actual NaN testing done via monkeypatch
- **Expected**: Raises `InvalidGeotransformError` (when transform contains NaN)

## Notes

1. **Fixtures are committed to VCS** for CI reproducibility
2. **Total size**: ~1MB (mostly `dem_large.tif`)
3. **Regeneration**: Safe to re-run `gen_fixtures.py` (idempotent)
4. **Not real terrain**: Synthetic data only - do not use for actual terrain analysis

## References

- FEAT-001 Spec: `spec/features/FEAT-001-load-dem.md` (v1.6.1)
- Generator: `scripts/gen_fixtures.py`
- Prompt: `PROMPTS/001-create-test-fixtures.md`
