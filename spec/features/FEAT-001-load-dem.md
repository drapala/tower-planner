# FEAT-001: LoadDEM

**Status**: Draft
**BC**: terrain
**Priority**: P0 (Foundation)
**Version**: 1.9.0

---

## Summary

Load a Digital Elevation Model (DEM) from a GeoTIFF file, normalize to system CRS (EPSG:4326), and return a validated TerrainGrid.

> **Ports & Adapters**
> - Domain Port: `TerrainRepository` (interface) in `domain/terrain/repositories.py`
> - Infrastructure Adapter: `GeoTiffTerrainAdapter` in `src/infrastructure/terrain/geotiff_adapter.py`
> - Responsibility: Adapter implements `load_dem()` using rasterio; Domain remains format‑agnostic.

---

## Function Signature

```python
def load_dem(file_path: Path | str) -> TerrainGrid:
    ...
```

### Domain Port

```python
class TerrainRepository(Protocol):
    def load_dem(self, file_path: Path | str) -> TerrainGrid: ...
```

> The infrastructure adapter (`GeoTiffTerrainAdapter`) implements this port.

### Input Normalization

> `file_path` accepts both `Path` and `str`. Internally, the function MUST convert to `Path` immediately:
> ```python
> file_path = Path(file_path)  # First line of implementation
> ```

---

## Input

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | `Path \| str` | Yes | Path to GeoTIFF file (normalized to `Path` internally) |

---

## Output

Returns: `TerrainGrid` (see definition below)

---

## TerrainGrid Definition

> **DDD Classification**: Value Object (immutable, identity by attributes, no UUID)
> **Location**: `domain/terrain/value_objects.py`

```python
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

class TerrainGrid(BaseModel):
    """Immutable elevation grid with geographic metadata (Value Object)."""

    data: NDArray[np.float32]         # 2D float32 array (height x width)
    bounds: BoundingBox               # Geographic extent in EPSG:4326
    crs: str                          # Always "EPSG:4326" (system CRS)
    resolution: tuple[float, float]   # (x_res, y_res) absolute values in degrees
    source_crs: str | None            # Original CRS before normalization (metadata only)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_grid(self) -> "TerrainGrid":
        # INV-1: 2D array
        if self.data.ndim != 2:
            raise ValueError(f"Data must be 2D, got {self.data.ndim}D")
        # INV-2, INV-3: Non-empty
        if self.data.shape[0] == 0 or self.data.shape[1] == 0:
            raise ValueError(f"Data cannot be empty: {self.data.shape}")
        # INV-4: dtype
        if self.data.dtype != np.float32:
            raise ValueError(f"Data must be float32, got {self.data.dtype}")
        # INV-5: CRS
        if self.crs != "EPSG:4326":
            raise ValueError(f"CRS must be EPSG:4326, got {self.crs}")
        # INV-7, INV-11: Resolution positive (absolute)
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValueError(f"Resolution must be positive: {self.resolution}")
        # INV-8: At least one valid pixel
        if not np.any(~np.isnan(self.data)):
            raise ValueError("Grid contains 100% NoData")
        return self
```

### TerrainGrid Invariants

| ID | Invariant | Validation |
|----|-----------|------------|
| INV-1 | `data.ndim == 2` | Must be 2D array |
| INV-2 | `data.shape[0] > 0` | Height > 0 |
| INV-3 | `data.shape[1] > 0` | Width > 0 |
| INV-4 | `data.dtype == np.float32` | Consistent type |
| INV-5 | `crs == "EPSG:4326"` | **System CRS enforced** |
| INV-6 | `bounds.is_valid()` | Validated by BoundingBox constructor |
| INV-7 | `resolution[0] > 0 and resolution[1] > 0` | Positive resolution |
| INV-8 | `np.any(~np.isnan(data))` | **At least one valid pixel** |
| INV-9 | `bounds.min_y >= -90 and bounds.max_y <= 90` | Validated by BoundingBox |
| INV-10 | `bounds.min_x >= -180 and bounds.max_x <= 180` | Validated by BoundingBox |
| INV-11 | `resolution` stores absolute values | **Independent of raster orientation** |

### Resolution Note

> `resolution` is expressed in CRS units (degrees for EPSG:4326).
> **Important**: 1° does NOT correspond to constant meters—distance per degree varies with latitude.
> Metric calculations (meters/pixel) belong to a separate feature (`FEAT-XXX: ReprojectToUTM`).

### Resolution Orientation Note

> Many GeoTIFFs have negative y_resolution (origin at top-left).
> **TerrainGrid always stores absolute (positive) resolution values**, independent of raster orientation.
> Implementation MUST normalize: `resolution = (abs(x_res), abs(y_res))`.
>
> Orientation guarantees after reprojection to EPSG:4326:
> - Grid is NORTH-UP (no rotation or flip).
> - Y-resolution is `abs(transform.e)` (always positive in the VO).
> - Row 0 corresponds to the northern edge (`bounds.max_y`).
>
> Domain assumes North‑Up consistently. Pixel→world mapping details (including sign in the affine transform) remain in the adapter; the domain stores only magnitudes and bounds.

---

## BoundingBox Definition

> **DDD Classification**: Value Object (Pydantic, validated on construction)
> **Location**: `domain/terrain/value_objects.py`

```python
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
        # BB-6: Longitude range
        if not (-180 <= self.min_x <= 180):
            raise ValueError(f"min_x longitude out of range: {self.min_x}")
        if not (-180 <= self.max_x <= 180):
            raise ValueError(f"max_x longitude out of range: {self.max_x}")
        # BB-5: Latitude range
        if not (-90 <= self.min_y <= 90):
            raise ValueError(f"min_y latitude out of range: {self.min_y}")
        if not (-90 <= self.max_y <= 90):
            raise ValueError(f"max_y latitude out of range: {self.max_y}")
        # BB-3: X ordering (see Antimeridian Note for ±180° crossing)
        if not (self.min_x < self.max_x):
            raise ValueError(f"Invalid x ordering: min_x={self.min_x} >= max_x={self.max_x}")
        # BB-4: Y ordering
        if not (self.min_y < self.max_y):
            raise ValueError(f"Invalid y ordering: min_y={self.min_y} >= max_y={self.max_y}")
        return self

    def is_valid(self) -> bool:
        """Returns True (always valid if constructed successfully).

        Provided for API compatibility - since validation occurs at construction,
        any existing BoundingBox instance is guaranteed valid.
        """
        return True
```

### BoundingBox Invariants

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| BB-1 | Order is `(min_x, min_y, max_x, max_y)` | **Fixed by field order** |
| BB-2 | Units match CRS | Degrees for EPSG:4326 |
| BB-3 | `min_x < max_x` | **Validated at construction** |
| BB-4 | `min_y < max_y` | **Validated at construction** |
| BB-5 | Latitude in `[-90, 90]` | **Validated at construction** |
| BB-6 | Longitude in `[-180, 180]` | **Validated at construction** |

> **Key difference from dataclass**: Invalid BoundingBox **cannot be created**.
> `BoundingBox(min_x=200, ...)` raises `ValidationError` immediately.

### Pixel Validity

> A "valid pixel" is any float32 value that is NOT `NaN`.
> - `NaN` denotes NoData (converted from source nodata).
> - Zero is NOT treated as NoData.
> - INV-8 requires at least one valid (non-NaN) pixel in the grid.

### Antimeridian (±180°) Handling

> **Current limitation**: The invariant `min_x < max_x` prevents representing rasters
> that cross the antimeridian (International Date Line at ±180° longitude).
>
> **For FEAT-001**: This is acceptable—Brazilian rasters do not cross ±180°.
>
> **Future consideration** (out of scope): If antimeridian crossing is needed:
> - Option A: Split into two BoundingBox instances
> - Option B: Allow `min_x > max_x` when crossing antimeridian (requires validator change)
> - Option C: Normalize to [0, 360] range internally

---

## Preconditions

| ID | Condition | Error |
|----|-----------|-------|
| PRE-1 | File exists at `file_path` | `FileNotFoundError` |
| PRE-2 | File is readable | `PermissionError` |
| PRE-3 | File is valid GeoTIFF (not corrupted) | `InvalidRasterError` |
| PRE-4 | File has exactly 1 band | `InvalidRasterError("Expected 1 band, got N")` |
| PRE-5 | CRS is defined in file | `MissingCRSError` |
| PRE-6 | File has valid geotransform | `InvalidGeotransformError` |
| PRE-7 | TIFF contains at least one valid IFD entry defining a raster band | `InvalidRasterError("Empty or bandless file")` |

### PRE-3 Clarification: Corrupted Files

> If the file exists, has `.tif` extension, but is corrupted (truncated, invalid headers, unreadable blocks), rasterio will raise `RasterioIOError` or similar.
> **Implementation MUST catch these and wrap in `InvalidRasterError`**:
> ```python
> except (rasterio.errors.RasterioIOError, rasterio.errors.RasterioError) as e:
>     raise InvalidRasterError(f"Corrupted or invalid raster: {e}") from e
> ```

---

## Postconditions

| ID | Condition | Validation |
|----|-----------|------------|
| POST-1 | `result.crs == "EPSG:4326"` | **CRS normalized to system standard** |
| POST-2 | `result.bounds` in EPSG:4326 coordinates | Reprojected if source differs |
| POST-3 | `result.bounds` follows `(min_x, min_y, max_x, max_y)` | Fixed order |
| POST-4 | `result.resolution > 0` | Positive values in degrees |
| POST-5 | NoData pixels are `np.nan` | Original nodata converted |
| POST-6 | **At least one pixel is NOT NoData** | `AllNoDataError` if 100% NaN |
| POST-7 | Non-nodata values unchanged if source is EPSG:4326 | No interpolation for same-CRS |
| POST-8 | `result.source_crs` contains original CRS | Metadata preserved |
| POST-9 | `result.bounds.is_valid() == True` | Bounds within WGS84 limits |

### POST-7 Clarification (Bit-Exact Conditions)

> **Bit-exact preservation** requires ALL of:
> 1. Source CRS == EPSG:4326 (no reprojection)
> 2. Source dtype == float32 (no type conversion)
>
> If source CRS == EPSG:4326 AND source dtype == float32:
>   → Non-nodata values MUST be **bit-identical** to source.
>
> If source CRS == EPSG:4326 BUT source dtype != float32:
>   → Values are **numerically equivalent** after cast to float32 (use `pytest.approx` with `rel=1e-6`).
>
> If source CRS != EPSG:4326:
>   → Values MAY differ due to reprojection interpolation.

### Reprojection Shape Note

> If source CRS != EPSG:4326, the resulting grid shape **MAY differ** from the source raster due to warping/resampling. This is expected behavior—do not write tests that assume shape preservation across CRS changes.

---

## CRS Normalization

```
┌─────────────────────┐
│ Source: EPSG:31983  │
│ Source: EPSG:4674   │  ──→  Output: EPSG:4326 (always)
│ Source: EPSG:4326   │
│ Source: Custom WKT  │
└─────────────────────┘
```

**Rationale**: All downstream calculations (LoS, distance, RF) require consistent CRS. EPSG:4326 is the system standard.

### Reprojection Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Resampling** | `Resampling.bilinear` | Good balance accuracy/speed for continuous data |
| **Target Resolution** | **Auto-calculated (library)** | Delegate to rasterio/PROJ for accuracy |
| **Bounds Alignment** | None (no snap-to-grid) | Preserve source extent |

#### Target Resolution Derivation (Preferred)

Use `rasterio.warp.calculate_default_transform` to derive the target transform, width, and height for EPSG:4326, letting PROJ decide appropriate resolution and shape. Extract positive resolution magnitudes from the resulting transform.

```python
from rasterio.warp import calculate_default_transform

dst_transform, dst_width, dst_height = calculate_default_transform(
    src_crs, "EPSG:4326", src_width, src_height, *src_bounds
)
x_res = abs(dst_transform.a)
y_res = abs(dst_transform.e)
```

> Fallback to approximate per‑axis conversion only if library derivation is unavailable.

---

## Adapter Lifecycle

Implementations should follow this lifecycle to avoid resource leaks and runtime warnings:

1. Open dataset with context manager: `with rasterio.open(file_path) as src:`
2. Optionally wrap operations in `with rasterio.Env(...):` for GDAL/PROJ config.
3. Read metadata (CRS, transform, width, height, nodata) and validate preconditions.
4. If CRS != EPSG:4326, compute target transform/shape via `calculate_default_transform` and reproject.
5. Convert nodata → `np.nan`, cast to `float32`, ensure at least one valid pixel.
6. Build `BoundingBox` from EPSG:4326 bounds and compute positive `resolution`.
7. Exit context(s) to close dataset and release GDAL handles.
8. Return immutable `TerrainGrid` VO.

## NoData Handling

| Scenario | Behavior |
|----------|----------|
| Source NoData value exists | Convert to `np.nan` |
| Source has no NoData defined | Preserve all values as-is |
| 100% pixels are NoData | **Raise `AllNoDataError`** |
| >80% pixels are NoData | Load successfully + **log warning** |
| <80% NoData | Load normally |

### Logging Specification

| Condition | Log Level | Message Format |
|-----------|-----------|----------------|
| >80% NoData | `WARNING` | `"DEM {file_path}: {pct}% NoData pixels detected"` |
| Reprojection performed | `INFO` | `"DEM {file_path}: Reprojected from {src_crs} to EPSG:4326"` |
| Load successful | `DEBUG` | `"DEM {file_path}: Loaded {width}x{height} grid"` |

> Use Python standard `logging` module. Logger name: `src.infrastructure.terrain.geotiff_adapter`
> (Infrastructure layer, not domain—file I/O is not business logic)

### Memory and Resource Limits

> Implementations MUST avoid process hangs under memory pressure. For extremely large rasters, estimate memory (`width × height × 4 bytes` for float32) before allocation. If the estimate exceeds a configurable budget, raise `InsufficientMemoryError`. If the runtime raises `MemoryError`, catch and wrap as `InsufficientMemoryError`.

### Thread Safety Note

> For consistent GDAL/PROJ configuration and better isolation across threads,
> adapters SHOULD wrap I/O operations in `rasterio.Env(...)` where appropriate.

---

## Error Hierarchy

```python
class TerrainError(Exception):
    """Base error for terrain operations."""

class InvalidRasterError(TerrainError):
    """File is not a valid raster, wrong format, or corrupted."""

class MissingCRSError(TerrainError):
    """Raster has no CRS defined."""

class InvalidGeotransformError(TerrainError):
    """Raster has invalid or missing geotransform."""

class AllNoDataError(TerrainError):
    """Raster contains 100% NoData pixels - unusable."""

class InvalidBoundsError(TerrainError):
    """Raster bounds are outside valid WGS84 range after reprojection."""

class InsufficientMemoryError(TerrainError):
    """Operation requires more memory than allowed or available."""
```

---

## Edge Cases

| Case | Behavior | Test |
|------|----------|------|
| Empty file (0 bytes) | Raise `InvalidRasterError("Empty file")` | TC-010 |
| **100% pixels are NoData** | Raise `AllNoDataError` | TC-005 |
| >80% NoData | Load + log warning | TC-011 |
| Large file (>1GB) | Load normally (memory is caller's concern) | TC-012 |
| Extremely large (exceeds memory budget) | Raise `InsufficientMemoryError` | TC-018 |
| Negative elevations (Dead Sea: -430m) | Valid, preserve as-is | TC-013 |
| Extreme elevations (Everest: 8849m) | Valid, preserve as-is | TC-013 |
| Non-GeoTIFF raster (.png, .jpg) | Raise `InvalidRasterError` | TC-014 |
| Corrupted GeoTIFF | Raise `InvalidRasterError("Corrupted...")` | TC-015 |
| Source CRS != EPSG:4326 | Reproject to EPSG:4326 | TC-002 |
| Source CRS == EPSG:4326 | No reprojection, bit-exact copy | TC-009 |
| Bounds outside WGS84 after reproject | Raise `InvalidBoundsError` | TC-016 |
| Geotransform contains NaN or Inf | Raise `InvalidGeotransformError` | TC-019 |

---

## Test Cases

### TC-001: Happy Path (EPSG:4326 source)
```python
def test_load_dem_valid_geotiff_4326():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    assert grid.data.shape == (100, 100)
    assert grid.data.dtype == np.float32
    assert grid.crs == "EPSG:4326"
    assert grid.bounds.is_valid()
    assert grid.source_crs == "EPSG:4326"
```

### TC-002: CRS Reprojection
```python
def test_load_dem_reprojects_to_4326():
    grid = load_dem("tests/fixtures/dem_utm23s.tif")
    assert grid.crs == "EPSG:4326"
    assert grid.source_crs == "EPSG:31983"
    assert -180 <= grid.bounds.min_x <= 180
    assert -90 <= grid.bounds.min_y <= 90
```

### TC-003: File Not Found
```python
def test_load_dem_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dem("nonexistent.tif")
```

### TC-004: NoData Conversion
```python
def test_load_dem_nodata_becomes_nan():
    grid = load_dem("tests/fixtures/dem_with_nodata.tif")
    assert np.isnan(grid.data[0, 0])  # Known nodata pixel
    assert not np.isnan(grid.data[50, 50])  # Known valid pixel
```

### TC-005: All NoData Rejected
```python
def test_load_dem_all_nodata_raises():
    with pytest.raises(AllNoDataError):
        load_dem("tests/fixtures/dem_all_nodata.tif")
```

### TC-006: Missing CRS
```python
def test_load_dem_missing_crs():
    with pytest.raises(MissingCRSError):
        load_dem("tests/fixtures/dem_no_crs.tif")
```

### TC-007: Multi-band Raster
```python
def test_load_dem_multiband_rejected():
    with pytest.raises(InvalidRasterError, match="Expected 1 band"):
        load_dem("tests/fixtures/rgb_image.tif")
```

### TC-008: BoundingBox Order and Validity
```python
def test_load_dem_bounds_valid():
    grid = load_dem("tests/fixtures/dem_100x100_4326.tif")
    assert grid.bounds.min_x < grid.bounds.max_x
    assert grid.bounds.min_y < grid.bounds.max_y
    assert grid.bounds.is_valid()
```

### TC-009: Bit-exact for same CRS
```python
def test_load_dem_bitexact_same_crs():
    grid = load_dem("tests/fixtures/dem_known_values_4326.tif")
    assert grid.data[10, 10] == pytest.approx(150.5, abs=0)
```

### TC-009b: Same CRS, non-float32 source
```python
def test_load_dem_same_crs_non_float32_uses_approx():
    grid = load_dem("tests/fixtures/dem_known_values_4326_int16.tif")
    # After cast to float32, values are numerically equivalent
    assert grid.data[10, 10] == pytest.approx(150.5, rel=1e-6)
```

### TC-010: Empty File
```python
def test_load_dem_empty_file():
    with pytest.raises(InvalidRasterError, match="Empty file"):
        load_dem("tests/fixtures/empty.tif")
```

### TC-011: High NoData Warning (>80%)
```python
def test_load_dem_high_nodata_logs_warning(caplog):
    with caplog.at_level(logging.WARNING):
        grid = load_dem("tests/fixtures/dem_85pct_nodata.tif")
    assert "NoData pixels detected" in caplog.text
    assert grid is not None  # Still loads successfully
```

### TC-012: Large File Loads
```python
@pytest.mark.slow
def test_load_dem_large_file():
    # 1GB+ test file (skip in CI if not available)
    grid = load_dem("tests/fixtures/dem_large.tif")
    assert grid.data.shape[0] > 0
```

### TC-013: Extreme Elevations Valid
```python
def test_load_dem_extreme_elevations():
    grid = load_dem("tests/fixtures/dem_extreme_elevations.tif")
    assert np.nanmin(grid.data) < 0  # Below sea level OK
    assert np.nanmax(grid.data) > 8000  # Everest OK
```

### TC-014: Non-GeoTIFF Rejected
```python
def test_load_dem_non_geotiff():
    with pytest.raises(InvalidRasterError):
        load_dem("tests/fixtures/image.png")
```

### TC-015: Corrupted File
```python
def test_load_dem_corrupted_file():
    with pytest.raises(InvalidRasterError, match="Corrupted"):
        load_dem("tests/fixtures/dem_corrupted.tif")
```

### TC-016: Invalid Bounds After Reprojection
```python
def test_load_dem_invalid_bounds_after_reproject():
    # Synthetic raster with bounds that become invalid in EPSG:4326
    with pytest.raises(InvalidBoundsError):
        load_dem("tests/fixtures/dem_polar_extreme.tif")
```

### TC-017: Path vs String Input
```python
def test_load_dem_accepts_path_and_string():
    from pathlib import Path

    path_str = "tests/fixtures/dem_100x100_4326.tif"
    path_obj = Path(path_str)

    grid1 = load_dem(path_str)
    grid2 = load_dem(path_obj)

    assert np.array_equal(grid1.data, grid2.data, equal_nan=True)
```

### TC-018: Memory Budget Exceeded
```python
def test_load_dem_memory_budget_exceeded(monkeypatch):
    # Simulate adapter estimating memory above budget
    from src.infrastructure.terrain.geotiff_adapter import GeoTiffTerrainAdapter

    adapter = GeoTiffTerrainAdapter(max_bytes=10_000)  # Tiny budget
    with pytest.raises(InsufficientMemoryError):
        adapter.load_dem("tests/fixtures/dem_large.tif")
```

### TC-019: Invalid Geotransform (NaN/Inf)
```python
def test_load_dem_nan_inf_geotransform():
    # Geotransform with NaN or Inf values must be rejected
    with pytest.raises(InvalidGeotransformError, match="NaN|Inf"):
        load_dem("tests/fixtures/dem_nan_transform.tif")
```

> **Note**: PRE-6 requires valid geotransform. NaN/Inf values in affine
> transform coefficients indicate corrupted or invalid raster metadata.

---

## SEC-01: Performance Verification Requirements

Performance requirements enable benchmarks, CI regression detection, and informed decisions
about future native kernels (C++/Rust).

| ID | Requirement | Target | Measurement |
|----|-------------|--------|-------------|
| PERF-1 | Same-CRS raster load (100MB) | < 2 seconds | `pytest-benchmark` |
| PERF-2 | Reprojection load (100MB UTM → EPSG:4326) | < 5 seconds | `pytest-benchmark` |
| PERF-3 | Memory overhead | < 2× raster size (float32) | Peak RSS during load |
| PERF-4 | OOM handling | Fail fast with `InsufficientMemoryError` | Budget check before allocation |
| PERF-5 | Version regression tolerance | < 20% degradation between versions | CI benchmark comparison |
| PERF-6 | Thread safety | Safe for concurrent reads (different files) | No global state; `rasterio.Env()` isolation |
| PERF-7 | Pre-flight size check | Fail fast if file > 2× budget | `stat().st_size` before `rasterio.open()` |

### Performance Notes

- **PERF-1/PERF-2**: Targets assume SSD storage; HDD may be slower.
- **PERF-3**: Memory budget is `width × height × 4` bytes for float32.
- **PERF-5**: Significant regressions should block merge and require investigation.
- **PERF-6**: Same-file concurrent reads are OS-dependent (file locking); use separate handles or process parallelism.
- **PERF-7**: ✅ Implemented - checks `st_size > max_bytes * 2` before opening with rasterio.

---

## SEC-02: Security Considerations

Security requirements protect against malicious files, DoS via oversized rasters,
and unintended behavior in CI/cluster environments.

| ID | Requirement | Rationale |
|----|-------------|-----------|
| SEC-1 | No external command execution | Adapter must not shell out to GDAL CLI or other tools |
| SEC-2 | Corrupted files raise `InvalidRasterError` immediately | Fail fast; no partial processing of malformed data |
| SEC-3 | Oversized files raise `InsufficientMemoryError` | Prevent OOM crashes via pre-allocation budget check |
| SEC-4 | Use `rasterio.Env()` for PROJ isolation | Prevent environment variable leakage between threads |
| SEC-5 | No symlink traversal outside sandbox | Adapter should not follow symlinks without explicit validation |
| SEC-6 | Reject non-GeoTIFF extensions | Files like `.png`, `.jpg`, `.bin` must raise `InvalidRasterError` |
| SEC-7 | No sensitive paths in logs | Avoid info leakage; log only filenames, not full paths |
| SEC-8 | Thread-safe (no global state) | Safe for multithreaded execution |
| SEC-9 | No silent failure masking | Every inconsistency must raise a specific exception |
| SEC-10 | Extension allowlist | Only `.tif`/`.tiff` extensions accepted; others raise `InvalidRasterError` |

### Security Notes

- **SEC-1**: All raster operations use rasterio's Python API only.
- **SEC-5**: ✅ Implemented - explicit symlink rejection (adapter raises `InvalidRasterError("Symlinks are not permitted")`).
- **SEC-6**: Extension validation is implicit via rasterio driver detection; explicit check may be added.
- **SEC-7**: ✅ Implemented - logging uses `path.name` (filename only) and avoids embedding absolute paths; stat errors log only filename, errno, and strerror.
- **SEC-10**: ✅ Implemented - explicit extension check before processing (`.tif`, `.tiff` only).

---

## Dependencies

- `rasterio>=1.3.0` — GeoTIFF reading and reprojection
- `numpy>=1.24.0` — Array handling
- `pyproj>=3.4.0` — CRS validation (via rasterio)

---

## Test Fixtures

The test suite references GeoTIFF fixtures under `tests/fixtures/`. To ensure reproducibility and avoid large binaries in VCS, provide generation scripts and/or documented steps for each fixture:

```
tests/fixtures/
├── dem_100x100_4326.tif                 # Synthetic float32 grid, EPSG:4326
├── dem_utm23s.tif                        # Synthetic grid in EPSG:31983 (UTM 23S)
├── dem_with_nodata.tif                   # Includes defined nodata value
├── dem_all_nodata.tif                    # 100% nodata
├── dem_no_crs.tif                        # Missing CRS tag
├── rgb_image.tif                         # Multi-band (3-band) image
├── dem_known_values_4326.tif             # Known pixel values for assertions
├── dem_known_values_4326_int16.tif       # Same content as above but int16 dtype
├── empty.tif                             # 0-byte file
├── dem_85pct_nodata.tif                  # ~85% nodata coverage
├── dem_large.tif                         # Large raster (size varies; mark as slow)
├── dem_extreme_elevations.tif            # Contains negative and >8000m elevations
├── image.png                             # Non-GeoTIFF
├── dem_polar_extreme.tif                 # Reprojection pushes bounds outside WGS84
├── dem_corrupted.tif                     # Truncated/invalid TIFF for TC-015
└── dem_nan_transform.tif                 # NaN in geotransform for TC-019
```

Include a `tests/fixtures/README.md` or script (e.g., `scripts/gen_fixtures.py`) detailing how to generate each file with rasterio/numpy.

---

## Decisions Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| Target CRS? | **EPSG:4326** | Global standard, simple for LoS/distance |
| Unit conversion? | **No** | Assume meters; document requirement |
| Lazy loading? | **No** | Out of scope; chunking is separate feature |
| Other formats? | **No** | GeoTIFF only for FEAT-001 |
| NoData handling? | **→ np.nan** | Numpy standard for missing data |
| 100% NoData? | **Error** | Prevents silent failures downstream |
| Resampling method? | **Bilinear** | Good for continuous elevation data |
| Target resolution? | **Auto-calculated** | Preserve approximate ground distance |
| Thread safety? | **Read-safe** | Different files OK, same file undefined |

---

## Acceptance Criteria

- [ ] All 19 test cases pass (TC-001 through TC-019)
- [ ] `load_dem()` returns normalized EPSG:4326 grid
- [ ] Port/Adapter: `TerrainRepository` (domain) and `GeoTiffTerrainAdapter` (infrastructure)
- [ ] CRS reprojection works for common Brazilian CRS (31983, 4674)
- [ ] TerrainGrid is immutable (Pydantic V2 frozen model)
- [ ] BoundingBox is immutable (Pydantic V2 frozen model)
- [ ] BoundingBox.is_valid() returns True for all valid instances
- [ ] 100% NoData raises `AllNoDataError`
- [ ] >80% NoData logs warning but loads
- [ ] Corrupted files raise `InvalidRasterError`
- [ ] Extremely large rasters fail gracefully with `InsufficientMemoryError`
- [ ] Type hints complete and mypy passes
- [ ] Docstring with examples
- [ ] Performance within baseline (non-blocking)
- [ ] Symlinks are rejected with `InvalidRasterError`

---

<!--
FEAT-001 v1.9.0 | Last updated: 2025-12-06

IMPORTANT: Always update the "Last updated" date above when modifying this file.
Format: YYYY-MM-DD

Changelog:
- v1.9.0 (2025-12-06): Added PERF-7 (pre-flight size check) and SEC-10 (extension allowlist)
- v1.8.0 (2025-12-06): Added SEC-7 (log filenames only)
-->
