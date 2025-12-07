# CLAUDE.md — AI Development Contract

This document defines **how Claude (or any AI agent)** must generate code,
tests, and documentation for the **Tower Planner** project using:

- **SDD** (Specification-Driven Development)
- **TDD** (Test-Driven Development)
- **DDD** (Domain-Driven Design)

Claude must **obey this document above all others**, except `spec/SPEC.md`,
which is the definitive source of truth.

---

## 1. Fundamental Rules

1. Claude MUST NOT generate code outside the SPEC.
2. Every new feature must first be defined in `spec/SPEC.md`.
3. Claude ALWAYS generates TESTS before implementation code (TDD).
4. Claude must preserve DDD architecture and bounded context boundaries.
5. Claude MUST NOT introduce new libraries without technical justification.
6. Claude must document important decisions in generated PRs.
7. Claude MUST NOT move files without explicit reason.
8. Claude must maintain consistent style with the existing codebase.
9. Claude MUST NOT sign commits (no "Generated with Claude Code" or "Co-Authored-By").

---

## 2. Directory Structure

```
tower-planner/
├── CLAUDE.md              # This file - AI development contract
├── README.md              # Project overview
├── pyproject.toml         # Python project configuration
├── spec/
│   ├── SPEC.md            # Source of truth for specifications
│   └── features/          # Feature specifications (FEAT-XXX)
├── domain/                # Domain layer (DDD) - NO I/O HERE
│   ├── __init__.py
│   ├── terrain/           # BC: Physical geography
│   │   ├── __init__.py
│   │   ├── value_objects.py       # GeoPoint, BoundingBox, TerrainGrid (VO)
│   │   ├── repositories.py        # TerrainRepository (Port interface)
│   │   ├── services.py            # LoSCalculator (pure domain logic)
│   │   └── errors.py              # TerrainError hierarchy
│   ├── coverage/          # BC: RF propagation
│   │   ├── __init__.py
│   │   ├── value_objects.py
│   │   └── services.py
│   └── siting/            # BC: Site selection
│       ├── __init__.py
│       ├── entities.py        # CandidateSite, Tower (have UUID)
│       ├── value_objects.py
│       └── services.py
├── src/                   # Infrastructure layer (I/O, APIs)
│   ├── __init__.py
│   └── infrastructure/
│       └── terrain/
│           ├── __init__.py
│           └── geotiff_adapter.py   # GeoTiffTerrainAdapter implements TerrainRepository.load_dem()
└── tests/
    ├── __init__.py
    ├── terrain/
    ├── coverage/
    ├── siting/
    └── gis/               # Infrastructure tests
```

### Path Prefix Note

- Domain code resides at project root under `domain/…` (not `src/domain/…`).
- Infrastructure code resides under `src/infrastructure/…`.
- This convention keeps the domain independent from application/infrastructure concerns.

### Layer Responsibilities

| Layer | Location | Contains | I/O Allowed |
|-------|----------|----------|-------------|
| **Domain** | `domain/` | Value Objects, Entities, Domain Services | **NO** |
| **Infrastructure** | `src/` | File loaders, API clients, CLI | **YES** |
| **Tests** | `tests/` | All test code | YES |

### Test Stubs Strategy

For TDD without installing heavy external dependencies (e.g., `rasterio`, `GDAL`),
lightweight stubs are placed in `tests/stubs/` and added to `sys.path` during test runs:

```
tests/
├── stubs/
│   ├── rasterio/          # Stub for rasterio (monkeypatched in tests)
│   │   ├── __init__.py
│   │   ├── enums.py
│   │   ├── transform.py
│   │   └── warp.py
│   └── affine/            # Stub for affine library
│       └── __init__.py
└── conftest.py            # Adds tests/stubs to sys.path via pytest_configure
```

**Rules**:
1. Stubs provide **only symbols** required for imports — no real functionality
2. All behavior is **monkeypatched** in individual test functions
3. Stubs take precedence during tests: `tests/conftest.py` inserts `tests/stubs/`
   at `sys.path[0]`, so stubs are loaded before any installed packages
4. Stubs are for **testing only** — they live under `tests/`, not `src/`

**Precedence**:
    `tests/conftest.py` inserts `tests/stubs/` at `sys.path` position 0, so these stubs
    take precedence over any installed rasterio in site-packages. This ensures
    unit tests use the stub by default.

    Real rasterio is only used when tests explicitly opt out via:
    - `real_rasterio_path` fixture (session scope, excludes stubs from sys.path)
    - `exclude_stubs_path` fixture (function scope, via monkeypatch)
    Both fixtures are defined in `tests/gis/conftest.py`.

**Completed (TD-002)**: Stubs moved from `src/` to `tests/stubs/` to avoid import
conflicts with real packages. See `spec/tech-debt/FEAT-001-tech-debt.md`.

---

## 3. Bounded Contexts

### 3.1 terrain/ — Physical Geography

**Responsibility**: Represents the physical world - terrain, elevation, coordinates, line-of-sight.

| Type | Components |
|------|------------|
| **Entities** | (none yet) |
| **Value Objects** | GeoPoint, Elevation, BoundingBox, TerrainGrid |
| **Services** | LoSCalculator |

**Invariants**:
- GeoPoint coordinates must be valid WGS84 (lat: -90 to 90, lon: -180 to 180)
- Elevation must be in meters above sea level
- TerrainGrid resolution must be positive (absolute values)
- BoundingBox validates bounds at construction (Pydantic)

### 3.2 coverage/ — RF Propagation

**Responsibility**: Radio frequency signal propagation, pathloss calculations, coverage analysis.

| Type | Components |
|------|------------|
| **Entities** | CoverageMap |
| **Value Objects** | PathlossResult, SignalStrength, Frequency |
| **Services** | PropagationModel |

**Invariants**:
- Frequency must be positive (in Hz, MHz, or GHz with explicit unit)
- PathlossResult must include the model used for calculation
- SignalStrength must be in dBm

### 3.3 siting/ — Site Selection

**Responsibility**: Candidate site evaluation, constraints, optimization, decision logic.

| Type | Components |
|------|------------|
| **Entities** | CandidateSite, Tower |
| **Value Objects** | Score, Constraint, Zone |
| **Services** | SiteSelector, Optimizer |

**Invariants**:
- Score must be normalized (0.0 to 1.0) or explicitly documented otherwise
- Constraints must be composable (AND/OR logic)
- CandidateSite must reference a valid GeoPoint from terrain BC

### 3.4 Evolution Path

When complexity grows, split bounded contexts:

```
Current:              Future (when needed):
terrain/         →    terrain/ + geo/ (coordinate systems, projections)
coverage/        →    coverage/ + propagation/ (multiple RF models)
siting/          →    siting/ + constraints/ + regulations/
```

---

## 4. DDD Rules for Claude

### DO:

**Create Value Objects for**:
- GeoPoint, Elevation, BoundingBox, TerrainGrid
- PathlossResult, SignalStrength, Frequency, CoverageMap
- Score, Constraint, Zone

> **Value Object Rule**: All validation MUST occur at construction time via Pydantic `@model_validator`.
> Invalid Value Objects cannot be instantiated.

**Create Entities for**:
- CandidateSite, Tower (have identity/UUID)

**Note**: TerrainGrid and CoverageMap are **Value Objects** (immutable, identity by attributes, no UUID).

**Create Domain Services for**:
- LoS (Line of Sight) calculations
- Pathloss/propagation simulation
- Candidate site selection and scoring

### DO NOT:

- Place domain logic in controllers
- Place domain logic in utility functions
- Place domain logic in infrastructure classes (GIS I/O, file readers)
- Create "god" services that span multiple bounded contexts
- Expose external library exceptions (e.g., `rasterio.errors`) to the domain. **Always wrap** in Domain Exceptions.

### Protocols for Ports:

> **Rule**: Domain Ports (Repositories/Gateways) MUST be defined as `typing.Protocol`, not abstract base classes (`abc.ABC`).
> This ensures structural typing and better decoupling.

### Separation:

| Layer | Location | Contains |
|-------|----------|----------|
| Domain | `/domain` | Business logic, rules, calculations |
| Infrastructure | `/src/infrastructure` | I/O, APIs, orchestration |
| Tests | `/tests` | All test code |

---

## 5. SDD Rules for Claude

When generating code:

1. **Read** `spec/SPEC.md` and understand the requirement
2. **Validate** the spec covers the requested functionality
3. **If spec is incomplete** → Request spec update before coding
4. **Only generate code** after tests exist (see TDD rules)
5. **Verify** pre-conditions and post-conditions from SPEC are enforced

### Spec Completeness Checklist:

- [ ] Feature has a unique identifier (e.g., `FEAT-001`)
- [ ] Inputs and outputs are defined
- [ ] Edge cases are documented
- [ ] Invariants are specified
- [ ] Dependencies on other features are noted
- [ ] Ports (interfaces) and Adapters (implementations) are identified when I/O is involved

---

## 6. TDD Rules for Claude

For every new functionality:

1. **Create tests** in `/tests/{bounded_context}/`
2. **Run tests** to verify they fail (red phase)
3. **Implement** minimal code to pass tests (green phase)
4. **Refactor** if necessary (refactor phase)

### Test Coverage Requirements:

| Type | Required |
|------|----------|
| Happy path | Yes |
| Error cases | Yes |
| Edge cases | Yes |
| Known synthetic data | Yes |
| SPEC invariants | Yes |
| Ports/Adapters behavior | Yes |

### Test Naming Convention:

```python
def test_{method}_{scenario}_{expected_result}():
    """Example: test_calculate_los_obstructed_terrain_returns_false"""
    pass
```

---

## 7. Code Style

- **Python**: 3.12+
- **Type hints**: Required (PEP 484)
- **Data classes**: Pydantic V2 for value objects when validation needed
- **Functions**: Pure functions in domain layer when possible
- **Side effects**: Avoided in domain; allowed only in infrastructure
- **Naming**: Explicit, clear, no abbreviations (except standard: `lat`, `lon`, `db`)

### Example Value Object:

```python
from pydantic import BaseModel, ConfigDict, Field, model_validator

class GeoPoint(BaseModel):
    """Immutable geographic coordinate in WGS84 (Value Object)."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    model_config = ConfigDict(frozen=True)

    # Optional: complex validation via model_validator
    @model_validator(mode="after")
    def validate_point(self) -> "GeoPoint":
        # Add complex invariants here if needed
        return self
```

### Example Value Object with NDArray:

```python
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np

class TerrainGrid(BaseModel):
    """Immutable elevation grid (Value Object)."""

    data: NDArray[np.float32]
    crs: str

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_grid(self) -> "TerrainGrid":
        if self.data.ndim != 2:
            raise ValueError("Data must be 2D")
        return self

```

### Example Port (Repository) and Adapter

```python
# domain/terrain/repositories.py
from typing import Protocol
from pathlib import Path

class TerrainRepository(Protocol):
    def load_dem(self, file_path: Path | str): ...

# src/infrastructure/terrain/geotiff_adapter.py
class GeoTiffTerrainAdapter:
    def __init__(self, max_bytes: int | None = None):
        self.max_bytes = max_bytes

    def load_dem(self, file_path: Path | str):
        """Implements TerrainRepository.load_dem using rasterio."""
        ...
```

### Example Entity:

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4

@dataclass
class CandidateSite:
    """A potential tower installation site."""

    id: UUID = field(default_factory=uuid4)
    location: GeoPoint
    scores: dict[str, float] = field(default_factory=dict)

    def overall_score(self) -> float:
        """Calculate weighted average of all scores."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)
```

---

## 8. PR Workflow

Each Claude-generated PR must include:

### Required Sections:

```markdown
## Summary
Brief description of what changed and why.

## SPEC Reference
Which spec items this implements (e.g., FEAT-001, FEAT-002).

## Tests Added
- test_xxx_yyy_zzz
- test_aaa_bbb_ccc

## DDD Compliance
- [ ] Logic is in correct bounded context
- [ ] No domain logic in infrastructure
- [ ] Value objects are immutable
- [ ] Entities have identity

## Refactoring Notes
Any structural changes made and reasoning.
```

---

## 9. Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks (run once after clone)
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Run all tests
pytest

# Run tests for specific BC
pytest tests/terrain/
pytest tests/coverage/
pytest tests/siting/

# Type checking
mypy domain/ src/

# Linting + Format check
black --check domain/ src/ tests/
isort --check-only domain/ src/ tests/

# Format code
black domain/ src/ tests/
isort domain/ src/ tests/
```

---

## 10. Security and Consistency Rules

### Claude MUST NOT:

- Invent geographic data (use only provided or synthetic test data)
- Invent RF formulas (use only documented propagation models)
- Modify the SPEC without explicit user approval
- Create classes with multiple responsibilities
- Generate code that breaks existing tests
- Use deprecated or unmaintained libraries

### Claude MUST:

- Validate all external inputs at system boundaries
- Use parameterized queries if any database interaction is added
- Log security-relevant operations
- Log security-relevant operations
- Document any assumptions made

### File Extension Allowlist (SEC-10):

> **Rule**: Adapters handling file inputs MUST enforce an allowlist of permitted extensions (e.g., `.tif`, `.tiff`) **before** opening the file.
> Reject invalid extensions with a Domain Exception (e.g., `InvalidRasterError`).

### Approved Libraries

- Geospatial I/O and CRS: `rasterio`, `pyproj`
- Numerical arrays: `numpy`
- Testing and performance: `pytest`, `pytest-benchmark`

---

## 11. Conflict Resolution

When code conflicts with SPEC:

1. **SPEC wins** — code must be updated to match SPEC
2. If SPEC seems wrong → **Ask user** before changing SPEC
3. Document the conflict and resolution in PR

When existing code conflicts with new requirements:

1. **Update tests first** to reflect new requirements
2. **Then update code** to pass new tests
3. **Verify** no regressions in existing tests

---

## 12. Versioning

### SPEC.md Versioning:

- Use semantic versioning: `SPEC v1.0.0`
- Document breaking changes
- Keep changelog in SPEC.md

### CLAUDE.md Updates:

- Any structural change requires explicit PR
- Update version comment at end of file
- Document what changed and why

---

## 13. Priority Hierarchy

When in doubt, follow this priority:

1. **`spec/SPEC.md`** — The absolute source of truth
2. **`CLAUDE.md`** — This file (development rules)
3. **Existing code** — Patterns already established
4. **Existing tests** — Expected behaviors

---

## 14. Code Review Patterns

Common patterns to follow during code review and refactoring:

### Logging

- **Module-level loggers**: Always define `logger = logging.getLogger(__name__)` at module level, outside of classes
- **Never create instance loggers**: Avoid `self.logger = ...` inside `__init__` or methods
- **Lazy formatting**: Use `%s` formatting in log calls, not f-strings

```python
# Good
logger = logging.getLogger(__name__)

class MyAdapter:
    def do_work(self):
        logger.debug("Processing %s", data)

# Bad
class MyAdapter:
    def do_work(self):
        logger = logging.getLogger(__name__)  # Don't recreate logger per call
        logger.debug(f"Processing {data}")   # Don't use f-strings in logs
```

### Exception Handling

- **Narrow exceptions**: Catch specific exceptions like `(AttributeError, TypeError)` instead of broad `Exception`
- **Document fallback behavior**: Add comments explaining why fallback is needed

```python
# Good
try:
    bounds_tuple = (sb.left, sb.bottom, sb.right, sb.top)
except (AttributeError, TypeError):
    # Fallback for bounds returned as tuple instead of object
    bounds_tuple = tuple(sb)

# Bad
except Exception:  # Too broad - hides real errors
    bounds_tuple = tuple(sb)
```

### Test Fixtures and Assertions

- **Scoped fixtures over global mutations**: Use pytest fixtures with `monkeypatch` instead of modifying globals in `pytest_configure`
- **Exact assertions**: Use exact values (`== -430.0`) instead of loose comparisons (`< 0`) when testing known values
- **Descriptive function names**: Name generators to describe purpose (e.g., `gen_dem_for_nan_transform_test`)

```python
# Good
assert np.nanmin(grid.data) == -430.0
assert np.nanmax(grid.data) == 8849.0

# Bad
assert np.nanmin(grid.data) < 0
assert np.nanmax(grid.data) > 8000
```

### Modern Python Patterns (3.12+)

- **Builtin generics**: Use `tuple[T, T]` instead of `Tuple[T, T]` from `typing` module
- **Union syntax**: Use `X | None` instead of `Optional[X]`
- **Import order**: Import order should match `__all__` order for consistency

```python
# Good
from __future__ import annotations
resolution: tuple[float, float]
source_crs: str | None = None

# Bad
from typing import Tuple, Optional
resolution: Tuple[float, float]
source_crs: Optional[str] = None
```

### Script Entry Points

- **Return exit codes**: Scripts should return `int` (0 success, 1 failure)
- **Error handling**: Wrap I/O operations in try/except with informative messages
- **Verification**: Validate expected outputs (e.g., file counts)

```python
def main() -> int:
    try:
        ensure_dir()
    except OSError as e:
        print(f"ERROR: Cannot create directory: {e}")
        return 1
    # ... do work ...
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### Symlink Policy (SEC-5)

- **Explicitly reject symlinks**: Adapters handling filesystem inputs must reject symlinks to avoid traversal outside sandboxed areas.
- **Use `Path.is_symlink()` before opening**: Raise a domain error (e.g., `InvalidRasterError`) with a generic message (no target path in message or logs).

```python
path = Path(file_path)
if path.is_symlink():
    raise InvalidRasterError("Symlinks are not permitted")
```

### Security in Logging (SEC-7)

- **Log filenames, not full paths**: Use `path.name` instead of `path` to avoid info leakage
- **No sensitive data in logs**: Avoid logging credentials, tokens, or internal paths
- **Use lazy formatting**: Combine with `%s` formatting for deferred evaluation

```python
# Good - SEC-7 compliant
logger.warning("DEM %s: %.1f%% NoData detected", path.name, nodata_pct)
logger.info("DEM %s: Reprojected from %s to EPSG:4326", path.name, src_crs)

# Bad - exposes full filesystem path
logger.warning(f"DEM {path}: {nodata_pct:.1f}% NoData detected")
logger.info(f"DEM {path}: Reprojected from {src_crs} to EPSG:4326")
```

### File Validation Order and TOCTOU (I/O Boundaries)

- Error precedence: Check existence first; missing files raise `FileNotFoundError` regardless of extension. Only then check extension allowlist for existing files (`.tif`/`.tiff`) and raise a domain error for disallowed types.
- Avoid TOCTOU: Do not pre-check permissions with `os.access()`. Attempt the operation and catch `PermissionError`; re-raise `PermissionError(path.name)` to avoid leaking paths in logs.
- Exceptions vs logs: Exceptions may include full paths for debugging; at logging boundaries apply SEC-7 (log filename only, include errno/strerror; no absolute paths).

### Redundant Conditionals After Validation

- **Remove conditionals for already-validated values**: If a value is validated (e.g., checked for None) and raises on failure, subsequent conditionals for the same check are redundant
- **Trust the validation**: Code after validation can assume the invariant holds

```python
# Good - validation guarantees src.crs is not None
if src.crs is None:
    raise MissingCRSError("Raster has no CRS defined")
src_crs_str = src.crs.to_string()  # Direct call, no conditional needed

# Bad - redundant conditional after validation
if src.crs is None:
    raise MissingCRSError("Raster has no CRS defined")
src_crs_str = src.crs.to_string() if src.crs else None  # Redundant check
```

### Redundant Type Casts

- **Trust dtype parameters**: If a function accepts `dtype` or `out_dtype` parameter, trust it returns that type
- **Preserve dtype in operations**: Use typed NaN values (e.g., `np.float32(np.nan)`) in `np.where` to avoid implicit upcasting
- **Avoid chained `.astype()` calls**: One cast is sufficient; additional casts add overhead

```python
# Good - out_dtype="float32" already returns float32
data = src.read(1, masked=True, out_dtype="float32")
if hasattr(data, "mask"):
    data = np.where(data.mask, np.float32(np.nan), data.data)  # Preserves float32

# Bad - redundant .astype() after out_dtype already specified
data = src.read(1, masked=True, out_dtype="float32").astype(np.float32, copy=False)
if hasattr(data, "mask"):
    data = np.where(data.mask, np.nan, data.data).astype(np.float32, copy=False)
```

### Pre-Allocation Validation (PERF-7)

- **Check budget BEFORE allocation**: Validate memory constraints before `src.read()` or `np.full()`
- **Use dimensions, not data shape**: Use `src.width * src.height * bytes_per_element` before reading
- **Fail fast**: Raise early to avoid wasting resources on allocations that will be rejected

```python
# Good - check BEFORE allocation
if self.max_bytes is not None:
    est_bytes = src.width * src.height * 4  # float32 = 4 bytes
    if est_bytes > self.max_bytes:
        raise InsufficientMemoryError(...)
data = src.read(1, masked=True, out_dtype="float32")

# Bad - check AFTER allocation (memory already wasted)
data = src.read(1, masked=True, out_dtype="float32")
height, width = data.shape
if self.max_bytes is not None:
    est_bytes = width * height * 4
    if est_bytes > self.max_bytes:
        raise InsufficientMemoryError(...)  # Too late!
```

### Narrow OS Error Handling

- **Catch specific OS errors**: Use `(FileNotFoundError, NotADirectoryError)` for path issues, not bare `OSError`
- **Re-raise unexpected errors**: Log and re-raise other OSError types (permissions, I/O)
- **Use path.name in exceptions**: Avoid full path leakage in exception messages (SEC-7)

```python
# Good - narrow handling with logging
try:
    st = path.stat()
except (FileNotFoundError, NotADirectoryError):
    # Benign: path structure issues - let downstream handle
    pass
except OSError as e:
    # Unexpected: log and fail fast
    # SEC-7: Log filename only, avoid embedding exception text which may contain full path
    logger.error("Failed to stat %s (errno=%s)", path.name, getattr(e, "errno", "?"))
    raise

# Good - Exception messages may contain full path for debugging, but LOGS must be sanitized
if not path.exists():
    raise FileNotFoundError(str(path))  # Full path allowed in exception

# Bad - swallows unexpected errors
except OSError:
    pass  # Hides permissions errors, I/O failures, etc.
```

### Test Fixture Constants

- **Define expected fixtures as constant**: Use `EXPECTED_FIXTURES` list at module level
- **Single source of truth**: All fixture-related tests reference the same constant
- **Update one place**: Adding/removing fixtures requires changing only the constant

```python
# Good - single source of truth
EXPECTED_FIXTURES = [
    "dem_100x100_4326.tif",
    "dem_utm23s.tif",
    # ... all fixtures
]

@pytest.mark.parametrize("filename", EXPECTED_FIXTURES)
def test_fixture_exists(filename):
    assert (FIXTURES_DIR / filename).exists()

def test_fixture_count():
    files = [f for f in FIXTURES_DIR.iterdir() if f.suffix == ".tif"]
    assert len(files) == len(EXPECTED_FIXTURES)

# Bad - hardcoded counts duplicated everywhere
assert len(files) == 16  # Magic number, easy to forget updating
```

### Session Fixture Cleanup

- **Use try/finally for session fixtures**: Session-scoped fixtures can't use monkeypatch
- **Modify sys.path IN PLACE**: Reassignment (`sys.path = ...`) breaks external references; use `clear()` + `extend()`
- **Prefer function scope with monkeypatch**: When isolation matters more than efficiency
- **Document scope reasoning**: Explain why session vs function scope

```python
# Good - exception-safe session fixture with in-place modification
@pytest.fixture(scope="session")
def real_rasterio_path():
    original = sys.path.copy()
    try:
        # Modify IN PLACE to preserve external references
        filtered = [p for p in sys.path if "stubs" not in p]
        sys.path.clear()
        sys.path.extend(filtered)
        yield
    finally:
        # Restore original contents IN PLACE
        sys.path.clear()
        sys.path.extend(original)

# Good - function scope with in-place modification + monkeypatch cleanup
@pytest.fixture(scope="function")
def exclude_stubs_path(monkeypatch):
    original = sys.path.copy()

    def restore():
        sys.path.clear()
        sys.path.extend(original)

    monkeypatch.callback(restore)

    filtered = [p for p in sys.path if "stubs" not in p]
    sys.path.clear()
    sys.path.extend(filtered)

# Bad - reassigns sys.path, breaking external references
@pytest.fixture(scope="session")
def some_fixture():
    original = sys.path.copy()
    sys.path = modified  # WRONG: Creates new list, external refs see old list
    yield
    sys.path = original  # WRONG: Doesn't restore the original list object
```

### Fixture Generator Naming

- **Name describes output, not test purpose**: `gen_dem_valid_for_transform_patch_test` not `gen_dem_nan_transform`
- **Clarify when fixture is indirect**: If tests use monkeypatch to inject behavior, document it
- **Output should be self-evident**: File contents should match function name expectations

```python
# Good - name clarifies what's actually created
def gen_dem_valid_for_transform_patch_test():
    """Generate valid DEM used with monkeypatch for NaN geotransform test.

    Creates a structurally valid GeoTIFF. Tests inject NaN/Inf transforms
    at runtime since rasterio cannot write invalid transforms directly.
    """

# Misleading - suggests file has NaN transforms (it doesn't)
def gen_dem_nan_transform():
    """Generate DEM for NaN geotransform test."""
    # Actually creates valid file...
```

### Import Path Consistency for Monkeypatching

- **Monkeypatch paths must match import paths**: Python creates different module objects for different import paths
- **Same file, different paths = different modules**: `import a.b.c` and `import x.b.c` create separate module objects
- **Patch the same path you import**: If test imports `src.infrastructure.X`, patch `src.infrastructure.X.func`

```python
# Project structure with PYTHONPATH=src:.
# domain/terrain/errors.py        -> import as domain.terrain.errors
# src/infrastructure/terrain/x.py -> import as src.infrastructure.terrain.x

# Good - consistent paths
from src.infrastructure.terrain.geotiff_adapter import GeoTiffTerrainAdapter
monkeypatch.setattr("src.infrastructure.terrain.geotiff_adapter.reproject", fake)

# Bad - different paths create different module objects
from infrastructure.terrain.geotiff_adapter import GeoTiffTerrainAdapter  # via 'src' in path
monkeypatch.setattr("src.infrastructure.terrain.geotiff_adapter.reproject", fake)  # patches DIFFERENT module!
```

### Stub Function Documentation

- **Document stub purpose**: Explain what the stub mimics and why
- **State whether monkeypatched or used directly**: Clarify expected usage pattern
- **Document assumptions**: Explain what the stub expects from callers

```python
# Good - clear documentation
def band(dataset: Any, index: int) -> tuple[Any, int]:
    """Stub for rasterio.band - returns (dataset, band_index) tuple.

    This stub is used directly (not monkeypatched) because the tuple return
    value is compatible with how tests mock the reproject() function.
    """
    return (dataset, index)

# Bad - no context for future readers
def band(dataset, index):
    return (dataset, index)
```

### Trivial Methods with Invariants

- **Document why trivial**: Explain the method exists but is guaranteed to succeed
- **Reference validation source**: Point to where actual validation occurs
- **Use pragma for coverage**: Mark as no-cover with explanation

```python
# Good - explains the invariant
def is_valid(self) -> bool:  # pragma: no cover - trivial, always True
    """Check if this BoundingBox is valid.

    Always returns True because Pydantic validation at construction time
    guarantees that invalid BoundingBox instances cannot exist.

    Note:
        This method performs no runtime checks. All validation occurs in
        the @model_validator during __init__.
    """
    return True

# Bad - no explanation of why it's always True
def is_valid(self) -> bool:
    return True
```

### Documentation Freshness

- **Mark completed work as "Completed"**: Replace "Future" and "TODO" with "Completed" when done
- **Reference related tickets**: Include TD-XXX or ticket numbers for traceability
- **Remove stale content**: Delete outdated instructions that no longer apply

```markdown
# Good - reflects current state
**Completed (TD-002)**: Stubs moved from `src/` to `tests/stubs/` to avoid import
conflicts with real packages.

# Bad - outdated, causes confusion
**Future**: Consider moving stubs to `tests/stubs/` and configuring `conftest.py`...
```

### NoData Conversion Completeness

- **Handle both masked arrays and explicit nodata**: `src.read(masked=True)` may return masked array or plain array with nodata
- **Check mask existence AND content**: `hasattr(data, "mask") and np.any(data.mask)`
- **Fall back to explicit nodata check**: When no mask, check `src.nodata is not None`
- **Preserve dtype**: Use `np.float32(np.nan)` in `np.where` to avoid upcasting

```python
# Good - handles both cases
if hasattr(data, "mask") and np.any(data.mask):
    # Masked array: convert masked values to NaN
    data = np.where(data.mask, np.float32(np.nan), data.data)
elif src.nodata is not None:
    # Plain array with explicit nodata value: convert to NaN
    data = np.where(data == src.nodata, np.float32(np.nan), data)

# Bad - only handles masked arrays, misses explicit nodata
if hasattr(data, "mask"):
    data = np.where(data.mask, np.nan, data.data)
```

### Exception Propagation (No Silent Swallowing)

- **Don't silently swallow exceptions**: Avoid `except SomeError: pass` patterns
- **Log and re-raise**: If you catch, log context then re-raise
- **Let caller decide**: Don't hide errors from callers who may want to handle them

```python
# Good - logs context and propagates
try:
    st = path.stat()
except OSError as e:
    logger.error("Failed to stat %s: %s", path.name, e)
    raise

# Bad - silently swallows, hides race conditions
try:
    st = path.stat()
except (FileNotFoundError, NotADirectoryError):
    pass  # Silently ignored - caller never knows
```

### Immutable Value Objects with NDArray

- **Make arrays read-only**: Set `flags.writeable = False` after validation
- **Use contiguous copy**: `np.ascontiguousarray()` ensures memory layout
- **Bypass frozen model**: Use `object.__setattr__()` in validator to replace data
- **Document immutability**: State in docstring that array cannot be modified

```python
# Good - truly immutable array (avoid unnecessary copies when possible)
@model_validator(mode="after")
def validate_grid(self) -> "TerrainGrid":
    # ... validation ...
    data = self.data
    if not data.flags.c_contiguous:
        immutable = np.ascontiguousarray(data, dtype=np.float32)
        immutable.flags.writeable = False
        object.__setattr__(self, "data", immutable)
    elif data.flags.writeable:
        data.flags.writeable = False
    return self

# Bad - Pydantic frozen but array still mutable
model_config = ConfigDict(frozen=True)  # Only prevents attribute reassignment
# External code can still do: grid.data[0, 0] = 999  # Works!
```

### Deprecated Methods

- **Use docstring deprecation notice**: Add `.. deprecated::` section
- **Explain migration path**: Tell users what to use instead
- **Add inline comment**: Mark with `# DEPRECATED:` for quick scanning
- **Keep pragma no-cover**: Deprecated code shouldn't affect coverage

```python
# Good - clear deprecation with migration guidance
def is_valid(self) -> bool:  # pragma: no cover - deprecated
    """Check validity.

    .. deprecated::
        This method is deprecated. Rely on construction-time validation
        instead - if you have an instance, it is already valid.
    """
    # DEPRECATED: Retained for API compatibility only.
    return True
```

### Cross-File Synchronization Comments

- **Document sync requirements**: When two files must stay in sync, say so explicitly
- **Reference the other file**: Include path to the file that must match
- **Explain what to update**: Be specific about what changes require sync

```python
# Good - clear sync requirement
# Expected fixtures - KEEP IN SYNC with scripts/gen_fixtures.py
# When adding/removing fixtures, update BOTH this list AND the generator script.
EXPECTED_FIXTURES = [...]

# Bad - no indication of sync requirement
# List of fixtures
EXPECTED_FIXTURES = [...]  # Easy to forget updating gen_fixtures.py
```

### Pytest Configuration Type Hints

- **Use `pytest.Config` for hooks**: Type hint `pytest_configure(config: pytest.Config)`
- **Import pytest explicitly**: Add `import pytest` even if only used for type hints
- **Document hook purpose**: Explain what the configuration hook does

```python
# Good - properly typed hook
import pytest

def pytest_configure(config: pytest.Config) -> None:
    """Add stubs to sys.path for test imports."""
    stubs_path = Path(__file__).parent / "stubs"
    if str(stubs_path) not in sys.path:
        sys.path.insert(0, str(stubs_path))

# Bad - untyped parameter
def pytest_configure(config):  # What type is config?
    ...
```

### Import Order Documentation

- **Document non-obvious ordering**: When imports follow a convention (isort, alphabetical), say so
- **Explain rationale**: Helps reviewers understand intentional ordering vs accidents
- **Match `__all__` order**: When defining `__all__`, import order should match

```python
# Good - explains the ordering convention
# Imports alphabetized per project style (isort)
from domain import coverage, siting, terrain

__all__ = ["terrain", "coverage", "siting"]  # Matches import order

# Bad - no explanation, looks arbitrary
from domain import coverage, siting, terrain
```

### Module Cache Clearing for Path Fixtures

- **Clear sys.modules when changing sys.path**: Cached imports bypass sys.path changes
- **Store removed entries for restoration**: Session fixtures need to restore state
- **Use monkeypatch.delitem for function scope**: Automatic cleanup on test end

```python
# Good - clears cached modules so fresh imports use new path
@pytest.fixture(scope="session")
def real_rasterio_path():
    saved_modules: dict[str, object] = {}
    for key in list(sys.modules.keys()):
        if key == "rasterio" or key.startswith("rasterio."):
            saved_modules[key] = sys.modules.pop(key)
    try:
        sys.path = [p for p in sys.path if "stubs" not in p]
        yield
    finally:
        sys.path = original_path
        for key, module in saved_modules.items():
            sys.modules[key] = module

# Good - function scope with monkeypatch
@pytest.fixture(scope="function")
def exclude_stubs_path(monkeypatch):
    monkeypatch.setattr(sys, "path", new_path)
    for key in list(sys.modules.keys()):
        if key == "rasterio" or key.startswith("rasterio."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    yield

# Bad - only changes path, cached stub module still used
sys.path = [p for p in sys.path if "stubs" not in p]
import rasterio  # Still gets cached stub!
```

### Owned Array Copies for Value Objects

- **Always copy to own the data**: Value Objects must not mutate external arrays
- **Setting writeable=False on external array mutates caller's data**: Violates encapsulation
- **Use ascontiguousarray unconditionally**: Ensures both ownership and correct layout

```python
# Good - always create owned copy, never mutate external arrays
immutable_data = np.ascontiguousarray(self.data, dtype=np.float32)
immutable_data.flags.writeable = False
object.__setattr__(self, "data", immutable_data)

# Bad - mutates caller's array if already contiguous
if not self.data.flags.c_contiguous:
    immutable_data = np.ascontiguousarray(self.data, dtype=np.float32)
    immutable_data.flags.writeable = False
    object.__setattr__(self, "data", immutable_data)
elif self.data.flags.writeable:
    self.data.flags.writeable = False  # Mutates external reference!
```

### Generator Fixtures with Yield

- **Use yield for cleanup logic**: Even if fixture has no explicit teardown
- **Enables monkeypatch cleanup**: Without yield, monkeypatch may not restore state
- **Document the yield purpose**: Clarify why yield is needed

```python
# Good - yield allows monkeypatch to restore state after test
@pytest.fixture(scope="function")
def exclude_stubs_path(monkeypatch):
    monkeypatch.setattr(sys, "path", new_path)
    for key in list(sys.modules.keys()):
        if key.startswith("rasterio"):
            monkeypatch.delitem(sys.modules, key, raising=False)
    yield  # Test runs here, then monkeypatch cleanup

# Bad - no yield, cleanup may not run properly
@pytest.fixture(scope="function")
def exclude_stubs_path(monkeypatch):
    monkeypatch.setattr(sys, "path", new_path)
    # Missing yield - test runs immediately after fixture returns
```

### In-Place sys.path Modification

- **Modify sys.path in place**: External references point to the same list object
- **Use clear() and extend()**: Preserves the list identity while changing contents
- **Never reassign sys.path**: `sys.path = new_list` breaks external references

```python
# Good - modifies in place, preserves list identity
original_path = sys.path.copy()
filtered = [p for p in sys.path if p != stubs_path]
sys.path.clear()
sys.path.extend(filtered)
# ... later restore ...
sys.path.clear()
sys.path.extend(original_path)

# Bad - replaces list object, breaks external references
sys.path = [p for p in sys.path if p != stubs_path]  # New list!
```

### Import Path Consistency

- **Use consistent import style**: Don't mix `src.infrastructure.*` with `domain.*`
- **Prefer package-relative imports**: `infrastructure.*` over `src.infrastructure.*`
- **Monkeypatch paths must match imports**: Inconsistent paths break monkeypatching

```python
# Good - consistent style (both use package name directly)
from domain.terrain.errors import AllNoDataError
from infrastructure.terrain.geotiff_adapter import GeoTiffTerrainAdapter
monkeypatch.setattr("infrastructure.terrain.geotiff_adapter.reproject", fake)

# Bad - inconsistent style
from domain.terrain.errors import AllNoDataError  # No prefix
from src.infrastructure.terrain.geotiff_adapter import GeoTiffTerrainAdapter  # Has src. prefix
```

### Variable Naming Clarity

- **Name variables for their content**: Use `nodata_pct` not `100 - valid_pct`
- **Avoid double negatives in computations**: Makes code harder to read
- **Print statements should match variable names**: Reduces confusion

```python
# Good - direct computation with clear name
nodata_pct = 100 * (np.sum(data == nodata) / data.size)
print(f"Created: {path.name} ({nodata_pct:.0f}% nodata)")

# Bad - indirect computation with confusing output
valid_pct = 100 * (1 - np.sum(data == nodata) / data.size)
print(f"Created: {path.name} ({100-valid_pct:.0f}% nodata)")  # Confusing!
```

### File Listing Filters

- **Apply same filters in listing as validation**: Keeps output consistent
- **Define allowed suffixes once**: Use a constant or set for reuse
- **Filter before iterating**: Reduces noise from unrelated files

```python
# Good - same filter as validation
allowed_suffixes = {".tif", ".png"}
for f in sorted(FIXTURES_DIR.iterdir()):
    if f.is_file() and f.suffix.lower() in allowed_suffixes:
        print(f"  {f.name}")

# Bad - shows files that validation ignores
for f in sorted(FIXTURES_DIR.iterdir()):
    if f.is_file():  # Shows README.md, etc.
        print(f"  {f.name}")
```

### Shared Module for Cross-Package Constants

- **Avoid scripts→tests dependency**: Don't import from `tests/` in `scripts/`
- **Use `shared/` directory**: Place constants needed by both in `shared/`
- **Re-export for backwards compatibility**: Original module can re-export from shared

```python
# Good - shared module avoids circular dependencies
# shared/fixtures_expected.py
EXPECTED_FIXTURES: list[str] = sorted([...])

# scripts/gen_fixtures.py
from shared.fixtures_expected import EXPECTED_FIXTURES

# tests/fixtures_expected.py (re-export for backwards compat)
from shared.fixtures_expected import EXPECTED_FIXTURES
__all__ = ["EXPECTED_FIXTURES"]

# Bad - scripts importing from tests
from tests.fixtures_expected import EXPECTED_FIXTURES  # Dependency inversion!
```

### Robust CRS Equality Comparison

- **Try object equality first**: Use rasterio CRS `==` operator
- **Fall back to string comparison**: For test mocks and equivalent representations
- **Handle None safely**: Check for None before comparison

```python
# Good - robust CRS comparison with fallback
def _is_wgs84(crs: Any) -> bool:
    if crs is None:
        return False
    # Try rasterio CRS equality (handles projjson, wkt, etc.)
    try:
        if crs == _TARGET_CRS:
            return True
    except (TypeError, AttributeError):
        pass
    # Fall back to string comparison for test mocks
    crs_str = str(crs).upper()
    return crs_str in ("EPSG:4326", "OGC:CRS84")

# Bad - brittle string-only comparison
if str(src.crs).upper() in ("EPSG:4326", "OGC:CRS84"):  # Misses equivalent CRS
```

### NoReturn for Always-Raising Stubs

- **Use `typing.NoReturn`**: When function never returns normally
- **Add clear error messages**: Help developers understand they need to monkeypatch
- **Document in docstring**: Explain the stub's purpose

```python
# Good - NoReturn annotation with helpful message
from typing import NoReturn

def reproject(*args: Any, **kwargs: Any) -> NoReturn:
    """Stub that always raises - must be monkeypatched in tests."""
    raise NotImplementedError("reproject is a stub; monkeypatch in tests")

# Bad - misleading None return type
def reproject(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError  # Never returns, but annotation says None
```

### Helper Functions to Reduce Script Duplication

- **Extract common patterns**: When multiple functions share similar logic
- **Accept optional parameters**: Use `None` defaults for optional fields
- **Handle variants via parameters**: e.g., `driver="GTiff"` vs `driver="PNG"`

```python
# Good - helper reduces duplication across generators
def write_raster(
    path: Path,
    data: NDArray[Any],
    transform: Affine,
    crs: CRS | None = None,
    dtype: str | None = None,
    nodata: float | None = None,
    driver: str = "GTiff",
) -> None:
    """Write raster file - handles 2D/3D arrays, optional CRS/nodata."""
    ...

# Then each generator is simple:
def gen_dem_100x100_4326() -> None:
    data = np.linspace(0, 1000, 10000, dtype=np.float32).reshape(100, 100)
    transform = Affine.translation(-50.0, -15.0) * Affine.scale(0.1, -0.1)
    write_raster(path, data, transform, crs=CRS.from_epsg(4326))

# Bad - duplicated rasterio.open boilerplate in every generator
```

---

## 15. Architecture and Patterns (GIS / FEAT-001)

### GIS Conventions

- North-Up normalization: Grids are North-Up; row 0 maps to the northern edge (`bounds.max_y`).
- System CRS: All outputs normalized to `EPSG:4326`; `TerrainGrid.crs == "EPSG:4326"`.
- Resolution magnitudes: Store positive magnitudes only: `(abs(transform.a), abs(transform.e))`.
- Bounds calculation: Use `rasterio.transform.array_bounds(height, width, transform)`; do not hand-roll formulas.

### Adapter Lifecycle

1. Preflight checks (extension allowlist, symlink rejection, stat size / budget)
2. Enter `rasterio.Env()` context
3. `rasterio.open(path)` dataset context
4. Validate metadata (band count, CRS present, transform finite & non-zero)
5. CRS decision: same-CRS fast path vs reprojection path
6. NoData normalization and dtype preservation (float32)
7. Build `BoundingBox`, positive `resolution`, construct `TerrainGrid`
8. Log with SEC-7 safe patterns; exit contexts

### Input Normalization

- Normalize file arguments on entry: `path = Path(file_path)`.
- Extension allowlist (SEC-10): Accept only `.tif`/`.tiff`; reject others early with a domain error.
- Symlink rejection (SEC-5): `if path.is_symlink(): raise InvalidRasterError("Symlinks are not permitted")`.

### CRS Equality and Normalization

- Helper `_is_wgs84(crs)` (module-level): try CRS object equality first, fall back to string checks for stubs (`"EPSG:4326" | "OGC:CRS84"`).
- Reprojection shape/transform: Prefer `calculate_default_transform` to derive destination transform, width, and height.
- Resampling: Use `Resampling.bilinear` for continuous elevation.

### NoData Handling

- Masked arrays: If `hasattr(data, "mask") and np.any(data.mask)`, convert masked cells to `np.float32(np.nan)`.
- Explicit nodata: Else if `src.nodata is not None`, replace exact matches with `np.float32(np.nan)`.
- Preserve dtype: Always use typed NaN to avoid upcasting to float64.

### Bounds and Resolution Semantics

- Use `array_bounds` for bounds; validate via `BoundingBox` (Pydantic) on construction.
- Store `resolution` as positive magnitudes independent of orientation.

### Memory Budget and Preflight Checks

- Preflight size check (PERF-7): If `max_bytes` set and `path.stat().st_size > 2 * max_bytes`, raise `InsufficientMemoryError` before `rasterio.open()`.
- Estimate allocation BEFORE reading/reprojecting: `est_bytes = width * height * 4` (float32) for both same-CRS and reprojection paths.

### Security Hardening

- SEC-5 symlinks: Reject symlinks explicitly (no target path in message or logs).
- SEC-7 logging: Use `path.name`; for OS errors, log only filename with `(errno, strerror)`; no absolute paths in messages.
- No shell-outs: Do not invoke GDAL CLI; use rasterio Python API.

### Exception Taxonomy and Message Policy

- Raster I/O failures: `RasterioIOError | RasterioError` → `InvalidRasterError("Corrupted or invalid raster: {e}")` (with `from e`).
- Missing CRS → `MissingCRSError`; NaN/Inf/zero transform → `InvalidGeotransformError`.
- 100% NoData → `AllNoDataError`; invalid bounds after reprojection → `InvalidBoundsError`.
- Empty file: Use precise message `"Empty file"` for 0-byte; bandless files → `"Empty or bandless file"`.
- Avoid leaking sensitive info in messages (no absolute paths).

### Domain Boundaries and Imports

- Domain purity: Domain layer uses numpy/pydantic only; no I/O, no imports from `src/`.
- Ports & Adapters: Protocols in `domain/.../repositories.py`; implementations in `src/infrastructure/...`.

### Value Object Array Ownership and Immutability

- Conditional copy (PERF-3):

```python
@model_validator(mode="after")
def validate_grid(self) -> "TerrainGrid":
    # ... invariants ...
    # Always create an owned contiguous copy to guarantee encapsulation.
    # We do not rely on "writeable=False" on the input array because
    # that would mutate the caller's data.
    immutable = np.array(self.data, dtype=np.float32, copy=True, order="C")
    immutable.flags.writeable = False
    object.__setattr__(self, "data", immutable)
    return self
```

- Tradeoff: Minimizes peak RSS for large rasters while guaranteeing immutability and dtype.

### Thread Safety

- Wrap operations in `rasterio.Env()` to isolate GDAL/PROJ.
- Avoid global mutable state; adapters are stateless aside from config (e.g., `max_bytes`).

### Testing Strategy

- Stubs live under `tests/stubs/`; unit tests monkeypatch behavior; integration tests (marker `integration`) use real rasterio.
- Path hygiene: Patch the same module path you import to avoid patching a different module object.
- Module cache hygiene: When switching between stubs and real packages, clear `sys.modules` entries for `rasterio*` before re-importing.
 - sys.path filtering: When excluding a directory (e.g., stubs), compare resolved paths (`Path(p).resolve() != Path(stubs).resolve()`) and update `sys.path` in place (`clear()` then `extend()`) to preserve the list object.
- Determinism: Seed RNGs in fixture generators (`np.random.default_rng(42)`) to stabilize artifact sizes and test behavior.

### Performance Notes

- Avoid upcasting by using typed NaN in `np.where`.
- Avoid redundant casts when APIs already provide `dtype`/`out_dtype`.
- Use `np.isnan(arr).all()` to detect 100% NaN without extra allocations.
- NoData warning threshold is strictly `> 80.0` (not `>=`).

### Reprojection Decisions and Tradeoffs

- Shape may change after reprojection; tests must not assume shape/extent preservation across CRS changes.
- `Resampling.bilinear` chosen for continuous DEM data; nearest/categorical handling is out of FEAT-001 scope.

### RF Propagation Requirements (FEAT-002+)

When implementing RF propagation features:

1. **Earth curvature correction**: Apply correction for distances > 1 km (4/3 Earth radius model for standard atmosphere).
2. **Fresnel zone calculations**: Use correct wavelength formulas: `λ = c / f`, `F1 = sqrt(λ * d1 * d2 / (d1 + d2))`.
3. **Path loss model validity**: Document regime constraints for each model (frequency range, terrain type, distance limits).


<!--
CLAUDE.md v1.16.0 | Last updated: 2025-12-07

IMPORTANT: Always update the "Last updated" date above when modifying this file.
Format: YYYY-MM-DD

Changelog:
- v1.16.0 (2025-12-07): Added RF Propagation Requirements section for FEAT-002+ (Earth curvature, Fresnel zones, path loss validity)
- v1.15.0 (2025-12-06): Fixed Rule 3 stub precedence (stubs take precedence, not installed packages); updated session fixture example to use in-place sys.path modification
- v1.14.0 (2025-12-06): Added consolidated GIS/FEAT-001 architecture & patterns section for maintainers
- v1.13.0 (2025-12-06): Added SEC-5 symlink policy and conditional VO immutability pattern; refined SEC-7 logging example with errno/strerror
- v1.12.0 (2025-12-06): Added shared module pattern, robust CRS comparison, NoReturn for stubs, helper function pattern
- v1.11.0 (2025-12-06): Added owned array copies, in-place sys.path, import consistency, variable naming, file filters
- v1.10.0 (2025-12-06): Added module cache clearing, generator fixtures patterns
- v1.9.0 (2025-12-06): Added pytest configuration type hints, import order documentation patterns
- v1.8.0 (2025-12-06): Added nodata conversion, exception propagation, immutable arrays, deprecation, sync comments
- v1.7.0 (2025-12-06): Added import consistency, stub docs, trivial methods, doc freshness patterns
- v1.6.0 (2025-12-06): Added PERF-7, narrow OSError, fixture patterns
- v1.5.0 (2025-12-06): Added Security in Logging (SEC-7) pattern
-->
