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

### Layer Responsibilities

| Layer | Location | Contains | I/O Allowed |
|-------|----------|----------|-------------|
| **Domain** | `domain/` | Value Objects, Entities, Domain Services | **NO** |
| **Infrastructure** | `src/` | File loaders, API clients, CLI | **YES** |
| **Tests** | `tests/` | All test code | YES |

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
- Document any assumptions made

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

<!-- CLAUDE.md v1.1.0 - Align with FEAT-001: Pydantic v2 examples, infrastructure placement for load_dem, approved geospatial libs -->
