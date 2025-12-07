# Prompt: Implement FEAT-002 (TerrainProfile)

**ID**: PROMPT-002
**Target**: FEAT-002 (TerrainProfile)
**Type**: Feature Implementation (TDD)
**Prerequisites**: `PROMPT-001` (FEAT-001 fixtures) completed.

---

## Context

You are Claude Code working in the repository "tower-planner".
Your task is to implement **FEAT-002: TerrainProfile** following a strict **Test-Driven Development (TDD)** workflow.

**Specification**: `spec/features/FEAT-002-terrain-profile.md` (v1.3.0)

---

## Goal

Implement the `terrain_profile` domain service and its dependencies (Value Objects) by following the **Implementation Order** defined in the spec.

### Phase 1: Test Data (Step 1)

1.  **Update Fixture Generator**:
    *   Edit `scripts/gen_fixtures.py`.
    *   Add the `gen_dem_all_nodata_partial()` function exactly as defined in the "Appendix: Fixture Bounds Reference" of FEAT-002.
    *   Run the script to generate `tests/fixtures/dem_all_nodata_partial.tif`.

### Phase 2: Red (Step 2)

2.  **Create Test Suite**:
    *   Create `tests/terrain/test_terrain_profile.py`.
    *   Implement **ALL 19 Test Cases** (TC-001 to TC-019) defined in the spec.
    *   **CRITICAL**: Use the exact coordinates provided in the spec for each TC. They are calibrated to match the fixture bounds.
    *   Run tests: `pytest tests/terrain/test_terrain_profile.py`.
    *   **Expectation**: Tests must fail (ImportError or NameError) because the domain code doesn't exist yet.

### Phase 3: Green (Steps 3-5)

3.  **Implement Value Objects**:
    *   Create/Update `domain/terrain/value_objects.py`.
    *   Add `GeoPoint` (if not exists), `ProfileSample`, `TerrainProfile`.
    *   Implement all invariants and validators (e.g., `DISTANCE_TOLERANCE_M`, `math.isnan` check).

4.  **Implement Errors**:
    *   Update `domain/terrain/errors.py`.
    *   Add `PointOutOfBoundsError`, `InvalidProfileError`.

5.  **Implement Domain Service**:
    *   Create `domain/terrain/services.py`.
    *   Implement `terrain_profile(...)`.
    *   Implement helpers: `derive_step_m`, `geodesic_distance`, `interpolate_geodesic_path`, `bilinear_interpolate`.
    *   Ensure `pyproj.Geod` is used for distance and path interpolation.
    *   Ensure edge clamping is used for bilinear interpolation.

### Phase 4: Refactor & Verify (Step 6)

6.  **Verification**:
    *   Run `pytest tests/terrain/test_terrain_profile.py`.
    *   All tests must PASS.
    *   Run `mypy domain/terrain`.
    *   Ensure code follows `CLAUDE.md` (no I/O in domain, type safety).

---

## Constraints

| Rule | Description |
|------|-------------|
| **Strict TDD** | Write tests BEFORE implementation. |
| **Spec Adherence** | Do not deviate from v1.3.0 spec signatures or logic. |
| **No I/O in Domain** | `services.py` must be pure computation. I/O stays in tests (loading fixtures). |
| **Dependencies** | Use `pyproj` and `numpy` as specified. No new deps. |
| **Coordinates** | Respect the "Fixture Bounds Reference" in the spec. |
| **Immutability** | All VOs must be frozen Pydantic models. |

---

## Deliverables

1.  `scripts/gen_fixtures.py` (updated)
2.  `tests/fixtures/dem_all_nodata_partial.tif` (generated)
3.  `tests/terrain/test_terrain_profile.py` (new)
4.  `domain/terrain/value_objects.py` (updated)
5.  `domain/terrain/errors.py` (updated)
6.  `domain/terrain/services.py` (new)

---

<!-- PROMPT-002 v1.0.0 - Implement FEAT-002 -->
