"""Tower Planner Domain Layer.

This package contains the core business logic organized by bounded contexts:
- terrain: Physical geography, elevation, line-of-sight calculations
- coverage: RF propagation, signal strength, pathloss models
- siting: Site selection, constraints, optimization
"""

from domain import coverage, siting, terrain

__all__ = ["terrain", "coverage", "siting"]
