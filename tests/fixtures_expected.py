"""Re-export fixture constants from shared module.

The canonical source of truth is shared/fixtures_expected.py.
This module re-exports for backwards compatibility with existing imports.

See shared/fixtures_expected.py for the actual constant definitions.
"""

from __future__ import annotations

from shared.fixtures_expected import EXPECTED_FIXTURE_COUNT, EXPECTED_FIXTURES

__all__ = ["EXPECTED_FIXTURES", "EXPECTED_FIXTURE_COUNT"]
