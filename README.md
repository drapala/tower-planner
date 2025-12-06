# Tower Planner

Telecommunications tower site planning and optimization system.

## Overview

Tower Planner analyzes terrain, calculates RF coverage, and recommends optimal
tower placement based on configurable constraints and scoring criteria.

## Project Structure

```
tower-planner/
├── CLAUDE.md          # AI development contract (SDD/TDD/DDD rules)
├── spec/SPEC.md       # Source of truth for specifications
├── domain/            # Domain layer (business logic)
│   ├── terrain/       # Geographic calculations
│   ├── coverage/      # RF propagation
│   └── siting/        # Site selection
├── src/               # Application layer
└── tests/             # Test suite
```

## Development Principles

This project follows:

- **SDD** (Specification-Driven Development) - All features defined in `spec/SPEC.md`
- **TDD** (Test-Driven Development) - Tests written before implementation
- **DDD** (Domain-Driven Design) - Business logic in bounded contexts

See [CLAUDE.md](./CLAUDE.md) for detailed development rules.

## Getting Started

### Requirements

- Python 3.12+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tower-planner

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests
pytest

# Specific bounded context
pytest tests/terrain/
pytest tests/coverage/
pytest tests/siting/

# With coverage
pytest --cov=domain --cov=src
```

### Pre-commit Hooks

```bash
# Install hooks (run once)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### Code Quality

```bash
# Type checking
mypy domain/ src/

# Format check
black --check domain/ src/ tests/
isort --check-only domain/ src/ tests/

# Format
black domain/ src/ tests/
isort domain/ src/ tests/
```

## Bounded Contexts

| Context | Responsibility |
|---------|----------------|
| **terrain** | Physical geography, elevation, line-of-sight |
| **coverage** | RF propagation, signal strength, pathloss |
| **siting** | Site selection, constraints, optimization |

## Contributing

1. Read `CLAUDE.md` for development rules
2. Check `spec/SPEC.md` for feature specifications
3. Write tests first (TDD)
4. Keep domain logic in appropriate bounded context

## License

MIT
