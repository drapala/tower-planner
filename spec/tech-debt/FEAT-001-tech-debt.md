# Tech Debt: FEAT-001 LoadDEM

**Feature**: FEAT-001 (Load DEM from GeoTIFF)
**Branch**: `FEAT-001/load-dem`
**Created**: 2025-12-06
**Priority**: Low (não bloqueia merge)
**Status**: ✅ All Resolved

---

## Resumo

Débitos técnicos identificados durante code review da implementação FEAT-001.
Nenhum item é crítico ou bloqueia o merge — são melhorias para manter
consistência com os padrões do projeto.

---

## TD-001: Logger criado dentro do método

**Severidade**: Baixa
**Arquivo**: `src/infrastructure/terrain/geotiff_adapter.py`
**Linha**: 63

### Código Atual

```python
def load_dem(self, file_path: Path | str) -> TerrainGrid:
    logger = logging.getLogger("src.infrastructure.terrain.geotiff_adapter")
```

### Problema

O logger é criado em cada chamada de `load_dem()`, violando a convenção
do projeto (CLAUDE.md) de usar module-level loggers.

### Correção Proposta

```python
# No topo do módulo (após imports)
logger = logging.getLogger(__name__)


class GeoTiffTerrainAdapter:
    def load_dem(self, file_path: Path | str) -> TerrainGrid:
        # Usar o logger do módulo
        logger.debug(...)
```

### Referência

- CLAUDE.md → "Module-level Loggers: Always use module-level `logger = logging.getLogger(__name__)` outside classes"

---

## TD-002: Stubs de rasterio/affine em `src/`

**Severidade**: Média
**Arquivos**:
- `src/rasterio/__init__.py`
- `src/rasterio/enums.py`
- `src/rasterio/transform.py`
- `src/rasterio/warp.py`
- `src/affine/__init__.py`

### Problema

Os stubs para simular rasterio/affine estão em `src/`, que é a camada de
infrastructure. Quando as dependências reais forem instaladas via pip,
haverá conflito de importação (Python pode importar o stub ao invés do
pacote real).

### Correção Proposta

**Opção A** (Recomendada): Mover para `tests/stubs/` e usar `conftest.py`:

```
tests/
├── stubs/
│   ├── rasterio/
│   │   ├── __init__.py
│   │   ├── enums.py
│   │   ├── transform.py
│   │   └── warp.py
│   └── affine/
│       └── __init__.py
└── conftest.py  # Adicionar tests/stubs ao sys.path
```

**Opção B**: Renomear para `_rasterio_stub` e ajustar imports nos testes.

### Notas

- Considerar adicionar ao `.gitignore` quando rasterio real for instalado
- Ou documentar no README que os stubs são para TDD sem dependências externas

---

## TD-003: Exception handler muito amplo

**Severidade**: Baixa
**Arquivo**: `src/infrastructure/terrain/geotiff_adapter.py`
**Linhas**: 177-180

### Código Atual

```python
try:
    bounds_tuple = (sb.left, sb.bottom, sb.right, sb.top)
except Exception:
    bounds_tuple = tuple(sb)
```

### Problema

`except Exception` é muito amplo e pode mascarar bugs inesperados
(ex: MemoryError, SystemExit, etc.).

### Correção Proposta

```python
try:
    bounds_tuple = (sb.left, sb.bottom, sb.right, sb.top)
except AttributeError:
    # Fallback para quando bounds é uma tupla simples ao invés de objeto
    bounds_tuple = tuple(sb)
```

---

## TD-004: Adapter não exportado no `__init__.py`

**Severidade**: Baixa
**Arquivo**: `src/infrastructure/terrain/__init__.py`
**Linha**: 1-2

### Código Atual

```python
"""Infrastructure adapters for the terrain bounded context."""
```

### Problema

O `GeoTiffTerrainAdapter` não está sendo exportado, forçando imports
com path completo:

```python
from src.infrastructure.terrain.geotiff_adapter import GeoTiffTerrainAdapter
```

### Correção Proposta

```python
"""Infrastructure adapters for the terrain bounded context."""

from .geotiff_adapter import GeoTiffTerrainAdapter

__all__ = ["GeoTiffTerrainAdapter"]
```

Permite import simplificado:
```python
from src.infrastructure.terrain import GeoTiffTerrainAdapter
```

---

## TD-005: Falta teste para InvalidGeotransformError com NaN/Inf

**Severidade**: Baixa
**Arquivo**: `tests/gis/test_geotiff_adapter.py`
**Relacionado**: `src/infrastructure/terrain/geotiff_adapter.py:95-108`

### Código Existente (adapter)

```python
if any(
    math.isnan(v) or math.isinf(v)
    for v in (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )
):
    raise InvalidGeotransformError(
        "Invalid (NaN/Inf) transform values"
    )
```

### Problema

Não há teste explícito que verifique o comportamento quando o transform
contém valores NaN ou Inf.

### Correção Proposta

Adicionar ao `test_geotiff_adapter.py`:

```python
def test_nan_transform_rejected(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "nan_transform.tif"
    p.write_bytes(b"x")

    # Affine com NaN no scale
    transform = Affine(float('nan'), 0.0, 0.0, 0.0, -0.01, 0.0)
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=10, height=10)

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(InvalidGeotransformError, match="NaN"):
        adapter.load_dem(p)


def test_inf_transform_rejected(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "inf_transform.tif"
    p.write_bytes(b"x")

    transform = Affine(float('inf'), 0.0, 0.0, 0.0, -0.01, 0.0)
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=10, height=10)

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(InvalidGeotransformError, match="Inf"):
        adapter.load_dem(p)
```

---

## Checklist de Resolução

| ID | Descrição | Status | Responsável | PR |
|----|-----------|--------|-------------|-----|
| TD-001 | Logger module-level | ✅ Resolved | Claude | - |
| TD-002 | Mover stubs para tests/ | ✅ Resolved | Claude | - |
| TD-003 | Especificar AttributeError | ✅ Resolved | Claude | - |
| TD-004 | Exportar adapter no __init__ | ✅ Resolved | Claude | - |
| TD-005 | Teste NaN/Inf transform | ✅ Resolved | Claude | - |

---

## Histórico

| Data | Ação |
|------|------|
| 2025-12-06 | Documento criado após code review da branch FEAT-001/load-dem |
| 2025-12-06 | TD-001 resolvido: Logger já estava em module-level (linha 43) |
| 2025-12-06 | TD-002 resolvido: Stubs movidos de src/ para tests/stubs/ |
| 2025-12-06 | TD-003 resolvido: Exception já era `(AttributeError, TypeError)` |
| 2025-12-06 | TD-004 resolvido: GeoTiffTerrainAdapter exportado em __init__.py |
| 2025-12-06 | TD-005 resolvido: 3 testes adicionados (NaN, Inf, -Inf) |

---

<!--
Tech Debt FEAT-001 v1.1.0 | Last updated: 2025-12-06

IMPORTANT: Always update the "Last updated" date above when modifying this file.
Format: YYYY-MM-DD

Changelog:
- v1.1.0 (2025-12-06): All 5 items resolved (TD-001 to TD-005)
- v1.0.0 (2025-12-06): Document created after code review
-->
