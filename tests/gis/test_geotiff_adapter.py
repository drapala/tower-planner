import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from domain.terrain.errors import (
    AllNoDataError,
    InsufficientMemoryError,
    InvalidBoundsError,
    InvalidRasterError,
    MissingCRSError,
)
from src.infrastructure.terrain.geotiff_adapter import GeoTiffTerrainAdapter


class FakeCRS:
    def __init__(self, code: str | None):
        self._code = code

    def to_string(self) -> str:
        return self._code or ""

    def __str__(self) -> str:  # rasterio may call str()
        return self._code or ""


class FakeDataset:
    def __init__(self, *, count: int, crs: str | None, transform, width: int, height: int, nodata=None):
        self.count = count
        self.crs = FakeCRS(crs) if crs is not None else None
        self.transform = transform
        self.width = width
        self.height = height
        self.nodata = nodata
        # bounds: (left, bottom, right, top)
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        # using formula for array_bounds would be circular; set via transform and dims
        # We'll rely on adapter's array_bounds for calculations.
        self.bounds = SimpleNamespace(left=c, bottom=f + e * height, right=c + a * width, top=f)

    def read(self, band: int, *, masked: bool, out_dtype: str):
        # Return a masked array with a gradient and some masked NoData at top-left
        data = np.linspace(0, 1, self.width * self.height, dtype=np.float32).reshape(self.height, self.width)
        if masked:
            import numpy.ma as ma

            m = np.zeros_like(data, dtype=bool)
            m[0, 0] = True  # one nodata
            return ma.MaskedArray(data, mask=m)
        return data.astype(np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_file_not_found_raises(tmp_path, monkeypatch):
    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(FileNotFoundError):
        adapter.load_dem(tmp_path / "missing.tif")


def test_permission_error_raises(tmp_path, monkeypatch):
    p = tmp_path / "file.tif"
    p.write_bytes(b"x")

    monkeypatch.setattr("os.access", lambda path, mode: False)

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(PermissionError):
        adapter.load_dem(p)


def test_empty_file_raises(tmp_path, monkeypatch):
    p = tmp_path / "empty.tif"
    p.write_bytes(b"")

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(InvalidRasterError):
        adapter.load_dem(p)


def test_same_crs_happy_path(monkeypatch, tmp_path, caplog):
    from affine import Affine

    p = tmp_path / "dem_4326.tif"
    p.write_bytes(b"x")

    transform = Affine.translation(-10.0, 10.0) * Affine.scale(0.01, -0.01)
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=100, height=100, nodata=None)

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *x: False))

    adapter = GeoTiffTerrainAdapter()
    grid = adapter.load_dem(p)

    assert grid.crs == "EPSG:4326"
    assert grid.data.dtype == np.float32
    assert grid.data.shape == (100, 100)
    assert math.isclose(grid.resolution[0], 0.01)
    assert math.isclose(grid.resolution[1], 0.01)
    assert grid.source_crs == "EPSG:4326"


def test_multiband_rejected(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "rgb.tif"
    p.write_bytes(b"x")

    transform = Affine.identity()
    ds = FakeDataset(count=3, crs="EPSG:4326", transform=transform, width=10, height=10)
    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *x: False))

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(InvalidRasterError):
        adapter.load_dem(p)


def test_missing_crs_rejected(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "nocrs.tif"
    p.write_bytes(b"x")

    transform = Affine.identity()
    ds = FakeDataset(count=1, crs=None, transform=transform, width=10, height=10)
    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *x: False))

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(MissingCRSError):
        adapter.load_dem(p)


def test_reprojection_path(monkeypatch, tmp_path, caplog):
    from affine import Affine

    p = tmp_path / "utm.tif"
    p.write_bytes(b"x")

    src_transform = Affine.translation(500000.0, 7200000.0) * Affine.scale(30.0, -30.0)
    ds = FakeDataset(count=1, crs="EPSG:31983", transform=src_transform, width=10, height=10, nodata=-9999)

    # Fake calculate_default_transform and reproject
    def fake_cdt(src_crs, dst_crs, w, h, *bounds):
        dst_transform = Affine.translation(-45.0, -15.0) * Affine.scale(0.001, 0.001)
        return dst_transform, 20, 20

    def fake_reproject(source, destination, **kwargs):
        destination[:] = 100.0  # fill with a constant

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *x: False))
    monkeypatch.setattr("src.infrastructure.terrain.geotiff_adapter.calculate_default_transform", fake_cdt)
    monkeypatch.setattr("src.infrastructure.terrain.geotiff_adapter.reproject", fake_reproject)

    adapter = GeoTiffTerrainAdapter()
    grid = adapter.load_dem(p)

    assert grid.crs == "EPSG:4326"
    assert grid.data.shape == (20, 20)
    assert np.isfinite(grid.data).all()


def test_all_nodata_after_reproject_raises(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "utm_all_nan.tif"
    p.write_bytes(b"x")

    src_transform = Affine.identity()
    ds = FakeDataset(count = 1, crs = "EPSG:31983", transform = src_transform, width = 10, height = 10)

    def fake_cdt(src_crs, dst_crs, w, h, *bounds):
        dst_transform = Affine.translation(0.0, 0.0) * Affine.scale(1.0, 1.0)
        return dst_transform, 5, 5

    def fake_reproject(source, destination, **kwargs):
        destination[:] = np.nan

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *x: False))
    monkeypatch.setattr("src.infrastructure.terrain.geotiff_adapter.calculate_default_transform", fake_cdt)
    monkeypatch.setattr("src.infrastructure.terrain.geotiff_adapter.reproject", fake_reproject)

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(AllNoDataError):
        adapter.load_dem(p)


def test_memory_budget_exceeded_same_crs(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "big_4326.tif"
    p.write_bytes(b"x")

    transform = Affine.identity()
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=5000, height=5000)

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *x: False))

    adapter = GeoTiffTerrainAdapter(max_bytes=10_000)  # too small
    with pytest.raises(InsufficientMemoryError):
        adapter.load_dem(p)


def test_invalid_bounds_raise(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "bad_bounds.tif"
    p.write_bytes(b"x")

    # Create transform offsetting bounds far beyond valid lon/lat
    transform = Affine.translation(1e6, 1e6) * Affine.scale(1.0, -1.0)
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=10, height=10)

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *x: False))

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(InvalidBoundsError):
        adapter.load_dem(p)

