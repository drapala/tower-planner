import math
from contextlib import nullcontext
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
    def __init__(
        self,
        *,
        count: int,
        crs: str | None,
        transform,
        width: int,
        height: int,
        nodata=None,
    ):
        self.count = count
        self.crs = FakeCRS(crs) if crs is not None else None
        self.transform = transform
        self.width = width
        self.height = height
        self.nodata = nodata
        # bounds: (left, bottom, right, top)
        a, b, c, d, e, f = (
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        )
        # using formula for array_bounds would be circular; set via transform and dims
        # We'll rely on adapter's array_bounds for calculations.
        self.bounds = SimpleNamespace(
            left=c, bottom=f + e * height, right=c + a * width, top=f
        )

    def read(self, band: int, *, masked: bool, out_dtype: str):
        # Default: gradient with single masked pixel at (0,0)
        data = np.linspace(0, 1, self.width * self.height, dtype=np.float32).reshape(
            self.height, self.width
        )
        if masked:
            import numpy.ma as ma

            m = np.zeros_like(data, dtype=bool)
            m[0, 0] = True
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
    ds = FakeDataset(
        count=1,
        crs="EPSG:4326",
        transform=transform,
        width=100,
        height=100,
        nodata=None,
    )

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())

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
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())

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
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(MissingCRSError):
        adapter.load_dem(p)


def test_reprojection_path(monkeypatch, tmp_path, caplog):
    from affine import Affine

    p = tmp_path / "utm.tif"
    p.write_bytes(b"x")

    src_transform = Affine.translation(500000.0, 7200000.0) * Affine.scale(30.0, -30.0)
    ds = FakeDataset(
        count=1,
        crs="EPSG:31983",
        transform=src_transform,
        width=10,
        height=10,
        nodata=-9999,
    )

    # Fake calculate_default_transform and reproject
    def fake_cdt(src_crs, dst_crs, w, h, *bounds):
        dst_transform = Affine.translation(-45.0, -15.0) * Affine.scale(0.001, 0.001)
        return dst_transform, 20, 20

    def fake_reproject(source, destination, **kwargs):
        destination[:] = 100.0  # fill with a constant

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        "src.infrastructure.terrain.geotiff_adapter.calculate_default_transform",
        fake_cdt,
    )
    monkeypatch.setattr(
        "src.infrastructure.terrain.geotiff_adapter.reproject", fake_reproject
    )

    adapter = GeoTiffTerrainAdapter()
    grid = adapter.load_dem(p)

    assert grid.crs == "EPSG:4326"
    assert grid.data.shape == (20, 20)
    assert np.isfinite(grid.data).all()


def test_nodata_conversion_masked(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "nodata_4326.tif"
    p.write_bytes(b"x")

    transform = Affine.identity()
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=5, height=5)

    # Force a masked array with a known nodata and a known valid pixel
    def fake_read(self, band, *, masked, out_dtype):
        import numpy.ma as ma

        data = np.arange(25, dtype=np.float32).reshape(5, 5)
        m = np.zeros((5, 5), dtype=bool)
        m[0, 0] = True  # nodata
        return ma.MaskedArray(data, mask=m)

    monkeypatch.setattr(FakeDataset, "read", fake_read)
    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())

    adapter = GeoTiffTerrainAdapter()
    grid = adapter.load_dem(p)
    assert np.isnan(grid.data[0, 0])
    assert not np.isnan(grid.data[2, 2])


def test_all_nodata_same_crs_raises(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "all_nodata.tif"
    p.write_bytes(b"x")

    transform = Affine.identity()
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=5, height=5)

    def fake_read(self, band, *, masked, out_dtype):
        import numpy.ma as ma

        data = np.zeros((5, 5), dtype=np.float32)
        m = np.ones((5, 5), dtype=bool)
        return ma.MaskedArray(data, mask=m)

    monkeypatch.setattr(FakeDataset, "read", fake_read)
    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(AllNoDataError):
        adapter.load_dem(p)


def test_logging_high_nodata_warning(monkeypatch, tmp_path, caplog):
    from affine import Affine

    p = tmp_path / "high_nodata.tif"
    p.write_bytes(b"x")

    transform = Affine.identity()
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=10, height=10)

    def fake_read(band, *, masked, out_dtype):
        import numpy.ma as ma

        data = np.zeros((10, 10), dtype=np.float32)
        m = np.zeros((10, 10), dtype=bool)
        m[:9, :] = True  # 90% masked
        return ma.MaskedArray(data, mask=m)

    monkeypatch.setattr(FakeDataset, "read", fake_read)
    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())

    caplog.set_level("WARNING")
    adapter = GeoTiffTerrainAdapter()
    adapter.load_dem(p)
    assert "NoData pixels detected" in caplog.text


def test_logging_reprojection_info(monkeypatch, tmp_path, caplog):
    from affine import Affine

    p = tmp_path / "utm_reprj.tif"
    p.write_bytes(b"x")

    ds = FakeDataset(
        count=1, crs="EPSG:31983", transform=Affine.identity(), width=5, height=5
    )

    def fake_cdt(src_crs, dst_crs, w, h, *bounds):
        return Affine.identity(), 5, 5

    def fake_reproject(source, destination, **kwargs):
        destination[:] = 1.0

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr("rasterio.Env", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        "src.infrastructure.terrain.geotiff_adapter.calculate_default_transform",
        fake_cdt,
    )
    monkeypatch.setattr(
        "src.infrastructure.terrain.geotiff_adapter.reproject", fake_reproject
    )

    caplog.set_level("INFO")
    adapter = GeoTiffTerrainAdapter()
    adapter.load_dem(p)
    assert "Reprojected from EPSG:31983 to EPSG:4326" in caplog.text


def test_extreme_elevations_valid(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "extreme.tif"
    p.write_bytes(b"x")

    transform = Affine.identity()
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=3, height=1)

    def fake_read(band, *, masked, out_dtype):
        import numpy.ma as ma

        data = np.array([[-430.0, 0.0, 8849.0]], dtype=np.float32)
        mask = np.zeros_like(data, dtype=bool)
        return ma.MaskedArray(data, mask=mask)

    monkeypatch.setattr(FakeDataset, "read", fake_read)
    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr(
        "rasterio.Env",
        lambda *a, **k: SimpleNamespace(
            __enter__=lambda s: s, __exit__=lambda s, *x: False
        ),
    )

    adapter = GeoTiffTerrainAdapter()
    grid = adapter.load_dem(p)
    assert np.nanmin(grid.data) < 0
    assert np.nanmax(grid.data) > 8000


def test_non_geotiff_rejected(monkeypatch, tmp_path):
    import rasterio

    p = tmp_path / "image.png"
    p.write_bytes(b"x")

    def fake_open(path):
        raise rasterio.errors.RasterioError("Not a GeoTIFF")

    monkeypatch.setattr("rasterio.open", fake_open)
    monkeypatch.setattr(
        "rasterio.Env",
        lambda *a, **k: SimpleNamespace(
            __enter__=lambda s: s, __exit__=lambda s, *x: False
        ),
    )

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(InvalidRasterError):
        adapter.load_dem(p)


def test_corrupted_file_rejected(monkeypatch, tmp_path):
    import rasterio

    p = tmp_path / "corrupted.tif"
    p.write_bytes(b"x")

    def fake_open(path):
        raise rasterio.errors.RasterioIOError("Corrupted data")

    monkeypatch.setattr("rasterio.open", fake_open)
    monkeypatch.setattr(
        "rasterio.Env",
        lambda *a, **k: SimpleNamespace(
            __enter__=lambda s: s, __exit__=lambda s, *x: False
        ),
    )

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(InvalidRasterError):
        adapter.load_dem(p)


def test_invalid_bounds_after_reproject_raises(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "bad_bounds_reproj.tif"
    p.write_bytes(b"x")

    ds = FakeDataset(
        count=1, crs="EPSG:31983", transform=Affine.identity(), width=10, height=10
    )

    def fake_cdt(src_crs, dst_crs, w, h, *bounds):
        # Produce obviously invalid lon/lat bounds via transform
        return Affine.translation(200.0, 100.0) * Affine.scale(1.0, 1.0), 5, 5

    def fake_reproject(source, destination, **kwargs):
        destination[:] = 1.0

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr(
        "rasterio.Env",
        lambda *a, **k: SimpleNamespace(
            __enter__=lambda s: s, __exit__=lambda s, *x: False
        ),
    )
    monkeypatch.setattr(
        "src.infrastructure.terrain.geotiff_adapter.calculate_default_transform",
        fake_cdt,
    )
    monkeypatch.setattr(
        "src.infrastructure.terrain.geotiff_adapter.reproject", fake_reproject
    )

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(InvalidBoundsError):
        adapter.load_dem(p)


def test_path_vs_string_input(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "dem_path.tif"
    p.write_bytes(b"x")

    transform = Affine.identity()
    ds = FakeDataset(count=1, crs="EPSG:4326", transform=transform, width=3, height=3)

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr(
        "rasterio.Env",
        lambda *a, **k: SimpleNamespace(
            __enter__=lambda s: s, __exit__=lambda s, *x: False
        ),
    )

    adapter = GeoTiffTerrainAdapter()
    grid1 = adapter.load_dem(str(p))
    grid2 = adapter.load_dem(p)
    assert np.array_equal(grid1.data, grid2.data, equal_nan=True)


@pytest.mark.slow
def test_large_file_loads_placeholder(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "large.tif"
    p.write_bytes(b"x")

    ds = FakeDataset(
        count=1, crs="EPSG:4326", transform=Affine.identity(), width=100, height=100
    )
    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr(
        "rasterio.Env",
        lambda *a, **k: SimpleNamespace(
            __enter__=lambda s: s, __exit__=lambda s, *x: False
        ),
    )

    adapter = GeoTiffTerrainAdapter()
    grid = adapter.load_dem(p)
    assert grid.data.shape[0] > 0


def test_all_nodata_after_reproject_raises(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "utm_all_nan.tif"
    p.write_bytes(b"x")

    src_transform = Affine.identity()
    ds = FakeDataset(
        count=1, crs="EPSG:31983", transform=src_transform, width=10, height=10
    )

    def fake_cdt(src_crs, dst_crs, w, h, *bounds):
        dst_transform = Affine.translation(0.0, 0.0) * Affine.scale(1.0, 1.0)
        return dst_transform, 5, 5

    def fake_reproject(source, destination, **kwargs):
        destination[:] = np.nan

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr(
        "rasterio.Env",
        lambda *a, **k: SimpleNamespace(
            __enter__=lambda s: s, __exit__=lambda s, *x: False
        ),
    )
    monkeypatch.setattr(
        "src.infrastructure.terrain.geotiff_adapter.calculate_default_transform",
        fake_cdt,
    )
    monkeypatch.setattr(
        "src.infrastructure.terrain.geotiff_adapter.reproject", fake_reproject
    )

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(AllNoDataError):
        adapter.load_dem(p)


def test_memory_budget_exceeded_same_crs(monkeypatch, tmp_path):
    from affine import Affine

    p = tmp_path / "big_4326.tif"
    p.write_bytes(b"x")

    transform = Affine.identity()
    ds = FakeDataset(
        count=1, crs="EPSG:4326", transform=transform, width=5000, height=5000
    )

    monkeypatch.setattr("rasterio.open", lambda path: ds)
    monkeypatch.setattr(
        "rasterio.Env",
        lambda *a, **k: SimpleNamespace(
            __enter__=lambda s: s, __exit__=lambda s, *x: False
        ),
    )

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
    monkeypatch.setattr(
        "rasterio.Env",
        lambda *a, **k: SimpleNamespace(
            __enter__=lambda s: s, __exit__=lambda s, *x: False
        ),
    )

    adapter = GeoTiffTerrainAdapter()
    with pytest.raises(InvalidBoundsError):
        adapter.load_dem(p)
