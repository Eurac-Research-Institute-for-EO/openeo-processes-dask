import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask.process_implementations.export.save_result import (
    _clean_unused_coordinates,
    _normalize_format,
    _rastercube_to_dataset,
    save_result,
)

def _make_test_cube() -> xr.DataArray:
    """
    Build a minimal RasterCube-compatible xarray.DataArray.

    In this codebase RasterCube is just an xr.DataArray, and the implementation
    relies on the .openeo and .rio accessors.
    """
    data = xr.DataArray(
        np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4),
        dims=("bands", "time", "y", "x"),
        coords={
            "bands": ["B01"],
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "y": [46.0, 45.5, 45.0],
            "x": [10.0, 10.5, 11.0, 11.5],
            "unused_coord": 123,
        },
        name="test_cube",
    )

    # The save_result implementation expects openEO dimension typing.
    data.openeo.add_dim_type("bands", "bands")
    data.openeo.add_dim_type("time", "temporal")
    data.openeo.add_dim_type("x", "spatial")
    data.openeo.add_dim_type("y", "spatial")

    # The implementation also tries to propagate CRS via .rio
    data = data.rio.write_crs("EPSG:4326", inplace=False)

    return data


def test_normalize_format_handles_aliases():
    assert _normalize_format("nc") == "netcdf"
    assert _normalize_format("NeTcDf") == "netcdf"
    assert _normalize_format("zarr") == "zarr"
    assert _normalize_format("GTIFF") == "gtiff"


def test_clean_unused_coordinates_removes_orphan_coords():
    ds = xr.Dataset(
        data_vars={
            "B01": (("time", "y", "x"), np.ones((2, 3, 4), dtype=np.float32)),
        },
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "y": [46.0, 45.5, 45.0],
            "x": [10.0, 10.5, 11.0, 11.5],
            "unused_coord": 123,
        },
    )

    cleaned = _clean_unused_coordinates(ds)

    assert "unused_coord" not in cleaned.coords
    assert "B01" in cleaned.data_vars


def test_rastercube_to_dataset_preserves_expected_attrs():
    cube = _make_test_cube()

    ds = _rastercube_to_dataset(cube)

    assert isinstance(ds, xr.Dataset)
    assert "B01" in ds.data_vars
    assert ds.attrs["openeo_band_dims"] == ["bands"]
    assert ds.attrs["openeo_temporal_dims"] == ["time"]
    assert ds.attrs["openeo_x_dim"] == "x"
    assert ds.attrs["openeo_y_dim"] == "y"
    assert ds.attrs["openeo_other_dims"] == []
    assert ds.attrs["crs"] == "EPSG:4326"
    assert "unused_coord" not in ds.coords


def test_save_result_requires_path_option():
    cube = _make_test_cube()

    with pytest.raises(ValueError, match=r"options\['path'\]"):
        save_result(cube, format="netcdf", options={})


def test_save_result_requires_collection_url():
    cube = _make_test_cube()

    with pytest.raises(ValueError, match=r"options\['collection_url'\]"):
        save_result(
            cube,
            format="netcdf",
            options={"path": "dummy-output"},
        )


def test_save_result_unsupported_format_raises(tmp_path: Path):
    cube = _make_test_cube()

    with pytest.raises(ValueError, match="Unsupported format"):
        save_result(
            cube,
            format="GTIFF",
            options={
                "path": str(tmp_path / "result"),
                "collection_url": "https://stac.openeo.eurac.edu/api/v1/pgstac/collections/",
            },
        )


def test_save_result_netcdf_missing_dependency_is_clear(tmp_path: Path, monkeypatch):
    cube = _make_test_cube()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "raster2stac":
            raise ImportError("No module named 'raster2stac'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(
        ImportError,
        match=r"openeo-processes-dask\[implementations,export\]",
    ):
        save_result(
            cube,
            format="netcdf",
            options={
                "path": str(tmp_path / "result"),
                "collection_url": "https://stac.openeo.eurac.edu/api/v1/pgstac/collections/",
            },
        )


def test_save_result_zarr_missing_dependency_is_clear(tmp_path: Path, monkeypatch):
    cube = _make_test_cube()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "raster2stac":
            raise ImportError("No module named 'raster2stac'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(
        ImportError,
        match=r"openeo-processes-dask\[implementations,export\]",
    ):
        save_result(
            cube,
            format="zarr",
            options={
                "path": str(tmp_path / "result"),
                "collection_url": "https://stac.openeo.eurac.edu/api/v1/pgstac/collections/",
            },
        )


def test_save_result_netcdf_returns_collection_when_available(tmp_path: Path):
    cube = _make_test_cube()

    try:
        import raster2stac  # noqa: F401
    except ImportError:
        pytest.skip("raster2stac not installed; cannot test netcdf export path.")

    result = save_result(
        cube,
        format="netcdf",
        options={
            "path": str(tmp_path / "netcdf_result"),
            "collection_id": "test-netcdf-result",
            "collection_url": "https://stac.openeo.eurac.edu/api/v1/pgstac/collections/",
            "description": "Test netcdf result",
            "title": "Test NetCDF Result",
            "license": "proprietary",
        },
    )

    # raster2stac writes {collection_id}.json, not metadata.json.
    output_folder = tmp_path / "netcdf_result"
    collection_json = output_folder / "test-netcdf-result.json"
    assert (
        collection_json.exists()
    ), f"Expected {collection_json}; found: {list(output_folder.iterdir())}"

    metadata = json.loads(collection_json.read_text())
    assert metadata["type"] == "Collection"
    assert metadata["id"] == "test-netcdf-result"

    # save_result now returns {"collection": ..., "items": ...}
    assert set(result.keys()) == {"collection", "items"}
    assert result["collection"]["type"] == "Collection"
    assert result["collection"]["id"] == "test-netcdf-result"
    assert isinstance(result["items"], dict)


def test_save_result_netcdf_is_case_insensitive(tmp_path: Path):
    cube = _make_test_cube()

    try:
        import raster2stac  # noqa: F401
    except ImportError:
        pytest.skip("raster2stac not installed; cannot test netcdf export path.")

    result = save_result(
        cube,
        format="Nc",
        options={
            "path": str(tmp_path / "netcdf_result_case"),
            "collection_id": "test-netcdf-case",
            "collection_url": "https://stac.openeo.eurac.edu/api/v1/pgstac/collections/",
            "description": "Case insensitive netcdf",
            "title": "Case NetCDF",
            "license": "proprietary",
        },
    )

    # raster2stac writes {collection_id}.json, not metadata.json.
    output_folder = tmp_path / "netcdf_result_case"
    collection_json = output_folder / "test-netcdf-case.json"
    assert (
        collection_json.exists()
    ), f"Expected {collection_json}; found: {list(output_folder.iterdir())}"

    assert result["collection"]["type"] == "Collection"
    assert result["collection"]["id"] == "test-netcdf-case"


def test_save_result_zarr_returns_collection_when_available(tmp_path: Path):
    cube = _make_test_cube()

    try:
        import raster2stac
    except ImportError:
        pytest.skip("raster2stac not installed; cannot test zarr export path.")

    output_path = tmp_path / "zarr_result"
    result = save_result(
        cube,
        format="zarr",
        options={
            "path": str(output_path),
            "collection_id": "test-zarr-result",
            "item_id": "test-zarr-item",
            "collection_url": "https://stac.openeo.eurac.edu/api/v1/pgstac/collections/",
            "description": "Test zarr result",
            "title": "Test Zarr Result",
            "license": "proprietary",
        },
    )

    assert output_path.exists(), f"Expected {output_path} to exist."
    assert not (tmp_path / "zarr_result.zarr").exists(), (
        "Path should not have been rewritten with a .zarr suffix."
    )

    # raster2stac writes {collection_id}.json, not metadata.json.
    collection_json = output_path / "test-zarr-result.json"
    assert (
        collection_json.exists()
    ), f"Expected {collection_json}; found: {list(output_path.iterdir())}"

    metadata = json.loads(collection_json.read_text())
    assert metadata["type"] == "Collection"
    assert metadata["id"] == "test-zarr-result"

    # save_result now returns {"collection": ..., "items": ...}
    assert set(result.keys()) == {"collection", "items"}
    assert result["collection"]["type"] == "Collection"
    assert result["collection"]["id"] == "test-zarr-result"
    assert isinstance(result["items"], dict)