import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

logger = logging.getLogger(__name__)


def create_multiresolution_rastercube(
    spatial_extent: BoundingBox,
    temporal_extent: TemporalInterval,
    backend="numpy",
    chunks=("auto", "auto", -1),
):
    fine_x = np.arange(
        min(spatial_extent.west, spatial_extent.east),
        max(spatial_extent.west, spatial_extent.east),
        0.01,
    )
    fine_y = np.arange(
        min(spatial_extent.south, spatial_extent.north),
        max(spatial_extent.south, spatial_extent.north),
        0.01,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        t_coords = pd.date_range(
            start=np.datetime64(temporal_extent.root[0].root),
            end=np.datetime64(temporal_extent.root[1].root),
            periods=4,
        ).values

    rng = np.random.default_rng(42)
    fine_ny, fine_nx = len(fine_y), len(fine_x)
    coarse_ny = max(1, fine_ny // 2)
    coarse_nx = max(1, fine_nx // 2)

    variables = {}
    for name, dtype, common_name in [
        ("B02", np.int16, "blue"),
        ("B03", np.int16, "green"),
    ]:
        data = rng.integers(-100, 100, size=(fine_ny, fine_nx, 4)).astype(dtype)
        variables[name] = xr.DataArray(
            data,
            name=name,
            dims=["y", "x", "t"],
            coords={"y": fine_y, "x": fine_x, "t": t_coords},
            attrs={"common_name": common_name},
        )

    coarse_x = fine_x[::2][:coarse_nx]
    coarse_y = fine_y[::2][:coarse_ny]
    coarse_data = rng.integers(
        -100, 100, size=(coarse_ny, coarse_nx, 4)
    ).astype(np.float32)
    variables["B08"] = xr.DataArray(
        coarse_data,
        name="B08",
        dims=["y", "x", "t"],
        coords={"y": coarse_y, "x": coarse_x, "t": t_coords},
        attrs={"common_name": "nir"},
    )

    ds = xr.merge(
        list(variables.values()),
        compat="override",
        combine_attrs="drop_conflicts",
    )
    ds.attrs["title"] = "multi-resolution test cube"

    for var_name in variables:
        ds[var_name].attrs["common_name"] = variables[var_name].attrs["common_name"]

    import odc.geo.xr

    ds = odc.geo.xr.assign_crs(ds, crs=spatial_extent.crs)

    if "dask" in backend:
        import dask.array as da

        ds = ds.chunk({dim: ch for dim, ch in zip(ds.dims, chunks)})

    return ds


def create_fake_rastercube(
    data,
    spatial_extent: BoundingBox,
    temporal_extent: TemporalInterval,
    bands: list,
    backend="numpy",
    chunks=("auto", "auto", "auto", -1),
    as_dataset=True,
):
    # Calculate the desired resolution based on how many samples we desire on the longest axis.
    len_x = max(spatial_extent.west, spatial_extent.east) - min(
        spatial_extent.west, spatial_extent.east
    )
    len_y = max(spatial_extent.south, spatial_extent.north) - min(
        spatial_extent.south, spatial_extent.north
    )

    x_coords = np.arange(
        min(spatial_extent.west, spatial_extent.east),
        max(spatial_extent.west, spatial_extent.east),
        step=len_x / data.shape[0],
    )
    y_coords = np.arange(
        min(spatial_extent.south, spatial_extent.north),
        max(spatial_extent.south, spatial_extent.north),
        step=len_y / data.shape[1],
    )

    # This line raises a deprecation warning, which according to this thread
    # will never actually be deprecated:
    # https://github.com/numpy/numpy/issues/23904
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        t_coords = pd.date_range(
            start=np.datetime64(temporal_extent.root[0].root),
            end=np.datetime64(temporal_extent.root[1].root),
            periods=data.shape[2],
        ).values

    coords = {"x": x_coords, "y": y_coords, "t": t_coords, "bands": bands}

    raster_cube = xr.DataArray(
        data=data,
        coords=coords,
        attrs={"crs": spatial_extent.crs},
    )
    import odc.geo.xr

    raster_cube = odc.geo.xr.assign_crs(raster_cube, crs=spatial_extent.crs)

    if "dask" in backend:
        import dask.array as da

        raster_cube.data = da.from_array(raster_cube.data, chunks=chunks)

    if as_dataset:
        raster_cube = raster_cube.to_dataset(dim="bands")

    return raster_cube
