import logging
from typing import Dict, Optional, Union

import numpy as np
import odc.geo.xr
import rioxarray  # needs to be imported to set .rio accessor on xarray objects.
import xarray as xr
from odc.geo.geobox import resolution_from_affine
from pyproj import Transformer
from pyproj.crs import CRS, CRSError

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionMissing,
    OpenEOException,
)

logger = logging.getLogger(__name__)

__all__ = ["resample_spatial", "resample_cube_spatial", "resample_cube_temporal"]

resample_methods_list = [
    "near",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "geocode",  # new-addition which uses xcube
]


def _find_lon_lat_vars(ds: xr.Dataset) -> tuple[str, str]:
    """
    Return names of lon/lat variables in ds.
    Accepts both (lon, lat) and (longitude, latitude) as coords or data_vars.
    """
    candidates = [
        ("lon", "lat"),
        ("longitude", "latitude"),
    ]
    for lon_name, lat_name in candidates:
        if (lon_name in ds.coords or lon_name in ds.data_vars) and (
            lat_name in ds.coords or lat_name in ds.data_vars
        ):
            return lon_name, lat_name
    raise OpenEOException(
        'method="geocode" requires 2D lon/lat layers present as variables or coordinates '
        "(expected names: lon/lat or longitude/latitude)."
    )


def _default_interp_methods_from_dtypes(ds: xr.Dataset) -> dict[str, str]:
    """
    xcube supports 'nearest' and 'bilinear' (among others), and applies them per variable.
    We default:
      - integer/flags -> nearest
      - float -> bilinear
    """
    methods: dict[str, str] = {}
    for var_name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.integer) or np.issubdtype(da.dtype, np.bool_):
            methods[var_name] = "nearest"
        else:
            methods[var_name] = "bilinear"
    return methods


def _build_target_gm_from_lonlat_bbox(
    lon2d: xr.DataArray,
    lat2d: xr.DataArray,
    target_crs: CRS,
    resolution: float,
    tile_size: int = 1024,
):
    """
    Build a regular xcube GridMapping using the lon/lat bbox as extent.
    If target_crs != EPSG:4326, bbox is transformed to target CRS.
    """
    # Lazy reductions (works with dask-backed lon/lat)
    west = float(lon2d.min().compute())
    east = float(lon2d.max().compute())
    south = float(lat2d.min().compute())
    north = float(lat2d.max().compute())

    # Transform bounds from EPSG:4326 into target CRS if needed
    src_crs = CRS.from_epsg(4326)
    if not target_crs.equals(src_crs) and not (
        target_crs.is_geographic and src_crs.is_geographic
    ):
        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
        west, south, east, north = transformer.transform_bounds(
            west, south, east, north
        )

    # Import here so the module still works even if xcube-resampling isn't installed
    from xcube_resampling.gridmapping import GridMapping

    return GridMapping.regular_from_bbox(
        bbox=(west, south, east, north),
        xy_res=resolution,
        crs=target_crs,
        tile_size=tile_size,
        is_j_axis_up=False,
    )


def resample_spatial(
    data: RasterCube,
    projection: Optional[Union[str, int]] = None,
    resolution: int = 0,
    method: str = "near",
    align: str = "upper-left",
):
    """Resamples the spatial dimensions (x,y) of the data cube to a specified resolution and/or warps the data cube to the target projection. At least resolution or projection must be specified."""

    if data.openeo.y_dim is None or data.openeo.x_dim is None:
        raise DimensionMissing(f"Spatial dimension missing for dataset: {data} ")

    if method not in resample_methods_list:
        raise Exception(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(resample_methods_list)}]"
        )

    dim_order = data.dims
    data_cp = data.transpose(..., data.openeo.y_dim, data.openeo.x_dim)

    # using xcube if method == "geocode" --> xcube-based "geocode"
    if method == "geocode":
        try:
            from xcube_resampling.rectify import rectify_dataset
            from xcube_resampling.spatial import resample_in_space
        except Exception as e:
            raise OpenEOException(
                'method="geocode" requires the optional dependency "xcube-resampling" '
                "to be installed."
            ) from e

        y_dim = data.openeo.y_dim
        x_dim = data.openeo.x_dim
        band_dim = getattr(data.openeo, "band_dim", None) or "bands"

        def _bands_da_to_vars_ds(
            da: xr.DataArray, band_dim_: str = "bands"
        ) -> xr.Dataset:
            """
            Convert DataArray with a bands dimension into Dataset with one data_var per band label.
            Examples:
              da(bands,y,x) with bands=['chl_nn'] -> ds with var 'chl_nn'(y,x)
              da(time,bands,y,x)                 -> ds with vars '...' (time,y,x)
            """
            # If there is no bands dim, fall back to a single-var dataset
            if band_dim_ not in da.dims:
                name = da.name or "data"
                return da.to_dataset(name=name)

            if band_dim_ not in da.coords:
                raise OpenEOException(
                    f'For method="geocode", band dimension {band_dim_!r} must have labels as a coordinate.'
                )

            labels = [str(v) for v in da[band_dim_].values.tolist()]

            vars_dict = {}
            for lbl in labels:
                # .sel keeps all remaining dims, drops only the selected band coordinate
                band_slice = da.sel({band_dim_: lbl}).drop_vars(
                    band_dim_, errors="ignore"
                )
                vars_dict[lbl] = band_slice

            # Keep all non-band coords (including time, lon/lat if present as coords)
            coords = {k: v for k, v in da.coords.items() if k != band_dim_}
            ds = xr.Dataset(vars_dict, coords=coords)

            # Preserve attrs at dataset level for convenience
            ds.attrs.update(getattr(da, "attrs", {}))
            return ds

        def _stack_extra_dims_for_xcube(
            ds: xr.Dataset, vars_: list[str]
        ) -> tuple[xr.Dataset, bool, list[str]]:
            """
            Ensure xcube sees at most one non-spatial dim before (y,x).
            If vars have >1 extra dims (e.g. time,bands,y,x), stack all extra dims into '__plane__'.
            Returns (new_ds, stacked?, extra_dims)
            """
            plane_dim_ = "__plane__"
            # Determine extra dims from the first variable (assume consistent for payload)
            v0 = vars_[0]
            extra = [d for d in ds[v0].dims if d not in (y_dim, x_dim)]
            stacked_ = False

            # Always enforce spatial dims at the end
            if len(extra) == 0:
                for v in vars_:
                    ds[v] = ds[v].transpose(y_dim, x_dim)
                return ds, False, extra

            if len(extra) == 1:
                for v in vars_:
                    ds[v] = ds[v].transpose(extra[0], y_dim, x_dim)
                return ds, False, extra

            # len(extra) > 1 -> stack
            stacked_ = True
            stacked_vars = {}
            for v in vars_:
                da = ds[v].transpose(*extra, y_dim, x_dim)
                stacked_vars[v] = da.stack({plane_dim_: extra})
            ds = ds.assign(stacked_vars)
            return ds, stacked_, extra

        def _unstack_after_xcube(
            da: xr.DataArray, extra_dims: list[str]
        ) -> xr.DataArray:
            """
            Unstack '__plane__' back to original extra dims order.
            """
            plane_dim_ = "__plane__"
            if plane_dim_ in da.dims:
                da = da.unstack(plane_dim_)
                # Restore extra dims order explicitly (xarray may reorder)
                da = da.transpose(*extra_dims, y_dim, x_dim)
            return da

        # build source_ds
        is_dataarray = isinstance(data_cp, xr.DataArray)

        if is_dataarray:
            # Convert bands->vars so payload isn't hidden inside a 'bands' dimension
            source_ds = _bands_da_to_vars_ds(data_cp, band_dim_=band_dim)
        else:
            source_ds = data_cp

        # find lon/lat
        lon_name, lat_name = _find_lon_lat_vars(source_ds)
        lon2d = source_ds[lon_name]
        lat2d = source_ds[lat_name]

        if lon2d.ndim != 2 or lat2d.ndim != 2:
            raise OpenEOException(
                f'For method="geocode", {lon_name!r} and {lat_name!r} must be 2D arrays '
                f"aligned to the spatial grid (got shapes {lon2d.shape} and {lat2d.shape})."
            )

        # force lon/lat to be coords (safest for GridMapping inference)
        source_ds = source_ds.assign_coords({lon_name: lon2d, lat_name: lat2d})

        # payload variables
        payload_vars = list(source_ds.data_vars)

        # remove lon/lat/spatial_ref if they are data_vars in some inputs
        payload_vars = [
            v for v in payload_vars if v not in (lon_name, lat_name, "spatial_ref")
        ]
        if not payload_vars:
            raise OpenEOException(
                'method="geocode": no payload variables found to rectify (only lon/lat present?).'
            )

        # Default per-variable interpolation (flags -> nearest, floats -> bilinear)
        interp_methods = _default_interp_methods_from_dtypes(source_ds[payload_vars])

        # Ensure xcube can handle dimensionality: stack if needed
        source_payload = source_ds[payload_vars]
        source_payload, stacked, extra_dims = _stack_extra_dims_for_xcube(
            source_payload, payload_vars
        )

        # Decide default vs explicit target grid
        user_passed_projection = projection is not None
        user_passed_resolution = resolution not in (0, None)

        if not user_passed_projection and not user_passed_resolution:
            # Default xcube behavior: derive regular grid from source GM internally (to_regular)
            out_ds = rectify_dataset(
                source_payload,
                interp_methods=interp_methods,
                tile_size=1024,
            )
        else:
            # Explicit grid mode: projection/resolution define the output grid
            if projection is None:
                target_crs = CRS.from_epsg(4326)
            else:
                try:
                    target_crs = CRS.from_user_input(projection)
                except CRSError as e:
                    raise CRSError(
                        f"Provided projection string: '{projection}' can not be parsed to CRS."
                    ) from e

            if not user_passed_resolution:
                raise OpenEOException(
                    'method="geocode": if "projection" is provided explicitly, you must also '
                    'provide a non-zero "resolution" so an explicit target grid can be built.'
                )

            target_gm = _build_target_gm_from_lonlat_bbox(
                lon2d=lon2d,
                lat2d=lat2d,
                target_crs=target_crs,
                resolution=float(resolution),
                tile_size=1024,
            )

            # resample_in_space will pick rectification (irregular -> regular) automatically
            out_ds = resample_in_space(
                source_payload,
                target_gm=target_gm,
                interp_methods=interp_methods,
            )

        # convert output back
        if is_dataarray:
            # Convert vars back into a banded DataArray, preserving time/etc.
            out_vars = [v for v in out_ds.data_vars if v != "spatial_ref"]
            if not out_vars:
                raise OpenEOException(
                    'method="geocode": rectification produced no output variables (only spatial_ref).'
                )

            out_bands = []
            for v in out_vars:
                da_v = out_ds[v]
                if stacked:
                    da_v = _unstack_after_xcube(da_v, extra_dims)
                out_bands.append(da_v)

            out = xr.concat(out_bands, dim=band_dim)
            out = out.assign_coords({band_dim: out_vars})

            rename_dims = {}
            if "lat" in out.dims and y_dim not in out.dims:
                rename_dims["lat"] = y_dim
            if "lon" in out.dims and x_dim not in out.dims:
                rename_dims["lon"] = x_dim
            if "latitude" in out.dims and y_dim not in out.dims:
                rename_dims["latitude"] = y_dim
            if "longitude" in out.dims and x_dim not in out.dims:
                rename_dims["longitude"] = x_dim
            if rename_dims:
                out = out.rename(rename_dims)

            out = out.transpose(*dim_order, missing_dims="ignore")

        else:
            # dataset input: just unstack variables if needed and restore dim order
            if stacked:
                fixed = {}
                for v in [v for v in out_ds.data_vars if v != "spatial_ref"]:
                    fixed[v] = _unstack_after_xcube(out_ds[v], extra_dims)
                out_ds = out_ds.assign(fixed)
            out = out_ds.transpose(*dim_order, missing_dims="ignore")

        # preserve original attrs (except CRS handled elsewhere)
        for k, v in data.attrs.items():
            if k.lower() != "crs":
                out.attrs[k] = v

        return out

    # ORIGINAL ODC-based behavior continues from here
    # Assert resampling method is correct.
    if method == "near":
        method = "nearest"

    elif method not in resample_methods_list:
        raise OpenEOException(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(resample_methods_list)}]"
        )

    if projection is None:
        projection = data_cp.rio.crs

    try:
        projection = CRS.from_user_input(projection)
    except CRSError as e:
        raise CRSError(
            f"Provided projection string: '{projection}' can not be parsed to CRS."
        ) from e

    if resolution == 0:
        resolution = resolution_from_affine(data_cp.odc.geobox.affine).x

    reprojected = data_cp.odc.reproject(
        how=projection, resolution=resolution, resampling=method
    )

    if reprojected.openeo.x_dim != data.openeo.x_dim:
        reprojected = reprojected.rename({reprojected.openeo.x_dim: data.openeo.x_dim})

    if reprojected.openeo.y_dim != data.openeo.y_dim:
        reprojected = reprojected.rename({reprojected.openeo.y_dim: data.openeo.y_dim})

    reprojected = reprojected.transpose(*dim_order)

    reprojected.attrs["crs"] = data_cp.rio.crs

    return reprojected


def resample_cube_spatial(
    data: RasterCube, target: RasterCube, method="near", options=None
) -> RasterCube:
    methods_list = [
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
    ]

    if (
        data.openeo.y_dim is None
        or data.openeo.x_dim is None
        or target.openeo.y_dim is None
        or target.openeo.x_dim is None
    ):
        raise DimensionMissing(
            f"Spatial dimension missing from data or target. Available dimensions for data: {data.dims} for target: {target.dims}"
        )

    # ODC reproject requires y to be before x
    required_dim_order = (..., data.openeo.y_dim, data.openeo.x_dim)

    data_reordered = data.transpose(*required_dim_order, missing_dims="ignore")
    target_reordered = target.transpose(*required_dim_order, missing_dims="ignore")

    if method == "near":
        method = "nearest"

    elif method not in methods_list:
        raise Exception(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(methods_list)}]"
        )

    resampled_data = data_reordered.odc.reproject(
        target_reordered.odc.geobox, resampling=method
    )

    resampled_data.rio.write_crs(target_reordered.rio.crs, inplace=True)

    try:
        # odc.reproject renames the coordinates according to the geobox, this undoes that.
        resampled_data = resampled_data.rename(
            {"longitude": data.openeo.x_dim, "latitude": data.openeo.y_dim}
        )
    except ValueError:
        pass

    # Order axes back to how they were before
    resampled_data = resampled_data.transpose(*data.dims)

    # Ensure that attrs except crs are copied over
    for k, v in data.attrs.items():
        if k.lower() != "crs":
            resampled_data.attrs[k] = v
    return resampled_data


def resample_cube_temporal(data, target, dimension=None, valid_within=None):
    if dimension is None:
        if len(data.openeo.temporal_dims) > 0:
            dimension = data.openeo.temporal_dims[0]
        else:
            raise Exception("DimensionNotAvailable")
    if dimension not in data.dims:
        raise Exception("DimensionNotAvailable")
    if dimension not in target.dims:
        if len(target.openeo.temporal_dims) > 0:
            target_time = target.openeo.temporal_dims[0]
        else:
            raise Exception("DimensionNotAvailable")
        target = target.rename({target_time: dimension})
    index = []
    for d in target[dimension].values:
        difference = np.abs(d - data[dimension].values)
        nearest = np.argwhere(difference == np.min(difference))
        # The rare case of ties is resolved by choosing the earlier timestamps. (index 0)
        if np.shape(nearest) == (2, 1):
            nearest = nearest[0]
        if np.shape(nearest) == (1, 2):
            nearest = nearest[:, 0]
        index.append(int(nearest))
    times_at_target_time = data[dimension].values[index]
    new_data = data.loc[{dimension: times_at_target_time}]
    filter_values = new_data[dimension].values
    new_data[dimension] = target[dimension].values
    # valid_within
    if valid_within is None:
        new_data = new_data
    else:
        minimum = np.timedelta64(valid_within, "D")
        filter_valid = np.abs(filter_values - new_data[dimension].values) <= minimum
        times_valid = new_data[dimension].values[filter_valid]
        valid_data = new_data.loc[{dimension: times_valid}]
        filter_nan = np.abs(filter_values - new_data[dimension].values) > minimum
        times_nan = new_data[dimension].values[filter_nan]
        nan_data = new_data.loc[{dimension: times_nan}] * np.nan
        combined = xr.concat([valid_data, nan_data], dim=dimension)
        new_data = combined.sortby(dimension)
    new_data.attrs = data.attrs
    return new_data
