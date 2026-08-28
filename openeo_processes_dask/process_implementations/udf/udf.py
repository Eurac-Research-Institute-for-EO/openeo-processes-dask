from typing import Optional, Union

import dask.array as da
import xarray as xr
from openeo.udf import UdfData
from openeo.udf.run_code import run_udf_code
from openeo.udf.xarraydatacube import XarrayDataCube

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.udf.dimension_helper import (
    fix_udf_dimensions,
)

__all__ = ["run_udf"]


def run_udf(
    data: Union[RasterCube, da.Array],
    udf: str,
    runtime: str,
    context: Optional[dict] = None,
) -> RasterCube:
    # Preserve dimension names and coordinates if input is already an xr.DataArray
    if isinstance(data, xr.DataArray):
        # Input is already a proper xr.DataArray (RasterCube), preserve its structure
        data_cube = XarrayDataCube(data)
    else:
        # Input is a dask/numpy array (this is how apply_dimension delivers it),
        # so xr.DataArray gives generic dim_0..dim_N names. Restore the semantic
        # names here from the metadata apply_dimension puts in the context, so
        # UDFs receive a properly named cube and stay portable — they must not
        # have to import backend internals to repair it themselves (issue #24).
        cube = fix_udf_dimensions(xr.DataArray(data), context)
        data_cube = XarrayDataCube(cube)
    data = UdfData(datacube_list=[data_cube], user_context=context)
    result = run_udf_code(code=udf, data=data)
    cubes = result.get_datacube_list()
    if len(cubes) != 1:
        raise ValueError(
            f"The provided UDF should return one datacube, but got: {result}"
        )
    result_array: xr.DataArray = cubes[0].array
    return result_array
