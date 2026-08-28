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
    # apply_dimension hands over an array whose dimension names are generic
    # (dim_0..dim_N) — sometimes already wrapped as a DataArray, sometimes raw —
    # so restore the semantic names from the _openeo_dimension_metadata that
    # apply_dimension puts in the context. Doing it here keeps UDFs portable:
    # they must not have to import backend internals to repair the cube (#24).
    # fix_udf_dimensions is a no-op when the names are already semantic.
    cube = data if isinstance(data, xr.DataArray) else xr.DataArray(data)
    cube = fix_udf_dimensions(cube, context)
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
