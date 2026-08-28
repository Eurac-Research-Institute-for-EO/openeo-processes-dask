"""run_udf must hand UDFs semantic dimension names (issue #24).

apply_dimension passes a raw array into run_udf, so the cube used to arrive as
dim_0..dim_N and every UDF had to call fix_udf_dimensions itself — which welds
the UDF to this backend and breaks portability. run_udf now restores the names
from the _openeo_dimension_metadata that apply_dimension puts in the context.
"""

import numpy as np
import xarray as xr

from openeo_processes_dask.process_implementations.udf import run_udf

UDF_SEMANTIC = """
import xarray as xr

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    # No helper import: the backend must already have restored the names.
    return cube.rolling({"t": 2}).sum().fillna(0)
"""

UDF_REPORT_DIMS = """
import xarray as xr

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    return xr.DataArray(list(cube.dims))
"""


def _context(dims, shape):
    """The metadata apply_dimension attaches to the context."""
    return {
        "_openeo_dimension_metadata": {
            "current_dimension": "t",
            "all_dimensions": list(dims),
            "data_shape": tuple(shape),
            "dimension_coords": {d: None for d in dims},
        }
    }


class TestRunUdfRestoresDimensions:
    def test_raw_array_gets_semantic_names(self):
        """A UDF using a real dimension name works without any helper call."""
        # as apply_dimension delivers it: raw array, target dim moved last
        raw = np.arange(2 * 3 * 4, dtype="float64").reshape(2, 3, 4)
        out = run_udf(
            raw, UDF_SEMANTIC, "Python", context=_context(("t", "y", "x"), (4, 2, 3))
        )
        assert isinstance(out, xr.DataArray)
        assert out.shape == raw.shape

    def test_dims_visible_to_udf_are_not_generic(self):
        raw = np.zeros((2, 3, 4))
        out = run_udf(
            raw, UDF_REPORT_DIMS, "Python", context=_context(("t", "y", "x"), (4, 2, 3))
        )
        seen = [str(d) for d in out.values]
        assert not any(d.startswith("dim_") for d in seen), seen
        assert "t" in seen

    def test_dataarray_input_keeps_its_own_dims(self):
        """An input that already has names must be passed through untouched."""
        cube = xr.DataArray(np.zeros((2, 3)), dims=("bands", "x"))
        out = run_udf(cube, UDF_REPORT_DIMS, "Python", context={})
        assert [str(d) for d in out.values] == ["bands", "x"]

    def test_no_metadata_does_not_raise(self):
        """Without metadata the cube stays generic, but nothing blows up."""
        out = run_udf(np.zeros((2, 3)), UDF_REPORT_DIMS, "Python", context={})
        assert len(out.values) == 2
