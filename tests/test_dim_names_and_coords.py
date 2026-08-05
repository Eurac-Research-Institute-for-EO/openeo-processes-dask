"""Regression tests for openeo-argoworkflows#172.

Two distinct defects, both surfacing on the same user process graph
(load_stac -> apply_dimension(run_udf) -> save_result):

1. Dimension naming: a collection carrying a `cube:dimensions` block WITHOUT
   declaring the datacube extension made the backend name the temporal
   dimension "time" while the openEO client (which checks the extension
   before trusting cube:dimensions) called it "t" — so neither name worked.

2. apply_dimension dropped the coordinate labels of the target dimension when
   the UDF returned the same number of elements (`==` typo instead of `=`),
   producing results whose time axis had no values (files stamped 1970).
"""

import numpy as np
import xarray as xr

from openeo_processes_dask.process_implementations.cubes.apply import apply_dimension
from openeo_processes_dask.process_implementations.cubes.load import (
    _get_dimension_names_from_stac,
)


class _Validator:
    """Minimal stand-in for stac_validator.StacValidate."""

    def __init__(self, content):
        self.stac_content = content


_CUBE_DIMS = {
    "x": {"axis": "x", "type": "spatial"},
    "y": {"axis": "y", "type": "spatial"},
    "time": {"type": "temporal"},
}

DATACUBE_EXT = "https://stac-extensions.github.io/datacube/v2.2.0/schema.json"


class TestDimensionNames:
    def test_cube_dimensions_ignored_when_extension_not_declared(self):
        """Undeclared cube:dimensions must not rename the temporal dimension —
        the client ignores it too, so honouring it desynced the two (#172)."""
        names = _get_dimension_names_from_stac(
            _Validator({"cube:dimensions": _CUBE_DIMS, "stac_extensions": []})
        )
        assert names["t"] == "t"

    def test_cube_dimensions_honoured_when_extension_declared(self):
        """Properly declared datacube metadata is still authoritative."""
        names = _get_dimension_names_from_stac(
            _Validator(
                {"cube:dimensions": _CUBE_DIMS, "stac_extensions": [DATACUBE_EXT]}
            )
        )
        assert names["t"] == "time"

    def test_missing_stac_extensions_key_falls_back(self):
        names = _get_dimension_names_from_stac(_Validator({"cube:dimensions": _CUBE_DIMS}))
        assert names["t"] == "t"


class TestApplyDimensionCoords:
    def _cube(self):
        t = np.array(
            ["2020-06-01", "2020-06-02", "2020-06-03"], dtype="datetime64[ns]"
        )
        return xr.DataArray(
            np.arange(3 * 2 * 2, dtype="float64").reshape(3, 2, 2),
            dims=("t", "y", "x"),
            coords={"t": t, "y": [0, 1], "x": [0, 1]},
        )

    def test_labels_preserved_when_length_unchanged(self):
        """Same-length result must keep the original labels, not lose them."""
        cube = self._cube()
        result = apply_dimension(
            data=cube, process=lambda data, **kw: data, dimension="t"
        )
        np.testing.assert_array_equal(result["t"].values, cube["t"].values)
