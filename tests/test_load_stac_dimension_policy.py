import pytest

from openeo_processes_dask.process_implementations.cubes.load import load_stac


def _cube_dims_in_order(cube):
    """
    Return dimension names in order from the underlying xarray object.

    - If cube is a DataArray: dims is already ordered
    - If cube is a Dataset: pick the first data_var and use its ordered dims
      (Dataset.dims is a mapping => no guaranteed order)
    """
    # DataArray
    if hasattr(cube, "dims") and isinstance(cube.dims, tuple):
        return tuple(cube.dims)

    # Dataset
    if hasattr(cube, "data_vars") and len(cube.data_vars) > 0:
        first_var = next(iter(cube.data_vars))
        return tuple(cube[first_var].dims)

    # Fallback (shouldn't happen)
    return tuple(getattr(cube, "dims", ()))


@pytest.mark.parametrize(
    "case_name, kwargs, expected_dims",
    [
        (
            "case1_eopf_core_local_item",
            dict(
                url="./tests/data/stac/s2_sample_dimension_policy_case_1.json",
                spatial_extent=dict(
                    west=9.669372670305636,
                    south=53.64026948239441,
                    east=9.701345402674315,
                    north=53.66341039786631,
                ),
                temporal_extent=["2025-05-01", "2025-06-01"],
                bands=["b02", "b08"],
            ),
            ("t", "bands", "y", "x"),
        ),
        (
            "case2_planetarycomputer_local_item",
            dict(
                url="./tests/data/stac/s2_sample_dimension_policy_case_2.json",
                spatial_extent=dict(west=11, east=12, south=46, north=47),
                temporal_extent=["2019-01-01", "2019-06-15"],
                bands=["B04"],
                properties={"eo:cloud_cover": dict(lt=50)},
            ),
            ("t", "bands", "y", "x"),
        ),
        (
            "case3_eurac_local_item",
            dict(
                url="./tests/data/stac/s2_sample_dimension_policy_case_3.json",
                bands=["B04"],
            ),
            ("time", "band", "y", "x"),
        ),
    ],
)
def test_load_stac_dimension_names_and_order(case_name, kwargs, expected_dims):
    cube = load_stac(**kwargs)

    dims = _cube_dims_in_order(cube)
    assert dims == tuple(
        expected_dims
    ), f"{case_name}: expected dims {expected_dims}, got {dims}"
