import pytest
from openeo.local import LocalConnection


@pytest.fixture(scope="module")
def local_conn():
    # Use current folder as local backend root (same as your snippets)
    return LocalConnection("./")


@pytest.mark.parametrize(
    "case_name, kwargs, expected_dims",
    [
        (
            "case1_eopf_core",
            dict(
                url="https://stac.core.eopf.eodc.eu/collections/sentinel-2-l2a",
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
            "case2_planetarycomputer",
            dict(
                url="https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a",
                spatial_extent=dict(west=11, east=12, south=46, north=47),
                temporal_extent=["2019-01-01", "2019-06-15"],
                bands=["B04"],
                properties={"eo:cloud_cover": dict(lt=50)},
            ),
            ("t", "bands", "y", "x"),
        ),
        (
            "case3_eurac_sample",
            dict(
                url="https://stac.eurac.edu/collections/SENTINEL2_L2A_SAMPLE_2",
                bands=["B04"],
            ),
            ("time", "band", "y", "x"),
        ),
    ],
)
def test_load_stac_dimension_names_and_order(
    local_conn, case_name, kwargs, expected_dims
):
    cube = local_conn.load_stac(**kwargs)
    arr = cube.execute()

    # xarray DataArray: dims is a tuple in order
    assert tuple(arr.dims) == tuple(
        expected_dims
    ), f"{case_name}: expected dims {expected_dims}, got {arr.dims}"
