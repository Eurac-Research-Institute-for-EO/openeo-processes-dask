# Known Issues

This document tracks confirmed current issues and technical debt in the codebase. It was audited against the current repository state on 2026-06-10 with Python 3.12.

The main historical migration from `xr.DataArray` raster cubes with a `bands` dimension to `xr.Dataset` raster cubes with bands as data variables is largely complete at the public raster-cube process boundary. `RasterCube` is defined as `xr.Dataset` in `data_model.py`, and the main cube processes now reject `xr.DataArray` raster-cube inputs through `ensure_raster_cube`.

## 1. Virtual-Band Bridge Constraints

Several processes still need to temporarily convert Dataset variables into a synthetic `bands` dimension:

- `apply_dimension(..., dimension="bands")`
- `reduce_dimension(..., dimension="bands")`
- `fit_curve` / `predict_curve`
- `run_udf`

The bridge is implemented in `cubes/dataset_bridge.py` via:

```python
array, meta = dataset_to_virtual_bands(dataset, dim="bands")
result = virtual_bands_to_dataset(array, meta, dim="bands")
```

This is intentional, but it has limits:

- Heterogeneous band dtypes are coerced by `Dataset.to_array`; the test suite documents this in `tests/test_dataset_bridge.py::TestVirtualBandsToDataset::test_heterogeneous_dtype_round_trip`.
- Multi-resolution or differently gridded variables must be representable as one aligned virtual array before the reducer/UDF/curve process runs.
- Per-variable attrs, dataset attrs, variable order, and CRS depend on explicit capture/restore logic.

Current coverage exists in `tests/test_dataset_bridge.py` and `tests/test_multiresolution.py`, so the older claim that this bridge has no dedicated tests is stale. The remaining issue is architectural: bridge users should stay local and explicit, and new virtual-band processes should add tests for dtype coercion, metadata round-trips, dask laziness, and multi-resolution behavior.

## 2. Fragile Band Permutation Detection

`detect_band_permutation` in `cubes/dataset_bridge.py` tries to infer whether an `apply_dimension(..., dimension="bands")` callback reordered bands. It samples the first element along every non-band dimension and matches values with `np.allclose`.

Risk: if the sampled position has identical values for multiple bands, for example all zeros or all NaN at the first `t/y/x` location, the matching can be ambiguous. The function then returns the first valid matching order it can infer, or `None`, and `apply_dimension` falls back to the original label order.

This can mislabel bands after a callback permutation when data values are not sufficiently distinctive at the sampled point.

Recommendation: prefer explicit label propagation from callback outputs where possible. If inference must remain, sample more than one position or reject ambiguous matches instead of silently retaining the original order.

## 3. Dataset `merge_cubes` Still Uses DataArray Conflict Logic

Public `merge_cubes` no longer accepts `xr.DataArray` raster-cube inputs. However, the Dataset implementation still resolves same-named variable conflicts by delegating each variable pair to `merge_dataarray_cubes`:

```python
result_vars[var] = merge_dataset_variable_conflict(
    var,
    cube1[var],
    cube2[var],
    overlap_resolver=overlap_resolver,
    context=context,
)
```

This is not a public legacy-raster-cube fallback, but it does mean Dataset conflict behavior inherits the DataArray merge implementation for dimension alignment, overlap resolution, broadcasting, and chunking.

Multi-resolution impact: non-conflicting variables are preserved at the Dataset boundary, but conflicts between same-named variables still use per-variable DataArray semantics. This remains a high-priority area for review in multi-resolution workflows.

## 4. Per-Variable Loop Overhead

Processes that iterate over Dataset variables, such as `apply_kernel` and `mask`, pay Python and task-graph overhead proportional to the number of bands. Each variable can create a separate `apply_ufunc`, `where`, or related dask graph segment.

This is a correctness-preserving trade-off of the Dataset model, but it can be slower or produce larger graphs than the old single 4D DataArray representation for many-band cubes.

Current coverage includes graph-size checks in `tests/test_graph_size.py`; the remaining work is performance characterization and thresholds for representative many-band workloads.

## 5. `predict_random_forest` Ordering Depends on Model Metadata

The stale issue claimed that `predict_random_forest` silently fell back to Dataset variable order when Dataset variables did not exactly match `model.feature_names`. That has been fixed: the current implementation raises on missing or extra variables and orders Dataset variables according to `model.feature_names`.

The remaining risk is limited to models without `feature_names`. In that case the implementation uses `context["feature_order"]` if provided; otherwise it warns and falls back to Dataset variable order:

```python
Model has no feature_names and context['feature_order'] is not set.
Using Dataset variable order.
```

Recommendation: callers should provide `context={"feature_order": [...]}` for models without feature metadata, and tests should keep covering the warning path.

## 6. Test Environment Notes

In the Python 3.12 micromamba environment used for this audit:

- Installed package metadata with `pip install -e . --no-deps`; a full extras install tried to build `gdal==3.13.1` against system `libgdal 3.8.4` and failed.
- Reinstalled the pinned Zarr dependency from `pyproject.toml`: `zarr 2.18.7`.
- `tests/test_dataset_bridge.py tests/test_multiresolution.py -q`: passed, 48 tests.
- `tests/test_rastercube_boundary.py tests/test_mask.py tests/test_filter.py tests/test_merge.py -q`: passed, 49 tests.
- `tests/test_ml.py -q`: passed, 10 tests.
- `tests/test_load_stac.py::test_load_stac -q`: passed, 1 test.
- Full `pytest -q`: passed, 418 tests and 3 skipped.

The older documented xgboost/dask test-isolation failure did not reproduce as of this audit.

## Resolved or Stale Historical Notes

The following older claims are no longer accurate:

- The main cube processes still have broad untested legacy `xr.DataArray` raster-cube fallback paths. Current boundary tests in `tests/test_rastercube_boundary.py` verify rejection for many cube processes.
- `ensure_raster_cube` is unused dead code. It is now called by many cube processes, including filtering, masking, applying, reducing, indexing, geometry-related, and general dimension processes.
- The Dataset bridge lacks dedicated tests. `tests/test_dataset_bridge.py` now covers metadata capture/restore, CRS, dask laziness, dtype coercion behavior, round-tripping, and band permutation detection.
- The full `tests/test_ml.py` module is currently unstable in Python 3.12. It passed during this audit.
- The full-suite failure seen during the first audit run was caused by an environment mismatch (`zarr 3.2.1`). It passed after reinstalling `zarr 2.18.7`, which is the version range pinned by `pyproject.toml`.
