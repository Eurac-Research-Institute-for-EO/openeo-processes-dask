# RasterCube: `xr.Dataset` as the native raster type

Date: 2026-05-23

Status: accepted and implemented on `dev-remodel`

## Decision

`RasterCube` is `xr.Dataset`.

Bands are represented as Dataset data variables, not as a physical xarray
dimension:

```text
Dataset
  B02: (t, y, x)
  B03: (t, y, x)
  B04: (t, y, x)
  coords: {t, y, x}
  attrs: {crs, ...}
```

The logical openEO cube still has a virtual `(t, bands, y, x)` shape. In this
repository, `bands` is interpreted as `list(dataset.data_vars)`.

## Current Contract

- Public RasterCube process inputs and outputs are `xr.Dataset`.
- Public raster process boundaries use `ensure_raster_cube` where appropriate
  to reject RasterCube-shaped `xr.DataArray` inputs with a clear error.
- Dataset data variables are the user-visible band labels.
- Real dimensions such as `t`, `y`, and `x` use Dataset-aware xarray APIs or
  explicit per-variable iteration.
- Virtual band operations may temporarily bridge through `xr.DataArray` via
  `cubes/dataset_bridge.py`.
- `load_stac` returns an `xr.Dataset` RasterCube.
- VectorCube, scalar, array, and UDF adapter internals may still use
  `xr.DataArray` where that is their actual API contract. That is not public
  RasterCube fallback support.

## Bounded DataArray Bridges

Some operations structurally need a band axis or an external DataArray-shaped
API. These paths must stay local and explicit:

- `apply_dimension(..., dimension="bands")`
- `reduce_dimension(..., dimension="bands")`
- `fit_curve` / `predict_curve`
- `run_udf`

Bridge helpers live in `openeo_processes_dask/process_implementations/cubes/dataset_bridge.py`:

- `capture_dataset_metadata`
- `restore_dataset_metadata`
- `dataset_to_virtual_bands`
- `virtual_bands_to_dataset`
- `detect_band_permutation`

These helpers preserve variable order, per-variable attributes, Dataset
attributes, CRS where possible, and dask laziness. `Dataset.to_array` still has
real xarray semantics: heterogeneous dtypes can be coerced, and variables must
be alignable as one virtual array.

## Remaining Deviations and Risks

- Virtual-band bridges can coerce dtype and align/broadcast variables before
  process-specific logic runs.
- `detect_band_permutation` infers band reordering from sampled data values and
  can be ambiguous when sampled values are identical across bands.
- `merge_cubes` has a Dataset boundary path, but same-name variable conflicts
  still delegate to per-variable `DataArray` merge logic.
- Per-variable Dataset operations can create task graph overhead proportional
  to band count.
- `predict_random_forest` is safe when `model.feature_names` is present. If a
  model has no feature metadata, callers should provide
  `context={"feature_order": [...]}`; otherwise Dataset variable order is used
  with a warning.

## Verification

Current audit environment:

- Python 3.12
- `zarr 2.18.7`, matching `pyproject.toml`
- editable install with `pip install -e . --no-deps`

Relevant verification:

```bash
python -m pytest tests/test_dataset_bridge.py tests/test_multiresolution.py -q
python -m pytest tests/test_rastercube_boundary.py tests/test_mask.py tests/test_filter.py tests/test_merge.py -q
python -m pytest tests/test_ml.py -q
python -m pytest tests/test_load_stac.py::test_load_stac -q
python -m pytest -q
```

Latest full-suite result from the audit:

```text
418 passed, 3 skipped
```

## Related Documents

- [RasterCube Dataset migration history](rastercube-dataset-migration-history.md)
- [Scalability notes](../scalability/README.md)
- [Known issues](../../KNOWN_ISSUES.md)
