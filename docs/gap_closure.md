# Gap Closure: `openeo-processes-dask` ↔ `openeo-processes-dask-slim`

## Context

`openeo-processes-dask-slim` was created by stripping GDAL/rasterio/rioxarray dependencies and niche features from `openeo-processes-dask`. Later, the slim repo's `dev_remodel` branch performed a Dataset migration (xr.DataArray → xr.Dataset). PR #17 inherited that migration work into `openeo-processes-dask`.

This document describes how the removed processes were re-added with a **Dataset-first contract** (xr.Dataset, `ensure_raster_cube` guards, odc CRS).

## Removed Processes — Status

| Process | Was Removed From | Now Lives In | Approach |
|---------|-----------------|-------------|----------|
| `filter_spatial` | `cubes/_filter.py` | `cubes/_filter.py` | Re-added with `ensure_raster_cube`, delegates to `filter_bbox` + `mask_polygon` |
| `apply_polygon` | `cubes/apply.py` | `cubes/apply.py` | Re-added with `ensure_raster_cube`, calls `mask_polygon` + `apply` |
| `aggregate_spatial` | `cubes/aggregate.py` | `cubes/aggregate.py` | Re-added with `ensure_raster_cube`, uses `odc.crs` instead of `rio.crs`, calls `xvec.zonal_stats` |
| `load_stac` / `load_url` | `cubes/load.py` | `cubes/load.py` | Refactored: returns `xr.Dataset`, uses `odc.geo.xr.assign_crs` |
| `mask_polygon` | `cubes/mask_polygon.py` | `cubes/mask_polygon.py` | Refactored: `odc.crs` + `odc.geobox.transform`, per-variable Dataset masking |

## Migration Pattern

Every re-added or refactored process follows these rules:

1. **`ensure_raster_cube` guard** — The first executable line calls `ensure_raster_cube(data, ...)`, rejecting `DataArray` inputs with a clear error.
2. **Per-variable operations** — Dataset operations iterate over `data.data_vars` instead of assuming a `bands` dimension, e.g.:
   ```python
   result_vars = {}
   for var_name in data.data_vars:
       result_vars[var_name] = data[var_name].where(...)
   return xr.Dataset(result_vars, coords=data.coords, attrs=data.attrs)
   ```
3. **odc CRS** — All `.rio.crs` / `.rio.write_crs()` replaced with `data.odc.crs` / `odc.geo.xr.assign_crs()`. Transform objects come from `data.odc.geobox.transform`.
4. **`__all__` registration** — Each function is listed in its module's `__all__`, auto-exported via `cubes/__init__.py`'s `from .module import *`.

## Key Files Changed

| File | Changes |
|------|---------|
| `openeo_processes_dask/process_implementations/cubes/_filter.py` | Added `filter_spatial` (+ import of `mask_polygon`) |
| `openeo_processes_dask/process_implementations/cubes/apply.py` | Added `apply_polygon` (+ imports of `shape`, `unary_union`, `MultiPolygon`) |
| `openeo_processes_dask/process_implementations/cubes/aggregate.py` | Added `aggregate_spatial` (+ `ensure_raster_cube`, `odc.crs`, `urlopen`/`json`) |
| `openeo_processes_dask/process_implementations/cubes/mask_polygon.py` | Replaced `rio.crs` / `rio.transform` with `odc.crs` / `odc.geobox.transform`; added `ensure_raster_cube`; per-variable Dataset where; dropped `rioxarray` import |
| `openeo_processes_dask/process_implementations/cubes/load.py` | `load_stac` returns `xr.Dataset`; uses `odc.geo.xr.assign_crs`; removed `.to_dataarray(dim="bands")` |
| `tests/test_load_stac.py` | Updated assertions for Dataset-first (check `data_vars` instead of `band_dims`) |
| `tests/test_reduce.py` | Skip `test_reduce_rqa` when `rqadeforestation` not installed |
| `.gitmodules` | Submodule URL switched to HTTPS; submodule initialized |

## Remaining Optional Dependencies

The following dependencies are kept optional in `pyproject.toml` but required by specific re-added processes:

- **rasterio** — Required by `mask_polygon` (`rasterio.features.geometry_mask`) and `aggregate_spatial` (transitive via xvec)
- **rioxarray** — No longer imported in any process; can be removed from extras if confirmed unused elsewhere
- **rqadeforestation** — Optional, guarded via `try/except ImportError`; test skips when absent

## Verification

```bash
# Create environment with all extras
micromamba create -n test-ci python=3.12
micromamba run -n test-ci pip install poetry
micromamba install -n test-ci -c conda-forge gdal
micromamba run -n test-ci poetry install --extras implementations --extras ml
micromamba run -n test-ci poetry run pytest -q

# Expected: 417 passed, 4 skipped, 0 failed
```
