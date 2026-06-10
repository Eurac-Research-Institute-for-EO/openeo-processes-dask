# RasterCube Dataset migration history

Date: 2026-06-10

Status: historical implementation record for `dev-remodel`

## Purpose

This document preserves the chronological record of the migration from
`xr.DataArray` RasterCubes to `xr.Dataset` RasterCubes. The current contract is
defined in [Dataset RasterCube migration](dataset-rastercube-migration.md).

Older review, quality-plan, process-inventory, and gap-closure notes were
consolidated here to avoid stale duplicate descriptions of the current state.

## Background

The repository historically represented openEO raster cubes as
`xr.DataArray` with a physical `bands` dimension. That model made it difficult
to preserve per-band metadata, CRS, nodata policy, and multi-resolution
behavior.

The remodel work adopted `xr.Dataset` as the native RasterCube carrier:

- bands are Dataset data variables;
- `bands` is a virtual openEO dimension;
- public RasterCube inputs are Dataset-only;
- bounded DataArray bridges remain only where a band axis or external API
  requires them.

## Chronological Record

| Commit / phase | Purpose |
|---|---|
| `2fc230a` | Modernization baseline: dependency ranges, Python metadata, Poetry/GitHub Actions refresh, NumPy 2 replacements, XGBoost Dask import update. |
| `93e72e6` | Pydantic `parse_obj` to `model_validate`, and replacement of private `np.core` exception handling. |
| `d880d90` | Merge of the modernization checkpoint into `dev-remodel`. |
| `30ad4c5` | Minimal Dataset migration: introduced `ensure_raster_cube`, Dataset-aware `apply_dimension` and `reduce_dimension`, Dataset-aware test comparisons. |
| `b6608dd`, `56f9feb`, `47481a0` | Process coverage for dimension helpers, filters, NDVI, temporal aggregation, mask, and kernel handling. |
| `d1172dc`, `022a912` | Merge, mask, and random-forest Dataset coverage. |
| `b5ed758`, `2f4ec02`, `9c90626`, `4711606` | Final enforcement pass: `RasterCube = xr.Dataset`, DataArray rejection at public boundaries, Dataset fixture default. |
| `fd075db` | Dataset `dims` compatibility fix for `apply_neighborhood_intertwin`. |
| `3e3ad49` through `11612fd` | Review-fix cycle: Dataset merge boundary path, removed eager `.compute()` in prediction, central bridge metadata preservation, documented `array_find` return type. |
| `c102554` | Gap closure after restoring processes from the full repository into the Dataset-first branch. |
| `9eb770d` | Current known-issues audit: removed stale migration claims, verified boundary enforcement, bridge coverage, ML stability, and full suite with pinned Zarr. |

## Gap Closure Summary

The slim remodel branch had removed some dependency-heavy processes. The gap
closure restored them under the Dataset-first contract:

| Process | Current approach |
|---|---|
| `filter_spatial` | Re-added with `ensure_raster_cube`; delegates to `filter_bbox` and `mask_polygon`. |
| `apply_polygon` | Re-added with `ensure_raster_cube`; calls `mask_polygon` and `apply`. |
| `aggregate_spatial` | Re-added with `ensure_raster_cube`; uses `odc.crs` and `xvec.zonal_stats`. |
| `load_stac` / `load_url` | Refactored to return `xr.Dataset`; uses `odc.geo.xr.assign_crs`. |
| `mask_polygon` | Uses `odc.crs` / `odc.geobox.transform`; applies per-variable Dataset masking. |

Optional dependencies remain process-specific. `rioxarray` is still listed in
the `implementations` extra, but current process code no longer imports it
directly.

## Current Process Classification

| Category | Processes |
|---|---|
| Dataset-native public RasterCube paths | `filter_temporal`, `filter_labels`, `filter_bands`, `filter_bbox`, `apply`, `apply_kernel`, `reduce_spatial`, `aggregate_temporal`, `aggregate_temporal_period`, `mask`, `ndvi`, dimension helpers, resampling processes, `mask_polygon`, `filter_spatial`, `apply_polygon`, `aggregate_spatial` |
| Bounded virtual-band bridges | `apply_dimension(..., "bands")`, `reduce_dimension(..., "bands")`, `fit_curve`, `run_udf` |
| Dataset boundary with DataArray conflict internals | `merge_cubes` |
| Non-raster / vector / scalar paths | geometry loaders and vector processes, dates, text, math, array utilities |

## Test Milestones

The migration added or updated tests for:

- Dataset bridge metadata, CRS, dask laziness, dtype coercion, and band
  permutation detection;
- public rejection of RasterCube-shaped `xr.DataArray` inputs;
- multi-resolution Dataset fixtures and current virtual-band behavior;
- per-variable Dataset graph-size behavior;
- random-forest feature-order validation and dask prediction laziness;
- Dataset-first `load_stac`.

Latest audit results:

```text
tests/test_dataset_bridge.py tests/test_multiresolution.py -q
48 passed

tests/test_rastercube_boundary.py tests/test_mask.py tests/test_filter.py tests/test_merge.py -q
49 passed

tests/test_ml.py -q
10 passed

tests/test_load_stac.py::test_load_stac -q
1 passed

pytest -q
418 passed, 3 skipped
```

## Superseded Documents

The following older notes were removed or folded into this history to avoid
diverging descriptions of the migration state:

- `docs/dataset_refactor_review.md`
- `docs/gap_closure.md`
- `docs/decisions/rastercube-dataset-quality-plan.md`
- `docs/decisions/rastercube-process-status.md`

The authoritative current-state document is
`docs/decisions/dataset-rastercube-migration.md`.
