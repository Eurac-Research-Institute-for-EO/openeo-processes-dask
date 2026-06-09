# Dataset Refactor Review

**Verdict: PASS_WITH_DEVIATIONS**

**Date**: 2026-05-11
**Branch**: `mod_datamodel_review`
**Base**: `main` (merge-base `0a2f4fe`)
**Spec source**: openeo-processes submodule at `2024.10.0-3-gfcc9e54`

## Summary

- 39 files changed (+1141/-711)
- 301 tests pass, 0 fail, 476 pre-existing deprecation warnings
- 1 bug fixed during review: `apply.py:136` `==` → `=` (dimension labels not restored)

## Component Matrix

| Component | Files | Status | Notes |
|---|---|---|---|
| Data model helpers | `data_model.py` | PASS | variables-as-bands (Convention A), central helpers |
| Dataset accessor | `_xr_interop.py` | PASS | `band_dims` virtual, `band_names` from data_vars |
| Dimension helpers | `general.py` | PASS | virtual bands handled in labels/rename/drop/add |
| `apply` | `apply.py` | PASS | Dataset-native xr.apply_ufunc; bug fixed |
| `apply_dimension` | `apply.py` | PASS | per-variable iteration + stack/unstack for bands |
| `apply_kernel` | `apply.py` | PASS | per-variable convolution |
| `reduce_dimension` | `reduce.py` | PASS | Dataset-native reduce + stack/unstack for bands |
| `filter_bands` | `_filter.py` | PASS | delegates to `select_bands()` |
| `filter_labels` | `_filter.py` | PASS | virtual bands path via data_vars |
| `filter_temporal` | `_filter.py` | PASS | Dataset-native slicing |
| `filter_bbox` | `_filter.py` | PASS | Dataset-native slicing |
| `mask` | `mask.py` | PASS | 3 modes: single-var, matching-vars, ambiguous error |
| `mask_polygon` | `mask_polygon.py` | PASS_WITH_DEVIATION | removed .data/.shape; dim order heuristic |
| `merge_cubes` | `merge.py` | PASS | Dataset-native concat/merge; stack/unstack resolvers |
| `ndvi` | `indices.py` | PASS | Dataset variable names + common_name fallback |
| `trim_cube` | `general.py` | PASS | per-variable notnull OR logic |
| `ndvi` | `indices.py` | PASS | Dataset var names + target_band |
| `fit_curve` | `ml/curve_fitting.py` | PASS | per-variable curvefit; rejects virtual bands |
| `predict_random_forest` | `ml/random_forest.py` | PASS | stack_bands; feature count validation |
| `run_udf` | `udf/udf.py` | PASS | stack/unstack boundary conversion |
| `load_stac` | `load.py` | PASS | returns Dataset natively |
| Fixtures | `mockdata.py` | PASS | returns Dataset with per-band data_vars |
| Test helpers | `general_checks.py` | PASS | per-variable comparison for Dataset |

## Test Execution

```bash
python -m compileall openeo_processes_dask     # OK
python -m pytest --collect-only -q              # 301 tests
python -m pytest -q                             # 301 passed, 0 failed
python -m pytest -q tests/test_apply.py         # 13 passed
```

## Dataset Convention

**Convention A (variables-as-bands)** — `xr.Dataset` with band names as `data_vars`:

```python
xr.Dataset(
    data_vars={"B02": (("t", "y", "x"), ...), "B03": (("t", "y", "x"), ...)},
    coords={"t": ..., "y": ..., "x": ...},
)
```

- `bands` is a **virtual** dimension: `band_dims` returns `("bands",)` but no physical `bands` dim exists.
- `band_names` returns `list(data.data_vars)`.
- `is_raster_cube()` detects spatial dims `x`/`y` in Dataset dims.

## Deviations

| Deviation | Severity | Compatibility | Required Action |
|---|---|---|---|
| Variables-as-bands convention | S1-Documentation | Compatible | Add standalone dataset-model doc |
| Virtual `bands` dimension | S1-Documentation | Compatible | Already in migration plan |
| Scattered `isinstance(data, xr.Dataset)` checks | S2-Test gap | Acceptable during migration | Consolidate before upstream PR |
| DataArray no longer primary input | S1-Documentation | Breaking change | Document in README |
| `_unstack_bands` coord-dropping heuristic | S3-Semantic risk | Risky for aux coords | Document or improve heuristic |
| `mask_polygon` spatial dim order assumption | S2-Test gap | Fragile | Add explicit test |

## Missing Test Coverage (S2)

- No dedicated Dataset Dask laziness tests
- No auxiliary coordinate regression tests
- No inconsistent-variables Dataset tests
- No DataArray ↔ Dataset equivalence tests
- No multi-variable operation edge-case tests

## Bug Fix Applied

**`apply.py:136`**: `==` (no-op comparison) changed to `=` (assignment). Dimension coordinate labels were never restored in the `apply_dimension` path when `dimension == target_dimension` and sizes matched.

## Merge Readiness

The refactor is functionally complete and the test suite passes. Before upstream PR, address:
1. **Must-fix**: None remaining (bug already fixed).
2. **Should-fix**: Add dedicated Dataset test files for Dask laziness, auxiliary coords, inconsistent variables, and DataArray equivalence.
3. **Nice-to-have**: Create `docs/dataset_datacube_model.md` documenting the dataset convention.
4. **Nice-to-have**: Consolidate scattered `isinstance(data, xr.Dataset)` checks into adapters.
