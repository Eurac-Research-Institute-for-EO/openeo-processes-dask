# Known Issues

This document tracks known issues and technical debt in the codebase, primarily related to the data model migration from `xr.DataArray` (bands as a dimension axis) to `xr.Dataset` (bands as named data variables).

## 1. Incomplete Migration — Untested Legacy DataArray Paths

`RasterCube` is defined as `xr.Dataset` in `data_model.py`, but the codebase still contains **~23 `isinstance(xr.Dataset)` guards** with corresponding legacy `xr.DataArray` fallback paths.

Affected files:

| File | Issue |
|---|---|
| `cubes/mask.py` | Full ~100-line DataArray path alongside `_mask_dataset` |
| `cubes/_filter.py` | `filter_bands` and `filter_labels` have `isinstance` guards with `.sel(band_dim=...)` fallback |
| `cubes/apply.py` | `apply_dimension` band path uses `isinstance` guard |
| `cubes/reduce.py` | `reduce_dimension` band path uses `isinstance` guard |
| `cubes/merge.py` | Dataset path delegates per-variable back to DataArray path recursively |
| `cubes/general.py` | `dimension_labels`, `rename_labels`, `trim_cube` have band-specific guards |
| `cubes/indices.py` | `ndvi` has `isinstance` guard |
| `cubes/geometries.py` | Lines 126, 157 have `isinstance(xr.DataArray)` checks |
| `arrays.py` | Lines 53, 415 have `isinstance(xr.DataArray)` checks |

**Impact**: The test infrastructure defaults to `as_dataset=True`, so these DataArray paths receive **no test coverage**. They are untested dead code.

**Recommendation**: Strip all `isinstance(data, xr.DataArray)` fallback paths. If `RasterCube` is `xr.Dataset`, enforce it uniformly.

---

## 2. `to_array` / `to_dataset` Bridge Risks

Any cross-band operation (`reduce_dimension` on bands, `apply_dimension` on bands, UDF adaptation, `fit_curve`) must convert via:

```python
band_array = data.to_array(dim="bands")    # Dataset → DataArray
# ... operate ...
result = result.to_dataset(dim="bands")    # DataArray → Dataset
```

### 2a. Dtype Coercion

`xr.DataArray` requires all values to share a single dtype. If bands have heterogeneous dtypes (e.g., `int16` reflectance vs `float32` thermal), `to_array` upcasts everything to the most general type, potentially doubling memory.

### 2b. Metadata Round-Trip Fragility

Per-variable attributes and variable order must be manually preserved via `_capture_var_metadata` / `_restore_var_metadata` in `cubes/utils.py`. This is the caller's responsibility and is not enforced. Missing or corrupted attribute round-trips pass silently.

### 2c. No Dedicated Test Coverage

The bridge pattern, metadata round-trip, and dtype coercion behavior lack dedicated test coverage.

### 2d. Multi-Resolution Limits

`xr.Dataset` can hold variables with different dimensionality and coordinate grids, which is one reason for the migration. The bridge pattern partially gives up that advantage: `Dataset.to_array(dim="bands")` must align variables into one array-like shape. For genuinely multi-resolution datasets, this can trigger implicit alignment, broadcasting, missing values, dtype coercion, or graph growth before the process-specific reducer even runs.

**Recommendation**: Keep `to_array` bridges local and explicit, add tests with bands on different spatial grids/resolutions, and document which virtual-band processes require prior harmonization.
---

## 3. Dead Validation: `ensure_raster_cube`

`ensure_raster_cube` in `cubes/utils.py` rejects non-Dataset inputs with a clear error message but is **never called** from any process. It is unused dead code.

**Recommendation**: Either call it as a guard in every process, or remove it.

---

## 4. Fragile Band Permutation Detection

`_detect_band_permutation` in `cubes/utils.py` tries to determine if a callback reordered bands by sampling the first valid element along non-band dimensions and matching via `np.allclose`.

**Risk**: If the first element along non-band dimensions has identical values for multiple bands (e.g., all zeros or all NaN at `t=0, y=0, x=0`), the matching is ambiguous. The function returns `None` in this case, which silently falls back to the original label order — potentially mislabeling all bands after a permutation.

---

## 5. `merge_cubes` — Dataset Path Depends on DataArray Logic

The Dataset path in `merge_cubes` (line 71) delegates per-variable merging back to the **DataArray path recursively**:

```python
result_vars[var] = merge_cubes(cube1[var], cube2[var], overlap_resolver, context)
```

This means all ~230 lines of the DataArray merge logic (dimension alignment, overlap resolution, broadcast) are still exercised even for Dataset inputs. Bugs in the DataArray path affect Dataset users.

**Multi-resolution impact**: `merge_cubes` is one of the most important processes for combining cubes with different spatial or temporal grids. The current Dataset path preserves data variables, attrs, order, and CRS at the Dataset boundary, but same-named variable conflicts still inherit the old DataArray alignment semantics. This should be treated as a high-priority review area for multi-resolution workflows.
---

## 6. Per-Variable Loop Overhead

Processes that iterate over data variables (`apply_kernel`, `mask`) pay Python overhead proportional to the number of bands. Each iteration creates a separate `apply_ufunc` (or `where`) call with its own task graph. The DataArray approach processed the full 4D cube in a single operation.

This is a correctness-preserving trade-off but has measurable performance implications for cubes with many bands.


---

## 7. `predict_random_forest` Feature Ordering Risk

`predict_random_forest` supports Dataset input by stacking data variables into a feature axis:

```python
feature_names = model.feature_names
data_vars = list(data.data_vars)
if set(data_vars) != set(feature_names):
    ordered_vars = data_vars
else:
    ordered_vars = feature_names
```

If the Dataset variable names do not match the model feature names exactly, the implementation silently falls back to the Dataset variable order. This can produce valid-looking predictions with the wrong feature ordering.

**Impact**: The output remains an `xr.Dataset` and can preserve dask laziness, so this error may not surface structurally. It is a semantic correctness risk.

**Recommendation**: Fail fast when model feature names and Dataset variables differ, unless the caller explicitly provides a feature mapping or ordered variable list.

---

## 8. Random-Forest ML Test Isolation Instability

The random-forest tests pass individually in a Python 3.12 environment with local dask networking enabled, but the full `tests/test_ml.py` module is unstable. In the investigation environment:

- `test_fit_regr_random_forest` passed.
- `test_fit_regr_random_forest_inline_geojson` passed when run alone.
- `test_predict_random_forest_dask` passed when run alone.
- Running the full module failed in `test_fit_regr_random_forest_inline_geojson` with an `xgboost.dask` worker-address `KeyError` after an earlier random-forest training test.

This points to dask/xgboost lifecycle or test-isolation problems rather than a direct Dataset migration failure.

**Recommendation**: Use a smaller deterministic dask client fixture for xgboost tests, ensure each test fully closes workers before the next training run, and consider separating expensive xgboost integration tests from Dataset-shape unit tests.
