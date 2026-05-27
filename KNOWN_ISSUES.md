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

Any cross-band operation (`reduce_dimension` on bands, `apply_dimension` on bands, `ndvi`, `fit_curve`) must convert via:

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

---

## 6. Per-Variable Loop Overhead

Processes that iterate over data variables (`apply_kernel`, `mask`) pay Python overhead proportional to the number of bands. Each iteration creates a separate `apply_ufunc` (or `where`) call with its own task graph. The DataArray approach processed the full 4D cube in a single operation.

This is a correctness-preserving trade-off but has measurable performance implications for cubes with many bands.
