#!/usr/bin/env python3
"""
Static check script for RasterCube Dataset migration.

Reports on patterns that indicate incomplete migration from xr.DataArray
to xr.Dataset as the RasterCube contract.

Phase 0 — informational only, does not fail CI.

Usage:
    python scripts/check_rastercube_migration.py
"""

import ast
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RASTER_DIR = (
    PROJECT_ROOT / "openeo_processes_dask_slim" / "process_implementations" / "cubes"
)
DATA_MODEL = (
    PROJECT_ROOT
    / "openeo_processes_dask_slim"
    / "process_implementations"
    / "data_model.py"
)

findings = []


def check_rastercube_type():
    """Check that RasterCube = xr.Dataset, not Union or DataArray."""
    src = DATA_MODEL.read_text()
    tree = ast.parse(src, filename=str(DATA_MODEL))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RasterCube":
                    if (
                        isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "xr"
                        and node.value.attr == "Dataset"
                    ):
                        return  # OK — RasterCube = xr.Dataset
                    findings.append(
                        {
                            "file": str(DATA_MODEL.relative_to(PROJECT_ROOT)),
                            "line": node.lineno,
                            "pattern": "RasterCube type is not xr.Dataset",
                            "severity": "ERROR",
                        }
                    )
    findings.append(
        {
            "file": str(DATA_MODEL.relative_to(PROJECT_ROOT)),
            "line": 0,
            "pattern": "RasterCube assignment not found",
            "severity": "ERROR",
        }
    )


def check_isinstance_dataarray():
    """Find isinstance(..., xr.DataArray) in raster process files."""
    for pyfile in sorted(RASTER_DIR.rglob("*.py")):
        src = pyfile.read_text()
        tree = ast.parse(src, filename=str(pyfile))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "isinstance") or (
                    isinstance(func, ast.Name) and func.id == "isinstance"
                ):
                    if len(node.args) >= 2:
                        arg2 = node.args[1]
                        if (
                            isinstance(arg2, ast.Attribute)
                            and isinstance(arg2.value, ast.Name)
                            and arg2.value.id == "xr"
                            and arg2.attr == "DataArray"
                        ):
                            findings.append(
                                {
                                    "file": str(pyfile.relative_to(PROJECT_ROOT)),
                                    "line": node.lineno,
                                    "pattern": "isinstance(..., xr.DataArray)",
                                    "severity": "WARN",
                                }
                            )


def check_to_array_bridge():
    """Find .to_array(dim=...) calls in raster process files."""
    for pyfile in sorted(RASTER_DIR.rglob("*.py")):
        src = pyfile.read_text()
        tree = ast.parse(src, filename=str(pyfile))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "to_array":
                    findings.append(
                        {
                            "file": str(pyfile.relative_to(PROJECT_ROOT)),
                            "line": node.lineno,
                            "pattern": "to_array(",
                            "severity": "INFO",
                        }
                    )


def check_direct_data_access():
    """Find direct .data access in raster process files."""
    for pyfile in sorted(RASTER_DIR.rglob("*.py")):
        src = pyfile.read_text()
        tree = ast.parse(src, filename=str(pyfile))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "data":
                parent = getattr(node, "parent", None) is None  # ast doesn't set parent
                findings.append(
                    {
                        "file": str(pyfile.relative_to(PROJECT_ROOT)),
                        "line": node.lineno,
                        "pattern": ".data access",
                        "severity": "INFO",
                    }
                )


def check_dataarray_register():
    """Find xr.register_dataarray_accessor references."""
    for pyfile in sorted(RASTER_DIR.rglob("*.py")):
        src = pyfile.read_text()
        if "register_dataarray_accessor" in src:
            tree = ast.parse(src, filename=str(pyfile))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "register_dataarray_accessor"
                    ):
                        findings.append(
                            {
                                "file": str(pyfile.relative_to(PROJECT_ROOT)),
                                "line": node.lineno,
                                "pattern": "register_dataarray_accessor",
                                "severity": "INFO",
                            }
                        )


def main():
    check_rastercube_type()
    check_isinstance_dataarray()
    check_to_array_bridge()
    check_direct_data_access()
    check_dataarray_register()

    # Remove duplicate .data findings (keep first occurrence per line per file)
    seen = set()
    unique_findings = []
    for f in findings:
        key = (f["file"], f["line"], f["pattern"])
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    # Print report
    print("=" * 72)
    print("  RasterCube Migration Static Check — Phase 0 (Informational)")
    print("=" * 72)

    if not unique_findings:
        print("\n  No issues found.")
        sys.exit(0)

    print()

    for f in sorted(
        unique_findings, key=lambda x: (x["severity"], x["file"], x["line"])
    ):
        print(f"  [{f['severity']:5s}] " f"{f['file']}:{f['line']}  " f"{f['pattern']}")

    info_count = sum(1 for f in unique_findings if f["severity"] == "INFO")
    warn_count = sum(1 for f in unique_findings if f["severity"] == "WARN")
    error_count = sum(1 for f in unique_findings if f["severity"] == "ERROR")

    print(
        f"\n  Summary: {len(unique_findings)} total — "
        f"{error_count} ERROR, {warn_count} WARN, {info_count} INFO"
    )
    print("  Phase 0 is informational only. No exit code failure.\n")

    # Phase 0 does not fail CI
    sys.exit(0)


if __name__ == "__main__":
    main()
