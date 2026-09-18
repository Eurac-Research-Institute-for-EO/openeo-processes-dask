"""Every exported process implementation must have a matching spec.

The backend executor registers processes by walking the public functions of
`openeo_processes_dask.process_implementations` and looking up a spec of the
same name in `openeo_processes_dask.specs`. Any *utility* that leaks into the
implementations namespace via a star-import therefore looks like a process
with no spec, and takes executor startup down with

    AttributeError: module 'openeo_processes_dask.specs' has no attribute
    'assign_dimension_labels'

That outage came from `udf/dimension_helper.py`, whose helpers were re-exported
by `udf/__init__.py` (`from .dimension_helper import *`) with no `__all__` to
stop them.

This test asserts the invariant itself rather than the one module that broke
it, so a leak from any future helper module is caught here instead of in
production.
"""

import inspect

import pytest

import openeo_processes_dask.process_implementations as impls
import openeo_processes_dask.specs as specs


def _exported_functions():
    return [
        (name, obj)
        for name, obj in inspect.getmembers(impls, inspect.isfunction)
        if not name.startswith("_")
    ]


def test_some_processes_are_exported():
    """Guard the guard: if this is empty the test below passes vacuously."""
    assert len(_exported_functions()) > 0


@pytest.mark.parametrize("name", [n for n, _ in _exported_functions()])
def test_exported_implementation_has_a_spec(name):
    assert hasattr(specs, name), (
        f"{name!r} is exported from process_implementations but has no spec in "
        "openeo_processes_dask.specs. If it is a utility rather than an openEO "
        "process, keep it out of the namespace (set __all__ = [] in its module)."
    )


def test_dimension_helpers_are_not_exported():
    """Regression: the specific helpers that caused the production outage."""
    leaked = [
        n
        for n in (
            "fix_udf_dimensions",
            "assign_dimension_labels",
            "restore_semantic_dimensions",
            "fix_dimensions",
        )
        if hasattr(impls, n)
    ]
    assert not leaked, (
        f"UDF dimension helpers leaked into the process namespace: {leaked}. "
        "They are utilities, not processes -- dimension_helper.py needs __all__ = []."
    )
