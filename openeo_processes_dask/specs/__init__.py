import builtins
import json
import keyword
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

json_path = Path(__file__).parent / "openeo-processes"
custom_json_path = Path(__file__).parent / "custom-processes"
process_json_paths = [pg_path for pg_path in (json_path).glob("*.json")]
# Proposals and missing-processes hold specs for implementations we export too.
# Without them those processes have no spec, so the backend registry skips them
# and they are silently unavailable.
for sub in ("proposals", "missing-processes"):
    sub_path = json_path / sub
    if sub_path.exists():
        process_json_paths.extend(sub_path.glob("*.json"))
if custom_json_path.exists():
    process_json_paths.extend(custom_json_path.glob("*.json"))

# Go through all the jsons in the top-level of the specs folder and add them to be importable from here
# E.g. from openeo_processes_dask.specs import *
# This is frowned upon in most python code, but I think here it's fine and allows a nice way of importing these

__all__ = []

for spec_path in process_json_paths:
    # Upstream ships at least one empty spec file
    # (missing-processes/export_collection.json). One malformed spec must not
    # take down every import of this package.
    try:
        with open(spec_path) as spec_file:
            spec_json = json.load(spec_file)
        process_name = spec_json["id"]
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        logger.warning("Skipping unreadable process spec %s: %s", spec_path, exc)
        continue

    # Make sure we don't overwrite any builtins
    if spec_json["id"] in dir(builtins) or keyword.iskeyword(spec_json["id"]):
        process_name = "_" + spec_json["id"]

    locals()[process_name] = spec_json
    __all__.append(process_name)
