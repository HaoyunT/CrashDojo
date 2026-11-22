"""Transport only the observed suffix of a left-padded ego history.

The native tracker consumes history[-1] and history[-2:]. The planner and
generator observations are untouched. Finite requests pass through unchanged;
interior gaps and missing recent states remain errors.
"""
import json
from pathlib import Path

import numpy as np


def prepare_request(request):
    if not isinstance(request, dict) or not {
        "positions", "yaws", "history", "extent", "step_index"
    }.issubset(request):
        return request, None
    history = np.asarray(request["history"], dtype=float)
    if history.ndim != 2 or history.shape[1] != 8 or len(history) < 2:
        raise ValueError("Native tracker requires at least two [T,8] history rows")
    finite_rows = np.isfinite(history).all(axis=1)
    if finite_rows.all():
        return request, None
    if not finite_rows[-2:].all():
        raise ValueError("Native tracker recent two history rows are nonfinite")
    first = int(np.flatnonzero(finite_rows)[0])
    if not finite_rows[first:].all():
        raise ValueError("Native tracker history contains an interior nonfinite gap")
    trimmed = dict(request)
    trimmed["history"] = history[first:].tolist()
    audit = {
        "step_index": int(request["step_index"]),
        "original_rows": len(history),
        "dropped_leading_rows": first,
        "transported_rows": len(history) - first,
        "last_two_rows_unchanged": bool(np.array_equal(
            np.asarray(trimmed["history"])[-2:], history[-2:])),
        "planner_and_generator_observations_modified": False,
        "protocol": "native_tracker_observed_history_transport_v1",
    }
    return trimmed, audit


class _TrackerJSON:
    def __init__(self, output):
        self.output = Path(output)

    def dumps(self, obj, *args, **kwargs):
        prepared, audit = prepare_request(obj)
        if audit is not None:
            with (self.output / "native_tracker_history_transport.jsonl").open("a") as f:
                f.write(json.dumps(audit, allow_nan=False) + "\n")
        return json.dumps(prepared, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(json, name)


def install(output):
    import ccdiff.evaluation.native_tracker_client as tracker_client
    tracker_client.json = _TrackerJSON(output)
