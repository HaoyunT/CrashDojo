"""Preserve the original interpolation guard and account for rejected episodes."""
import json
from pathlib import Path

import h5py
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from evaluate_selected_rollout import evaluate as original_evaluate, box


def dynamic_diagnostic(xy, yaw, limits):
    xy, yaw = np.asarray(xy, dtype=float), np.asarray(yaw, dtype=float)
    if len(xy) < 4 or not np.isfinite(xy).all() or not np.isfinite(yaw).all():
        return dict(evaluable=False, common_dynamic_envelope_pass=False)
    velocity = np.diff(xy, axis=0) / .1
    accel = np.diff(velocity, axis=0) / .1
    u = np.stack((np.cos(yaw[1:-1]), np.sin(yaw[1:-1])), axis=-1)
    along = (accel * u).sum(-1)
    lateral = (accel * np.stack((-u[:, 1], u[:, 0]), axis=-1)).sum(-1)
    speed = float(np.linalg.norm(velocity, axis=-1).max())
    vector_jerk = float(np.linalg.norm(np.diff(accel, axis=0) / .1, axis=-1).max())
    long_jerk = float(abs(np.diff(along) / .1).max())
    passed = (speed <= limits["max_speed_mps"]
              and along.max() <= limits["max_acceleration_mps2"]
              and along.min() >= -limits["max_braking_mps2"]
              and abs(lateral).max() <= limits["max_lateral_mps2"]
              and max(vector_jerk, long_jerk) <= limits["max_jerk_mps3"])
    return dict(evaluable=True, frames=len(xy), speed_max_mps=speed,
                long_accel_min=float(along.min()), long_accel_max=float(along.max()),
                lat_accel_abs_max=float(abs(lateral).max()),
                vector_jerk_max_mps3=vector_jerk, long_jerk_max_mps3=long_jerk,
                common_dynamic_envelope_pass=bool(passed),
                scope="raw_10hz_finite_differences_no_startup_exclusion_no_smoothing")


def evaluate(directory, request):
    directory = Path(directory)
    r = json.loads(Path(request).read_text())
    bind = json.loads((directory / "selection_binding.json").read_text())
    a, e = bind["runtime_target_index"], bind["runtime_ego_index"]
    with h5py.File(directory / "data.hdf5") as f:
        if len(f) != 1:
            raise ValueError("Expected one selected scene")
        group = f[next(iter(f))]
        xy = group["centroid"][:]
        yaw = group["yaw"][:]
        extent = group["extent"][:]
    # Match the existing evaluator's float32 calculation and thresholds exactly.
    finite = np.isfinite(xy).all() and np.isfinite(yaw).all()
    violations = []
    if finite:
        delta = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
        angle = (np.diff(yaw, axis=1) + np.pi) % (2 * np.pi) - np.pi
        for k in range(xy.shape[1] - 1):
            n = max(1, int(np.ceil(float(delta[:, k].max()) / .05)),
                    int(np.ceil(float(abs(angle[:, k]).max()) / .02)))
            if n > 200:
                i = int(delta[:, k].argmax())
                violations.append(dict(frame=k, required_substeps=n, actor_index=i,
                                       role="E" if i == e else "A" if i == a else "background",
                                       displacement_m=float(delta[i, k])))
    if finite and not violations:
        result = original_evaluate(directory, request)
        result.update(geometry_evaluable=True, geometry_rejection_reason=None)
    else:
        corridor = unary_union([Polygon(p) for p in r["corridor_world_polygons"]])
        coverage = None
        if np.isfinite(xy[a]).all() and np.isfinite(yaw[a]).all():
            body = [box(x, y, sz) for x, y, sz in zip(xy[a], yaw[a], extent[a])]
            coverage = min(b.intersection(corridor).area / max(b.area, 1e-8) for b in body)
        result = dict(
            scope="rejected_interpolation_with_sampled_A_coverage_diagnostic_only",
            geometry_evaluable=False,
            geometry_rejection_reason="unreasonable_between_frame_motion" if finite else "nonfinite_trajectory",
            interpolation_guard_violations=violations, frames=int(xy.shape[1]),
            target_contact=None, ego_any_contact=None, scene_any_contact=None,
            first_recorded_target_overlap=None, first_target_contact_s=None,
            first_non_target_contact_s=None, min_target_body_distance_m=None,
            target_route_body_coverage_min=coverage,
            coverage_scope="recorded_frames_only_not_interpolated",
            formal_validated=False,
            caveat="Unknown contact is not a negative contact result. Keep this episode in the planned denominator.",
        )
    (directory / "executed_geometry_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result

# Versioned availability-aware evaluator; dynamics thresholds are unchanged.
from available_geometry import evaluate
