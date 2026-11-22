"""Contact diagnostics on explicit observed lifecycles and continuous valid edges."""

import json
import math
from pathlib import Path

import h5py
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def box(position, yaw, extent):
    u = np.array([np.cos(yaw), np.sin(yaw)]) * extent[0] / 2
    v = np.array([-np.sin(yaw), np.cos(yaw)]) * extent[1] / 2
    return Polygon([position + u + v, position + u - v,
                    position - u - v, position - u + v])


def evaluate_arrays(xy, yaw, extent, available, target, ego, corridor, dt=.1):
    xy, yaw, extent = np.asarray(xy), np.asarray(yaw), np.asarray(extent)
    available = np.asarray(available, dtype=bool)
    shape = xy.shape[:2]
    if available.shape != shape or yaw.shape != shape or extent.shape[:2] != shape:
        raise ValueError('Pose and availability shapes differ')
    result = dict(scope='linear_interpolation_on_available_lifecycle_intervals',
                  frames=shape[1], time_origin='first_recorded_frame',
                  unavailable_actor_frames=int((~available).sum()),
                  lifecycle_policy='observed_native_track_support_no_gap_bridging_or_tail_extension',
                  formal_validated=False, qualified_conflict_rate=None)

    def reject(reason, **details):
        return dict(result, geometry_evaluable=False, geometry_rejection_reason=reason,
                    target_contact=None, ego_any_contact=None, scene_any_contact=None,
                    first_recorded_target_overlap=None, first_target_contact_s=None,
                    first_non_target_contact_s=None, min_target_body_distance_m=None,
                    target_route_body_coverage_min=None, **details)

    if not available[[target, ego]].all():
        return reject('controlled_actor_missing')
    finite = np.isfinite(xy).all(-1) & np.isfinite(yaw) & np.isfinite(extent[..., :2]).all(-1)
    if not finite[available].all() or not (extent[..., :2][available] > 0).all():
        return reject('invalid_available_pose_or_extent')
    # A row that exits may not reappear without an explicit lifecycle model.
    if ((~available[:, :-1]) & available[:, 1:]).any():
        return reject('unsupported_reappearance')
    subdivisions = []
    for k in range(shape[1] - 1):
        active = available[:, k] & available[:, k + 1]
        displacement = np.linalg.norm(xy[active, k + 1] - xy[active, k], axis=-1)
        angle = (yaw[active, k + 1] - yaw[active, k] + np.pi) % (2 * np.pi) - np.pi
        n = max(1, int(math.ceil(float(displacement.max()) / .05)),
                int(math.ceil(float(abs(angle).max()) / .02)))
        if n > 200:
            return reject('unreasonable_between_frame_motion', frame=k, required_substeps=n)
        subdivisions.append(n)
    first_target = first_other = None
    ego_hit = scene_hit = initial_overlap = False
    minimum = math.inf
    coverage = 1.

    def inspect(time_s, position, heading, size, active):
        nonlocal first_target, first_other, ego_hit, scene_hit, initial_overlap, minimum, coverage
        bodies = {i: box(position[i], heading[i], size[i]) for i in np.flatnonzero(active)}
        minimum = min(minimum, bodies[target].distance(bodies[ego]))
        coverage = min(coverage, bodies[target].intersection(corridor).area / bodies[target].area)
        indices = list(bodies)
        for offset, i in enumerate(indices):
            for j in indices[offset + 1:]:
                if not bodies[i].intersects(bodies[j]):
                    continue
                scene_hit = True
                if ego in (i, j):
                    ego_hit = True
                if {i, j} == {target, ego}:
                    if first_target is None:
                        first_target = time_s
                    if time_s == 0:
                        initial_overlap = True
                elif target in (i, j) and first_other is None:
                    first_other = time_s

    for k in range(shape[1]):
        inspect(k * dt, xy[:, k], yaw[:, k], extent[:, k], available[:, k])
        if k < len(subdivisions):
            active = available[:, k] & available[:, k + 1]
            indices = np.flatnonzero(active)
            angle = (yaw[indices, k + 1] - yaw[indices, k] + np.pi) % (2 * np.pi) - np.pi
            for step in range(1, subdivisions[k]):
                alpha = step / subdivisions[k]
                position = np.zeros_like(xy[:, k])
                heading = np.zeros_like(yaw[:, k])
                size = np.zeros_like(extent[:, k])
                position[indices] = (1 - alpha) * xy[indices, k] + alpha * xy[indices, k + 1]
                heading[indices] = yaw[indices, k] + alpha * angle
                size[indices] = (1 - alpha) * extent[indices, k] + alpha * extent[indices, k + 1]
                inspect((k + alpha) * dt, position, heading, size, active)
    return dict(result, geometry_evaluable=True, geometry_rejection_reason=None,
                target_contact=first_target is not None, ego_any_contact=bool(ego_hit),
                scene_any_contact=bool(scene_hit), first_recorded_target_overlap=bool(initial_overlap),
                first_target_contact_s=first_target, first_non_target_contact_s=first_other,
                min_target_body_distance_m=minimum, target_route_body_coverage_min=coverage,
                caveat='Observed lifecycle and linear interpolation diagnostics, not certified continuous physical trajectories')


def evaluate(directory, request):
    directory = Path(directory)
    r = json.loads(Path(request).read_text())
    binding = json.loads((directory / 'selection_binding.json').read_text())
    with h5py.File(directory / 'data.hdf5') as handle:
        group = handle[next(iter(handle))]
        if 'availability' not in group:
            raise ValueError('Explicit availability required for this evaluator')
        result = evaluate_arrays(group['centroid'][:], group['yaw'][:], group['extent'][:],
                                 group['availability'][:], binding['runtime_target_index'],
                                 binding['runtime_ego_index'],
                                 unary_union([Polygon(p) for p in r['corridor_world_polygons']]))
    (directory / 'executed_geometry_diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
