"""Exact body-area checks for candidate execution prefixes, before selection."""
import math
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def make_corridor(polygons):
    corridor = unary_union([Polygon(p) for p in polygons])
    if corridor.is_empty or not corridor.is_valid:
        raise ValueError("Candidate gate requires valid native corridor polygons")
    return corridor


def body_box(position, yaw, extent):
    length, width = extent[:2]
    u = np.array([np.cos(yaw), np.sin(yaw)]) * length / 2
    v = np.array([-np.sin(yaw), np.cos(yaw)]) * width / 2
    return Polygon([position + u + v, position + u - v,
                    position - u - v, position - u + v])


def prefix_coverages(positions, yaws, transform, extent, corridor, prefix_frames=5):
    """Return one min area ratio per candidate; no clipping or trajectory edits.

    Uses the actual current pose, then the next five executed poses. Linear
    interpolation has <=0.025 m / <=0.01 rad increments. An excessive jump or
    nonfinite pose is ineligible, never silently skipped.
    """
    positions, yaws = np.asarray(positions, float), np.asarray(yaws, float)
    transform, extent = np.asarray(transform, float), np.asarray(extent, float)
    if positions.ndim != 3 or positions.shape[-1] != 2 or yaws.shape != positions.shape[:2]:
        raise ValueError("Expected candidate positions [K,T,2] and yaws [K,T]")
    if min(extent[:2]) <= 0 or not np.isfinite(transform).all():
        raise ValueError("Invalid extent or world transform")
    count = min(prefix_frames, positions.shape[1])
    angle = np.arctan2(transform[1, 0], transform[0, 0])
    world = positions[:, :count] @ transform[:2, :2].T + transform[:2, 2]
    heading = yaws[:, :count] + angle
    result = []
    for future, angles in zip(world, heading):
        if not np.isfinite(future).all() or not np.isfinite(angles).all():
            result.append(0.)
            continue
        xy = np.concatenate([transform[None, :2, 2], future])
        yy = np.concatenate([[angle], angles])
        minimum = 1.
        for i in range(count):
            delta = xy[i + 1] - xy[i]
            turn = (yy[i + 1] - yy[i] + np.pi) % (2 * np.pi) - np.pi
            substeps = max(1, math.ceil(np.linalg.norm(delta) / .025),
                           math.ceil(abs(turn) / .01))
            if substeps > 400:
                minimum = 0.
                break
            for alpha in np.linspace(0, 1, substeps + 1):
                body = body_box(xy[i] + alpha * delta, yy[i] + alpha * turn, extent)
                minimum = min(minimum, body.intersection(corridor).area / body.area)
        result.append(minimum)
    return np.asarray(result)
