"""Reference metric primitives shared by nuScenes and nuPlan adapters."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


DEFAULT_ABSOLUTE_GATES = {
    "max_speed_mps": 20.0,
    "max_abs_acc_mps2": 4.0,
    "max_abs_jerk_mps3": 8.37,
    "max_abs_yaw_rate_rps": 0.95,
}

GATE_TO_SERIES = {
    "max_speed_mps": "speed_mps",
    "max_abs_acc_mps2": "abs_acc_mps2",
    "max_abs_jerk_mps3": "abs_jerk_mps3",
    "max_abs_yaw_rate_rps": "abs_yaw_rate_rps",
}


def _as_track(value: Any, trailing: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != trailing:
        raise ValueError(f"expected rank {trailing}, got shape {array.shape}")
    return array


def _continuous_mask(active: np.ndarray, track_id: Optional[np.ndarray], order: int) -> np.ndarray:
    if order < 2:
        raise ValueError("order must be at least two")
    length = len(active) - order + 1
    if length <= 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(length, dtype=bool)
    for offset in range(order):
        mask &= active[offset : offset + length]
    if track_id is not None:
        for offset in range(order - 1):
            mask &= (
                track_id[offset : offset + length]
                == track_id[offset + 1 : offset + 1 + length]
            )
    return mask


def compute_track_kinematics(
    position: Any,
    yaw: Any,
    active: Any,
    dt: float,
    track_id: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """Compute derivatives only on finite, identity-continuous support.

    Position is ``[T, 2]`` and yaw/active/track_id are ``[T]``.  Vector
    acceleration and vector jerk magnitudes are used.  Yaw differences are
    wrapped to ``[-pi, pi]`` before division by ``dt``.
    """

    position_array = _as_track(position, 2)
    yaw_array = _as_track(yaw, 1)
    active_array = np.asarray(active, dtype=bool)
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    if position_array.shape[1] != 2:
        raise ValueError("position must have shape [T, 2]")
    if yaw_array.shape != active_array.shape or len(yaw_array) != len(position_array):
        raise ValueError("position, yaw, and active must share the time dimension")
    active_array = (
        active_array
        & np.isfinite(position_array).all(axis=-1)
        & np.isfinite(yaw_array)
    )
    track_array = None
    if track_id is not None:
        track_array = np.asarray(track_id)
        if track_array.shape != yaw_array.shape:
            raise ValueError("track_id must have shape [T]")

    velocity = np.diff(position_array, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    yaw_delta = np.arctan2(np.sin(np.diff(yaw_array)), np.cos(np.diff(yaw_array))) / dt
    pair = _continuous_mask(active_array, track_array, 2)
    triple = _continuous_mask(active_array, track_array, 3)
    quad = _continuous_mask(active_array, track_array, 4)
    return {
        "speed_mps": np.linalg.norm(velocity, axis=-1)[pair],
        "abs_acc_mps2": np.linalg.norm(acceleration, axis=-1)[triple],
        "abs_jerk_mps3": np.linalg.norm(jerk, axis=-1)[quad],
        "abs_yaw_rate_rps": np.abs(yaw_delta[pair]),
        "speed_support_mask": pair,
        "acc_support_mask": triple,
        "jerk_support_mask": quad,
    }


def _maximum_or_none(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else None


def marginal_naturalness(
    kinematics: Mapping[str, np.ndarray],
    gates: Mapping[str, float] = DEFAULT_ABSOLUTE_GATES,
    empirical_q99: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Score one actor while retaining absolute and empirical gates."""

    unknown_gates = sorted(set(gates) - set(GATE_TO_SERIES))
    if unknown_gates:
        raise ValueError(f"unsupported naturalness gates: {unknown_gates}")
    maxima = {
        gate: _maximum_or_none(
            np.asarray(kinematics.get(GATE_TO_SERIES[gate], []), dtype=np.float64)
        )
        for gate in gates
    }
    sufficient = all(value is not None for value in maxima.values())
    absolute_checks = {
        key: maxima[key] is not None and maxima[key] <= float(gates[key])
        for key in gates
    }
    empirical_checks = None
    empirical_pass = None
    ratios = None
    if empirical_q99 is not None:
        missing = sorted(set(gates) - set(empirical_q99))
        if missing:
            raise ValueError(f"empirical_q99 is missing {missing}")
        ratios = {
            key: None
            if maxima[key] is None
            else maxima[key] / max(float(empirical_q99[key]), 1.0e-12)
            for key in gates
        }
        empirical_checks = {
            key: ratios[key] is not None and ratios[key] <= 1.0 for key in gates
        }
        empirical_pass = sufficient and all(empirical_checks.values())
    absolute_pass = sufficient and all(absolute_checks.values())
    return {
        "sufficient_support": sufficient,
        "maxima": maxima,
        "absolute_checks": absolute_checks,
        "absolute_pass": absolute_pass,
        "empirical_q99_ratios": ratios,
        "empirical_checks": empirical_checks,
        "empirical_pass": empirical_pass,
        "joint_pass": absolute_pass and (empirical_pass if empirical_q99 is not None else True),
    }


def obb_corners(position: Any, yaw: float, extent: Any) -> np.ndarray:
    """Return rectangle corners for full length/width extents."""

    center = np.asarray(position, dtype=np.float64)
    size = np.asarray(extent, dtype=np.float64)
    if center.shape != (2,) or size.shape != (2,) or np.any(size <= 0):
        raise ValueError("position/extent must be positive [2] vectors")
    half_length, half_width = size / 2.0
    local = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cosine, sine = np.cos(float(yaw)), np.sin(float(yaw))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + center


def obb_intersects(
    position_a: Any,
    yaw_a: float,
    extent_a: Any,
    position_b: Any,
    yaw_b: float,
    extent_b: Any,
    tolerance: float = 1.0e-9,
) -> bool:
    """Separating-axis OBB intersection; tangency counts as collision."""

    corners_a = obb_corners(position_a, yaw_a, extent_a)
    corners_b = obb_corners(position_b, yaw_b, extent_b)
    axes = []
    for corners in (corners_a, corners_b):
        for start, end in ((0, 1), (1, 2)):
            edge = corners[end] - corners[start]
            normal = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            axes.append(normal / np.linalg.norm(normal))
    for axis in axes:
        projection_a = corners_a @ axis
        projection_b = corners_b @ axis
        if projection_a.max() < projection_b.min() - tolerance:
            return False
        if projection_b.max() < projection_a.min() - tolerance:
            return False
    return True


def obb_sat_margin(
    position_a: Any,
    yaw_a: float,
    extent_a: Any,
    position_b: Any,
    yaw_b: float,
    extent_b: Any,
) -> float:
    """Signed separating-axis margin for two OBBs.

    Positive values indicate separation, zero is tangency, and negative values
    indicate overlap.  This is the maximum interval gap over the four OBB axes,
    not Euclidean polygon distance.
    """

    corners_a = obb_corners(position_a, yaw_a, extent_a)
    corners_b = obb_corners(position_b, yaw_b, extent_b)
    gaps = []
    for corners in (corners_a, corners_b):
        for start, end in ((0, 1), (1, 2)):
            edge = corners[end] - corners[start]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            projection_a = corners_a @ axis
            projection_b = corners_b @ axis
            gaps.append(
                max(
                    float(projection_b.min() - projection_a.max()),
                    float(projection_a.min() - projection_b.max()),
                )
            )
    return float(max(gaps))


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator <= 0.0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))


def obb_signed_distance(
    position_a: Any,
    yaw_a: float,
    extent_a: Any,
    position_b: Any,
    yaw_b: float,
    extent_b: Any,
) -> float:
    """Exact boundary distance when separate, SAT penetration when overlapping."""

    sat_margin = obb_sat_margin(
        position_a, yaw_a, extent_a, position_b, yaw_b, extent_b
    )
    if sat_margin <= 0.0:
        return sat_margin
    corners_a = obb_corners(position_a, yaw_a, extent_a)
    corners_b = obb_corners(position_b, yaw_b, extent_b)
    distances = []
    for point in corners_a:
        for index in range(4):
            distances.append(
                _point_segment_distance(
                    point, corners_b[index], corners_b[(index + 1) % 4]
                )
            )
    for point in corners_b:
        for index in range(4):
            distances.append(
                _point_segment_distance(
                    point, corners_a[index], corners_a[(index + 1) % 4]
                )
            )
    return float(min(distances))


def _first_conflict_arrival(
    position: np.ndarray,
    valid: np.ndarray,
    conflict_point: np.ndarray,
    radius: float,
    dt: float,
) -> Optional[float]:
    hits = np.flatnonzero(valid & (np.linalg.norm(position - conflict_point, axis=-1) <= radius))
    return float(hits[0] * dt) if hits.size else None


def joint_timing_criticality(
    ego_position: Any,
    ego_yaw: Any,
    ego_extent: Any,
    ego_active: Any,
    target_position: Any,
    target_yaw: Any,
    target_extent: Any,
    target_active: Any,
    conflict_point: Any,
    dt: float,
    conflict_radius_m: float = 2.0,
    ttc_horizon_seconds: float = 5.0,
    start_frame: int = 0,
) -> Dict[str, Any]:
    """Measure joint arrival, SAT margin, and constant-velocity OBB TTC.

    Arrival is the first sampled center entry into a fixed conflict disk. TTC
    freezes current yaw/extents and linearly extrapolates the most recent
    finite velocity; it is intentionally distinct from time-to-observed-impact.
    """

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(conflict_radius_m) or conflict_radius_m <= 0:
        raise ValueError("conflict_radius_m must be finite and positive")
    ego_pos = np.asarray(ego_position, dtype=np.float64)
    target_pos = np.asarray(target_position, dtype=np.float64)
    ego_yaw_array = np.asarray(ego_yaw, dtype=np.float64)
    target_yaw_array = np.asarray(target_yaw, dtype=np.float64)
    ego_extent_array = np.asarray(ego_extent, dtype=np.float64)
    target_extent_array = np.asarray(target_extent, dtype=np.float64)
    ego_valid = np.asarray(ego_active, dtype=bool)
    target_valid = np.asarray(target_active, dtype=bool)
    point = np.asarray(conflict_point, dtype=np.float64)
    if ego_pos.ndim != 2 or ego_pos.shape[1] != 2 or target_pos.shape != ego_pos.shape:
        raise ValueError("ego/target positions must share shape [T, 2]")
    frame_count = len(ego_pos)
    if not isinstance(start_frame, int) or isinstance(start_frame, bool):
        raise ValueError("start_frame must be an integer")
    if not 0 <= start_frame <= frame_count:
        raise ValueError("start_frame must be inside or immediately after the trajectory")
    expected = (frame_count,)
    if ego_yaw_array.shape != expected or target_yaw_array.shape != expected:
        raise ValueError("ego/target yaw must have shape [T]")
    if ego_extent_array.shape != (frame_count, 2) or target_extent_array.shape != (frame_count, 2):
        raise ValueError("ego/target extents must have shape [T, 2]")
    if ego_valid.shape != expected or target_valid.shape != expected or point.shape != (2,):
        raise ValueError("active must have shape [T] and conflict_point shape [2]")
    ego_valid &= (
        np.isfinite(ego_pos).all(axis=-1)
        & np.isfinite(ego_yaw_array)
        & np.isfinite(ego_extent_array).all(axis=-1)
        & np.all(ego_extent_array > 0, axis=-1)
    )
    target_valid &= (
        np.isfinite(target_pos).all(axis=-1)
        & np.isfinite(target_yaw_array)
        & np.isfinite(target_extent_array).all(axis=-1)
        & np.all(target_extent_array > 0, axis=-1)
    )
    joint_valid = ego_valid & target_valid
    post_mask = np.arange(frame_count) >= start_frame
    ego_arrival_absolute = _first_conflict_arrival(
        ego_pos, ego_valid & post_mask, point, conflict_radius_m, dt
    )
    target_arrival_absolute = _first_conflict_arrival(
        target_pos, target_valid & post_mask, point, conflict_radius_m, dt
    )
    ego_arrival = (
        ego_arrival_absolute - start_frame * dt
        if ego_arrival_absolute is not None
        else None
    )
    target_arrival = (
        target_arrival_absolute - start_frame * dt
        if target_arrival_absolute is not None
        else None
    )
    sat_margins = [
        obb_sat_margin(
            ego_pos[frame],
            ego_yaw_array[frame],
            ego_extent_array[frame],
            target_pos[frame],
            target_yaw_array[frame],
            target_extent_array[frame],
        )
        for frame in np.flatnonzero(joint_valid & post_mask)
    ]
    signed_distances = [
        obb_signed_distance(
            ego_pos[frame],
            ego_yaw_array[frame],
            ego_extent_array[frame],
            target_pos[frame],
            target_yaw_array[frame],
            target_extent_array[frame],
        )
        for frame in np.flatnonzero(joint_valid & post_mask)
    ]

    horizon_steps = int(np.floor(ttc_horizon_seconds / dt))
    ttc_values = []
    for frame in range(max(1, start_frame), frame_count):
        if not (joint_valid[frame - 1] and joint_valid[frame]):
            continue
        ego_velocity = (ego_pos[frame] - ego_pos[frame - 1]) / dt
        target_velocity = (target_pos[frame] - target_pos[frame - 1]) / dt
        for horizon in range(horizon_steps + 1):
            delta_t = horizon * dt
            if obb_intersects(
                ego_pos[frame] + ego_velocity * delta_t,
                ego_yaw_array[frame],
                ego_extent_array[frame],
                target_pos[frame] + target_velocity * delta_t,
                target_yaw_array[frame],
                target_extent_array[frame],
            ):
                ttc_values.append(delta_t)
                break
    arrival_difference = None
    if ego_arrival is not None and target_arrival is not None:
        arrival_difference = abs(ego_arrival - target_arrival)
    return {
        "frame_denominator": frame_count,
        "evaluation_start_frame": int(start_frame),
        "post_start_frame_denominator": int(frame_count - start_frame),
        "joint_active_frame_count": int((joint_valid & post_mask).sum()),
        "ego_arrival_time_seconds": ego_arrival,
        "target_arrival_time_seconds": target_arrival,
        "arrival_time_difference_seconds": arrival_difference,
        "minimum_obb_signed_distance_meters": (
            float(min(signed_distances)) if signed_distances else None
        ),
        "minimum_obb_sat_margin_meters": (
            float(min(sat_margins)) if sat_margins else None
        ),
        "minimum_constant_velocity_obb_ttc_seconds": (
            float(min(ttc_values)) if ttc_values else None
        ),
        "ttc_state_count": len(ttc_values),
        "conflict_radius_m": float(conflict_radius_m),
        "ttc_horizon_seconds": float(ttc_horizon_seconds),
    }


def emergent_cascade_metrics(
    joint_risk: float,
    single_agent_counterfactual_risks: Sequence[float],
    leave_one_event_out_risks: Optional[Sequence[float]] = None,
    high_risk_threshold: float = 1.0,
    low_risk_threshold: float = 0.0,
    necessity_delta: float = 0.1,
) -> Dict[str, Any]:
    """Counterfactual emergence and event-necessity metrics."""

    joint = float(joint_risk)
    singles = np.asarray(single_agent_counterfactual_risks, dtype=np.float64)
    if not np.isfinite(joint) or singles.ndim != 1 or singles.size == 0:
        raise ValueError("joint risk and at least one finite single-agent risk are required")
    if not np.isfinite(singles).all():
        raise ValueError("single-agent risks must be finite")
    maximum_single = float(singles.max())
    result: Dict[str, Any] = {
        "single_agent_counterfactual_denominator": int(singles.size),
        "joint_risk": joint,
        "maximum_single_agent_counterfactual_risk": maximum_single,
        "emergence_gap": joint - maximum_single,
        "strict_emergence_success": (
            joint >= float(high_risk_threshold)
            and maximum_single <= float(low_risk_threshold)
        ),
    }
    if leave_one_event_out_risks is not None:
        removals = np.asarray(leave_one_event_out_risks, dtype=np.float64)
        if removals.ndim != 1 or removals.size == 0 or not np.isfinite(removals).all():
            raise ValueError("leave-one-event-out risks must be a non-empty finite vector")
        effects = joint - removals
        result.update(
            event_removal_denominator=int(removals.size),
            event_necessity_effects=effects.tolist(),
            chain_necessity_rate=float(np.mean(effects > float(necessity_delta))),
        )
    else:
        result.update(
            event_removal_denominator=0,
            event_necessity_effects=None,
            chain_necessity_rate=None,
        )
    return result


def collision_events(
    position: Any,
    yaw: Any,
    extent: Any,
    active: Any,
    is_ego: Any,
    track_id: Any,
    target_track_id: Optional[int] = None,
    intervention_frame: int = 0,
) -> Dict[str, Any]:
    """Classify target and collateral OBB events with fixed denominators."""

    position_array = np.asarray(position, dtype=np.float64)
    yaw_array = np.asarray(yaw, dtype=np.float64)
    extent_array = np.asarray(extent, dtype=np.float64)
    active_array = np.asarray(active, dtype=bool)
    ego_array = np.asarray(is_ego, dtype=bool)
    track_array = np.asarray(track_id)
    if position_array.ndim != 3 or position_array.shape[-1] != 2:
        raise ValueError("position must have shape [A, T, 2]")
    actor_count, frame_count = position_array.shape[:2]
    expected = (actor_count, frame_count)
    if yaw_array.shape != expected or active_array.shape != expected or track_array.shape != expected:
        raise ValueError("yaw, active, and track_id must have shape [A, T]")
    if extent_array.shape != (actor_count, frame_count, 2):
        raise ValueError("extent must have shape [A, T, 2]")
    if ego_array.shape == (actor_count,):
        ego_rows = np.flatnonzero(ego_array)
    elif ego_array.shape == expected:
        ego_rows = np.flatnonzero(np.any(ego_array, axis=1))
    else:
        raise ValueError("is_ego must have shape [A] or [A, T]")
    if len(ego_rows) != 1:
        raise ValueError("exactly one ego actor is required")
    ego_row = int(ego_rows[0])
    pairs_by_frame = []
    target_frames = []
    new_target_frames = []
    ego_any_frames = []
    new_ego_any_frames = []
    collateral_frames = []
    new_collateral_frames = []
    preexisting_pairs = set()
    for frame in range(frame_count):
        valid_rows = [
            row
            for row in range(actor_count)
            if active_array[row, frame]
            and np.isfinite(position_array[row, frame]).all()
            and np.isfinite(yaw_array[row, frame])
            and np.isfinite(extent_array[row, frame]).all()
            and np.all(extent_array[row, frame] > 0)
        ]
        frame_pairs = []
        for row_a, row_b in combinations(valid_rows, 2):
            if not obb_intersects(
                position_array[row_a, frame],
                yaw_array[row_a, frame],
                extent_array[row_a, frame],
                position_array[row_b, frame],
                yaw_array[row_b, frame],
                extent_array[row_b, frame],
            ):
                continue
            ids = tuple(sorted((int(track_array[row_a, frame]), int(track_array[row_b, frame]))))
            frame_pairs.append(ids)
            if frame < intervention_frame:
                preexisting_pairs.add(ids)
            is_ego_pair = ego_row in (row_a, row_b)
            other_row = row_b if row_a == ego_row else row_a
            is_target = (
                is_ego_pair
                and target_track_id is not None
                and int(track_array[other_row, frame]) == int(target_track_id)
            )
            if frame >= intervention_frame:
                is_new_pair = ids not in preexisting_pairs
                if is_target:
                    target_frames.append(frame)
                    if is_new_pair:
                        new_target_frames.append(frame)
                else:
                    collateral_frames.append(frame)
                    if is_new_pair:
                        new_collateral_frames.append(frame)
                if is_ego_pair:
                    ego_any_frames.append(frame)
                    if is_new_pair:
                        new_ego_any_frames.append(frame)
        pairs_by_frame.append(frame_pairs)
    unique_target_frames = sorted(set(target_frames))
    unique_new_target_frames = sorted(set(new_target_frames))
    unique_collateral_frames = sorted(set(collateral_frames))
    unique_new_collateral_frames = sorted(set(new_collateral_frames))
    unique_ego_frames = sorted(set(ego_any_frames))
    unique_new_ego_frames = sorted(set(new_ego_any_frames))
    return {
        "frame_denominator": frame_count,
        "actor_denominator": actor_count,
        "intervention_frame": int(intervention_frame),
        "target_collision": bool(unique_target_frames),
        "target_collision_frames": unique_target_frames,
        "new_target_collision": bool(unique_new_target_frames),
        "new_target_collision_frames": unique_new_target_frames,
        "ego_any_collision": bool(unique_ego_frames),
        "ego_any_collision_frames": unique_ego_frames,
        "new_ego_any_collision": bool(unique_new_ego_frames),
        "new_ego_any_collision_frames": unique_new_ego_frames,
        "collateral_collision_any": bool(unique_collateral_frames),
        "collateral_collision_frames": unique_collateral_frames,
        "new_collateral_collision": bool(unique_new_collateral_frames),
        "new_collateral_collision_frames": unique_new_collateral_frames,
        "preexisting_collision_pairs": [list(pair) for pair in sorted(preexisting_pairs)],
        "pairs_by_frame": pairs_by_frame,
    }


def progress_metrics(position: Any, yaw0: float) -> Dict[str, float]:
    """Report forward, net, and path progress without conflating them."""

    position_array = np.asarray(position, dtype=np.float64)
    if position_array.ndim != 2 or position_array.shape[1] != 2 or len(position_array) < 2:
        raise ValueError("position must have shape [T, 2] with T >= 2")
    if not np.isfinite(position_array).all():
        raise ValueError("progress position must be finite")
    displacement = position_array[-1] - position_array[0]
    heading = np.asarray([np.cos(float(yaw0)), np.sin(float(yaw0))])
    return {
        "forward_m": float(displacement @ heading),
        "net_m": float(np.linalg.norm(displacement)),
        "path_m": float(np.linalg.norm(np.diff(position_array, axis=0), axis=-1).sum()),
    }
