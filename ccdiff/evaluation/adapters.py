"""Domain adapters and denominator aggregation for frozen evaluation inputs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .manifest import BenchmarkManifest, validate_observed_identities
from .metrics import (
    collision_events,
    compute_track_kinematics,
    joint_timing_criticality,
    marginal_naturalness,
    progress_metrics,
)


DENOMINATOR_FLAGS = (
    "reset_valid",
    "eligible_native_safe",
    "attempted",
    "completed",
    "contract_valid",
    "naturalness_valid",
    "target_event_success",
    "new_planner_failure_success",
    "zero_collateral_success",
)


def aggregate_denominators(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    """Aggregate all requested outcomes without silently dropping failures."""

    rows = list(records)
    result = {"requested": len(rows)}
    for flag in DENOMINATOR_FLAGS:
        values = []
        for index, row in enumerate(rows):
            value = row.get(flag)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"record {index} field {flag!r} must be boolean")
            values.append(bool(value))
        result[flag] = sum(values)
    result["paired_primary_success"] = sum(
        bool(row["eligible_native_safe"])
        and bool(row["completed"])
        and bool(row["contract_valid"])
        and bool(row["naturalness_valid"])
        and bool(row["zero_collateral_success"])
        and (
            bool(row["target_event_success"])
            or bool(row["new_planner_failure_success"])
        )
        for row in rows
    )
    return result


def adapt_nuplan_official_audit(
    audit: Mapping[str, Any],
    manifest: BenchmarkManifest,
) -> Dict[str, Any]:
    """Validate exact99 coverage and expose native-domain planner metrics."""

    if manifest.domain != "nuplan":
        raise ValueError("nuPlan audit requires a nuPlan manifest")
    records = audit.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("nuPlan audit records must be a non-empty list")
    normalized = []
    seen = set()
    for record in records:
        identity = (record.get("profile"), record.get("planner"))
        if identity in seen or not all(isinstance(item, str) and item for item in identity):
            raise ValueError(f"invalid or duplicate nuPlan evaluation identity: {identity}")
        seen.add(identity)
        coverage = validate_observed_identities(manifest, record.get("scenario_ids", []))
        if not coverage["exact"]:
            raise ValueError(f"nuPlan scenario coverage mismatch: {identity}: {coverage}")
        if int(record.get("scenario_count", -1)) != len(manifest.included):
            raise ValueError(f"nuPlan scenario count mismatch: {identity}")
        if int(record.get("succeeded", -1)) != len(manifest.included):
            raise ValueError(f"nuPlan run is not complete: {identity}")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"nuPlan metrics missing: {identity}")
        normalized.append(
            {
                "profile": identity[0],
                "planner": identity[1],
                "scenario_coverage": coverage,
                "native_domain_metrics": metrics,
                "planner_runtime_mean_s": record.get("planner_runtime_mean_s"),
                "planner_runtime_median_s": record.get("planner_runtime_median_s"),
                "planner_runtime_p95_s": record.get("planner_runtime_p95_s"),
                "crashdojo_generation_metrics_available": False,
                "unavailable_reason": (
                    "official native planner audit has no CrashDojo-generated "
                    "multi-agent traffic branches"
                ),
            }
        )
    return {
        "schema": "crashdojo_domain_adapter_v1",
        "domain": "nuplan",
        "manifest": manifest.summary(),
        "evaluation_record_count": len(normalized),
        "evaluations": normalized,
    }


def evaluate_nuscenes_scene_arrays(
    *,
    position: Any,
    yaw: Any,
    extent: Any,
    active: Any,
    is_ego: Any,
    track_id: Any,
    target_track_id: int,
    conflict_point: Any,
    intervention_frame: int,
    dt: float,
    empirical_q99_by_actor: Optional[Mapping[int, Mapping[str, float]]] = None,
    onroad: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate one decoded nuScenes rollout without mixing data supports."""

    position_array = np.asarray(position, dtype=np.float64)
    yaw_array = np.asarray(yaw, dtype=np.float64)
    extent_array = np.asarray(extent, dtype=np.float64)
    active_array = np.asarray(active, dtype=bool)
    track_array = np.asarray(track_id)
    actor_count, frame_count = position_array.shape[:2]
    if yaw_array.shape != (actor_count, frame_count):
        raise ValueError("yaw shape does not match position")
    ego_flags = np.asarray(is_ego, dtype=bool)
    if ego_flags.shape == (actor_count, frame_count):
        ego_rows = np.flatnonzero(np.any(ego_flags, axis=1))
    elif ego_flags.shape == (actor_count,):
        ego_rows = np.flatnonzero(ego_flags)
    else:
        raise ValueError("is_ego must have shape [A] or [A, T]")
    if len(ego_rows) != 1:
        raise ValueError("exactly one ego actor is required")
    ego_row = int(ego_rows[0])
    target_rows = np.flatnonzero(np.any(track_array == int(target_track_id), axis=1))
    if len(target_rows) != 1:
        raise ValueError("target_track_id must resolve to exactly one actor row")
    target_row = int(target_rows[0])

    actor_scores = []
    for actor in range(actor_count):
        kinematics = compute_track_kinematics(
            position_array[actor],
            yaw_array[actor],
            active_array[actor],
            dt,
            track_array[actor],
        )
        empirical = None
        if empirical_q99_by_actor is not None:
            empirical = empirical_q99_by_actor.get(actor)
        score = marginal_naturalness(kinematics, empirical_q99=empirical)
        if empirical_q99_by_actor is not None and empirical is None:
            score["empirical_calibration_missing"] = True
            score["empirical_pass"] = False
            score["joint_pass"] = False
        else:
            score["empirical_calibration_missing"] = False
        score.update(actor_row=actor, is_ego=actor == ego_row, is_target=actor == target_row)
        actor_scores.append(score)

    collision = collision_events(
        position=position_array,
        yaw=yaw_array,
        extent=extent_array,
        active=active_array,
        is_ego=ego_flags,
        track_id=track_array,
        target_track_id=target_track_id,
        intervention_frame=intervention_frame,
    )
    timing = joint_timing_criticality(
        ego_position=position_array[ego_row],
        ego_yaw=yaw_array[ego_row],
        ego_extent=extent_array[ego_row],
        ego_active=active_array[ego_row],
        target_position=position_array[target_row],
        target_yaw=yaw_array[target_row],
        target_extent=extent_array[target_row],
        target_active=active_array[target_row],
        conflict_point=conflict_point,
        dt=dt,
        start_frame=intervention_frame,
    )
    ego_support = active_array[ego_row] & np.isfinite(position_array[ego_row]).all(axis=-1)
    ego_positions = position_array[ego_row][ego_support]
    progress = None
    if len(ego_positions) >= 2:
        progress = progress_metrics(ego_positions, yaw_array[ego_row][ego_support][0])

    onroad_result = None
    if onroad is not None:
        onroad_array = np.asarray(onroad, dtype=bool)
        if onroad_array.shape != active_array.shape:
            raise ValueError("onroad must have shape [A, T]")
        active_count = int(active_array.sum())
        onroad_result = {
            "active_frame_denominator": active_count,
            "active_frame_onroad_rate": (
                float(onroad_array[active_array].mean()) if active_count else None
            ),
            "all_active_frames_onroad": bool(active_count and onroad_array[active_array].all()),
        }
    all_agent_pass = bool(actor_scores) and all(score["joint_pass"] for score in actor_scores)
    return {
        "schema": "crashdojo_nuscenes_scene_metrics_v1",
        "actor_denominator": actor_count,
        "frame_denominator": frame_count,
        "dt_seconds": float(dt),
        "actor_naturalness": actor_scores,
        "all_agent_plausibility": all_agent_pass,
        "collision": collision,
        "joint_criticality": timing,
        "ego_progress": progress,
        "onroad": onroad_result,
    }
