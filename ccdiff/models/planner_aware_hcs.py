"""Planner-aware hard-case selector for one closed-loop ego planner.

The selector is causal: it consumes the actual ego planner output plus only
the current state of surrounding agents.  It never reads logged futures or an
adversarial rollout.  Candidate motion is extrapolated with constant velocity
solely for event selection; the frozen joint FM remains responsible for the
generated traffic trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import torch


class PlannerAwareHCSSelectionError(RuntimeError):
    """Raised when no causal, phase-feasible target can be selected."""


@dataclass(frozen=True)
class PlannerAwareHCSEvent:
    target_local_index: int
    target_global_index: int
    reference_local_index: int
    reference_global_index: int
    collision_type: str
    conflict_step: int
    conflict_time_s: float
    conflict_point_world: tuple
    predicted_obb_gap_m: float
    predicted_bystander_gap_m: float
    score: float
    details: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class PlannerAwareHCSCoalitionMember:
    local_index: int
    global_index: int
    role: str
    corridor_point_world: tuple
    predicted_center_world: tuple
    reach_error_m: float
    required_speed_mps: float
    speed_change_mps: float
    heading_error_rad: float
    primary_clearance_m: float
    score: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class PlannerAwareHCSCoalitionEvent:
    primary: PlannerAwareHCSEvent
    members: Tuple[PlannerAwareHCSCoalitionMember, ...]
    controlled_local_indices: Tuple[int, ...]
    controlled_global_indices: Tuple[int, ...]
    score: float
    details: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def _as_tensor(value, *, device=None, dtype=torch.float32):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _world_points(local_xy: torch.Tensor, world_from_agent: torch.Tensor):
    ones = torch.ones_like(local_xy[..., :1])
    homogeneous = torch.cat((local_xy, ones), dim=-1)
    return torch.matmul(homogeneous, world_from_agent.transpose(-1, -2))[..., :2]


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _obb_signed_gap(
    center_a: torch.Tensor,
    yaw_a: torch.Tensor,
    half_a: torch.Tensor,
    center_b: torch.Tensor,
    yaw_b: torch.Tensor,
    half_b: torch.Tensor,
) -> torch.Tensor:
    """Separating-axis signed gap; non-positive values indicate overlap."""
    forward_a = torch.stack((torch.cos(yaw_a), torch.sin(yaw_a)), dim=-1)
    lateral_a = torch.stack((-torch.sin(yaw_a), torch.cos(yaw_a)), dim=-1)
    forward_b = torch.stack((torch.cos(yaw_b), torch.sin(yaw_b)), dim=-1)
    lateral_b = torch.stack((-torch.sin(yaw_b), torch.cos(yaw_b)), dim=-1)
    axes = torch.stack((forward_a, lateral_a, forward_b, lateral_b), dim=-2)
    delta = center_a - center_b
    projection = torch.abs(torch.sum(delta[..., None, :] * axes, dim=-1))
    radius_a = (
        torch.abs(torch.sum(axes * forward_a[..., None, :], dim=-1))
        * half_a[..., 0, None]
        + torch.abs(torch.sum(axes * lateral_a[..., None, :], dim=-1))
        * half_a[..., 1, None]
    )
    radius_b = (
        torch.abs(torch.sum(axes * forward_b[..., None, :], dim=-1))
        * half_b[..., 0, None]
        + torch.abs(torch.sum(axes * lateral_b[..., None, :], dim=-1))
        * half_b[..., 1, None]
    )
    return torch.max(projection - radius_a - radius_b, dim=-1).values


class PlannerAwareHCS:
    """Select ``(target, type, time, point)`` against one actual ego plan."""

    def __init__(
        self,
        dt: float = 0.1,
        min_target_speed_mps: float = 0.5,
        min_initial_distance_m: float = 2.0,
        max_predicted_gap_m: float = 10.0,
        min_bystander_gap_m: float = 1.0,
        min_conflict_step: int = 2,
        max_conflict_step: Optional[int] = None,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.min_target_speed_mps = float(min_target_speed_mps)
        self.min_initial_distance_m = float(min_initial_distance_m)
        self.max_predicted_gap_m = float(max_predicted_gap_m)
        self.min_bystander_gap_m = float(min_bystander_gap_m)
        self.min_conflict_step = int(min_conflict_step)
        self.max_conflict_step = max_conflict_step

    @staticmethod
    def _classify_collision_type(ego_yaw, target_yaw, relative_position):
        heading = abs(float(_wrap(target_yaw - ego_yaw)))
        ego_forward = torch.stack((torch.cos(ego_yaw), torch.sin(ego_yaw)))
        rel_norm = torch.linalg.norm(relative_position).clamp(min=1e-6)
        bearing = float(torch.dot(relative_position / rel_norm, ego_forward))
        if heading >= math.radians(150.0):
            return "head_on", heading, bearing
        if math.radians(60.0) <= heading <= math.radians(120.0):
            return "side", heading, bearing
        if heading <= math.radians(30.0):
            return "rear_end", heading, bearing
        return "cut_in", heading, bearing

    @staticmethod
    def _planner_yaw(plan_world, fallback_yaw):
        delta = plan_world[1:] - plan_world[:-1]
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        speed = torch.linalg.norm(delta, dim=-1)
        yaw = torch.where(speed > 1e-4, yaw, fallback_yaw.expand_as(yaw))
        return torch.cat((yaw, yaw[-1:]), dim=0)

    def select(
        self,
        observation: Dict,
        planner_positions_local,
        reference_agent_idx: int = 0,
        candidate_local_indices: Optional[Sequence[int]] = None,
    ) -> PlannerAwareHCSEvent:
        required = ("scene_index", "world_from_agent", "curr_speed", "extent")
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError("planner-aware HCS missing fields {}".format(missing))

        world_from_agent = _as_tensor(observation["world_from_agent"])
        device = world_from_agent.device
        scene_index = _as_tensor(
            observation["scene_index"], device=device, dtype=torch.long
        ).reshape(-1)
        unique_scene = torch.unique(scene_index)
        if unique_scene.numel() != 1:
            raise ValueError("planner-aware HCS requires one scene per batch")
        agents = torch.nonzero(scene_index == unique_scene[0], as_tuple=True)[0]
        if not 0 <= int(reference_agent_idx) < int(agents.numel()):
            raise IndexError("reference_agent_idx exceeds scene agent count")
        reference_global = int(agents[int(reference_agent_idx)])

        plan_local = _as_tensor(planner_positions_local, device=device)
        if plan_local.ndim == 3:
            if plan_local.shape[0] != 1:
                raise ValueError("expected one ego plan")
            plan_local = plan_local[0]
        if plan_local.ndim != 2 or plan_local.shape[-1] != 2 or plan_local.shape[0] < 2:
            raise ValueError("planner_positions_local must have shape [T,2], T>=2")
        if not torch.isfinite(plan_local).all():
            raise ValueError("ego planner output contains NaN/Inf")
        plan_world = _world_points(plan_local, world_from_agent[reference_global])

        centers = world_from_agent[:, :2, 2]
        speeds = _as_tensor(observation["curr_speed"], device=device).reshape(-1)
        extents = _as_tensor(observation["extent"], device=device)[..., :2]
        if "yaw" in observation:
            yaws = _as_tensor(observation["yaw"], device=device).reshape(-1)
        else:
            yaws = torch.atan2(world_from_agent[:, 1, 0], world_from_agent[:, 0, 0])
        if not (torch.isfinite(centers).all() and torch.isfinite(speeds).all()
                and torch.isfinite(extents).all() and torch.isfinite(yaws).all()):
            raise ValueError("current scene state contains NaN/Inf")

        planner_yaw = self._planner_yaw(plan_world, yaws[reference_global])
        horizon = int(plan_world.shape[0])
        first_step = max(0, self.min_conflict_step)
        last_step = horizon - 1
        if self.max_conflict_step is not None:
            last_step = min(last_step, int(self.max_conflict_step))
        if first_step > last_step:
            raise PlannerAwareHCSSelectionError("empty conflict-time window")
        time_s = (
            torch.arange(horizon, device=device, dtype=plan_world.dtype) + 1.0
        ) * self.dt

        if candidate_local_indices is None:
            candidate_local_indices = [
                index for index in range(int(agents.numel()))
                if index != int(reference_agent_idx)
            ]
        else:
            candidate_local_indices = [int(index) for index in candidate_local_indices]

        best = None
        for target_local in candidate_local_indices:
            if not 0 <= target_local < int(agents.numel()):
                raise IndexError("candidate index exceeds scene agent count")
            if target_local == int(reference_agent_idx):
                continue
            target_global = int(agents[target_local])
            target_speed = float(speeds[target_global])
            initial_distance = float(
                torch.linalg.norm(centers[target_global] - centers[reference_global])
            )
            if target_speed < self.min_target_speed_mps:
                continue
            if initial_distance <= self.min_initial_distance_m:
                continue

            direction = torch.stack(
                (torch.cos(yaws[target_global]), torch.sin(yaws[target_global]))
            )
            target_world = (
                centers[target_global][None]
                + time_s[:, None] * speeds[target_global] * direction[None]
            )
            target_yaw = yaws[target_global].expand(horizon)
            pair_gap = _obb_signed_gap(
                target_world,
                target_yaw,
                0.5 * extents[target_global],
                plan_world,
                planner_yaw,
                0.5 * extents[reference_global],
            )
            window = pair_gap[first_step:last_step + 1]
            conflict_step = int(first_step + torch.argmin(window))
            predicted_gap = float(pair_gap[conflict_step])
            if predicted_gap > self.max_predicted_gap_m:
                continue

            bystander_gap = float("inf")
            for other_local in range(int(agents.numel())):
                if other_local in {target_local, int(reference_agent_idx)}:
                    continue
                other_global = int(agents[other_local])
                other_direction = torch.stack(
                    (torch.cos(yaws[other_global]), torch.sin(yaws[other_global]))
                )
                other_position = (
                    centers[other_global]
                    + time_s[conflict_step] * speeds[other_global] * other_direction
                )
                gap = _obb_signed_gap(
                    target_world[conflict_step],
                    target_yaw[conflict_step],
                    0.5 * extents[target_global],
                    other_position,
                    yaws[other_global],
                    0.5 * extents[other_global],
                )
                bystander_gap = min(bystander_gap, float(gap))
            if bystander_gap < self.min_bystander_gap_m:
                continue

            collision_type, heading, bearing = self._classify_collision_type(
                planner_yaw[conflict_step],
                target_yaw[conflict_step],
                target_world[conflict_step] - plan_world[conflict_step],
            )
            reach = math.exp(
                -(max(predicted_gap, 0.0) ** 2) / (2.0 * 4.0 ** 2)
            )
            event_time = float(time_s[conflict_step])
            timing = math.exp(-((event_time - 4.0) ** 2) / (2.0 * 2.5 ** 2))
            motion = min(max(target_speed, 0.0) / 4.0, 1.0)
            clearance = 1.0 if math.isinf(bystander_gap) else min(
                max(bystander_gap, 0.0) / 8.0, 1.0
            )
            geometry = {
                "rear_end": 0.95,
                "side": 1.0,
                "cut_in": 0.85,
                "head_on": 0.80,
            }[collision_type]
            score = (
                0.34 * reach + 0.20 * timing + 0.14 * motion
                + 0.17 * geometry + 0.15 * clearance
            )
            conflict_point = 0.5 * (
                plan_world[conflict_step] + target_world[conflict_step]
            )
            logged_bystander_gap = (
                bystander_gap if math.isfinite(bystander_gap) else 1.0e9
            )
            event = PlannerAwareHCSEvent(
                target_local_index=target_local,
                target_global_index=target_global,
                reference_local_index=int(reference_agent_idx),
                reference_global_index=reference_global,
                collision_type=collision_type,
                conflict_step=conflict_step,
                conflict_time_s=event_time,
                conflict_point_world=(
                    float(conflict_point[0]), float(conflict_point[1])
                ),
                predicted_obb_gap_m=predicted_gap,
                predicted_bystander_gap_m=logged_bystander_gap,
                score=score,
                details={
                    "planner_reference_used": 1.0,
                    "target_speed_mps": target_speed,
                    "initial_distance_m": initial_distance,
                    "relative_heading_rad": heading,
                    "target_bearing_cos": bearing,
                    "reach_score": reach,
                    "timing_score": timing,
                    "motion_score": motion,
                    "geometry_score": geometry,
                    "bystander_clearance_score": clearance,
                },
            )
            key = (score, -max(predicted_gap, 0.0), -event_time)
            if best is None or key > best[0]:
                best = (key, event)

        if best is None:
            raise PlannerAwareHCSSelectionError(
                "no candidate passed planner-aware reachability and collateral gates"
            )
        return best[1]

    def select_escape_coalition(
        self,
        observation: Dict,
        planner_positions_local,
        primary: PlannerAwareHCSEvent,
        candidate_local_indices: Optional[Sequence[int]] = None,
        min_blockers: int = 1,
        max_blockers: int = 2,
        corridor_offset_m: float = 3.2,
        max_reach_error_m: float = 12.0,
        max_speed_change_mps: float = 6.0,
        max_heading_error_rad: float = 2.35,
        min_primary_clearance_m: float = 0.5,
    ) -> PlannerAwareHCSCoalitionEvent:
        """Select natural moving blockers for the ego plan's escape corridors.

        The primary HCS event fixes the causal conflict time.  Two lateral
        corridor points are then constructed from the *same actual ego plan*.
        Candidate blockers are scored only from their current state under a
        constant-velocity reachability probe; logged futures are never read.
        The method fails closed if the requested coalition cannot be formed.
        """
        if min_blockers < 1 or max_blockers < min_blockers or max_blockers > 2:
            raise ValueError("escape coalition requires 1 <= min <= max <= 2")
        if corridor_offset_m <= 0.0 or max_reach_error_m <= 0.0:
            raise ValueError("escape corridor geometry must be positive")

        required = ("scene_index", "world_from_agent", "curr_speed", "extent")
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError("planner-aware coalition missing fields {}".format(missing))
        world_from_agent = _as_tensor(observation["world_from_agent"])
        device = world_from_agent.device
        scene_index = _as_tensor(
            observation["scene_index"], device=device, dtype=torch.long
        ).reshape(-1)
        unique_scene = torch.unique(scene_index)
        if unique_scene.numel() != 1:
            raise ValueError("planner-aware coalition requires one scene per batch")
        agents = torch.nonzero(scene_index == unique_scene[0], as_tuple=True)[0]
        if not 0 <= primary.reference_local_index < int(agents.numel()):
            raise IndexError("primary reference is outside the current scene")
        if not 0 <= primary.target_local_index < int(agents.numel()):
            raise IndexError("primary target is outside the current scene")

        plan_local = _as_tensor(planner_positions_local, device=device)
        if plan_local.ndim == 3:
            if plan_local.shape[0] != 1:
                raise ValueError("expected one ego plan")
            plan_local = plan_local[0]
        if plan_local.ndim != 2 or plan_local.shape[-1] != 2:
            raise ValueError("planner_positions_local must have shape [T,2]")
        reference_global = int(agents[primary.reference_local_index])
        plan_world = _world_points(plan_local, world_from_agent[reference_global])
        centers = world_from_agent[:, :2, 2]
        speeds = _as_tensor(observation["curr_speed"], device=device).reshape(-1)
        extents = _as_tensor(observation["extent"], device=device)[..., :2]
        if "yaw" in observation:
            yaws = _as_tensor(observation["yaw"], device=device).reshape(-1)
        else:
            yaws = torch.atan2(world_from_agent[:, 1, 0], world_from_agent[:, 0, 0])
        planner_yaw = self._planner_yaw(plan_world, yaws[reference_global])
        conflict_step = min(max(int(primary.conflict_step), 0), plan_world.shape[0] - 1)
        conflict_time = float((conflict_step + 1) * self.dt)
        center = plan_world[conflict_step]
        lateral = torch.stack(
            (-torch.sin(planner_yaw[conflict_step]), torch.cos(planner_yaw[conflict_step]))
        )
        corridor_points = {
            "block_left": center + float(corridor_offset_m) * lateral,
            "block_right": center - float(corridor_offset_m) * lateral,
        }

        if candidate_local_indices is None:
            candidates = [
                index for index in range(int(agents.numel()))
                if index not in {
                    int(primary.reference_local_index), int(primary.target_local_index)
                }
            ]
            locked_candidates = None
        else:
            candidates = [int(index) for index in candidate_local_indices]
            locked_candidates = set(candidates)
            if len(locked_candidates) != len(candidates):
                raise ValueError("coalition candidate identities must be unique")
            min_blockers = len(candidates)
            max_blockers = len(candidates)

        primary_global = int(agents[primary.target_local_index])
        primary_direction = torch.stack(
            (torch.cos(yaws[primary_global]), torch.sin(yaws[primary_global]))
        )
        primary_at_conflict = (
            centers[primary_global]
            + conflict_time * speeds[primary_global] * primary_direction
        )
        options = []
        for local_index in candidates:
            if not 0 <= local_index < int(agents.numel()):
                raise IndexError("coalition candidate exceeds scene agent count")
            if local_index in {
                int(primary.reference_local_index), int(primary.target_local_index)
            }:
                continue
            global_index = int(agents[local_index])
            speed = float(speeds[global_index])
            if speed < self.min_target_speed_mps:
                continue
            if float(torch.linalg.norm(centers[global_index] - centers[reference_global])) <= self.min_initial_distance_m:
                continue
            direction = torch.stack(
                (torch.cos(yaws[global_index]), torch.sin(yaws[global_index]))
            )
            predicted = centers[global_index] + conflict_time * speeds[global_index] * direction
            for role, point in corridor_points.items():
                to_point = point - centers[global_index]
                distance = torch.linalg.norm(to_point).clamp(min=1e-6)
                required_speed = float(distance / max(conflict_time, 1e-6))
                speed_change = abs(required_speed - speed)
                required_heading = torch.atan2(to_point[1], to_point[0])
                heading_error = abs(float(_wrap(required_heading - yaws[global_index])))
                reach_error = float(torch.linalg.norm(predicted - point))
                primary_clearance = float(_obb_signed_gap(
                    predicted,
                    yaws[global_index],
                    0.5 * extents[global_index],
                    primary_at_conflict,
                    yaws[primary_global],
                    0.5 * extents[primary_global],
                ))
                if (
                    reach_error > max_reach_error_m
                    or speed_change > max_speed_change_mps
                    or heading_error > max_heading_error_rad
                    or primary_clearance < min_primary_clearance_m
                ):
                    continue
                reach_score = math.exp(-0.5 * (reach_error / 4.0) ** 2)
                speed_score = math.exp(-0.5 * (speed_change / 2.5) ** 2)
                heading_score = math.exp(-0.5 * (heading_error / 0.9) ** 2)
                clearance_score = min(max(primary_clearance, 0.0) / 8.0, 1.0)
                score = (
                    0.45 * reach_score + 0.20 * speed_score
                    + 0.20 * heading_score + 0.15 * clearance_score
                )
                member = PlannerAwareHCSCoalitionMember(
                    local_index=local_index,
                    global_index=global_index,
                    role=role,
                    corridor_point_world=(float(point[0]), float(point[1])),
                    predicted_center_world=(float(predicted[0]), float(predicted[1])),
                    reach_error_m=reach_error,
                    required_speed_mps=required_speed,
                    speed_change_mps=speed_change,
                    heading_error_rad=heading_error,
                    primary_clearance_m=primary_clearance,
                    score=score,
                )
                options.append((score, -reach_error, -speed_change, member))

        selected = []
        selected_locals = set()
        selected_roles = set()
        for _, _, _, member in sorted(options, key=lambda item: item[:3], reverse=True):
            if member.local_index in selected_locals or member.role in selected_roles:
                continue
            selected.append(member)
            selected_locals.add(member.local_index)
            selected_roles.add(member.role)
            if len(selected) == max_blockers:
                break
        if len(selected) < min_blockers:
            raise PlannerAwareHCSSelectionError(
                "no phase-feasible escape-corridor coalition passed hard gates"
            )
        if locked_candidates is not None and selected_locals != locked_candidates:
            raise PlannerAwareHCSSelectionError(
                "locked coalition member failed the current reachability gates"
            )
        selected = tuple(sorted(selected, key=lambda member: member.role))
        controlled_local = (int(primary.target_local_index),) + tuple(
            member.local_index for member in selected
        )
        controlled_global = (int(primary.target_global_index),) + tuple(
            member.global_index for member in selected
        )
        coalition_score = float(
            0.6 * primary.score
            + 0.4 * sum(member.score for member in selected) / len(selected)
        )
        return PlannerAwareHCSCoalitionEvent(
            primary=primary,
            members=selected,
            controlled_local_indices=controlled_local,
            controlled_global_indices=controlled_global,
            score=coalition_score,
            details={
                "planner_reference_used": 1.0,
                "escape_corridor_offset_m": float(corridor_offset_m),
                "minimum_blockers": float(min_blockers),
                "maximum_blockers": float(max_blockers),
                "selected_blockers": float(len(selected)),
                "conflict_time_s": conflict_time,
                "logged_future_used": 0.0,
            },
        )
