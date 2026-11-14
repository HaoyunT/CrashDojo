"""CCDiff-native ego planners using SafeSim/STRIVE planning semantics.

The adapters in this file deliberately consume CCDiff's unified 8-D trajdata
observation contract.  STRIVE's 9-D ``StateTensor`` contract is introduced at
one named boundary only; keeping that conversion explicit prevents velocity
channels from silently shifting.
"""

from copy import deepcopy

import numpy as np
import torch
from trajdata.data_structures.state import StateTensor

from ccdiff.planners.base import EgoPlannerPolicy
from tbsim.policies.base import Policy
from tbsim.policies.common import Action
from tbsim.policies.strive_planner_trajdata import (
    CONFIG_DICT,
    DEF_CONFIG,
    HardcodeNuscPlanner,
    PlannerConfig,
    load_lane_graphs_trajdata,
)
import tbsim.utils.geometry_utils as GeoUtils


CCDIFF_STATE_FORMAT = "x,y,xd,yd,xdd,ydd,s,c"
SAFESIM_STATE_FORMAT = "x,y,z,xd,yd,xdd,ydd,s,c"


def _ccdiff_state_8d_to_safesim_9d(state):
    """Insert SafeSim's z channel without changing any motion channel.

    CCDiff: ``x,y,xd,yd,xdd,ydd,s,c``
    SafeSim: ``x,y,z,xd,yd,xdd,ydd,s,c``
    """
    state = torch.as_tensor(state)
    if state.shape[-1] == 9:
        return state
    if state.shape[-1] != 8:
        raise ValueError(
            "Expected CCDiff 8-D or SafeSim 9-D state, got {}".format(
                state.shape[-1]
            )
        )
    zero_z = torch.zeros_like(state[..., :1])
    converted = torch.cat((state[..., :2], zero_z, state[..., 2:]), dim=-1)
    if converted.shape[-1] != 9:
        raise AssertionError("8-D to 9-D state conversion failed")
    return converted


class LoggedReplayTrafficPolicy(Policy):
    """Replay the logged future for the unified scene batch.

    ``PlannerOverrideRolloutWrapper`` replaces the ego row afterwards, so this
    policy intentionally returns all rows, including the logged ego row.
    """

    def __init__(self, device):
        super().__init__(device)

    def eval(self):
        return self

    def get_action(self, obs, **kwargs):
        positions = torch.as_tensor(
            obs["target_positions"], device=self.device
        ).clone()
        yaws = torch.as_tensor(obs["target_yaws"], device=self.device).clone()
        positions = torch.nan_to_num(positions)
        yaws = torch.nan_to_num(yaws)
        action = Action(positions=positions, yaws=yaws)
        return action, {
            "action_samples": Action(
                positions=positions[:, None],
                yaws=yaws[:, None],
            ).to_dict()
        }


class IDMPlannerPolicy(EgoPlannerPolicy):
    """CCDiff-native port of SafeSim's IDM ego-planning baseline."""

    name = "idm"

    def __init__(
        self,
        device,
        dt=0.1,
        horizon=52,
        max_accel=1.0,
        comfortable_brake=3.0,
        min_gap=5.0,
        time_headway=1.0,
        desired_speed=10.0,
        lane_half_width=3.5,
        max_yaw_rate=1.2,
    ):
        super().__init__(device)
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.max_accel = float(max_accel)
        self.comfortable_brake = float(comfortable_brake)
        self.min_gap = float(min_gap)
        self.time_headway = float(time_headway)
        self.desired_speed = float(desired_speed)
        self.lane_half_width = float(lane_half_width)
        self.max_yaw_rate = float(max_yaw_rate)

    @staticmethod
    def _as_tensor(value, device, dtype=torch.float32):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _reference_paths(self, obs, batch_size):
        lane_data = obs.get("extras", {}).get("closest_lane_point")
        if lane_data is None:
            return [None] * batch_size
        lane_data = self._as_tensor(lane_data, self.device)
        paths = []
        for batch_idx in range(batch_size):
            best_path = None
            best_score = None
            for candidate in lane_data[batch_idx]:
                valid = torch.isfinite(candidate[..., :2]).all(dim=-1)
                points = candidate[valid]
                if points.shape[0] < 2:
                    continue
                closest = torch.linalg.norm(points[:, :2], dim=-1).min()
                if points.shape[-1] >= 4:
                    headings = points[:, 3]
                else:
                    delta = points[1:, :2] - points[:-1, :2]
                    headings = torch.atan2(delta[:, 1], delta[:, 0])
                heading_error = torch.abs(
                    torch.atan2(torch.sin(headings), torch.cos(headings))
                ).median()
                score = closest + 2.0 * heading_error
                if best_score is None or score < best_score:
                    best_score = score
                    best_path = points

            if best_path is None:
                paths.append(None)
                continue

            # CCDiff caches nearby lane samples by distance.  In the ego frame,
            # sorting by longitudinal x restores the forward route order.
            best_path = best_path[torch.argsort(best_path[:, 0])]
            # Never prepend a behind-ego point after the local origin. Doing so
            # creates a short reverse segment and a pi-radian yaw discontinuity
            # at the next receding-horizon query.
            best_path = best_path[best_path[:, 0] > 1e-3]
            if best_path.shape[0] < 2:
                paths.append(None)
                continue
            origin = torch.zeros(
                (1, best_path.shape[-1]),
                device=best_path.device,
                dtype=best_path.dtype,
            )
            best_path = torch.cat((origin, best_path), dim=0)
            segment_lengths = torch.linalg.norm(
                best_path[1:, :2] - best_path[:-1, :2], dim=-1
            )
            keep = torch.cat(
                (
                    torch.ones(1, dtype=torch.bool, device=best_path.device),
                    segment_lengths > 1e-3,
                )
            )
            best_path = best_path[keep]
            paths.append(best_path if best_path.shape[0] >= 2 else None)
        return paths

    def _idm_acceleration(self, obs, ego_speed):
        free_accel = self.max_accel * (
            1.0 - (ego_speed / self.desired_speed).clamp_min(0.0).pow(4)
        )
        neigh_hist = obs.get("neigh_hist")
        neigh_types = obs.get("neigh_types")
        if neigh_types is None:
            neigh_types = obs.get("all_other_agents_types")
        if neigh_hist is None or neigh_types is None:
            return free_accel.clamp(
                min=-2.0 * self.comfortable_brake,
                max=self.max_accel,
            )

        neigh_hist = self._as_tensor(neigh_hist, self.device)
        neigh_types = self._as_tensor(neigh_types, self.device)
        current = neigh_hist[..., -1, :]
        rel_pos = current[..., :2]
        lead_speed = torch.linalg.norm(current[..., 2:4], dim=-1)
        valid = (
            (neigh_types > 0)
            & torch.isfinite(rel_pos).all(dim=-1)
            & (rel_pos[..., 0] > 0.0)
            & (rel_pos[..., 1].abs() < self.lane_half_width)
        )
        distance = torch.where(
            valid,
            rel_pos[..., 0],
            torch.full_like(rel_pos[..., 0], float("inf")),
        )
        min_distance, lead_idx = distance.min(dim=1)
        has_lead = torch.isfinite(min_distance)
        selected_lead_speed = torch.gather(
            lead_speed, 1, lead_idx[:, None]
        ).squeeze(1)
        delta_v = ego_speed - selected_lead_speed
        denominator = 2.0 * np.sqrt(
            self.max_accel * self.comfortable_brake
        )
        desired_gap = self.min_gap + torch.clamp_min(
            ego_speed * self.time_headway + ego_speed * delta_v / denominator,
            0.0,
        )
        interaction = (
            desired_gap / min_distance.clamp_min(0.5)
        ).pow(2)
        following_accel = self.max_accel * (
            1.0
            - (ego_speed / self.desired_speed).clamp_min(0.0).pow(4)
            - interaction
        )
        accel = torch.where(has_lead, following_accel, free_accel)
        return accel.clamp(
            min=-2.0 * self.comfortable_brake,
            max=self.max_accel,
        )

    def _sample_path(self, path, distances):
        if path is None:
            positions = torch.stack(
                (distances, torch.zeros_like(distances)), dim=-1
            )
            return positions, torch.zeros_like(distances)

        xy = path[:, :2]
        segment = xy[1:] - xy[:-1]
        segment_length = torch.linalg.norm(
            segment, dim=-1
        ).clamp_min(1e-4)
        cumulative = torch.cat(
            (
                torch.zeros(1, device=xy.device, dtype=xy.dtype),
                torch.cumsum(segment_length, dim=0),
            )
        )
        required = distances[-1]
        if required > cumulative[-1]:
            tail_yaw = torch.atan2(segment[-1, 1], segment[-1, 0])
            tail_distance = required - cumulative[-1] + 5.0
            direction = torch.stack(
                (torch.cos(tail_yaw), torch.sin(tail_yaw))
            )
            xy = torch.cat((xy, (xy[-1] + tail_distance * direction)[None]))
            segment = xy[1:] - xy[:-1]
            segment_length = torch.linalg.norm(
                segment, dim=-1
            ).clamp_min(1e-4)
            cumulative = torch.cat(
                (
                    torch.zeros(1, device=xy.device, dtype=xy.dtype),
                    torch.cumsum(segment_length, dim=0),
                )
            )
        segment_index = torch.searchsorted(
            cumulative[1:], distances, right=False
        ).clamp(max=segment_length.shape[0] - 1)
        ratio = (
            (distances - cumulative[segment_index])
            / segment_length[segment_index]
        ).clamp(0.0, 1.0)
        positions = xy[segment_index] + ratio[:, None] * segment[segment_index]
        yaws = torch.atan2(
            segment[segment_index, 1], segment[segment_index, 0]
        )
        return positions, yaws

    def _limit_yaw_rate(self, yaws):
        limited = []
        previous = yaws.new_zeros(())
        max_delta = self.max_yaw_rate * self.dt
        for yaw in yaws:
            delta = torch.atan2(
                torch.sin(yaw - previous),
                torch.cos(yaw - previous),
            ).clamp(min=-max_delta, max=max_delta)
            previous = previous + delta
            limited.append(previous)
        return torch.stack(limited)

    def _integrate_rate_limited_path(self, distances, path_yaws):
        """Keep the executed positions consistent with the limited headings.

        ``_sample_path`` supplies a geometric reference heading.  Clipping only
        the reported yaw while retaining the original path positions makes the
        vehicle translate in a direction that can differ sharply from its
        orientation.  Integrating each path-length increment along the limited
        heading preserves the IDM speed profile and produces one physically
        self-consistent action trajectory.
        """
        limited_yaws = self._limit_yaw_rate(path_yaws)
        step_distances = torch.diff(
            torch.cat((distances.new_zeros(1), distances))
        ).clamp_min(0.0)
        directions = torch.stack(
            (torch.cos(limited_yaws), torch.sin(limited_yaws)), dim=-1
        )
        positions = torch.cumsum(step_distances[:, None] * directions, dim=0)
        return positions, limited_yaws

    def plan(
        self,
        ego_observation,
        candidate_agents_future=None,
        strategy_depth=0,
        agent_observation=None,
        **kwargs,
    ):
        if candidate_agents_future is not None or strategy_depth != 0:
            raise ValueError("Native IDM currently implements frozen K=0 only")
        obs = ego_observation
        if "curr_speed" in obs:
            ego_speed = self._as_tensor(
                obs["curr_speed"], self.device
            ).reshape(-1)
        else:
            agent_hist = self._as_tensor(obs["agent_hist"], self.device)
            ego_speed = torch.linalg.norm(
                agent_hist[..., -1, 2:4], dim=-1
            ).reshape(-1)

        accel = self._idm_acceleration(obs, ego_speed)
        time_steps = torch.arange(
            1,
            self.horizon + 1,
            device=self.device,
            dtype=ego_speed.dtype,
        )
        speeds = torch.clamp(
            ego_speed[:, None]
            + accel[:, None] * self.dt * time_steps[None],
            min=0.0,
        )
        distances = torch.cumsum(speeds * self.dt, dim=1)
        paths = self._reference_paths(obs, ego_speed.shape[0])
        positions = []
        yaws = []
        for batch_idx, path in enumerate(paths):
            _, path_yaw = self._sample_path(path, distances[batch_idx])
            position, yaw = self._integrate_rate_limited_path(
                distances[batch_idx], path_yaw
            )
            positions.append(position)
            yaws.append(self._limit_yaw_rate(yaw))
        return Action(
            positions=torch.stack(positions),
            yaws=torch.stack(yaws).unsqueeze(-1),
        ), {
            "idm_acceleration": accel,
            "planner_input_speed_mps": ego_speed,
            "candidate_future_consumed": False,
        }


class STRIVEPlannerPolicy(EgoPlannerPolicy):
    """SafeSim STRIVE adapted to CCDiff's unified observation contract."""

    name = "strive"

    def __init__(
        self,
        device,
        profile="final_tuned_val_1",
        dt=0.1,
        horizon=52,
    ):
        super().__init__(device)
        if profile not in CONFIG_DICT:
            raise KeyError("Unknown STRIVE profile: {}".format(profile))
        self.profile = profile
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.cfg = deepcopy(DEF_CONFIG)
        self.cfg.update(deepcopy(CONFIG_DICT[profile]))
        lane_graphs = load_lane_graphs_trajdata("nuscenes")
        self.planner = HardcodeNuscPlanner(
            lane_graphs,
            PlannerConfig(**self.cfg),
        )

    @staticmethod
    def _as_tensor(value, device, dtype=torch.float32):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _rotation_only(transform):
        rotation = transform.clone()
        rotation[..., :2, 2] = 0
        rotation[..., 2, :] = 0
        rotation[..., 2, 2] = 1
        return rotation

    @staticmethod
    def _map_name(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        return str(value).split(":")[-1]

    @staticmethod
    def _resample_plan(
        plan,
        source_dt,
        target_dt,
        target_horizon,
        initial_speed=0.0,
        max_speed=20.0,
        max_accel=4.0,
    ):
        """Resample a STRIVE path while enforcing its speed/acceleration cap.

        STRIVE returns absolute local poses every 0.2 s.  Linear pose
        interpolation creates piecewise-constant 0.1 s speeds and artificial
        acceleration impulses.  Instead, interpolate the desired path speed,
        rate-limit it by ``accmax``, integrate distance once, and sample that
        distance along the same geometric path.
        """
        if plan.ndim == 3:
            if plan.shape[0] != 1:
                raise ValueError("STRIVE planner returned an unexpected batch")
            plan = plan[0]
        if plan.ndim != 2 or plan.shape[-1] != 4:
            raise ValueError("STRIVE plan must have shape [T,4]")
        initial = plan.new_tensor([[0.0, 0.0, 1.0, 0.0]])
        samples = torch.cat((initial, plan), dim=0)
        segment = samples[1:, :2] - samples[:-1, :2]
        segment_length = torch.linalg.norm(segment, dim=-1).clamp_min(1e-6)
        cumulative = torch.cat(
            (plan.new_zeros(1), torch.cumsum(segment_length, dim=0))
        )
        source_speed = (segment_length / float(source_dt)).clamp(
            min=0.0, max=float(max_speed)
        )

        current_speed = torch.as_tensor(
            initial_speed, device=plan.device, dtype=plan.dtype
        ).reshape(())
        current_speed = current_speed.clamp(0.0, float(max_speed))
        current_distance = plan.new_zeros(())
        target_distances = []
        for target_index in range(target_horizon):
            source_index = min(
                int((target_index * float(target_dt)) // float(source_dt)),
                source_speed.shape[0] - 1,
            )
            desired_speed = source_speed[source_index]
            speed_delta = (desired_speed - current_speed).clamp(
                min=-float(max_accel) * float(target_dt),
                max=float(max_accel) * float(target_dt),
            )
            next_speed = (current_speed + speed_delta).clamp(
                0.0, float(max_speed)
            )
            current_distance = current_distance + 0.5 * (
                current_speed + next_speed
            ) * float(target_dt)
            current_distance = current_distance.clamp(max=cumulative[-1])
            target_distances.append(current_distance)
            current_speed = next_speed
        target_distances = torch.stack(target_distances)

        segment_index = torch.searchsorted(
            cumulative[1:], target_distances, right=False
        ).clamp(max=segment_length.shape[0] - 1)
        ratio = (
            (target_distances - cumulative[segment_index])
            / segment_length[segment_index]
        ).clamp(0.0, 1.0)
        output = samples[segment_index] + ratio[:, None] * (
            samples[segment_index + 1] - samples[segment_index]
        )
        heading_norm = torch.linalg.norm(
            output[:, 2:4], dim=-1
        ).clamp_min(1e-6)
        output[:, 2:4] = output[:, 2:4] / heading_norm[:, None]
        return output

    def _valid_neighbors(self, obs, batch_idx, neighbor_hist):
        if "num_neigh" in obs:
            count = int(torch.as_tensor(obs["num_neigh"][batch_idx]).item())
            neighbor_hist = neighbor_hist[:count]
        valid = torch.isfinite(neighbor_hist[..., -1, :]).all(dim=-1)
        if "neigh_types" in obs:
            types = self._as_tensor(
                obs["neigh_types"][batch_idx], self.device
            )[: neighbor_hist.shape[0]]
            valid &= types > 0
        elif "all_other_agents_types" in obs:
            types = self._as_tensor(
                obs["all_other_agents_types"][batch_idx], self.device
            )[: neighbor_hist.shape[0]]
            valid &= types > 0
        return neighbor_hist[valid], valid

    def plan(
        self,
        ego_observation,
        candidate_agents_future=None,
        strategy_depth=0,
        agent_observation=None,
        **kwargs,
    ):
        if candidate_agents_future is not None or strategy_depth != 0:
            raise ValueError("STRIVE adapter currently implements frozen K=0 only")
        obs = ego_observation
        batch_size = int(obs["agent_hist"].shape[0])
        output_positions = []
        output_yaws = []
        raw_input_speed_max = []

        for batch_idx in range(batch_size):
            agent_hist_8d = self._as_tensor(
                obs["agent_hist"][batch_idx], self.device
            )
            neighbor_hist_8d = self._as_tensor(
                obs["neigh_hist"][batch_idx], self.device
            )
            neighbor_hist_8d, valid_neighbor = self._valid_neighbors(
                obs, batch_idx, neighbor_hist_8d
            )

            # This is the sole 8-D -> 9-D boundary in the STRIVE adapter.
            agent_hist = _ccdiff_state_8d_to_safesim_9d(agent_hist_8d)
            neighbor_hist = _ccdiff_state_8d_to_safesim_9d(neighbor_hist_8d)
            all_local = torch.cat((agent_hist.unsqueeze(0), neighbor_hist), dim=0)
            history_index = agent_hist.shape[0] - 1
            all_world = StateTensor.from_array(
                all_local,
                format=SAFESIM_STATE_FORMAT,
            ).as_format("x,y,c,s,xd,yd")
            world_from_agent = self._as_tensor(
                obs["world_from_agent"][batch_idx], self.device
            )
            all_world[..., :2] = GeoUtils.transform_points_tensor(
                all_world[..., :2], world_from_agent
            )
            rotation = self._rotation_only(world_from_agent)
            all_world[..., 2:4] = GeoUtils.transform_points_tensor(
                all_world[..., 2:4], rotation
            )
            all_world[..., 4:6] = GeoUtils.transform_points_tensor(
                all_world[..., 4:6], rotation
            )
            all_world_np = all_world.detach().cpu().numpy()
            speed = np.linalg.norm(
                all_world_np[..., 4:6], axis=-1, keepdims=True
            )
            raw_input_speed_max.append(float(np.nanmax(speed)))
            speed = np.clip(speed, 0.0, float(self.cfg["smax"]))
            initial_state = np.concatenate(
                (all_world_np[..., :4], speed, speed * 0.0), axis=-1
            )

            ego_extent = self._as_tensor(
                obs["extent"][batch_idx, :2], self.device
            ).detach().cpu().numpy()
            other_extents = None
            if "all_other_agents_extents" in obs:
                candidate_extents = self._as_tensor(
                    obs["all_other_agents_extents"][batch_idx], self.device
                )[: valid_neighbor.shape[0]]
                other_extents = candidate_extents[valid_neighbor, :2].detach().cpu().numpy()
            if other_extents is None:
                other_extents = np.tile(
                    np.asarray(
                        [self.cfg["vehicle_length"], self.cfg["vehicle_width"]]
                    ),
                    (neighbor_hist.shape[0], 1),
                )
            vehicle_attributes = np.concatenate(
                (ego_extent[None], other_extents), axis=0
            )
            map_value = obs["map_names"][batch_idx]
            map_name = self._map_name(map_value)
            batch_mask = np.zeros(initial_state.shape[0], dtype=np.int64)
            self.planner.reset(
                initial_state[:, history_index],
                vehicle_attributes,
                batch_mask,
                1,
                [map_name],
                ego_idx=0,
            )

            future_len = int(self.cfg["fut_len"])
            planner_dt = float(self.cfg["dt"])
            agent_times = np.arange(1, future_len + 1) * planner_dt
            current_velocity = all_world_np[1:, history_index, 4:6]
            current_state = initial_state[1:, history_index, :4]
            neighbor_positions = (
                current_velocity[:, None] * agent_times[None, :, None]
                + current_state[:, None, :2]
            )
            neighbor_headings = np.repeat(
                current_state[:, None, 2:4], repeats=future_len, axis=1
            )
            neighbor_future = np.concatenate(
                (neighbor_positions, neighbor_headings), axis=-1
            )
            pointer = np.asarray([0, len(neighbor_future)])
            plan_world = self.planner.rollout(
                neighbor_future,
                agent_times,
                pointer,
                agent_times,
            )

            # Always use the transform belonging to this ego observation.
            agent_from_world = self._as_tensor(
                obs["agent_from_world"][batch_idx], self.device
            )
            plan_local = plan_world.to(
                device=agent_from_world.device,
                dtype=agent_from_world.dtype,
            )
            plan_local[..., :2] = GeoUtils.transform_points_tensor(
                plan_local[..., :2], agent_from_world
            )
            rotation_back = self._rotation_only(agent_from_world)
            plan_local[..., 2:4] = GeoUtils.transform_points_tensor(
                plan_local[..., 2:4], rotation_back
            )
            plan_local = self._resample_plan(
                plan_local,
                source_dt=planner_dt,
                target_dt=self.dt,
                target_horizon=self.horizon,
                initial_speed=float(speed[0, history_index, 0]),
                max_speed=float(self.cfg["smax"]),
                max_accel=float(self.cfg["accmax"]),
            )
            output_positions.append(plan_local[:, :2])
            output_yaws.append(
                torch.atan2(plan_local[:, 3], plan_local[:, 2])[:, None]
            )

        return Action(
            positions=torch.stack(output_positions),
            yaws=torch.stack(output_yaws),
        ), {
            "candidate_future_consumed": False,
            "strive_profile": self.profile,
            "strive_smax_mps": float(self.cfg["smax"]),
            "strive_accmax_mps2": float(self.cfg["accmax"]),
            "strive_raw_input_speed_max_mps": torch.as_tensor(
                raw_input_speed_max, device=self.device
            ),
        }


def build_ego_planner(name, device, strive_profile="final_tuned_val_1"):
    normalized = name.strip().lower()
    if normalized == "idm":
        return IDMPlannerPolicy(device=device)
    if normalized == "strive":
        return STRIVEPlannerPolicy(
            device=device,
            profile=strive_profile,
        )
    if normalized in {"diffusion", "diffusion_planner", "dp"}:
        from ccdiff.planners.official_nuplan import DiffusionPlannerPolicy

        return DiffusionPlannerPolicy(device=device)
    if normalized in {"flow", "flow_planner", "flow_matching"}:
        from ccdiff.planners.official_nuplan import FlowPlannerPolicy

        return FlowPlannerPolicy(device=device)
    raise ValueError("Unsupported ego planner: {}".format(name))
