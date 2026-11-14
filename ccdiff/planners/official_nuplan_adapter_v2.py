"""SafeSim adapters for the official Diffusion Planner and Flow Planner.

Both models consume NuPlan-style rear-axle coordinates. SafeSim observations
are agent-centred. This module makes the frame conversion and the limited
map/route proxy explicit before handing features to the isolated model worker.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from ccdiff.planners.base import EgoPlannerPolicy
from tbsim.policies.common import Action


class OfficialNuPlanPlanner(EgoPlannerPolicy):
    supports_candidate_future = False
    kind = None
    name = None

    def __init__(self, device, horizon=52, dt=0.1):
        super().__init__(device)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.base_seed = int(os.environ.get("CCDIFF_PLANNER_SEED", "1"))
        self._worker_log = None
        self._start_worker()

    def _start_worker(self):
        planner_home = Path(os.environ.get(
            "PLANNER_HOME", "/root/path/to/workspace/home/planner_sources_20260810"
        ))
        source_root = planner_home / "source_no_assets"
        worker = Path(__file__).with_name("official_nuplan_worker.py")
        python = os.environ.get(
            "PLANNER_PYTHON",
            "/root/path/to/workspace/home/miniconda3/envs/ctg/bin/python",
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CCDIFF_PLANNER_GPU", "1")
        env["PLANNER_SOURCES_ROOT"] = str(source_root)
        nuplan_paths = [
            str(source_root / "Diffusion-Planner-main"),
            str(source_root / "Flow-Planner-main"),
            str(planner_home / "nuplan_source" / "nuplan-devkit-master"),
            "/root/path/to/workspace/home/venvs/nuplan_planners_20260810/lib/python3.9/site-packages",
        ]
        env["PYTHONPATH"] = os.pathsep.join(
            nuplan_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        log_dir = Path(os.environ.get("CCDIFF_PLANNER_LOG_DIR", "/tmp"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self._worker_log = open(
            log_dir / "official_{}_planner_worker_{}.log".format(self.kind, os.getpid()),
            "a",
            encoding="utf-8",
        )
        self._worker = subprocess.Popen(
            [python, str(worker), self.kind],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._worker_log,
            text=True,
            bufsize=1,
            env=env,
        )
        while True:
            line = self._worker.stdout.readline()
            if not line:
                raise RuntimeError("planner worker exited during startup; see {}".format(self._worker_log.name))
            try:
                status = json.loads(line)
            except ValueError:
                self._worker_log.write(line)
                self._worker_log.flush()
                continue
            if status.get("ready"):
                self.worker_device = status.get("device")
                break
            if status.get("error"):
                raise RuntimeError("planner worker startup failed: {}".format(status))

    @staticmethod
    def _numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _get_batch_value(obs, key, index, default=None):
        value = obs.get(key, default)
        if value is None:
            return None
        value = OfficialNuPlanPlanner._numpy(value)
        return value[index]

    @staticmethod
    def _history_from_parsed(obs, batch_idx, agent_idx=None):
        if agent_idx is None:
            positions = OfficialNuPlanPlanner._get_batch_value(obs, "history_positions", batch_idx)
            yaws = OfficialNuPlanPlanner._get_batch_value(obs, "history_yaws", batch_idx)
            speeds = OfficialNuPlanPlanner._get_batch_value(obs, "history_speeds", batch_idx)
            avail = OfficialNuPlanPlanner._get_batch_value(obs, "history_availabilities", batch_idx)
        else:
            positions = OfficialNuPlanPlanner._get_batch_value(obs, "all_other_agents_history_positions", batch_idx)[agent_idx]
            yaws = OfficialNuPlanPlanner._get_batch_value(obs, "all_other_agents_history_yaws", batch_idx)[agent_idx]
            speeds = OfficialNuPlanPlanner._get_batch_value(obs, "all_other_agents_history_speeds", batch_idx)[agent_idx]
            avail = OfficialNuPlanPlanner._get_batch_value(obs, "all_other_agents_history_availabilities", batch_idx)[agent_idx]
        positions = np.asarray(positions, dtype=np.float32)
        yaws = np.asarray(yaws, dtype=np.float32).reshape(-1)
        speeds = np.asarray(speeds, dtype=np.float32).reshape(-1)
        avail = np.asarray(avail, dtype=bool).reshape(-1)
        n = min(len(positions), len(yaws), len(speeds), len(avail))
        positions, yaws, speeds, avail = positions[-n:], yaws[-n:], speeds[-n:], avail[-n:]
        vx = speeds * np.cos(yaws)
        vy = speeds * np.sin(yaws)
        ax = np.gradient(vx, 0.1) if n > 1 else np.zeros_like(vx)
        ay = np.gradient(vy, 0.1) if n > 1 else np.zeros_like(vy)
        raw = np.stack((positions[:, 0], positions[:, 1], vx, vy, ax, ay,
                        np.sin(yaws), np.cos(yaws)), axis=-1).astype(np.float32)
        raw[~avail] = np.nan
        return raw

    def _agent_history(self, obs, batch_idx, agent_idx=None):
        key = "agent_hist" if agent_idx is None else "neigh_hist"
        raw = self._get_batch_value(obs, key, batch_idx)
        if raw is None:
            raw = self._history_from_parsed(obs, batch_idx, agent_idx)
        elif agent_idx is not None:
            raw = raw[agent_idx]
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[-1] < 8:
            raise ValueError("{} must be [T,8+], got {}".format(key, raw.shape))
        return np.nan_to_num(raw[:, :8], nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _fixed_history(raw, steps=21):
        if raw.shape[0] >= steps:
            return raw[-steps:]
        pad = np.zeros((steps - raw.shape[0], raw.shape[-1]), dtype=np.float32)
        return np.concatenate((pad, raw), axis=0)

    @staticmethod
    def _rear_position(center_xy, yaw, vehicle_length, ego_length, is_ego):
        half_length = 0.5 * float(vehicle_length)
        ego_anchor = np.asarray([0.5 * float(ego_length), 0.0], dtype=np.float32)
        orientation = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
        if is_ego:
            return center_xy - half_length * orientation + ego_anchor
        # Official neighbor features describe tracked-object box centers in
        # the ego rear-axle frame; only the coordinate origin changes.
        return center_xy + ego_anchor

    @staticmethod
    def _type_one_hot(agent_type, namespace="trajdata"):
        if isinstance(agent_type, bytes):
            agent_type = agent_type.decode("utf-8", errors="ignore")
        if isinstance(agent_type, str):
            value = agent_type.lower()
            if "ped" in value:
                return (0.0, 1.0, 0.0)
            if "bicycle" in value or "bike" in value or "motorcycle" in value:
                return (0.0, 0.0, 1.0)
            return (1.0, 0.0, 0.0)
        try:
            value = int(agent_type)
        except (TypeError, ValueError):
            value = 1
        if namespace == "l5kit":
            if value == 14:
                return (0.0, 1.0, 0.0)
            if value in (10, 11):
                return (0.0, 0.0, 1.0)
            return (1.0, 0.0, 0.0)
        if namespace != "trajdata":
            raise ValueError("Unrecognized agent type namespace: " + namespace)
        if value == 2:
            return (0.0, 1.0, 0.0)
        if value in (3, 4):
            return (0.0, 0.0, 1.0)
        return (1.0, 0.0, 0.0)

    @staticmethod
    def _lane_feature(points, ego_length, n_points=20):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[-1] < 3:
            return None
        valid = np.isfinite(points[:, :3]).all(axis=-1)
        valid &= (np.abs(points[:, 0]) + np.abs(points[:, 1]) + np.abs(points[:, 2])) > 1e-5
        points = points[valid]
        if len(points) < 2:
            return None
        points = points[np.argsort(points[:, 0], kind="stable")]
        keep = np.concatenate(([True], np.linalg.norm(np.diff(points[:, :2], axis=0), axis=-1) > 1e-3))
        points = points[keep]
        if len(points) < 2:
            return None
        sample_idx = np.linspace(0, len(points) - 1, n_points)
        left = np.floor(sample_idx).astype(int)
        right = np.minimum(left + 1, len(points) - 1)
        ratio = (sample_idx - left)[:, None]
        sampled = points[left] * (1.0 - ratio) + points[right] * ratio
        x = sampled[:, 0] + 0.5 * float(ego_length)
        y = sampled[:, 1]
        yaw = sampled[:, 2]
        center = np.stack((x, y), axis=-1)
        tangent = np.zeros_like(center)
        tangent[:-1] = center[1:] - center[:-1]
        half_width = 1.75
        left_offset = np.stack((-np.sin(yaw), np.cos(yaw)), axis=-1) * half_width
        right_offset = -left_offset
        feature = np.zeros((n_points, 12), dtype=np.float32)
        feature[:, 0:2] = center
        feature[:, 2:4] = tangent
        feature[:, 4:6] = left_offset
        feature[:, 6:8] = right_offset
        feature[:, 11] = 1.0  # unknown traffic-light state, shared proxy
        return feature

    def _build_features(self, obs):
        agent_hist = self._get_batch_value(obs, "agent_hist", 0)
        if agent_hist is None:
            batch_size = int(self._numpy(obs["history_positions"]).shape[0])
        else:
            batch_size = int(self._numpy(obs["agent_hist"]).shape[0])
        ego_extent = self._numpy(obs["extent"]).astype(np.float32)
        neighbor_extent = self._numpy(obs.get("all_other_agents_extents", obs.get("neigh_extents", np.zeros((batch_size, 32, 3), dtype=np.float32)))).astype(np.float32)
        # Parsing maps trajdata IDs to L5Kit IDs in place. Prefer the explicit
        # parsed field to avoid confusing vehicle=3 with trajdata bicycle=3.
        parsed_types = "all_other_agents_types" in obs
        neighbor_types = obs.get("all_other_agents_types", obs.get("neigh_types"))
        if neighbor_types is not None:
            neighbor_types = self._numpy(neighbor_types)
        counts = self._numpy(obs.get("num_neigh", np.full((batch_size,), neighbor_extent.shape[1]))).reshape(-1)
        lane_data = obs.get("extras", {}).get("closest_lane_point")
        if lane_data is not None:
            lane_data = self._numpy(lane_data)

        neighbors = np.zeros((batch_size, 32, 21, 11), dtype=np.float32)
        ego_past = np.zeros((batch_size, 21, 14), dtype=np.float32)
        ego_current_dp = np.zeros((batch_size, 10), dtype=np.float32)
        ego_current_flow = np.zeros((batch_size, 16), dtype=np.float32)
        lanes = np.zeros((batch_size, 70, 20, 12), dtype=np.float32)
        routes = np.zeros((batch_size, 25, 20, 12), dtype=np.float32)
        lane_speed = np.full((batch_size, 70, 1), 13.9, dtype=np.float32)
        route_speed = np.full((batch_size, 25, 1), 13.9, dtype=np.float32)
        lane_has_speed = np.zeros((batch_size, 70, 1), dtype=bool)
        route_has_speed = np.zeros((batch_size, 25, 1), dtype=bool)
        static = np.zeros((batch_size, 5, 10), dtype=np.float32)

        for b in range(batch_size):
            ego_length = float(ego_extent[b, 0]) if ego_extent.shape[-1] else 4.8
            ego_width = float(ego_extent[b, 1]) if ego_extent.shape[-1] > 1 else 2.0
            ego_raw = self._fixed_history(self._agent_history(obs, b))
            ego_yaw = np.arctan2(ego_raw[:, 6], ego_raw[:, 7])
            ego_xy = self._rear_position(ego_raw[:, :2], ego_yaw, ego_length, ego_length, True)
            ego_hist = np.zeros((21, 14), dtype=np.float32)
            ego_hist[:, 0:2] = ego_xy
            ego_hist[:, 2] = np.cos(ego_yaw)
            ego_hist[:, 3] = np.sin(ego_yaw)
            ego_hist[:, 4:8] = ego_raw[:, 2:6]
            ego_hist[:, 8] = ego_width
            ego_hist[:, 9] = ego_length
            ego_hist[:, 10] = 1.0
            ego_past[b] = ego_hist
            ego_current_dp[b, 0:4] = ego_hist[-1, 0:4]
            ego_current_dp[b, 4:8] = ego_hist[-1, 4:8]
            ego_current_dp[b, 8:10] = 0.0
            ego_current_flow[b, 0:8] = ego_hist[-1, 0:8]
            ego_current_flow[b, 10] = ego_width
            ego_current_flow[b, 11] = ego_length
            ego_current_flow[b, 12] = 1.0

            num_neighbors = min(int(counts[b]), 32)
            candidates = []
            for n in range(num_neighbors):
                raw = self._agent_history(obs, b, n)
                extent = neighbor_extent[b, n] if n < neighbor_extent.shape[1] else np.asarray([4.8, 2.0, 1.5])
                length = float(extent[0]) if len(extent) else 4.8
                width = float(extent[1]) if len(extent) > 1 else 2.0
                raw = self._fixed_history(raw)
                yaw = np.arctan2(raw[:, 6], raw[:, 7])
                rear_xy = self._rear_position(raw[:, :2], yaw, length, ego_length, False)
                actor_type = neighbor_types[b, n] if neighbor_types is not None and n < neighbor_types.shape[1] else 1
                type_bits = self._type_one_hot(actor_type, "l5kit" if parsed_types else "trajdata")
                feat = np.zeros((21, 11), dtype=np.float32)
                feat[:, 0:2] = rear_xy
                feat[:, 2] = np.cos(yaw)
                feat[:, 3] = np.sin(yaw)
                feat[:, 4:6] = raw[:, 2:4]
                feat[:, 6] = width
                feat[:, 7] = length
                feat[:, 8:11] = type_bits
                history_valid = np.linalg.norm(raw[:, 6:8], axis=-1) > .5
                available = self._get_batch_value(obs, "all_other_agents_history_availabilities", b)
                if available is not None:
                    mask = np.asarray(available[n], dtype=bool).reshape(-1)[-21:]
                    history_valid &= np.pad(mask, (21 - len(mask), 0), constant_values=False)
                feat[~history_valid] = 0
                if not history_valid[-1]:
                    continue
                distance = float(np.linalg.norm(rear_xy[-1]))
                candidates.append((distance, feat))
            candidates.sort(key=lambda item: item[0])
            for n, (_, feat) in enumerate(candidates[:32]):
                neighbors[b, n] = feat

            if lane_data is not None:
                lanes_this_scene = lane_data[b]
                converted = []
                for lane in lanes_this_scene:
                    feature = self._lane_feature(lane, ego_length)
                    if feature is not None:
                        center = feature[:, :2]
                        score = float(np.min(np.linalg.norm(center - np.asarray([ego_length * 0.5, 0.0]), axis=-1)))
                        tangent_yaw = np.arctan2(feature[:, 3], feature[:, 2])
                        score += 2.0 * float(np.median(np.abs(tangent_yaw)))
                        converted.append((score, feature))
                converted.sort(key=lambda item: item[0])
                for i, (_, feature) in enumerate(converted[:70]):
                    lanes[b, i] = feature
                # SafeSim exposes nearby centerlines, not NuPlan route IDs.
                # Use the closest current lane as the same deterministic route
                # proxy for both official planners and all generation methods.
                if converted:
                    routes[b, 0] = converted[0][1]
                    route_has_speed[b, 0] = False
                    lane_has_speed[b, :len(converted[:70])] = False

        return {
            "neighbors": neighbors,
            "static": static,
            "lanes": lanes,
            "routes": routes,
            "lane_speed": lane_speed,
            "lane_has_speed": lane_has_speed,
            "route_speed": route_speed,
            "route_has_speed": route_has_speed,
            "ego_past": ego_past,
            "ego_current_dp": ego_current_dp,
            "ego_current_flow": ego_current_flow,
        }

    def plan(self, ego_observation, candidate_agents_future=None, strategy_depth=0,
             agent_observation=None, step_index=0, **kwargs):
        if candidate_agents_future is not None or strategy_depth != 0:
            raise ValueError("Official NuPlan planners currently use the shared K=0 interface")
        features = self._build_features(ego_observation)
        request = {key: value.tolist() for key, value in features.items()}
        request["seed"] = self.base_seed + int(step_index)
        self._worker.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._worker.stdin.flush()
        line = self._worker.stdout.readline()
        if not line:
            raise RuntimeError("planner worker stopped; see {}".format(self._worker_log.name))
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError("official planner inference failed: {}".format(result))
        trajectory = torch.as_tensor(result["trajectory"], device=self.device, dtype=torch.float32)
        trajectory = trajectory[:, :self.horizon]
        yaw = torch.atan2(trajectory[..., 3], trajectory[..., 2])
        length = torch.as_tensor(self._numpy(ego_observation["extent"])[..., 0], device=self.device, dtype=trajectory.dtype)
        # Model coordinates are rear-axle anchored; SafeSim actions are
        # centroid-relative. Convert the complete pose so turning actions stay
        # geometrically consistent with their reported yaw.
        half = length[:, None] * 0.5
        positions = trajectory[..., :2] + torch.stack((half * (torch.cos(yaw) - 1.0), half * torch.sin(yaw)), dim=-1)
        action = Action(positions=positions, yaws=yaw.unsqueeze(-1))
        return action, {
            "planner": self.name,
            "coordinate_input": "NuPlan rear-axle local frame",
            "coordinate_output": "SafeSim centroid-relative local frame",
            "route_source": "closest_lane_point first-lane proxy; route IDs unavailable",
            "speed_limit_source": "default 13.9 m/s",
            "traffic_light": "unknown",
            "candidate_future_consumed": "false",
            "adapter_version": "v2_neighbor_center_and_type_namespace",
            "ego_rear_axle_offset_source": "legacy_half_length_proxy_not_calibrated_axle_geometry",
        }

    def close(self):
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)
        if self._worker_log is not None:
            self._worker_log.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DiffusionPlannerPolicy(OfficialNuPlanPlanner):
    kind = "diffusion"
    name = "diffusion_planner"


class FlowPlannerPolicy(OfficialNuPlanPlanner):
    kind = "flow"
    name = "flow_planner"
