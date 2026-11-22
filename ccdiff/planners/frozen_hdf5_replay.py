"""Replay one generated traffic scene while allowing ego replanning."""

import h5py
import numpy as np
import torch

from tbsim.policies.base import Policy
from tbsim.policies.common import Action


class FrozenHDF5TrafficPolicy(Policy):
    """Convert saved world trajectories into each current agent frame.

    The ego row is returned for a unified scene action but is overwritten by
    PlannerOverrideRolloutWrapper.  Non-ego rows are the exact saved generated
    traffic trajectory, so two ego planners can be tested against one frozen
    counterfactual without regenerating traffic.
    """

    def __init__(self, hdf5_path, device, frame_offset=0):
        super().__init__(device)
        self.hdf5_path = str(hdf5_path)
        self.frame_offset = int(frame_offset)
        with h5py.File(self.hdf5_path, "r") as handle:
            groups = list(handle)
            if len(groups) != 1:
                raise ValueError("frozen replay requires exactly one HDF5 scene")
            group = handle[groups[0]]
            self.group_name = groups[0]
            self.centroid = np.asarray(group["centroid"], dtype=np.float32)
            self.yaw = np.asarray(group["yaw"], dtype=np.float32)
            if self.yaw.ndim == 3 and self.yaw.shape[-1] == 1:
                self.yaw = self.yaw[..., 0]
            track_id = np.asarray(group["track_id"])
            is_ego = np.asarray(group["is_ego"])
            self.track_id = track_id[:, 0] if track_id.ndim > 1 else track_id
            self.is_ego = is_ego[:, 0] if is_ego.ndim > 1 else is_ego
        if self.centroid.ndim != 3 or self.centroid.shape[-1] != 2:
            raise ValueError("expected centroid [M,F,2]")
        if self.yaw.shape != self.centroid.shape[:2]:
            raise ValueError("yaw and centroid histories disagree")
        ego_rows = np.flatnonzero(self.is_ego.astype(bool))
        if len(ego_rows) != 1:
            raise ValueError("frozen replay requires exactly one saved ego")
        self.saved_ego_index = int(ego_rows[0])
        self._printed_attestation = False

    def eval(self):
        return self

    @staticmethod
    def _names(obs):
        values = obs.get("agent_name")
        if values is None:
            return None
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        return [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in np.asarray(values).reshape(-1).tolist()
        ]

    def _saved_rows(self, obs, count):
        names = self._names(obs)
        track_ids = obs.get("track_id")
        if torch.is_tensor(track_ids):
            track_ids = track_ids.detach().cpu().numpy()
        track_ids = np.asarray(track_ids).reshape(-1)
        lookup = {int(track): index for index, track in enumerate(self.track_id)}
        rows = []
        for index in range(count):
            if names is not None and names[index] == "ego":
                rows.append(self.saved_ego_index)
                continue
            track = int(track_ids[index])
            if track not in lookup:
                raise KeyError("track {} absent from frozen HDF5".format(track))
            rows.append(lookup[track])
        return rows

    def get_action(self, obs, step_index=None, **kwargs):
        if step_index is None:
            raise ValueError("frozen replay requires step_index")
        target_positions = torch.as_tensor(
            obs["target_positions"], device=self.device, dtype=torch.float32
        )
        horizon = int(target_positions.shape[1])
        count = int(target_positions.shape[0])
        rows = self._saved_rows(obs, count)
        frames = np.clip(
            np.arange(horizon) + int(step_index) + self.frame_offset,
            0,
            self.centroid.shape[1] - 1,
        )
        world_positions = torch.as_tensor(
            self.centroid[np.asarray(rows)[:, None], frames[None, :]],
            device=self.device,
            dtype=torch.float32,
        )
        world_yaws = torch.as_tensor(
            self.yaw[np.asarray(rows)[:, None], frames[None, :]],
            device=self.device,
            dtype=torch.float32,
        )
        world_from_agent = torch.as_tensor(
            obs["world_from_agent"], device=self.device, dtype=torch.float32
        )
        if world_from_agent.ndim == 4:
            world_from_agent = world_from_agent[0]
        agent_from_world = torch.linalg.inv(world_from_agent)
        ones = torch.ones(
            count, horizon, 1, device=self.device, dtype=torch.float32
        )
        homogeneous = torch.cat((world_positions, ones), dim=-1)
        positions = torch.matmul(
            homogeneous, agent_from_world.transpose(-1, -2)
        )[..., :2]
        world_heading = torch.atan2(
            world_from_agent[:, 1, 0], world_from_agent[:, 0, 0]
        )
        yaws = torch.atan2(
            torch.sin(world_yaws - world_heading[:, None]),
            torch.cos(world_yaws - world_heading[:, None]),
        )[..., None]
        if not self._printed_attestation:
            print(
                "FROZEN_HDF5_TRAFFIC_ATTESTATION group={} frame_offset={} "
                "nonreactive_traffic=True source={}".format(
                    self.group_name, self.frame_offset, self.hdf5_path
                )
            )
            self._printed_attestation = True
        action = Action(positions=positions, yaws=yaws)
        return action, {
            "action_samples": Action(
                positions=positions[:, None], yaws=yaws[:, None]
            ).to_dict()
        }
