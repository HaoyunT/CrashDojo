"""Record expected world poses for independent comparison with executed logs."""
import json
from pathlib import Path

import numpy as np


def expected_poses(local, headings, centers, angles, action_dtype, state_dtype):
    local = np.asarray(local, dtype=action_dtype)
    headings = np.asarray(headings, dtype=action_dtype)
    world = np.empty(local.shape, dtype=state_dtype)
    world_yaw = np.empty(headings.shape, dtype=state_dtype)
    for actor, angle in enumerate(angles):
        # Use the simulator's yaw and centroid directly. Its float32 inverse
        # transform can have a different translation at large map coordinates.
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        for frame in range(local.shape[1]):
            world[actor, frame] = local[actor, frame] @ rotation + centers[actor]
            world_yaw[actor, frame] = angle + headings[actor, frame]
    return world, world_yaw


def record(directory, step, names, observation, positions, yaws, audit, *, action_dtype,
           selected_prefixes=None):
    local = positions[:, :5].detach().cpu().numpy()
    headings = yaws[:, :5, 0].detach().cpu().numpy()
    centers = observation['centroid'].detach().cpu().numpy()
    angles = observation['yaw'].detach().cpu().numpy().reshape(-1)
    state_dtype = observation['agent_fut'].detach().cpu().numpy().dtype
    world, headings = expected_poses(local, headings, centers, angles, action_dtype, state_dtype)
    ego = audit['protected_ego_index']
    if names[ego] != 'ego' or names.count('ego') != 1:
        raise ValueError('Protected ego index does not identify the unique ego')

    def nullable(values):
        return np.where(np.isfinite(values), values, None).tolist()

    tracks = observation['track_id'].detach().cpu().numpy().reshape(-1)
    value = dict(step_index=int(step), names=list(names), track_ids=tracks.tolist(), transport=audit,
                 selected_prefixes=list(selected_prefixes or []),
                 action_execution_dtype=np.dtype(action_dtype).name,
                 state_assignment_dtype=np.dtype(state_dtype).name,
                 supplied_ego_local_positions=local[ego:ego+1].tolist(),
                 supplied_ego_local_yaws=yaws[ego:ego+1, :5].detach().cpu().numpy().tolist(),
                 expected_world_positions=nullable(world), expected_world_yaws=nullable(headings))
    with (Path(directory) / 'common_execution_steps.jsonl').open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
