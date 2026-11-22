"""Check composed trajectories against actual HDF5 states and tracker output."""
import json
from pathlib import Path
import h5py
import numpy as np


def audit(directory, steps):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / 'common_execution_steps.jsonl').read_text().splitlines()]
    tracker = [json.loads(line) for line in (directory / 'native_tracker_execution.jsonl').read_text().splitlines()]
    expected_steps = list(range(0, steps, 5))
    if [row['step_index'] for row in rows] != expected_steps or [row['step_index'] for row in tracker] != expected_steps:
        raise ValueError('Incomplete common execution or native tracker records')
    paths = list(directory.rglob('data.hdf5'))
    if len(paths) != 1:
        raise ValueError('Expected one actual rollout HDF5')
    with h5py.File(paths[0]) as handle:
        if len(handle) != 1:
            raise ValueError('Expected one actual scene')
        group = handle[next(iter(handle))]
        xy, yaw, tracks, available = [np.asarray(group[key]) for key in
                                     ['centroid', 'yaw', 'track_id', 'availability']]
    if xy.shape[1] != steps or not np.all(tracks == tracks[:, :1]):
        raise ValueError('Actual frame budget or actor track identity changed')
    mapping = {int(track): index for index, track in enumerate(tracks[:, 0])}
    if len(mapping) != len(xy):
        raise ValueError('Duplicate actual track IDs')
    errors = dict(position_m=0., yaw_rad=0., tracked_ego_local_m=0., tracked_ego_local_yaw_rad=0.)
    slots, missing = 0, 0
    for row, supplied in zip(rows, tracker):
        if row['names'] != rows[0]['names']:
            raise ValueError('Actor identity ordering changed')
        indices = [mapping[int(track)] for track in row['track_ids']]
        ego_xy = np.asarray(row['supplied_ego_local_positions'], float)[0]
        ego_yaw = np.asarray(row['supplied_ego_local_yaws'], float)[0]
        errors['tracked_ego_local_m'] = max(errors['tracked_ego_local_m'],
            float(np.max(np.abs(ego_xy - np.asarray(supplied['tracked_positions'], float)[:5]))))
        errors['tracked_ego_local_yaw_rad'] = max(errors['tracked_ego_local_yaw_rad'],
            float(np.max(np.abs(ego_yaw - np.asarray(supplied['tracked_yaws'], float)[:5]))))
        expected = np.asarray(row['expected_world_positions'], float)
        heading = np.asarray(row['expected_world_yaws'], float)
        for offset in range(5):
            frame = row['step_index'] + offset + 1
            if frame >= steps:
                continue
            actual, actual_yaw = xy[indices, frame], yaw[indices, frame].reshape(-1)
            finite = np.isfinite(expected[:, offset]).all(-1) & np.isfinite(heading[:, offset])
            actual_finite = np.isfinite(actual).all(-1) & np.isfinite(actual_yaw)
            if not np.array_equal(finite, actual_finite):
                raise AssertionError('Actual occupancy differs at frame ' + str(frame))
            if not np.array_equal(available[indices, frame].reshape(-1).astype(bool), finite):
                raise AssertionError('Logged availability differs from actual occupancy')
            if finite.any():
                errors['position_m'] = max(errors['position_m'],
                    float(np.max(np.abs(actual[finite] - expected[finite, offset]))))
                delta = actual_yaw[finite] - heading[finite, offset]
                errors['yaw_rad'] = max(errors['yaw_rad'],
                    float(np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))))
            slots += len(indices)
            missing += int((~finite).sum())
    if (errors['position_m'] > .005 or errors['yaw_rad'] > .00005
            or errors['tracked_ego_local_m'] > .00001 or errors['tracked_ego_local_yaw_rad'] > .00001):
        raise AssertionError('Actual execution differs from composed/supplied trajectories: ' + repr(errors))
    return dict(passed=True, calls=len(rows), frames=steps, compared_frames=steps - 1,
                final_frame_not_recorded=True, compared_pose_slots=slots,
                absent_pose_slots_verified=missing, max_errors=errors,
                tracker_protocol='native_nuplan_lqr_bicycle_v1', physical_validity_not_implied=True)
