import h5py
import numpy as np
import torch

from ccdiff.planners.frozen_hdf5_replay import FrozenHDF5TrafficPolicy


def test_replays_saved_world_traffic_in_current_agent_frame(tmp_path):
    path = tmp_path / "traffic.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("scene-test_31")
        group.create_dataset(
            "centroid",
            data=np.asarray(
                [
                    [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                    [[11.0, 0.0], [12.0, 0.0], [13.0, 0.0]],
                ],
                dtype=np.float32,
            ),
        )
        group.create_dataset("yaw", data=np.zeros((2, 3), dtype=np.float32))
        group.create_dataset("track_id", data=np.asarray([[-1], [7]]))
        group.create_dataset("is_ego", data=np.asarray([[1], [0]]))

    transforms = torch.eye(3).repeat(2, 1, 1)
    transforms[1, 0, 2] = 10.0
    obs = {
        "target_positions": torch.zeros(2, 2, 2),
        "track_id": torch.tensor([-1, 7]),
        "agent_name": ["ego", "vehicle-7"],
        "world_from_agent": transforms,
    }
    policy = FrozenHDF5TrafficPolicy(path, device="cpu")
    action, info = policy.get_action(obs, step_index=0)

    torch.testing.assert_close(
        action.positions,
        torch.tensor([[[1.0, 0.0], [2.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]]]),
    )
    assert action.yaws.shape == (2, 2, 1)
    assert info["action_samples"]["positions"].shape == (2, 1, 2, 2)
