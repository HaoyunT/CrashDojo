import numpy as np
import pytest
import torch

from tbsim.policies.wrappers import select_fixed_native_joint_mode_action


def test_agent_centric_fixed_joint_mode_preserves_ego_and_uses_native_rows():
    samples = torch.arange(3 * 4 * 2 * 2, dtype=torch.float32).reshape(3, 4, 2, 2)
    yaws = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2, 1)
    baseline = torch.full((3, 2, 2), -1.0)
    baseline_yaw = torch.full((3, 2, 1), -2.0)
    positions, selected_yaws, audit = select_fixed_native_joint_mode_action(
        samples, yaws, baseline, baseline_yaw, ego_indices=[0], sample_index=2
    )
    torch.testing.assert_close(positions[0], baseline[0])
    torch.testing.assert_close(selected_yaws[0], baseline_yaw[0])
    torch.testing.assert_close(positions[1:], samples[1:, 2])
    torch.testing.assert_close(selected_yaws[1:], yaws[1:, 2])
    assert audit["native_joint_exact"]
    assert audit["full_horizon_verifier_required"]
    assert audit["non_ego_agent_count"] == 2


def test_scene_centric_numpy_fixed_joint_mode_uses_joint_sample_axis():
    samples = np.arange(1 * 3 * 4 * 2 * 2, dtype=np.float32).reshape(1, 3, 4, 2, 2)
    yaws = np.arange(1 * 3 * 4 * 2, dtype=np.float32).reshape(1, 3, 4, 2, 1)
    baseline = np.full((1, 4, 2, 2), -1.0, dtype=np.float32)
    baseline_yaw = np.full((1, 4, 2, 1), -2.0, dtype=np.float32)
    positions, selected_yaws, audit = select_fixed_native_joint_mode_action(
        samples, yaws, baseline, baseline_yaw, ego_indices=[0], sample_index=1
    )
    np.testing.assert_array_equal(positions[:, 0], baseline[:, 0])
    np.testing.assert_array_equal(selected_yaws[:, 0], baseline_yaw[:, 0])
    np.testing.assert_array_equal(positions[:, 1:], samples[:, 1, 1:])
    np.testing.assert_array_equal(selected_yaws[:, 1:], yaws[:, 1, 1:])
    assert audit["selected_sample_index"] == 1
    assert audit["num_samples"] == 3


def test_fixed_joint_mode_rejects_out_of_range_sample():
    samples = torch.zeros(2, 3, 2, 2)
    yaws = torch.zeros(2, 3, 2, 1)
    baseline = torch.zeros(2, 2, 2)
    baseline_yaw = torch.zeros(2, 2, 1)
    with pytest.raises(IndexError):
        select_fixed_native_joint_mode_action(
            samples, yaws, baseline, baseline_yaw, ego_indices=[0], sample_index=3
        )
