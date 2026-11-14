import torch

from ccdiff.algos.algos import apply_semantic_reaction_hold


def test_reaction_hold_integrates_constant_command_and_splices_tail():
    positions = torch.zeros(1, 2, 5, 2)
    positions[0, 1, :, 0] = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
    yaws = torch.zeros(1, 2, 5, 1)

    out_pos, out_yaw = apply_semantic_reaction_hold(
        positions,
        yaws,
        receiver_idx=1,
        hold_steps=3,
        speed=torch.tensor([10.0]),
        yaw_rate=torch.tensor([0.0]),
        dt=0.1,
    )

    torch.testing.assert_close(out_pos[0, 1, :3, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(out_pos[0, 1, :3, 1], torch.zeros(3))
    torch.testing.assert_close(out_yaw[0, 1], torch.zeros(5, 1))
    # Original tail is translated by the held/original boundary difference 1.5 m.
    torch.testing.assert_close(out_pos[0, 1, 3:, 0], torch.tensor([3.5, 4.0]))
    torch.testing.assert_close(out_pos[0, 0], positions[0, 0])


def test_reaction_hold_uses_unicycle_arc_for_nonzero_yaw_rate():
    positions = torch.zeros(1, 1, 4, 2)
    yaws = torch.zeros(1, 1, 4, 1)
    out_pos, out_yaw = apply_semantic_reaction_hold(
        positions,
        yaws,
        receiver_idx=0,
        hold_steps=4,
        speed=5.0,
        yaw_rate=0.2,
        dt=0.1,
    )
    assert torch.all(torch.isfinite(out_pos))
    assert torch.all(torch.diff(out_pos[0, 0, :, 0]) > 0)
    torch.testing.assert_close(out_yaw[0, 0, :, 0], torch.tensor([0.02, 0.04, 0.06, 0.08]))


def test_zero_hold_is_exact_noop():
    positions = torch.randn(2, 3, 5, 2)
    yaws = torch.randn(2, 3, 5, 1)
    out_pos, out_yaw = apply_semantic_reaction_hold(
        positions,
        yaws,
        receiver_idx=2,
        hold_steps=0,
        speed=torch.ones(2),
        yaw_rate=torch.zeros(2),
        dt=0.1,
    )
    torch.testing.assert_close(out_pos, positions)
    torch.testing.assert_close(out_yaw, yaws)


def test_reaction_hold_applies_longitudinal_acceleration_bias_without_touching_others():
    positions = torch.zeros(1, 2, 5, 2)
    yaws = torch.zeros(1, 2, 5, 1)
    out_pos, out_yaw = apply_semantic_reaction_hold(
        positions,
        yaws,
        receiver_idx=1,
        hold_steps=4,
        speed=5.0,
        yaw_rate=0.0,
        dt=0.1,
        accel_bias_mps2=1.0,
    )
    expected_step_speeds = torch.tensor([5.05, 5.15, 5.25, 5.35])
    torch.testing.assert_close(
        torch.diff(torch.cat((torch.zeros(1), out_pos[0, 1, :4, 0]))) / 0.1,
        expected_step_speeds,
    )
    torch.testing.assert_close(out_pos[0, 0], positions[0, 0])
    torch.testing.assert_close(out_yaw, yaws)


def test_acceleration_bias_continues_across_replanning_chunks():
    positions = torch.zeros(1, 1, 2, 2)
    yaws = torch.zeros(1, 1, 2, 1)
    out_pos, _ = apply_semantic_reaction_hold(
        positions,
        yaws,
        receiver_idx=0,
        hold_steps=2,
        speed=5.0,
        yaw_rate=0.0,
        dt=0.1,
        accel_bias_mps2=1.0,
        elapsed_steps=4,
    )
    expected_step_speeds = torch.tensor([5.45, 5.55])
    torch.testing.assert_close(
        torch.diff(torch.cat((torch.zeros(1), out_pos[0, 0, :, 0]))) / 0.1,
        expected_step_speeds,
    )
