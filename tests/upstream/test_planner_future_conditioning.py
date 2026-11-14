import pytest
import torch

from ccdiff.models.ccdiff import inject_planner_reference_future_states


def _inputs(batch_size=2, samples=3, agents=4, horizon=5):
    shape = (batch_size * samples, agents, horizon)
    positions = torch.randn(*shape, 2, requires_grad=True)
    speeds = torch.randn(*shape, requires_grad=True)
    yaws = torch.randn(*shape, 1, requires_grad=True)
    planner_positions = torch.randn(batch_size, agents, horizon, 2)
    planner_yaws = torch.randn(batch_size, agents, horizon, 1)
    planner_mask = torch.zeros(batch_size, agents, dtype=torch.bool)
    planner_mask[:, 0] = True
    return (
        positions,
        speeds,
        yaws,
        planner_positions,
        planner_yaws,
        planner_mask,
    )


def test_planner_future_replaces_only_executed_sender_rows_across_k():
    values = _inputs()
    positions, speeds, yaws, plan_pos, plan_yaw, plan_mask = values
    out_pos, out_speed, out_yaw, repeated_mask = (
        inject_planner_reference_future_states(*values, dt=0.1)
    )

    expected_pos = plan_pos.repeat_interleave(3, dim=0)
    expected_yaw = plan_yaw.repeat_interleave(3, dim=0)
    assert repeated_mask.shape == (6, 4)
    assert torch.equal(repeated_mask[:, 0], torch.ones(6, dtype=torch.bool))
    assert torch.equal(out_pos[:, 0], expected_pos[:, 0])
    assert torch.equal(out_yaw[:, 0], expected_yaw[:, 0])
    assert torch.equal(out_pos[:, 1:], positions[:, 1:])
    assert torch.equal(out_yaw[:, 1:], yaws[:, 1:])
    assert torch.equal(out_speed[:, 1:], speeds[:, 1:])

    origin = torch.zeros_like(expected_pos[:, :, :1])
    expected_speed = torch.linalg.vector_norm(
        torch.diff(torch.cat((origin, expected_pos), dim=2), dim=2), dim=-1
    ) / 0.1
    assert torch.allclose(out_speed[:, 0], expected_speed[:, 0])


def test_planner_future_keeps_background_source_gradients_and_masks_planner_row():
    values = _inputs(batch_size=1, samples=2)
    positions, speeds, yaws = values[:3]
    out_pos, out_speed, out_yaw, _ = inject_planner_reference_future_states(
        *values, dt=0.1
    )
    (out_pos.sum() + out_speed.sum() + out_yaw.sum()).backward()
    assert torch.count_nonzero(positions.grad[:, 0]) == 0
    assert torch.count_nonzero(speeds.grad[:, 0]) == 0
    assert torch.count_nonzero(yaws.grad[:, 0]) == 0
    assert torch.count_nonzero(positions.grad[:, 1:]) > 0
    assert torch.count_nonzero(speeds.grad[:, 1:]) > 0
    assert torch.count_nonzero(yaws.grad[:, 1:]) > 0


def test_planner_future_rejects_ambiguous_or_partial_contracts():
    values = list(_inputs(batch_size=1, samples=2))
    values[-1][:] = False
    with pytest.raises(ValueError, match="exactly one"):
        inject_planner_reference_future_states(*values)

    values = list(_inputs(batch_size=2, samples=2))
    with pytest.raises(ValueError, match="not divisible"):
        inject_planner_reference_future_states(
            values[0][:-1], values[1][:-1], values[2][:-1], *values[3:]
        )
