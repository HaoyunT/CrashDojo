import pytest
import torch

from ccdiff.models.scenetemporal import apply_directed_yield_belief


def _inputs():
    # B=1, two receivers/senders, four future steps, dt=0.5.
    future_positions = torch.zeros(1, 2, 2, 4, 2)
    future_speeds = torch.full((1, 2, 2, 4), 4.0)
    last_positions = torch.zeros(1, 2, 2, 2)
    initial_speeds = torch.full((1, 2, 2), 4.0)
    # Selected edge follows a straight predicted chord at constant speed.
    future_positions[0, 1, 0, :, 0] = torch.tensor([2.0, 4.0, 6.0, 8.0])
    return future_positions, future_speeds, last_positions, initial_speeds


def test_directed_yield_belief_only_changes_selected_edge():
    pos, speed, hist_pos, hist_speed = _inputs()
    deceleration = torch.zeros(1, 2, 2)
    deceleration[0, 1, 0] = 2.0
    out_pos, out_speed = apply_directed_yield_belief(
        pos, speed, hist_pos, hist_speed, deceleration, dt=0.5
    )
    expected_distance = torch.tensor([1.75, 3.0, 3.75, 4.0])
    expected_speed = torch.tensor([3.0, 2.0, 1.0, 0.0])
    torch.testing.assert_close(out_pos[0, 1, 0, :, 0], expected_distance)
    torch.testing.assert_close(out_speed[0, 1, 0], expected_speed)
    torch.testing.assert_close(out_pos[0, 0, 1], pos[0, 0, 1])
    torch.testing.assert_close(out_speed[0, 0, 1], speed[0, 0, 1])


def test_zero_yield_belief_is_exact_identity():
    values = _inputs()
    pos, speed = apply_directed_yield_belief(
        *values, torch.zeros(1, 2, 2), dt=0.5
    )
    assert torch.equal(pos, values[0])
    assert torch.equal(speed, values[1])


def test_yield_belief_rejects_negative_deceleration():
    values = _inputs()
    deceleration = torch.zeros(1, 2, 2)
    deceleration[0, 1, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        apply_directed_yield_belief(*values, deceleration, dt=0.5)
