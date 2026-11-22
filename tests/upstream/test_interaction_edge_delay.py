import pytest
import torch

from ccdiff.models.scenetemporal import apply_directed_interaction_edge_delay


def _time_features(num_agents=3, num_steps=5, num_features=2):
    time = torch.arange(num_steps, dtype=torch.float32)
    return time.view(1, 1, 1, num_steps, 1).repeat(
        1, num_agents, num_agents, 1, num_features
    )


def test_directed_delay_only_shifts_requested_receiver_sender_edge():
    neighbor_feat = _time_features()
    delay = torch.zeros((1, 3, 3), dtype=torch.long)
    delay[:, 2, 0] = 2

    result = apply_directed_interaction_edge_delay(neighbor_feat, delay)

    assert result[0, 2, 0, :, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
    assert result[0, 0, 2, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert torch.equal(result[0, 1, 2], neighbor_feat[0, 1, 2])


def test_zero_delay_is_value_identical_and_negative_delay_is_rejected():
    neighbor_feat = _time_features()
    zero = torch.zeros((1, 3, 3), dtype=torch.long)
    assert torch.equal(
        apply_directed_interaction_edge_delay(neighbor_feat, zero), neighbor_feat
    )

    negative = zero.clone()
    negative[:, 1, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        apply_directed_interaction_edge_delay(neighbor_feat, negative)
