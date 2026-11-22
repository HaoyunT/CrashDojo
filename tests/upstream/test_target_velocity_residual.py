import pytest
import torch

from ccdiff.models.target_velocity_residual import (
    TargetVelocityResidualConfig,
    apply_target_velocity_residual,
)


def bank():
    torch.manual_seed(7)
    return torch.randn(1, 20, 6, 52, 2)


def test_zero_action_is_bit_exact():
    native = bank()
    out, audit = apply_target_velocity_residual(
        native,
        TargetVelocityResidualConfig(target_actor_idx=3),
        flow_fraction=0.5,
        euler_step_size=0.05,
    )
    assert torch.equal(out, native)
    assert audit["status"] == "zero_action_native_identity"


def test_nonzero_is_local_and_bounded():
    native = bank()
    out, audit = apply_target_velocity_residual(
        native,
        TargetVelocityResidualConfig(
            target_actor_idx=3,
            coefficients=(0.2, -0.1, 0.05, 0.0),
        ),
        flow_fraction=0.5,
        euler_step_size=0.05,
    )
    assert audit["status"] == "accepted_velocity_residual"
    assert not torch.equal(out[:, 0, 3], native[:, 0, 3])
    assert torch.equal(out[:, 1:], native[:, 1:])
    assert torch.equal(out[:, 0, [0, 1, 2, 4, 5]], native[:, 0, [0, 1, 2, 4, 5]])
    assert audit["one_step_delta_rms"] <= 0.02


def test_flow_endpoint_window_is_identity():
    native = bank()
    config = TargetVelocityResidualConfig(target_actor_idx=3, coefficients=(0.2, 0, 0, 0))
    at_start, _ = apply_target_velocity_residual(
        native, config, flow_fraction=0.0, euler_step_size=0.05
    )
    assert torch.equal(at_start, native)


def test_rejected_action_falls_back():
    native = bank()
    config = TargetVelocityResidualConfig(
        target_actor_idx=3,
        coefficients=(0.3, 0, 0, 0),
        max_target_residual_rms=1e-8,
    )
    out, audit = apply_target_velocity_residual(
        native, config, flow_fraction=0.5, euler_step_size=0.05
    )
    assert torch.equal(out, native)
    assert audit["status"] == "native_fallback_trust_reject"


def test_invalid_coefficient_fails_closed():
    with pytest.raises(ValueError):
        apply_target_velocity_residual(
            bank(),
            TargetVelocityResidualConfig(target_actor_idx=3, coefficients=(0.4, 0, 0, 0)),
            flow_fraction=0.5,
            euler_step_size=0.05,
        )
