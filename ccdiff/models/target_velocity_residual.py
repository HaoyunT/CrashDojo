"""Trust-bounded residual controls for a frozen flow-matching vector field.

The residual changes one target actor in one complete joint K mode during the
native FM ODE.  It never edits a sampled endpoint, projects a trajectory, or
mixes rows between joint samples.  A zero action is bit-exact identity.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

import torch


@dataclass(frozen=True)
class TargetVelocityResidualConfig:
    target_actor_idx: int
    selected_sample_idx: int = 0
    coefficients: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    pulse_center_fraction: float = 0.5
    pulse_width_fraction: float = 0.25
    max_abs_coefficient: float = 0.35
    max_global_residual_rms: float = 0.05
    max_target_residual_rms: float = 0.25
    max_one_step_delta_rms: float = 0.02
    max_path_energy: float = 0.02

    def validate(self, velocity_shape: Sequence[int]) -> None:
        if len(velocity_shape) != 5:
            raise ValueError("velocity bank must have shape [B,K,M,T,A]")
        _, samples, actors, horizon, action_dim = velocity_shape
        if action_dim != 2 or horizon < 1:
            raise ValueError("expected a positive horizon and two action dimensions")
        if not 0 <= self.selected_sample_idx < samples:
            raise IndexError("selected sample is outside the K bank")
        if not 0 <= self.target_actor_idx < actors:
            raise IndexError("target actor is outside the actor bank")
        if len(self.coefficients) != 4:
            raise ValueError("coefficients must be (acc_pulse, acc_deriv, yaw_pulse, yaw_deriv)")
        if max(abs(float(x)) for x in self.coefficients) > self.max_abs_coefficient:
            raise ValueError("coefficient exceeds the residual action box")
        if not 0.0 <= self.pulse_center_fraction <= 1.0:
            raise ValueError("pulse center must be in [0,1]")
        if not 0.0 < self.pulse_width_fraction <= 1.0:
            raise ValueError("pulse width must be in (0,1]")
        for name in (
            "max_abs_coefficient",
            "max_global_residual_rms",
            "max_target_residual_rms",
            "max_one_step_delta_rms",
            "max_path_energy",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


def _residual_basis(
    horizon: int,
    coefficients: Sequence[float],
    center: float,
    width: float,
    flow_fraction: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    unit = torch.linspace(0.0, 1.0, horizon, device=device, dtype=dtype)
    normalized = (unit - float(center)) / float(width)
    pulse = torch.exp(-0.5 * normalized.square())
    derivative = normalized * pulse
    acc_pulse, acc_deriv, yaw_pulse, yaw_deriv = (
        torch.as_tensor(value, device=device, dtype=dtype) for value in coefficients
    )
    temporal = torch.stack(
        (
            acc_pulse * pulse + acc_deriv * derivative,
            yaw_pulse * pulse + yaw_deriv * derivative,
        ),
        dim=-1,
    )
    # Smoothly vanish at both FM endpoints; the maximum multiplier is one.
    flow_window = torch.sin(
        torch.as_tensor(float(flow_fraction), device=device, dtype=dtype) * torch.pi
    )
    return temporal * flow_window


def apply_target_velocity_residual(
    native_velocity_bank: torch.Tensor,
    config: TargetVelocityResidualConfig,
    *,
    flow_fraction: float,
    euler_step_size: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Apply one bounded residual and return a fail-closed audit."""
    if not torch.is_floating_point(native_velocity_bank):
        raise TypeError("native velocity must be floating point")
    if not torch.isfinite(native_velocity_bank).all():
        raise ValueError("native velocity must be finite")
    if not 0.0 <= float(flow_fraction) <= 1.0:
        raise ValueError("flow_fraction must be in [0,1]")
    if not 0.0 < float(euler_step_size) <= 1.0:
        raise ValueError("Euler step size must be in (0,1]")
    config.validate(native_velocity_bank.shape)

    native = native_velocity_bank.detach().clone()
    residual = _residual_basis(
        native.shape[-2],
        config.coefficients,
        config.pulse_center_fraction,
        config.pulse_width_fraction,
        flow_fraction,
        device=native.device,
        dtype=native.dtype,
    )
    zero = bool(torch.count_nonzero(residual).item() == 0)
    audit: Dict[str, object] = {
        "contract": "target_actor_fm_velocity_residual_v1",
        "config": asdict(config),
        "flow_fraction": float(flow_fraction),
        "euler_step_size": float(euler_step_size),
        "endpoint_projection": False,
        "trajectory_modified_after_sampling": False,
        "agent_row_mixing": False,
        "frozen_fm_backbone_required": True,
        "exact_kl_claim": False,
        "path_energy_regularizer": True,
        "zero_action": zero,
    }
    if zero:
        audit.update({
            "status": "zero_action_native_identity",
            "accepted": True,
            "global_residual_rms": 0.0,
            "target_residual_rms": 0.0,
            "one_step_delta_rms": 0.0,
            "path_energy": 0.0,
            "native_identity_bit_exact": True,
            "other_modes_bit_exact": True,
            "other_actors_bit_exact": True,
        })
        return native, audit

    candidate = native.clone()
    sample_idx = int(config.selected_sample_idx)
    actor_idx = int(config.target_actor_idx)
    candidate[:, sample_idx, actor_idx] += residual
    full_residual = candidate - native
    global_rms = full_residual.square().mean().sqrt()
    target_rms = residual.square().mean().sqrt()
    one_step_rms = (full_residual * float(euler_step_size)).square().mean().sqrt()
    path_energy = 0.5 * full_residual.square().mean()
    trust_pass = bool(
        torch.isfinite(candidate).all()
        and global_rms <= config.max_global_residual_rms
        and target_rms <= config.max_target_residual_rms
        and one_step_rms <= config.max_one_step_delta_rms
        and path_energy <= config.max_path_energy
    )
    result = candidate if trust_pass else native
    other_modes = torch.cat((result[:, :sample_idx], result[:, sample_idx + 1 :]), dim=1)
    native_other_modes = torch.cat((native[:, :sample_idx], native[:, sample_idx + 1 :]), dim=1)
    actor_mask = torch.ones(native.shape[2], device=native.device, dtype=torch.bool)
    actor_mask[actor_idx] = False
    audit.update({
        "status": "accepted_velocity_residual" if trust_pass else "native_fallback_trust_reject",
        "accepted": trust_pass,
        "global_residual_rms": float(global_rms.item()),
        "target_residual_rms": float(target_rms.item()),
        "one_step_delta_rms": float(one_step_rms.item()),
        "path_energy": float(path_energy.item()),
        "native_identity_bit_exact": bool(torch.equal(result, native)),
        "other_modes_bit_exact": bool(torch.equal(other_modes, native_other_modes)),
        "other_actors_bit_exact": bool(
            torch.equal(result[:, sample_idx, actor_mask], native[:, sample_idx, actor_mask])
        ),
    })
    if not audit["other_modes_bit_exact"] or not audit["other_actors_bit_exact"]:
        raise AssertionError("velocity residual escaped its target/K support")
    return result, audit
