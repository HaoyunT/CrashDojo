"""Low-dimensional trust-region actions for frozen flow-matching sources.

The intervention changes only one target actor in one complete joint K mode.
It operates on the initial Gaussian action source, before the unchanged FM ODE;
it never edits, projects, or splices generated trajectories.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

import torch


def estimate_online_planner_target_cpa(
    planner_positions_local: torch.Tensor,
    world_from_actor: torch.Tensor,
    current_speeds: torch.Tensor,
    *,
    planner_actor_idx: int,
    target_actor_idx: int,
    dt: float,
    max_horizon: int,
    max_required_target_speed_mps: float = 20.0,
) -> Dict[str, float]:
    """Estimate CPA from the executed planner and target constant-velocity state.

    This uses only information available at the current closed-loop step: the
    actual ego planner future, current actor transforms, and current target
    speed. It never reads logged or generated future target positions.
    """
    if planner_positions_local.ndim != 2 or planner_positions_local.shape[-1] != 2:
        raise ValueError("planner_positions_local must have shape [T,2]")
    if world_from_actor.ndim != 3 or world_from_actor.shape[-2:] != (3, 3):
        raise ValueError("world_from_actor must have shape [M,3,3]")
    if current_speeds.ndim != 1 or current_speeds.shape[0] != world_from_actor.shape[0]:
        raise ValueError("current_speeds must contain one scalar per actor")
    if not 0 <= planner_actor_idx < world_from_actor.shape[0]:
        raise IndexError("planner actor is outside transform bank")
    if not 0 <= target_actor_idx < world_from_actor.shape[0]:
        raise IndexError("target actor is outside transform bank")
    if dt <= 0 or max_horizon < 1 or max_required_target_speed_mps <= 0:
        raise ValueError("dt and max_horizon must be positive")
    horizon = min(int(planner_positions_local.shape[0]), int(max_horizon))
    local = planner_positions_local[:horizon]
    if not torch.isfinite(local).all():
        raise ValueError("executed planner future must be finite")
    planner_tf = world_from_actor[planner_actor_idx]
    target_tf = world_from_actor[target_actor_idx]
    planner_world = local @ planner_tf[:2, :2].T + planner_tf[:2, 2]
    target_origin = target_tf[:2, 2]
    target_forward = target_tf[:2, 0]
    steps = torch.arange(1, horizon + 1, device=local.device, dtype=local.dtype)
    target_world = target_origin + (
        steps[:, None]
        * float(dt)
        * current_speeds[target_actor_idx]
        * target_forward
    )
    distance = torch.linalg.vector_norm(planner_world - target_world, dim=-1)
    if not torch.isfinite(distance).all():
        raise ValueError("online CPA distance became non-finite")
    cpa_step = int(torch.argmin(distance).item())
    # A simultaneous CPA gives a time but not a scene-covariant control sign.
    # Project each executed-planner point onto the target's current forward
    # ray.  The required speed to reach that ray point at the planner time
    # determines whether the target must accelerate or brake.  This consumes
    # no logged/generated target future and keeps the action semantics stable
    # across crossing, following, and merging geometries.
    planner_from_target = planner_world - target_origin
    longitudinal = planner_from_target @ target_forward
    lateral_vector = planner_from_target - longitudinal[:, None] * target_forward
    lateral_distance = torch.linalg.vector_norm(lateral_vector, dim=-1)
    # Ignore the target's current footprint; a zero-longitudinal tie is not a
    # future conflict point and would collapse the arrival-speed signal.
    arrival_times = steps * float(dt)
    required_speed_bank = longitudinal / arrival_times
    forward = (
        (longitudinal > 0.5)
        & (required_speed_bank >= 0.0)
        & (required_speed_bank <= float(max_required_target_speed_mps))
    )
    if bool(forward.any()):
        gated_lateral = torch.where(
            forward,
            lateral_distance,
            torch.full_like(lateral_distance, torch.inf),
        )
        arrival_step = int(torch.argmin(gated_lateral).item())
        arrival_valid = True
    else:
        arrival_step = int(torch.argmin(lateral_distance).item())
        arrival_valid = False
    arrival_time = float((arrival_step + 1) * dt)
    required_speed = float(longitudinal[arrival_step].item()) / arrival_time
    current_target_speed = float(current_speeds[target_actor_idx].item())
    speed_error = required_speed - current_target_speed
    speed_deadband = 0.25
    control_sign = 0.0 if abs(speed_error) <= speed_deadband else (1.0 if speed_error > 0.0 else -1.0)
    return {
        "cpa_step": cpa_step,
        "cpa_time_s": float((cpa_step + 1) * dt),
        "cpa_center_distance_m": float(distance[cpa_step].item()),
        "pulse_center_fraction": float(cpa_step / max(horizon - 1, 1)),
        "arrival_alignment_valid": arrival_valid,
        "arrival_step": arrival_step,
        "arrival_time_s": arrival_time,
        "arrival_pulse_center_fraction": float(arrival_step / max(horizon - 1, 1)),
        "arrival_lateral_distance_m": float(lateral_distance[arrival_step].item()),
        "arrival_longitudinal_distance_m": float(longitudinal[arrival_step].item()),
        "arrival_required_target_speed_mps": required_speed,
        "arrival_max_required_target_speed_mps": float(max_required_target_speed_mps),
        "arrival_current_target_speed_mps": current_target_speed,
        "arrival_speed_error_mps": speed_error,
        "arrival_control_sign": control_sign,
        "arrival_speed_deadband_mps": speed_deadband,
        "horizon": horizon,
        "uses_logged_target_future": False,
        "uses_generated_target_future": False,
        "target_motion_model": "current_speed_constant_heading",
    }


@dataclass(frozen=True)
class TargetSourceMeanShiftConfig:
    target_actor_idx: int
    selected_sample_idx: int = 0
    coefficients: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    basis_kind: str = "constant_ramp"
    pulse_center_fraction: float = 0.5
    pulse_width_fraction: float = 0.15
    max_abs_coefficient: float = 0.25
    max_global_delta_rms: float = 0.05
    max_target_delta_rms: float = 0.20
    max_mean_kl_per_coordinate: float = 0.02
    max_source_energy_increase: float = 0.02

    def validate(self, source_shape: Sequence[int]) -> None:
        if len(source_shape) != 5:
            raise ValueError("source must have shape [B,K,M,T,A]")
        _, num_samples, num_actors, horizon, action_dim = source_shape
        if action_dim != 2 or horizon < 1:
            raise ValueError("expected a positive horizon and two action dimensions")
        if not 0 <= self.selected_sample_idx < num_samples:
            raise IndexError("selected_sample_idx is outside the source bank")
        if not 0 <= self.target_actor_idx < num_actors:
            raise IndexError("target_actor_idx is outside the actor bank")
        if len(self.coefficients) != 4:
            raise ValueError("coefficients must be (acc0, acc_ramp, yaw0, yaw_ramp)")
        if self.basis_kind not in {"constant_ramp", "gaussian_pulse"}:
            raise ValueError("unsupported source mean-shift basis")
        if not 0.0 <= self.pulse_center_fraction <= 1.0:
            raise ValueError("pulse_center_fraction must be in [0,1]")
        if not 0.0 < self.pulse_width_fraction <= 1.0:
            raise ValueError("pulse_width_fraction must be in (0,1]")
        if max(abs(float(v)) for v in self.coefficients) > self.max_abs_coefficient:
            raise ValueError("coefficient exceeds the preregistered action box")
        for name in (
            "max_abs_coefficient",
            "max_global_delta_rms",
            "max_target_delta_rms",
            "max_mean_kl_per_coordinate",
            "max_source_energy_increase",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


def _temporal_shift(
    horizon: int,
    coefficients: Sequence[float],
    basis_kind: str,
    pulse_center_fraction: float,
    pulse_width_fraction: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    unit = torch.linspace(0.0, 1.0, horizon, device=device, dtype=dtype)
    phase = 2.0 * unit - 1.0
    acc0, acc_ramp, yaw0, yaw_ramp = (
        torch.as_tensor(value, device=device, dtype=dtype) for value in coefficients
    )
    if basis_kind == "constant_ramp":
        acc = acc0 + acc_ramp * phase
        yaw = yaw0 + yaw_ramp * phase
    elif basis_kind == "gaussian_pulse":
        normalized = (unit - pulse_center_fraction) / pulse_width_fraction
        pulse = torch.exp(-0.5 * normalized.square())
        derivative = normalized * pulse
        acc = acc0 * pulse + acc_ramp * derivative
        yaw = yaw0 * pulse + yaw_ramp * derivative
    else:  # validate() makes this unreachable; retain fail-closed locality.
        raise ValueError("unsupported source mean-shift basis")
    return torch.stack((acc, yaw), dim=-1)


def apply_target_source_mean_shift(
    native_source: torch.Tensor,
    config: TargetSourceMeanShiftConfig,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Return a shifted source bank and a fail-closed contract audit.

    A rejected candidate returns the native source bit-for-bit. A zero action
    also takes an identity fast path, which is used as the common-random-number
    contract test before GPU experiments.
    """
    if not torch.is_floating_point(native_source):
        raise TypeError("native_source must be floating point")
    if not torch.isfinite(native_source).all():
        raise ValueError("native_source must be finite")
    config.validate(native_source.shape)

    base = native_source.detach().clone()
    shift = _temporal_shift(
        base.shape[-2],
        config.coefficients,
        config.basis_kind,
        config.pulse_center_fraction,
        config.pulse_width_fraction,
        device=base.device,
        dtype=base.dtype,
    )
    zero_action = bool(torch.count_nonzero(shift).item() == 0)
    audit: Dict[str, object] = {
        "contract": "target_actor_low_dimensional_gaussian_mean_shift_v1",
        "config": asdict(config),
        "source_shape": [int(v) for v in base.shape],
        "action_dimensions": 4,
        "basis": config.basis_kind,
        "target_actor_idx": int(config.target_actor_idx),
        "selected_sample_idx": int(config.selected_sample_idx),
        "endpoint_projection": False,
        "trajectory_modified_after_sampling": False,
        "agent_row_mixing": False,
        "frozen_fm_transport_required": True,
        "zero_action": zero_action,
    }
    if zero_action:
        audit.update({
            "status": "zero_action_native_identity",
            "accepted": True,
            "global_delta_rms": 0.0,
            "target_delta_rms": 0.0,
            "mean_kl_per_coordinate": 0.0,
            "source_energy_increase": 0.0,
            "native_identity_bit_exact": True,
            "other_modes_bit_exact": True,
            "other_actors_bit_exact": True,
        })
        return base, audit

    candidate = base.clone()
    sample_idx = int(config.selected_sample_idx)
    actor_idx = int(config.target_actor_idx)
    candidate[:, sample_idx, actor_idx] += shift
    delta = candidate - base
    global_rms = delta.square().mean().sqrt()
    target_rms = shift.square().mean().sqrt()
    mean_kl = 0.5 * shift.square().mean()
    energy_increase = 0.5 * (
        candidate.square().mean() - base.square().mean()
    )
    trust_pass = bool(
        torch.isfinite(candidate).all()
        and global_rms <= config.max_global_delta_rms
        and target_rms <= config.max_target_delta_rms
        and mean_kl <= config.max_mean_kl_per_coordinate
        and energy_increase <= config.max_source_energy_increase
    )
    result = candidate if trust_pass else base
    other_modes = torch.cat(
        (result[:, :sample_idx], result[:, sample_idx + 1 :]), dim=1
    )
    native_other_modes = torch.cat(
        (base[:, :sample_idx], base[:, sample_idx + 1 :]), dim=1
    )
    actor_mask = torch.ones(base.shape[2], device=base.device, dtype=torch.bool)
    actor_mask[actor_idx] = False
    audit.update({
        "status": "accepted_mean_shift" if trust_pass else "native_fallback_trust_reject",
        "accepted": trust_pass,
        "global_delta_rms": float(global_rms.item()),
        "target_delta_rms": float(target_rms.item()),
        "mean_kl_per_coordinate": float(mean_kl.item()),
        "source_energy_increase": float(energy_increase.item()),
        "native_identity_bit_exact": bool(torch.equal(result, base)),
        "other_modes_bit_exact": bool(torch.equal(other_modes, native_other_modes)),
        "other_actors_bit_exact": bool(
            torch.equal(result[:, sample_idx, actor_mask], base[:, sample_idx, actor_mask])
        ),
    })
    if trust_pass and audit["native_identity_bit_exact"]:
        raise AssertionError("non-zero accepted action unexpectedly preserved full source")
    if not audit["other_modes_bit_exact"] or not audit["other_actors_bit_exact"]:
        raise AssertionError("mean-shift action escaped its selected target/K support")
    return result, audit
