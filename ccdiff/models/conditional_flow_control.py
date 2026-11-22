"""Trust-region utilities for a future Conditional-FM source-space arm.

This module is intentionally independent from the production sampler.  It can
prepare and test the source-space optimization contract without authorizing a
new experimental method before the preregistered native-rejection kill gate.

The optimizer never accepts world-coordinate trajectories or clean endpoints.
Its only variable is one complete joint Gaussian FM source tensor.  The caller
supplies a differentiable scalar objective whose graph may run through the
frozen FM velocity field and dynamics.
"""

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class MonotoneJointVelocityConfig:
    """One-step B4 exact-acceptance contract for a complete joint FM field."""

    initial_step_rms: float = 0.01
    trust_max_rms: float = 0.01
    trust_max_linf: float = 0.05
    feasibility_weight: float = 1.0
    backtracking_steps: int = 8
    backtracking_factor: float = 0.5
    attack_tolerance: float = 1.0e-7
    feasibility_tolerance: float = 1.0e-7

    def validate(self) -> None:
        if min(
            self.initial_step_rms,
            self.trust_max_rms,
            self.trust_max_linf,
        ) <= 0.0:
            raise ValueError("B4 step and trust limits must be positive")
        if self.backtracking_steps <= 0:
            raise ValueError("B4 backtracking_steps must be positive")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("B4 backtracking_factor must be in (0,1)")
        if self.attack_tolerance < 0.0 or self.feasibility_tolerance < 0.0:
            raise ValueError("B4 tolerances must be non-negative")


def _finite_scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 0:
        raise ValueError("{} must be scalar".format(name))
    if not bool(torch.isfinite(value)):
        raise ValueError("{} must be finite".format(name))
    return value


def monotone_complete_joint_velocity_step(
    velocity0: torch.Tensor,
    energy_partition: Callable[
        [torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]
    ],
    config: MonotoneJointVelocityConfig,
) -> Tuple[torch.Tensor, Dict[str, object], Dict[str, object]]:
    """Return one exact monotone B4 update or native identity.

    The update direction spans the complete input tensor.  Trust enforcement
    scales that direction as a whole (per sample RMS and global L-infinity),
    so no agent row or coordinate is clipped/replaced independently.
    ``energy_partition`` returns attack, feasibility and component audit data.
    """

    config.validate()
    if not torch.is_floating_point(velocity0):
        raise TypeError("B4 velocity field must be floating point")
    if velocity0.ndim < 2 or velocity0.numel() == 0:
        raise ValueError("B4 velocity field must be a non-empty joint tensor")
    if not bool(torch.isfinite(velocity0).all()):
        raise ValueError("B4 velocity field must be finite")

    native = velocity0.detach().clone()
    variable = native.clone().requires_grad_(True)
    attack0_tensor, feasible0_tensor, components0 = energy_partition(variable)
    attack0_tensor = _finite_scalar(attack0_tensor, "B4 attack energy")
    feasible0_tensor = _finite_scalar(feasible0_tensor, "B4 feasibility energy")
    objective = attack0_tensor + config.feasibility_weight * feasible0_tensor
    gradient = torch.autograd.grad(objective, variable, only_inputs=True)[0]
    gradient_rms_tensor = gradient.square().mean().sqrt()
    gradient_rms = float(gradient_rms_tensor.detach())
    attack0 = float(attack0_tensor.detach())
    feasible0 = float(feasible0_tensor.detach())
    audit = {
        "contract": "b4_complete_joint_fm_velocity_monotone_v1",
        "optimized_variable": "complete_joint_flow_velocity",
        "joint_shape": list(native.shape),
        "direct_trajectory_edit": False,
        "row_replacement_or_mixing": False,
        "hard_gn_projection": False,
        "initial_attack_energy": attack0,
        "initial_feasibility_energy": feasible0,
        "gradient_rms": gradient_rms,
        "accepted": False,
        "fallback_native": True,
        "trace": [],
    }
    if not bool(torch.isfinite(gradient).all()) or gradient_rms <= 1.0e-12:
        audit.update({
            "status": "fail_closed_invalid_gradient",
            "final_attack_energy": attack0,
            "final_feasibility_energy": feasible0,
        })
        return native, components0, audit

    direction = -gradient.detach() / gradient_rms_tensor.detach()
    final_components = components0
    for trial in range(config.backtracking_steps):
        requested = (
            config.initial_step_rms
            * config.backtracking_factor ** trial
        )
        delta = requested * direction
        reduce_dims = tuple(range(1, delta.ndim))
        per_sample_rms = delta.square().mean(dim=reduce_dims).sqrt()
        trust_scale = min(
            1.0,
            config.trust_max_rms
            / max(float(per_sample_rms.max()), 1.0e-12),
            config.trust_max_linf
            / max(float(delta.abs().max()), 1.0e-12),
        )
        delta = delta * trust_scale
        candidate = native + delta
        with torch.no_grad():
            attack_tensor, feasible_tensor, components = energy_partition(candidate)
        finite = bool(
            torch.isfinite(candidate).all()
            and torch.isfinite(attack_tensor)
            and torch.isfinite(feasible_tensor)
        )
        attack = float(attack_tensor) if finite else float("nan")
        feasible = float(feasible_tensor) if finite else float("nan")
        attack_decreased = finite and (
            attack <= attack0 - config.attack_tolerance
        )
        feasibility_nonworsened = finite and (
            feasible <= feasible0 + config.feasibility_tolerance
        )
        accepted = bool(attack_decreased and feasibility_nonworsened)
        audit["trace"].append({
            "trial": trial,
            "requested_step_rms": requested,
            "actual_max_sample_delta_rms": float(
                delta.square().mean(dim=reduce_dims).sqrt().max()
            ),
            "actual_delta_linf": float(delta.abs().max()),
            "attack_energy": attack,
            "feasibility_energy": feasible,
            "finite": finite,
            "attack_decreased": attack_decreased,
            "feasibility_nonworsened": feasibility_nonworsened,
            "accepted": accepted,
        })
        if accepted:
            audit.update({
                "status": "accepted_monotone_feasible_step",
                "accepted": True,
                "fallback_native": False,
                "final_attack_energy": attack,
                "final_feasibility_energy": feasible,
            })
            final_components = components
            return candidate.detach(), final_components, audit

    audit.update({
        "status": "fail_closed_no_feasible_descent",
        "final_attack_energy": attack0,
        "final_feasibility_energy": feasible0,
    })
    return native, final_components, audit


@dataclass(frozen=True)
class ConditionalFlowSourceConfig:
    enabled: bool = False
    num_steps: int = 8
    step_size: float = 0.05
    backtracking_steps: int = 8
    backtracking_factor: float = 0.5
    min_objective_improvement: float = 1e-6
    max_source_delta_rms: float = 0.10
    max_per_actor_source_delta_rms: float = 0.10
    max_source_energy_increase: float = 0.02
    finite_gradient_required: bool = True
    normalize_masked_gradient_rms: bool = False
    optimization_scope: str = "all_background"
    optimization_start_fraction: float = 0.0
    target_mean_shift_coefficients: Optional[Tuple[float, float, float, float]] = None
    target_mean_shift_max_abs_coefficient: float = 0.25
    target_mean_shift_max_kl_per_coordinate: float = 0.02
    target_mean_shift_basis_kind: str = "constant_ramp"
    target_mean_shift_pulse_width_fraction: float = 0.15
    target_mean_shift_online_cpa: bool = False
    target_mean_shift_max_cpa_distance_m: float = 30.0
    target_velocity_residual: bool = False
    target_velocity_residual_arrival_aligned: bool = False
    target_velocity_residual_arrival_max_lateral_distance_m: float = 4.0

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.backtracking_steps <= 0:
            raise ValueError("backtracking_steps must be positive")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must be in (0, 1)")
        if self.min_objective_improvement < 0:
            raise ValueError("min_objective_improvement must be non-negative")
        if self.max_source_delta_rms < 0:
            raise ValueError("max_source_delta_rms must be non-negative")
        if self.max_per_actor_source_delta_rms < 0:
            raise ValueError(
                "max_per_actor_source_delta_rms must be non-negative"
            )
        if self.max_source_energy_increase < 0:
            raise ValueError("max_source_energy_increase must be non-negative")
        if self.optimization_scope not in {"all_background", "target_only"}:
            raise ValueError("unsupported optimization_scope")
        if not 0.0 <= self.optimization_start_fraction < 1.0:
            raise ValueError("optimization_start_fraction must be in [0,1)")
        if self.target_mean_shift_coefficients is not None:
            if len(self.target_mean_shift_coefficients) != 4:
                raise ValueError("target mean shift requires four coefficients")
            if max(abs(float(v)) for v in self.target_mean_shift_coefficients) > self.target_mean_shift_max_abs_coefficient:
                raise ValueError("target mean-shift coefficient exceeds action box")
        if self.target_mean_shift_max_abs_coefficient < 0:
            raise ValueError("target_mean_shift_max_abs_coefficient must be non-negative")
        if self.target_mean_shift_max_kl_per_coordinate < 0:
            raise ValueError("target_mean_shift_max_kl_per_coordinate must be non-negative")
        if self.target_mean_shift_basis_kind not in {
            "constant_ramp", "gaussian_pulse"
        }:
            raise ValueError("unsupported target mean-shift basis")
        if not 0.0 < self.target_mean_shift_pulse_width_fraction <= 1.0:
            raise ValueError("target pulse width must be in (0,1]")
        if self.target_mean_shift_online_cpa and self.target_mean_shift_basis_kind != "gaussian_pulse":
            raise ValueError("online CPA is defined only for gaussian_pulse basis")
        if self.target_mean_shift_max_cpa_distance_m <= 0:
            raise ValueError("online CPA distance gate must be positive")
        if self.target_velocity_residual:
            if self.target_mean_shift_coefficients is None:
                raise ValueError("velocity residual requires four coefficients")
            if self.target_mean_shift_basis_kind != "gaussian_pulse":
                raise ValueError("velocity residual requires gaussian_pulse basis")
            if not self.target_mean_shift_online_cpa:
                raise ValueError("velocity residual requires online CPA alignment")
        if self.target_velocity_residual_arrival_aligned:
            if not self.target_velocity_residual:
                raise ValueError("arrival alignment requires velocity residual")
            if self.target_velocity_residual_arrival_max_lateral_distance_m <= 0:
                raise ValueError("arrival lateral-distance gate must be positive")


def gaussian_energy(source: torch.Tensor) -> torch.Tensor:
    """Mean negative-log-density energy, up to an additive constant."""
    return 0.5 * source.square().mean()


def source_delta_rms(source: torch.Tensor, source0: torch.Tensor) -> torch.Tensor:
    return (source - source0).square().mean().sqrt()


def per_actor_source_delta_rms(
    source: torch.Tensor, source0: torch.Tensor
) -> torch.Tensor:
    """Return one RMS displacement per actor row.

    Production FM sources have shape ``[B*K,M,T,A]``.  The final two axes are
    the actor trajectory and action dimensions, so the returned shape is
    ``[B*K,M]``.  Lower-rank tensors used by unit tests are treated as one
    source group instead of silently skipping this trust gate.
    """
    if source.shape != source0.shape:
        raise ValueError("source and source0 must have identical shapes")
    if source.ndim >= 3:
        return (source - source0).square().mean(dim=(-1, -2)).sqrt()
    return source_delta_rms(source, source0).reshape(1)


def _scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 0:
        raise ValueError(f"{name} must return a scalar tensor, got {value.shape}")
    return value


def optimize_complete_joint_source(
    source0: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
    config: ConditionalFlowSourceConfig,
    optimization_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Minimize a differentiable objective inside a Gaussian source trust region.

    The complete source tensor is kept intact.  ``optimization_mask`` can freeze
    preregistered coordinates, but it cannot combine rows from different source
    particles.  Candidate steps are rejected through deterministic backtracking;
    there is no coordinate clipping or endpoint projection.
    """
    config.validate()
    if not torch.is_floating_point(source0):
        raise TypeError("source0 must be floating point")
    if not torch.isfinite(source0).all():
        raise ValueError("source0 must be finite")

    base = source0.detach().clone()
    if optimization_mask is None:
        mask = torch.ones_like(base)
    else:
        if optimization_mask.shape != base.shape:
            raise ValueError("optimization_mask must match source shape")
        mask = optimization_mask.to(device=base.device, dtype=base.dtype)
        if not torch.isfinite(mask).all():
            raise ValueError("optimization_mask must be finite")
        if not torch.logical_or(mask == 0, mask == 1).all():
            raise ValueError("optimization_mask must be binary")

    audit: Dict[str, object] = {
        "contract": "conditional_fm_complete_joint_gaussian_source_v1",
        "config": asdict(config),
        "source_shape": list(base.shape),
        "mask_nonzero_fraction": float(mask.mean().item()),
        "endpoint_projection": False,
        "world_coordinate_variable": False,
        "per_agent_row_mixing": False,
        "per_actor_source_trust_enforced": True,
        "accepted_steps": 0,
        "backtracking_trials": 0,
        "fallback_native": True,
        "trace": [],
    }
    if not config.enabled:
        audit.update({
            "status": "disabled_native_identity",
            "initial_objective": None,
            "final_objective": None,
            "source_delta_rms": 0.0,
            "source_energy_delta": 0.0,
        })
        return base, audit

    with torch.enable_grad():
        initial_value = _scalar(objective(base.clone().requires_grad_(True)), "objective")
        if not torch.isfinite(initial_value):
            raise ValueError("initial objective must be finite")
        initial_float = float(initial_value.detach().item())
        base_energy = gaussian_energy(base)
        current = base.clone()
        current_float = initial_float

        for iteration in range(config.num_steps):
            variable = current.detach().clone().requires_grad_(True)
            value = _scalar(objective(variable), "objective")
            gradient = torch.autograd.grad(
                value, variable, only_inputs=True, create_graph=False
            )[0]
            gradient = gradient * mask
            finite_gradient = bool(torch.isfinite(gradient).all().item())
            active = mask.bool()
            active_grad_rms_tensor = (
                gradient[active].square().mean().sqrt()
                if bool(active.any()) else gradient.new_zeros(())
            )
            raw_grad_rms = float(
                active_grad_rms_tensor.detach().item()
            )
            if (
                config.normalize_masked_gradient_rms
                and finite_gradient
                and raw_grad_rms > 0.0
            ):
                gradient = gradient / active_grad_rms_tensor
            grad_rms = float(gradient.square().mean().sqrt().detach().item())
            row = {
                "iteration": iteration,
                "objective_before": float(value.detach().item()),
                "gradient_rms": grad_rms,
                "raw_active_gradient_rms": raw_grad_rms,
                "gradient_normalized": bool(
                    config.normalize_masked_gradient_rms
                    and raw_grad_rms > 0.0
                ),
                "finite_gradient": finite_gradient,
                "accepted": False,
                "trials": [],
            }
            if not finite_gradient:
                audit["trace"].append(row)
                if config.finite_gradient_required:
                    break

            step = config.step_size
            accepted_candidate = None
            accepted_value = None
            for trial in range(config.backtracking_steps):
                audit["backtracking_trials"] += 1
                candidate = current - step * gradient
                delta = source_delta_rms(candidate, base)
                actor_delta = per_actor_source_delta_rms(candidate, base)
                max_actor_delta = actor_delta.max()
                energy_delta = gaussian_energy(candidate) - base_energy
                trust_pass = bool(
                    delta <= config.max_source_delta_rms
                    and max_actor_delta <= config.max_per_actor_source_delta_rms
                    and energy_delta <= config.max_source_energy_increase
                    and torch.isfinite(candidate).all()
                )
                candidate_value_float = None
                improvement = None
                if trust_pass:
                    candidate_value = _scalar(objective(candidate), "objective")
                    if torch.isfinite(candidate_value):
                        candidate_value_float = float(candidate_value.detach().item())
                        improvement = current_float - candidate_value_float
                        if improvement >= config.min_objective_improvement:
                            accepted_candidate = candidate.detach().clone()
                            accepted_value = candidate_value_float
                row["trials"].append({
                    "trial": trial,
                    "step_size": step,
                    "trust_pass": trust_pass,
                    "source_delta_rms": float(delta.detach().item()),
                    "max_per_actor_source_delta_rms": float(
                        max_actor_delta.detach().item()
                    ),
                    "source_energy_delta": float(energy_delta.detach().item()),
                    "objective": candidate_value_float,
                    "improvement": improvement,
                })
                if accepted_candidate is not None:
                    break
                step *= config.backtracking_factor

            if accepted_candidate is None:
                audit["trace"].append(row)
                break
            current = accepted_candidate
            current_float = float(accepted_value)
            row["accepted"] = True
            row["objective_after"] = current_float
            audit["accepted_steps"] += 1
            audit["trace"].append(row)

    final_delta = float(source_delta_rms(current, base).item())
    final_max_actor_delta = float(
        per_actor_source_delta_rms(current, base).max().item()
    )
    final_energy_delta = float((gaussian_energy(current) - gaussian_energy(base)).item())
    improved = (
        audit["accepted_steps"] > 0
        and initial_float - current_float >= config.min_objective_improvement
    )
    if not improved:
        current = base
        current_float = initial_float
        final_delta = 0.0
        final_max_actor_delta = 0.0
        final_energy_delta = 0.0
        status = "native_fallback_no_valid_improvement"
    else:
        audit["fallback_native"] = False
        status = "accepted_source_control"

    audit.update({
        "status": status,
        "initial_objective": initial_float,
        "final_objective": current_float,
        "source_delta_rms": final_delta,
        "max_per_actor_source_delta_rms": final_max_actor_delta,
        "source_energy_delta": final_energy_delta,
    })
    return current.detach(), audit
