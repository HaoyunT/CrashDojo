"""Production-boundary prototype for source-space Conditional Flow Matching.

This file is deliberately isolated from the running M2 worktree.  It defines
the differentiable boundary that may be wired into ``CCFlowMatching`` only
after the preregistered unchanged-native kill gate fires.

The controlled variable is a complete joint Gaussian source bank.  The code
never optimizes a clean endpoint or world-coordinate trajectory, never mixes
agent rows from different samples, and never changes the frozen FM weights or
20-step probability-flow ODE.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def _copy_tree(value):
    """Copy containers while intentionally sharing immutable tensor values."""
    if isinstance(value, dict):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    return value


def _functional_stationary_endpoint(model, endpoint: torch.Tensor) -> torch.Tensor:
    mask = getattr(model, "stationary_mask", None)
    if not getattr(model, "disable_control_on_stationary", False) or mask is None:
        return endpoint
    if mask.shape[0] != endpoint.shape[0]:
        if mask.shape[0] % endpoint.shape[0] != 0:
            raise ValueError("stationary mask sample dimension is incompatible")
        grouped = mask.reshape(
            endpoint.shape[0], mask.shape[0] // endpoint.shape[0], *mask.shape[1:]
        )
        if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
            raise ValueError("stationary mask differs across native K modes")
        mask = grouped[:, 0]
    if mask.shape != endpoint.shape[:2]:
        raise ValueError("stationary mask disagrees with flattened joint source")
    stationary = model.descale_traj(endpoint, [4, 5])
    stationary = torch.zeros_like(stationary)
    stationary = model.scale_traj(stationary, [4, 5])
    return torch.where(mask[..., None, None], stationary, endpoint)


def integrate_frozen_probability_flow(
    model,
    joint_source: torch.Tensor,
    data_batch: Dict[str, torch.Tensor],
    repeated_aux_info: Dict[str, object],
    class_free_guide_w: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, object]:
    """Integrate the unchanged learned ODE from a caller-provided source.

    Args:
        joint_source: Flattened ``[B*K, M, T, 2]`` Gaussian action source.

    Returns:
        ``(scaled_state_action, final_normalized_actions, attn_weights)``.

    This path intentionally rejects all endpoint-space mutations.  External
    hard acceptance remains responsible for on-road, dynamics, collateral,
    target-planner failure, and reference-planner safety.
    """
    if joint_source.ndim != 4 or joint_source.shape[-1] != model.action_dim:
        raise ValueError("joint_source must have shape [B*K,M,T,action_dim]")
    if not torch.is_floating_point(joint_source):
        raise TypeError("joint_source must be floating point")
    if not torch.isfinite(joint_source).all():
        raise ValueError("joint_source must be finite")
    if getattr(model, "typed_collision_projection", None) is not None:
        raise ValueError("Conditional-FM forbids endpoint projection")
    if getattr(model, "current_constraints", None) is not None:
        raise ValueError("Conditional-FM prototype forbids output constraints")

    action_flow = joint_source
    aux_info = _copy_tree(repeated_aux_info)
    x_flow = model.convert_action_to_state_and_action(
        action_flow, aux_info["curr_states"]
    )
    step_size = 1.0 / float(model.n_flow_steps)
    x_clean_model_out = None
    attn_weights = None
    batch_samples = joint_source.shape[0]

    for step_index in range(model.n_flow_steps):
        flow_value = step_index * step_size
        flow_time = torch.full(
            (batch_samples,), flow_value,
            device=joint_source.device, dtype=joint_source.dtype,
        )
        velocity, info = model._predict_flow_velocity(
            x_flow,
            flow_time,
            aux_info,
            data_batch,
            class_free_guide_w=class_free_guide_w,
            x_clean_model_out=x_clean_model_out,
        )
        remaining = 1.0 - flow_value
        endpoint = action_flow + remaining * velocity
        endpoint = _functional_stationary_endpoint(model, endpoint)

        # Numerically identical Euler form to the native FM sampler, without
        # detaching the source-to-trajectory graph needed by Conditional-FM.
        guided_velocity = (endpoint - action_flow) / remaining
        action_flow = action_flow + step_size * guided_velocity
        x_clean_model_out = endpoint
        x_flow = model.convert_action_to_state_and_action(
            action_flow, aux_info["curr_states"]
        )
        if step_index == model.n_flow_steps - 1:
            attn_weights = info.get("attn_weights")

    if not torch.isfinite(x_flow).all() or not torch.isfinite(action_flow).all():
        raise FloatingPointError("Conditional-FM integration became non-finite")
    return x_flow, action_flow, attn_weights


def verify_noisy_edge_gradient_contract(model) -> Dict[str, object]:
    """Fail closed until M2's relation-conditioning Jacobian is explicit.

    The current production M2 computes noisy future edge values from ``x_flow``
    but intentionally detaches that branch.  Conditional source optimization
    must not call the resulting partial derivative a full probability-flow
    gradient.  Production integration must first expose and test a value-identical
    differentiable-conditioning path.
    """
    noisy = bool(getattr(model, "use_noisy_fut_edge", False))
    path_available = hasattr(
        model, "conditional_flow_differentiable_conditioning"
    )
    path_enabled = bool(getattr(
        model, "conditional_flow_differentiable_conditioning", False
    ))
    relation_gradient = (not noisy) or (path_available and path_enabled)
    contract = {
        "use_noisy_fut_edge": noisy,
        "direct_state_gradient": True,
        "differentiable_conditioning_path_available": path_available,
        "differentiable_conditioning_path_enabled": path_enabled,
        "relation_conditioning_gradient": relation_gradient,
        "full_probability_flow_jacobian": relation_gradient,
    }
    if not relation_gradient:
        raise RuntimeError(
            "use_noisy_fut_edge=True currently detaches relation conditioning; "
            "install and value-match the differentiable edge path before enabling "
            "Conditional-FM"
        )
    return contract


def local_to_world(
    positions: torch.Tensor, world_from_agent: torch.Tensor
) -> torch.Tensor:
    """Transform ``[K,M,T,2]`` agent-local positions into world coordinates."""
    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [K,M,T,2]")
    if world_from_agent.shape != positions.shape[1:2] + (3, 3):
        raise ValueError("world_from_agent must have shape [M,3,3]")
    rotation = world_from_agent[:, :2, :2]
    translation = world_from_agent[:, :2, 2]
    return torch.einsum("kmti,mji->kmtj", positions, rotation) + translation[None, :, None]


def circle_surface_clearance(
    first_positions: torch.Tensor,
    second_positions: torch.Tensor,
    first_extent: torch.Tensor,
    second_extent: torch.Tensor,
) -> torch.Tensor:
    """Differentiable conservative clearance proxy used only for optimization."""
    first_radius = 0.5 * torch.linalg.vector_norm(first_extent[:2])
    second_radius = 0.5 * torch.linalg.vector_norm(second_extent[:2])
    return (
        torch.linalg.vector_norm(first_positions - second_positions, dim=-1)
        - first_radius - second_radius
    )


def persistent_soft_min(
    values: torch.Tensor, persistence_steps: int, temperature: float
) -> torch.Tensor:
    """Smooth minimum over consecutive-window mean clearances."""
    if values.ndim != 1:
        raise ValueError("persistent_soft_min expects one temporal vector")
    if persistence_steps <= 0 or persistence_steps > values.numel():
        raise ValueError("invalid persistence_steps")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    windows = values.unfold(0, persistence_steps, 1).mean(dim=-1)
    return -temperature * torch.logsumexp(-windows / temperature, dim=0)


@dataclass(frozen=True)
class PlannerSeparationObjectiveConfig:
    target_actor_idx: int
    executed_planner_actor_idx: int = 0
    selected_sample_idx: int = 0
    persistence_steps: int = 3
    temperature_m: float = 0.25
    reference_safety_margin_m: float = 1.0
    collateral_margin_m: float = 0.5
    reference_barrier_weight: float = 4.0
    collateral_barrier_weight: float = 4.0
    source_proximity_weight: float = 0.25
    global_conflict_step: int = None
    current_global_step: int = 0
    conflict_anchor_radius_steps: int = 2

    def validate(self, num_samples: int, num_agents: int) -> None:
        if not 0 <= self.selected_sample_idx < num_samples:
            raise IndexError("selected_sample_idx exceeds joint source bank")
        if not 0 <= self.target_actor_idx < num_agents:
            raise IndexError("target_actor_idx exceeds modeled agents")
        if not 0 <= self.executed_planner_actor_idx < num_agents:
            raise IndexError("planner actor index exceeds modeled agents")
        if self.target_actor_idx == self.executed_planner_actor_idx:
            raise ValueError("target actor must differ from planner actor")
        if self.temperature_m <= 0:
            raise ValueError("temperature_m must be positive")
        if self.global_conflict_step is not None:
            if int(self.global_conflict_step) < 0:
                raise ValueError("global_conflict_step must be non-negative")
            if int(self.current_global_step) < 0:
                raise ValueError("current_global_step must be non-negative")
            if int(self.conflict_anchor_radius_steps) < 0:
                raise ValueError("conflict_anchor_radius_steps must be non-negative")


def objective_time_slice(usable: int, config: PlannerSeparationObjectiveConfig) -> slice:
    """Return the local horizon slice tied to one simulator-global event.

    With no anchor this preserves the original all-horizon objective.  With an
    anchor, every receding-horizon call optimizes the same global conflict
    frame instead of repeatedly pushing the soft minimum to the horizon tail.
    """
    if usable <= 0:
        raise ValueError("usable horizon must be positive")
    if config.global_conflict_step is None:
        return slice(0, usable)
    local_conflict = int(config.global_conflict_step) - int(config.current_global_step)
    radius = int(config.conflict_anchor_radius_steps)
    start = max(0, local_conflict - radius)
    stop = min(usable, local_conflict + radius + 1)
    if stop - start < int(config.persistence_steps):
        raise ValueError(
            "global conflict anchor is outside the usable local horizon or "
            "does not contain the requested persistence window"
        )
    return slice(start, stop)


def smooth_max_violation(
    clearance: torch.Tensor, margin_m: float, temperature_m: float
) -> torch.Tensor:
    """Non-negative smooth approximation to max(margin-clearance, 0)."""
    raw = temperature_m * torch.logsumexp(
        (float(margin_m) - clearance) / temperature_m, dim=0
    )
    return temperature_m * F.softplus(raw / temperature_m)


def planner_separation_source_objective(
    model,
    joint_source: torch.Tensor,
    source_shape: Sequence[int],
    data_batch: Dict[str, torch.Tensor],
    repeated_aux_info: Dict[str, object],
    target_planner_positions_local: torch.Tensor,
    safety_planner_positions_local: torch.Tensor,
    world_from_agent: torch.Tensor,
    extents: torch.Tensor,
    config: PlannerSeparationObjectiveConfig,
    native_source: torch.Tensor = None,
    valid_actor_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Natural adversarial surrogate evaluated through the frozen FM ODE.

    It attracts one native target actor toward the attacked planner, while
    penalizing proximity to the safety-reference planner and all uninvolved
    generated actors.  These are optimization surrogates only: acceptance is
    decided by full closed-loop OBB/persistence/on-road/dynamics audits.
    """
    batch_size, num_samples, num_agents, horizon, action_dim = source_shape
    if batch_size != 1:
        raise ValueError("initial Conditional-FM protocol requires one scene")
    if tuple(joint_source.shape) != (
        batch_size * num_samples, num_agents, horizon, action_dim
    ):
        raise ValueError("joint_source disagrees with source_shape")
    config.validate(num_samples, num_agents)
    if valid_actor_mask is None:
        valid_actor_mask = torch.ones(
            num_agents, device=joint_source.device, dtype=torch.bool
        )
    else:
        if valid_actor_mask.shape != (num_agents,):
            raise ValueError("valid_actor_mask must have shape [M]")
        valid_actor_mask = valid_actor_mask.to(
            device=joint_source.device, dtype=torch.bool
        )
    if not bool(valid_actor_mask[config.target_actor_idx]):
        raise ValueError("target actor is invalid/padded")
    if not bool(valid_actor_mask[config.executed_planner_actor_idx]):
        raise ValueError("planner actor is invalid/padded")

    x_flow, _, _ = integrate_frozen_probability_flow(
        model, joint_source, data_batch, repeated_aux_info
    )
    descaled = model.descale_traj(x_flow)
    positions = descaled[..., :2].reshape(num_samples, num_agents, horizon, 2)
    world = local_to_world(positions, world_from_agent)
    sample = config.selected_sample_idx
    target = config.target_actor_idx
    planner = config.executed_planner_actor_idx
    target_world = world[sample, target]

    planner_tf = world_from_agent[planner]
    target_plan_world = local_to_world(
        target_planner_positions_local[None, None], planner_tf[None]
    )[0, 0]
    safety_plan_world = local_to_world(
        safety_planner_positions_local[None, None], planner_tf[None]
    )[0, 0]
    usable = min(
        horizon, target_plan_world.shape[0], safety_plan_world.shape[0]
    )
    target_world = target_world[:usable]
    target_plan_world = target_plan_world[:usable]
    safety_plan_world = safety_plan_world[:usable]
    time_slice = objective_time_slice(usable, config)
    target_world = target_world[time_slice]
    target_plan_world = target_plan_world[time_slice]
    safety_plan_world = safety_plan_world[time_slice]

    attacked_clearance = circle_surface_clearance(
        target_world, target_plan_world, extents[target], extents[planner]
    )
    attacked_cost = persistent_soft_min(
        attacked_clearance,
        config.persistence_steps,
        config.temperature_m,
    )
    safety_clearance = circle_surface_clearance(
        target_world, safety_plan_world, extents[target], extents[planner]
    )
    reference_barrier = smooth_max_violation(
        safety_clearance,
        config.reference_safety_margin_m,
        config.temperature_m,
    )

    collateral_terms = []
    for other in range(num_agents):
        if other in (target, planner) or not bool(valid_actor_mask[other]):
            continue
        clearance = circle_surface_clearance(
            target_world,
            world[sample, other, :usable][time_slice],
            extents[target],
            extents[other],
        )
        collateral_terms.append(
            smooth_max_violation(
                clearance,
                config.collateral_margin_m,
                config.temperature_m,
            )
        )
    collateral_barrier = (
        torch.stack(collateral_terms).sum()
        if collateral_terms else attacked_cost.new_zeros(())
    )
    source_proximity = (
        joint_source.new_zeros(())
        if native_source is None
        else (joint_source - native_source).square().mean()
    )
    return (
        attacked_cost
        + config.reference_barrier_weight * reference_barrier
        + config.collateral_barrier_weight * collateral_barrier
        + config.source_proximity_weight * source_proximity
    )


def selected_joint_background_mask(
    source: torch.Tensor,
    selected_sample_idx: int,
    ego_actor_idx: int,
    valid_actor_mask: torch.Tensor,
    target_actor_idx: int = None,
    optimization_scope: str = "all_background",
    optimization_start_fraction: float = 0.0,
) -> torch.Tensor:
    """Optimize all valid background actors of exactly one joint mode.

    Small, per-actor-bounded changes can jointly form a rare interaction while
    every actor remains individually normal.  The ego row is frozen because an
    external planner overwrites it, and no rows from different K samples mix.
    """
    if source.ndim != 5:
        raise ValueError("source must have shape [B,K,M,T,A]")
    if source.shape[0] != 1:
        raise ValueError("initial protocol requires one scene")
    if valid_actor_mask.shape != source.shape[2:3]:
        raise ValueError("valid_actor_mask must have shape [M]")
    if not 0 <= int(selected_sample_idx) < source.shape[1]:
        raise IndexError("selected_sample_idx exceeds K")
    if not 0 <= int(ego_actor_idx) < source.shape[2]:
        raise IndexError("ego_actor_idx exceeds M")
    actors = valid_actor_mask.to(device=source.device, dtype=torch.bool).clone()
    actors[int(ego_actor_idx)] = False
    if optimization_scope == "target_only":
        if target_actor_idx is None or not 0 <= int(target_actor_idx) < source.shape[2]:
            raise IndexError("target_only requires a valid target_actor_idx")
        if not bool(actors[int(target_actor_idx)]):
            raise ValueError("target actor is invalid, padded, or ego")
        actors.zero_()
        actors[int(target_actor_idx)] = True
    elif optimization_scope != "all_background":
        raise ValueError("unsupported optimization_scope")
    if not 0.0 <= float(optimization_start_fraction) < 1.0:
        raise ValueError("optimization_start_fraction must be in [0,1)")
    start = int(source.shape[3] * float(optimization_start_fraction))
    mask = torch.zeros_like(source)
    mask[0, int(selected_sample_idx), actors, start:] = 1
    return mask.reshape(-1, *source.shape[2:])


def per_actor_source_delta_rms(
    source: torch.Tensor, native_source: torch.Tensor
) -> torch.Tensor:
    """RMS source displacement for every ``[B,K,M]`` actor row."""
    if source.shape != native_source.shape or source.ndim != 5:
        raise ValueError("source/native_source must share [B,K,M,T,A]")
    return (source - native_source).square().mean(dim=(-1, -2)).sqrt()
