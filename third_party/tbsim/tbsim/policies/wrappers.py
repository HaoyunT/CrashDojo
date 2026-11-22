#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import json
import hashlib
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import OrderedDict, Tuple, Dict

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.batch_utils import batch_utils
from tbsim.policies.common import Action, Plan, RolloutAction
from ccdiff.algos.algo_utils import yaw_from_pos
from tbsim.utils.trajdata_utils import convert_scene_obs_to_agent_prep, convert_scene_data_to_agent_coordinates, check_consistency, full_keywords, major_keywords, prep_keywords, add_scene_dim_to_agent_data, check_action_consistency
import numpy as np
from trajdata.utils.arr_utils import angle_wrap


def select_fixed_native_joint_mode_action(
    sample_positions,
    sample_yaws,
    baseline_positions,
    baseline_yaws,
    ego_indices,
    sample_index,
):
    """Use one untouched M2 joint sample for every non-ego actor.

    This is a proposal primitive, not an online causal verifier.  The selected
    rows remain byte-for-byte native FM outputs; ego is preserved here and is
    overwritten by the configured planner at the unified action boundary.
    """
    if sample_positions.ndim not in (4, 5) or sample_yaws.ndim != sample_positions.ndim:
        raise ValueError(
            "expected action samples [M,K,T,D] or [B,K,M,T,D]; got {} and {}".format(
                tuple(sample_positions.shape), tuple(sample_yaws.shape)
            )
        )
    if sample_positions.shape[:-1] != sample_yaws.shape[:-1]:
        raise ValueError("native joint position/yaw shapes disagree")
    num_samples = int(sample_positions.shape[1])
    sample_index = int(sample_index)
    if not 0 <= sample_index < num_samples:
        raise IndexError(
            "native joint sample index {} outside K={}".format(
                sample_index, num_samples
            )
        )
    if baseline_positions.ndim != sample_positions.ndim - 1:
        raise ValueError("baseline/action-sample rank mismatch")
    selected_positions = sample_positions[:, sample_index]
    selected_yaws = sample_yaws[:, sample_index]
    if selected_positions.shape != baseline_positions.shape:
        raise ValueError(
            "selected joint mode shape {} does not match baseline {}".format(
                tuple(selected_positions.shape), tuple(baseline_positions.shape)
            )
        )
    if selected_yaws.shape != baseline_yaws.shape:
        raise ValueError("selected joint yaw shape does not match baseline")
    if torch.is_tensor(baseline_positions):
        positions = baseline_positions.clone()
        yaws = baseline_yaws.clone()
        finite = bool(
            torch.isfinite(selected_positions).all().item()
            and torch.isfinite(selected_yaws).all().item()
        )
    else:
        positions = np.array(baseline_positions, copy=True)
        yaws = np.array(baseline_yaws, copy=True)
        finite = bool(
            np.isfinite(selected_positions).all()
            and np.isfinite(selected_yaws).all()
        )
    if not finite:
        raise ValueError("native joint proposal contains non-finite values")

    # Agent-centric is [M,T,D]; scene-centric is [B,M,T,D].
    agent_axis = 0 if positions.ndim == 3 else 1
    num_agents = int(positions.shape[agent_axis])
    ego_set = {int(index) for index in ego_indices}
    if any(index < 0 or index >= num_agents for index in ego_set):
        raise IndexError("ego index exceeds modeled agent count")
    non_ego = [index for index in range(num_agents) if index not in ego_set]
    for agent_index in non_ego:
        if agent_axis == 0:
            positions[agent_index] = selected_positions[agent_index]
            yaws[agent_index] = selected_yaws[agent_index]
        else:
            positions[:, agent_index] = selected_positions[:, agent_index]
            yaws[:, agent_index] = selected_yaws[:, agent_index]
    return positions, yaws, {
        "selected_sample_index": sample_index,
        "num_samples": num_samples,
        "non_ego_agent_count": len(non_ego),
        "native_joint_exact": True,
        "proposal_only": True,
        "full_horizon_verifier_required": True,
        "trajectory_modified": False,
    }


def select_constrained_native_joint_mode_action(
    sample_positions,
    sample_yaws,
    baseline_positions,
    baseline_yaws,
    reference_ego_positions,
    reference_ego_yaws,
    target_ego_positions,
    target_ego_yaws,
    world_from_agent,
    extents,
    ego_indices,
    *,
    collateral_margin_m=0.25,
    reference_safety_margin_m=0.25,
    target_max_clearance_m=1.0,
    baseline_min_target_clearance_m=None,
    target_min_persistence_steps=1,
    min_risk_improvement_m=0.05,
    progressive_target_barrier=False,
    progressive_activation_max_clearance_m=None,
    progressive_risk_first=False,
    progressive_persistence_first=False,
    progressive_dose_first=False,
    progressive_exposure_clearance_m=2.0,
    max_agent_tail_fraction=0.01,
    dt=0.1,
    max_speed_mps=20.0,
    max_acc_mps2=4.0,
    max_jerk_mps3=8.37,
    max_yaw_rate_rps=0.95,
    max_lateral_acc_mps2=4.0,
    kinematic_horizon_steps=5,
    naturalness_profile=None,
    history_positions=None,
    history_yaws=None,
    history_availabilities=None,
    target_agent_indices=None,
):
    """Select one *complete* native FM joint mode under hard constraints.

    Every non-ego row comes from the same sample index.  The primitive never
    edits, projects, interpolates, or mixes agent rows.  It is deliberately a
    receding-horizon proposal selector; downstream full-rollout verification
    remains mandatory.
    """
    agent_centric = sample_positions.ndim == 4
    if agent_centric:
        # [M,K,T,D] -> [1,K,M,T,D]
        sample_positions = sample_positions.permute(1, 0, 2, 3).unsqueeze(0)
        sample_yaws = sample_yaws.permute(1, 0, 2, 3).unsqueeze(0)
        baseline_positions = baseline_positions.unsqueeze(0)
        baseline_yaws = baseline_yaws.unsqueeze(0)
    if sample_positions.ndim != 5 or sample_yaws.ndim != 5:
        raise ValueError("expected samples [B,K,M,T,D] or [M,K,T,D]")
    if sample_positions.shape[:4] != sample_yaws.shape[:4]:
        raise ValueError("joint-mode position/yaw shapes disagree")
    if sample_positions.shape[0] != 1 or baseline_positions.shape[0] != 1:
        raise ValueError("constrained joint-mode selection requires batch size one")
    _, num_samples, num_agents, horizon, _ = sample_positions.shape
    ego_set = {int(value) for value in ego_indices}
    if len(ego_set) != 1:
        raise ValueError("constrained joint-mode selection requires exactly one ego")
    ego = next(iter(ego_set))
    if not 0 <= ego < num_agents:
        raise IndexError("ego index exceeds modeled agent count")
    background = [index for index in range(num_agents) if index != ego]
    if not background:
        raise ValueError("joint-mode selection requires background agents")
    if target_agent_indices is None:
        target_background = list(background)
    else:
        target_background = sorted({int(value) for value in target_agent_indices})
        if not target_background:
            raise ValueError("target_agent_indices cannot be empty")
        if any(value not in background for value in target_background):
            raise ValueError(
                "target_agent_indices must contain modeled non-ego rows"
            )

    device, dtype = sample_positions.device, sample_positions.dtype
    transforms = torch.as_tensor(world_from_agent, device=device, dtype=dtype)
    if transforms.ndim == 4:
        if transforms.shape[0] != 1:
            raise ValueError("unexpected world_from_agent batch")
        transforms = transforms[0]
    if transforms.shape != (num_agents, 3, 3):
        raise ValueError("expected one 3x3 transform per agent")
    extent = torch.as_tensor(extents, device=device, dtype=dtype)
    if extent.ndim == 4:
        extent = extent[0, :, -1]
    elif extent.ndim == 3:
        extent = extent[0] if extent.shape[0] == 1 else extent[:, -1]
    if extent.ndim != 2 or extent.shape[0] != num_agents:
        raise ValueError("expected one extent per agent")

    def fit_ego(value, name):
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("{} must contain exactly one ego".format(name))
        return value[0, :horizon]

    reference_pos = fit_ego(reference_ego_positions, "reference_ego_positions")
    reference_yaw = fit_ego(reference_ego_yaws, "reference_ego_yaws")
    target_pos = fit_ego(target_ego_positions, "target_ego_positions")
    target_yaw = fit_ego(target_ego_yaws, "target_ego_yaws")
    effective_horizon = min(
        horizon, reference_pos.shape[0], target_pos.shape[0]
    )
    reference_pos, reference_yaw = (
        reference_pos[:effective_horizon], reference_yaw[:effective_horizon]
    )
    target_pos, target_yaw = (
        target_pos[:effective_horizon], target_yaw[:effective_horizon]
    )

    def local_to_world(points, transform):
        ones = torch.ones(*points.shape[:-1], 1, device=device, dtype=dtype)
        return torch.matmul(
            torch.cat((points, ones), dim=-1), transform.transpose(-1, -2)
        )[..., :2]

    def local_yaw_to_world(yaw, transform):
        return yaw.squeeze(-1) + torch.atan2(transform[1, 0], transform[0, 0])

    def signed_box_gap(center_a, yaw_a, half_a, center_b, yaw_b, half_b):
        center_a, center_b = torch.broadcast_tensors(center_a, center_b)
        yaw_a, yaw_b = torch.broadcast_tensors(yaw_a, yaw_b)
        fa = torch.stack((torch.cos(yaw_a), torch.sin(yaw_a)), dim=-1)
        la = torch.stack((-torch.sin(yaw_a), torch.cos(yaw_a)), dim=-1)
        fb = torch.stack((torch.cos(yaw_b), torch.sin(yaw_b)), dim=-1)
        lb = torch.stack((-torch.sin(yaw_b), torch.cos(yaw_b)), dim=-1)
        axes = torch.stack((fa, la, fb, lb), dim=-2)
        projection = torch.abs(
            torch.sum((center_a - center_b).unsqueeze(-2) * axes, dim=-1)
        )
        ra = (
            torch.abs(torch.sum(axes * fa.unsqueeze(-2), dim=-1)) * half_a[0]
            + torch.abs(torch.sum(axes * la.unsqueeze(-2), dim=-1)) * half_a[1]
        )
        rb = (
            torch.abs(torch.sum(axes * fb.unsqueeze(-2), dim=-1)) * half_b[0]
            + torch.abs(torch.sum(axes * lb.unsqueeze(-2), dim=-1)) * half_b[1]
        )
        return torch.amax(projection - ra - rb, dim=-1)

    world = []
    world_yaw = []
    for agent in range(num_agents):
        world.append(local_to_world(
            sample_positions[0, :, agent, :effective_horizon], transforms[agent]
        ))
        world_yaw.append(local_yaw_to_world(
            sample_yaws[0, :, agent, :effective_horizon], transforms[agent]
        ))
    world = torch.stack(world, dim=1)  # [K,M,T,2]
    world_yaw = torch.stack(world_yaw, dim=1)  # [K,M,T]
    baseline_world_all = []
    baseline_world_yaw_all = []
    for agent in range(num_agents):
        baseline_world_all.append(local_to_world(
            baseline_positions[0, agent, :effective_horizon], transforms[agent]
        ))
        baseline_world_yaw_all.append(local_yaw_to_world(
            baseline_yaws[0, agent, :effective_horizon], transforms[agent]
        ))
    baseline_world_all = torch.stack(baseline_world_all, dim=0)
    baseline_world_yaw_all = torch.stack(baseline_world_yaw_all, dim=0)
    reference_world = local_to_world(reference_pos, transforms[ego])
    reference_world_yaw = local_yaw_to_world(reference_yaw, transforms[ego])
    target_world = local_to_world(target_pos, transforms[ego])
    target_world_yaw = local_yaw_to_world(target_yaw, transforms[ego])

    target_gaps = []
    reference_gaps = []
    for agent in target_background:
        target_gaps.append(signed_box_gap(
            world[:, agent], world_yaw[:, agent], 0.5 * extent[agent, :2],
            target_world[None], target_world_yaw[None], 0.5 * extent[ego, :2],
        ))
    for agent in background:
        reference_gaps.append(signed_box_gap(
            world[:, agent], world_yaw[:, agent], 0.5 * extent[agent, :2],
            reference_world[None], reference_world_yaw[None], 0.5 * extent[ego, :2],
        ))
    target_gap_tensor = torch.stack(target_gaps, dim=1)
    reference_gap_tensor = torch.stack(reference_gaps, dim=1)
    target_time_min = torch.amin(target_gap_tensor, dim=1)
    reference_time_min = torch.amin(reference_gap_tensor, dim=1)
    target_min = torch.amin(target_time_min, dim=1)
    reference_min = torch.amin(reference_time_min, dim=1)

    persistence_steps = int(target_min_persistence_steps)
    if persistence_steps < 1:
        raise ValueError("target_min_persistence_steps must be positive")
    if persistence_steps > effective_horizon:
        raise ValueError(
            "target_min_persistence_steps exceeds proposal horizon"
        )
    below_target = target_time_min <= float(target_max_clearance_m)
    running = torch.zeros(num_samples, device=device, dtype=torch.long)
    max_running = torch.zeros_like(running)
    for time_index in range(effective_horizon):
        running = torch.where(
            below_target[:, time_index], running + 1,
            torch.zeros_like(running),
        )
        max_running = torch.maximum(max_running, running)
    exposure_clearance = float(progressive_exposure_clearance_m)
    if (
        bool(progressive_persistence_first or progressive_dose_first)
        and exposure_clearance < float(target_max_clearance_m)
    ):
        raise ValueError(
            "progressive_exposure_clearance_m must be at least the terminal "
            "target_max_clearance_m"
        )
    below_exposure = target_time_min <= exposure_clearance
    exposure_running = torch.zeros(
        num_samples, device=device, dtype=torch.long
    )
    max_exposure_running = torch.zeros_like(exposure_running)
    for time_index in range(effective_horizon):
        exposure_running = torch.where(
            below_exposure[:, time_index], exposure_running + 1,
            torch.zeros_like(exposure_running),
        )
        max_exposure_running = torch.maximum(
            max_exposure_running, exposure_running
        )
    target_risk_dose = torch.sum(
        torch.clamp(exposure_clearance - target_time_min, min=0.0), dim=1
    )

    finite_gate = torch.isfinite(world).all(dim=-1).all(dim=-1).all(dim=-1)
    finite_gate &= torch.isfinite(world_yaw).all(dim=-1).all(dim=-1)
    reference_gate = reference_min > float(reference_safety_margin_m)
    terminal_target_gate = max_running >= persistence_steps
    baseline_target_gaps = []
    baseline_reference_gaps = []
    for agent in target_background:
        baseline_target_gaps.append(signed_box_gap(
            baseline_world_all[agent], baseline_world_yaw_all[agent],
            0.5 * extent[agent, :2], target_world, target_world_yaw,
            0.5 * extent[ego, :2],
        ))
    for agent in background:
        baseline_reference_gaps.append(signed_box_gap(
            baseline_world_all[agent], baseline_world_yaw_all[agent],
            0.5 * extent[agent, :2], reference_world,
            reference_world_yaw, 0.5 * extent[ego, :2],
        ))
    baseline_target_time_min = torch.amin(
        torch.stack(baseline_target_gaps, dim=0), dim=0
    )
    baseline_target = float(torch.amin(baseline_target_time_min).detach().cpu())
    baseline_reference = float(torch.amin(
        torch.stack(baseline_reference_gaps, dim=0)
    ).detach().cpu())
    baseline_running = 0
    baseline_max_running = 0
    for value in baseline_target_time_min:
        if bool((value <= float(target_max_clearance_m)).item()):
            baseline_running += 1
            baseline_max_running = max(baseline_max_running, baseline_running)
        else:
            baseline_running = 0
    baseline_safe_gate = (
        baseline_min_target_clearance_m is None
        or baseline_target >= float(baseline_min_target_clearance_m)
    )

    # Every non-ego pair must remain separated.  This is the online collateral
    # gate; the exact full-horizon collision audit is still authoritative.
    min_collateral = torch.full(
        (num_samples,), torch.inf, device=device, dtype=dtype
    )
    collateral_nonworsening = torch.ones(
        num_samples, device=device, dtype=torch.bool
    )
    baseline_min_collateral = float("inf")
    for offset, first in enumerate(background):
        for second in background[offset + 1:]:
            gap = signed_box_gap(
                world[:, first], world_yaw[:, first], 0.5 * extent[first, :2],
                world[:, second], world_yaw[:, second], 0.5 * extent[second, :2],
            )
            pair_min = torch.amin(gap, dim=-1)
            min_collateral = torch.minimum(min_collateral, pair_min)
            baseline_pair_min = torch.amin(signed_box_gap(
                baseline_world_all[first], baseline_world_yaw_all[first],
                0.5 * extent[first, :2], baseline_world_all[second],
                baseline_world_yaw_all[second], 0.5 * extent[second, :2],
            ))
            baseline_min_collateral = min(
                baseline_min_collateral,
                float(baseline_pair_min.detach().cpu()),
            )
            # "No new collateral" is relative to the unselected native mode.
            # If mode-0 is separated, retain the conservative positive margin;
            # if it already overlaps in the proposal horizon, a candidate may
            # only preserve or improve that exact pair. Full-rollout collision
            # attribution remains the authoritative acceptance gate.
            pair_allowed = torch.where(
                baseline_pair_min > float(collateral_margin_m),
                pair_min > float(collateral_margin_m),
                pair_min >= baseline_pair_min - 1.0e-6,
            )
            collateral_nonworsening &= pair_allowed
    collateral_gate = collateral_nonworsening

    bg_world = world[:, background]
    executed_horizon = min(effective_horizon, int(kinematic_horizon_steps))
    if executed_horizon < 2:
        raise ValueError("kinematic_horizon_steps must expose at least two positions")

    def scene_agent_tensor(value, expected_last_dim=None):
        if value is None:
            return None
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim >= 1 and tensor.shape[0] == 1 and tensor.ndim == 4:
            tensor = tensor[0]
        if expected_last_dim is not None and tensor.shape[-1] != expected_last_dim:
            raise ValueError("unexpected history tensor shape {}".format(tuple(tensor.shape)))
        return tensor

    history_pos = scene_agent_tensor(history_positions, expected_last_dim=2)
    history_yaw = scene_agent_tensor(history_yaws, expected_last_dim=1)
    history_avail = scene_agent_tensor(history_availabilities)
    physical_gate = torch.ones(num_samples, device=device, dtype=torch.bool)
    # Diagnostics only: keep the exact authoritative conjunction in
    # ``physical_gate``, while retaining enough attribution to distinguish a
    # genuinely impossible native mode from a single unrelated actor/metric
    # veto.  These tensors never participate in selection.
    physical_metric_gates = {
        name: torch.ones(num_samples, device=device, dtype=torch.bool)
        for name in ("speed", "acceleration", "jerk", "yaw_rate", "lateral_acc")
    }
    physical_agent_gates = []
    tail_fraction_rows = []
    bounds = None
    q99 = None
    if naturalness_profile is not None:
        bounds = torch.as_tensor(
            naturalness_profile["speed_bin_upper_bounds_mps"],
            device=device, dtype=dtype,
        )
        q99 = naturalness_profile["q99"]

    for agent in background:
        agent_physical_gate = torch.ones(
            num_samples, device=device, dtype=torch.bool
        )
        executed_world = world[:, agent, :executed_horizon]
        executed_world_yaw = world_yaw[:, agent, :executed_horizon]
        history_count = 0
        if history_pos is not None and history_yaw is not None:
            if history_pos.ndim != 3 or history_pos.shape[0] != num_agents:
                raise ValueError("expected history positions [M,H,2]")
            available = torch.ones(
                history_pos.shape[1], device=device, dtype=torch.bool
            )
            if history_avail is not None:
                if history_avail.ndim != 2 or history_avail.shape[0] != num_agents:
                    raise ValueError("expected history availabilities [M,H]")
                available = history_avail[agent].bool()
            available_index = torch.nonzero(available, as_tuple=False).flatten()
            if available_index.numel() >= 3:
                available_index = available_index[-3:]
                past_world = local_to_world(
                    history_pos[agent].index_select(0, available_index),
                    transforms[agent],
                )
                past_yaw = local_yaw_to_world(
                    history_yaw[agent].index_select(0, available_index),
                    transforms[agent],
                )
                history_count = int(past_world.shape[0])
                executed_world = torch.cat((
                    past_world.unsqueeze(0).expand(num_samples, -1, -1),
                    executed_world,
                ), dim=1)
                executed_world_yaw = torch.cat((
                    past_yaw.unsqueeze(0).expand(num_samples, -1),
                    executed_world_yaw,
                ), dim=1)

        speed_all = torch.linalg.vector_norm(
            torch.diff(executed_world, dim=1) / float(dt), dim=-1
        )
        acceleration_all = torch.diff(speed_all, dim=1) / float(dt)
        jerk_all = torch.diff(acceleration_all, dim=1) / float(dt)
        yaw_delta = torch.diff(executed_world_yaw, dim=1)
        yaw_rate_all = torch.atan2(
            torch.sin(yaw_delta), torch.cos(yaw_delta)
        ) / float(dt)
        speed = speed_all[:, max(0, history_count - 1):]
        acceleration = acceleration_all[:, max(0, history_count - 2):]
        jerk = jerk_all[:, max(0, history_count - 3):]
        yaw_rate = yaw_rate_all[:, max(0, history_count - 1):]
        lateral_acc = speed * torch.abs(yaw_rate)

        if speed.shape[1]:
            gate = torch.amax(speed, dim=1) <= float(max_speed_mps)
            physical_gate &= gate
            physical_metric_gates["speed"] &= gate
            agent_physical_gate &= gate
        if acceleration.shape[1]:
            gate = torch.amax(
                torch.abs(acceleration), dim=1
            ) <= float(max_acc_mps2)
            physical_gate &= gate
            physical_metric_gates["acceleration"] &= gate
            agent_physical_gate &= gate
        if jerk.shape[1]:
            gate = torch.amax(
                torch.abs(jerk), dim=1
            ) <= float(max_jerk_mps3)
            # Raw 0.1 s third differences are retained as a conservative
            # regulatory diagnostic, but are not an independent hard veto.
            # Jerk remains hard-constrained by the speed-conditioned empirical
            # q99 gate below and by the final full-rollout filtered/raw audit.
            physical_metric_gates["jerk"] &= gate
        if yaw_rate.shape[1]:
            gate = torch.amax(
                torch.abs(yaw_rate), dim=1
            ) <= float(max_yaw_rate_rps)
            physical_gate &= gate
            physical_metric_gates["yaw_rate"] &= gate
            agent_physical_gate &= gate
            gate = torch.amax(
                lateral_acc, dim=1
            ) <= float(max_lateral_acc_mps2)
            physical_gate &= gate
            physical_metric_gates["lateral_acc"] &= gate
            agent_physical_gate &= gate

        violations = torch.zeros(num_samples, device=device, dtype=dtype)
        counts = torch.zeros_like(violations)
        if bounds is not None:
            def accumulate(metric, context, name):
                if metric.shape[1] == 0:
                    return
                context = context[:, -metric.shape[1]:].contiguous()
                bins = torch.bucketize(context, bounds, right=True)
                threshold = torch.as_tensor(q99[name], device=device, dtype=dtype)
                selected_threshold = threshold.index_select(
                    0, bins.reshape(-1)
                ).reshape_as(metric)
                violations.add_(torch.sum(
                    torch.abs(metric) > selected_threshold, dim=1
                ))
                counts.add_(torch.full_like(counts, metric.shape[1]))

            accumulate(speed, speed, "speed_mps")
            accumulate(acceleration, speed, "abs_acc_mps2")
            accumulate(jerk, speed, "abs_jerk_mps3")
            accumulate(yaw_rate, speed, "abs_yaw_rate_rps")
            accumulate(lateral_acc, speed, "abs_lateral_acc_mps2")
        tail_fraction_rows.append(violations / torch.clamp(counts, min=1.0))
        physical_agent_gates.append(agent_physical_gate)

    per_agent_tail_fraction = torch.stack(tail_fraction_rows, dim=1)
    naturalness_gate = torch.all(
        per_agent_tail_fraction <= float(max_agent_tail_fraction), dim=1
    )
    # Proposal 0 is the fixed-seed ordinary-policy draw for the pCN contract.
    # Report its qualification separately: a local pCN search is not a fair
    # natural-adversarial test when the neighbourhood centre itself violates
    # the preregistered q99 profile.  These fields are diagnostics only and do
    # not alter selection or any hard gate.
    proposal_sample0_tail_fraction = per_agent_tail_fraction[0]
    proposal_sample0_naturalness_gate = bool(naturalness_gate[0].item())
    proposal_sample0_physical_gate = bool(physical_gate[0].item())

    baseline_world = baseline_world_all[background].unsqueeze(0)
    displacement = torch.linalg.vector_norm(
        bg_world - baseline_world, dim=-1
    )
    mean_displacement = torch.mean(displacement, dim=(1, 2))
    maximum_displacement = torch.amax(displacement, dim=(1, 2))
    risk_improvement_gate = (
        target_min <= baseline_target - float(min_risk_improvement_m)
    )
    best_target_index = int(torch.argmin(target_min).item())
    best_target_clearance = float(
        target_min[best_target_index].detach().cpu()
    )
    best_target_improvement = float(baseline_target - best_target_clearance)
    # A terminal gate is appropriate for one-shot selection but too strict for
    # receding-horizon FM steering: a native mode may make a small safe step
    # toward conflict without reaching the final severity band in one 0.5 s
    # proposal.  Progressive mode uses a monotone clearance barrier online;
    # the downstream full-rollout verifier still requires persistent target
    # failure.  No trajectory is edited in either mode.
    activation_limit = (
        float("inf")
        if progressive_activation_max_clearance_m is None
        else float(progressive_activation_max_clearance_m)
    )
    progressive_activation_gate = bool(baseline_target <= activation_limit)
    target_gate = (
        risk_improvement_gate & progressive_activation_gate
        if bool(progressive_target_barrier)
        else terminal_target_gate
    )
    non_target_hard_gate = (
        finite_gate & reference_gate & collateral_gate
        & physical_gate & naturalness_gate
    )
    hard_safe_indices = torch.nonzero(
        non_target_hard_gate, as_tuple=False
    ).flatten()
    if hard_safe_indices.numel():
        hard_safe_target = target_min.index_select(0, hard_safe_indices)
        best_hard_safe_offset = int(torch.argmin(hard_safe_target).item())
        best_hard_safe_index = int(
            hard_safe_indices[best_hard_safe_offset].item()
        )
        best_hard_safe_clearance = float(
            target_min[best_hard_safe_index].detach().cpu()
        )
        best_hard_safe_improvement = float(
            baseline_target - best_hard_safe_clearance
        )
    else:
        best_hard_safe_index = None
        best_hard_safe_clearance = None
        best_hard_safe_improvement = None
    valid = (
        non_target_hard_gate & target_gate
    )
    if not baseline_safe_gate:
        valid &= torch.zeros_like(valid)
    admissible = valid & risk_improvement_gate & torch.isfinite(target_min)
    selection_stage = "minimum_displacement"
    if bool(torch.any(admissible).item()):
        terminal_admissible = (
            non_target_hard_gate & terminal_target_gate
            & risk_improvement_gate & torch.isfinite(target_min)
        )
        if bool(
            progressive_risk_first
            or progressive_persistence_first
            or progressive_dose_first
        ) and bool(
            torch.any(terminal_admissible).item()
        ):
            # Once a complete native mode reaches the preregistered terminal
            # severity, causal parsimony becomes the objective: choose the
            # successful mode nearest to the executable policy baseline.
            admissible = terminal_admissible
            cost = torch.where(
                admissible, mean_displacement,
                torch.full_like(mean_displacement, torch.inf),
            )
            selected = int(torch.argmin(cost).item())
            selection_stage = "terminal_severity_then_minimum_displacement"
        elif bool(progressive_dose_first):
            # Integrated clearance deficit rewards both depth and duration;
            # clearance and displacement only break numerical ties.
            best_dose = torch.amax(target_risk_dose[admissible])
            dose_ties = admissible & torch.isclose(
                target_risk_dose, best_dose, rtol=0.0, atol=1.0e-6
            )
            best_risk = torch.amin(target_min[dose_ties])
            risk_ties = dose_ties & torch.isclose(
                target_min, best_risk, rtol=0.0, atol=1.0e-6
            )
            cost = torch.where(
                risk_ties, mean_displacement,
                torch.full_like(mean_displacement, torch.inf),
            )
            selected = int(torch.argmin(cost).item())
            selection_stage = "maximum_safe_target_risk_dose"
        elif bool(progressive_persistence_first):
            # Single-frame clearance minima produce narrow spikes that a
            # receding-horizon planner can avoid. Prefer a complete native
            # joint mode that sustains the semantic exposure band longest;
            # clearance and displacement are deterministic tie-breaks only.
            best_exposure_run = torch.amax(
                max_exposure_running[admissible]
            )
            exposure_ties = admissible & (
                max_exposure_running == best_exposure_run
            )
            best_risk = torch.amin(target_min[exposure_ties])
            risk_ties = exposure_ties & torch.isclose(
                target_min, best_risk, rtol=0.0, atol=1.0e-6
            )
            cost = torch.where(
                risk_ties, mean_displacement,
                torch.full_like(mean_displacement, torch.inf),
            )
            selected = int(torch.argmin(cost).item())
            selection_stage = "maximum_safe_risk_persistence"
        elif bool(progressive_risk_first):
            # Before terminal severity exists, minimizing displacement is
            # counterproductive: it repeatedly accepts centimetre-scale
            # changes and stalls. Continue along the safest admissible native
            # joint mode with the strongest target-risk progress. Displacement
            # is only a deterministic tie-break among numerically equal risks.
            best_risk = torch.amin(target_min[admissible])
            risk_ties = admissible & torch.isclose(
                target_min, best_risk, rtol=0.0, atol=1.0e-6
            )
            cost = torch.where(
                risk_ties, mean_displacement,
                torch.full_like(mean_displacement, torch.inf),
            )
            selected = int(torch.argmin(cost).item())
            selection_stage = "maximum_safe_risk_progress"
        else:
            cost = torch.where(
                admissible, mean_displacement,
                torch.full_like(mean_displacement, torch.inf),
            )
            selected = int(torch.argmin(cost).item())
        if (
            float(target_min[selected].detach().cpu())
            > baseline_target - float(min_risk_improvement_m)
        ):
            selected = 0
            fallback = True
        else:
            fallback = False
    else:
        selected = 0
        fallback = True

    positions = baseline_positions.clone()
    yaws = baseline_yaws.clone()
    if not fallback:
        for agent in background:
            positions[0, agent] = sample_positions[0, selected, agent]
            yaws[0, agent] = sample_yaws[0, selected, agent]
    audit = {
        "selected_sample_index": selected,
        "num_samples": int(num_samples),
        "valid_joint_mode_count": int(admissible.sum().item()),
        "finite_gate_count": int(finite_gate.sum().item()),
        "target_gate_count": int(target_gate.sum().item()),
        "terminal_target_gate_count": int(terminal_target_gate.sum().item()),
        "target_gate_mode": (
            "progressive_clearance_barrier"
            if bool(progressive_target_barrier)
            else "terminal_persistent_severity"
        ),
        "progressive_risk_first": bool(progressive_risk_first),
        "progressive_persistence_first": bool(
            progressive_persistence_first
        ),
        "progressive_dose_first": bool(progressive_dose_first),
        "progressive_exposure_clearance_m": exposure_clearance,
        "selection_stage": selection_stage,
        "progressive_activation_max_clearance_m": (
            None if progressive_activation_max_clearance_m is None
            else float(progressive_activation_max_clearance_m)
        ),
        "progressive_activation_gate": progressive_activation_gate,
        "risk_improvement_gate_count": int(risk_improvement_gate.sum().item()),
        "non_target_hard_gate_count": int(non_target_hard_gate.sum().item()),
        "risk_and_non_target_hard_gate_count": int(
            (risk_improvement_gate & non_target_hard_gate).sum().item()
        ),
        "best_target_sample_index": best_target_index,
        "best_target_clearance_m": best_target_clearance,
        "best_target_improvement_m": best_target_improvement,
        "best_hard_safe_sample_index": best_hard_safe_index,
        "best_hard_safe_target_clearance_m": best_hard_safe_clearance,
        "best_hard_safe_target_improvement_m": best_hard_safe_improvement,
        "target_clearance_quantiles_m": {
            "q00": float(torch.quantile(target_min, 0.00).detach().cpu()),
            "q25": float(torch.quantile(target_min, 0.25).detach().cpu()),
            "q50": float(torch.quantile(target_min, 0.50).detach().cpu()),
            "q75": float(torch.quantile(target_min, 0.75).detach().cpu()),
            "q100": float(torch.quantile(target_min, 1.00).detach().cpu()),
        },
        "baseline_target_safe_gate": bool(baseline_safe_gate),
        "reference_gate_count": int(reference_gate.sum().item()),
        "collateral_gate_count": int(collateral_gate.sum().item()),
        "physical_gate_count": int(physical_gate.sum().item()),
        "physical_gate_counts_by_metric": {
            name: int(gate.sum().item())
            for name, gate in physical_metric_gates.items()
        },
        "physical_hard_gate_metrics": [
            "speed", "acceleration", "yaw_rate", "lateral_acc",
            "empirical_q99_including_jerk",
        ],
        "raw_jerk_regulatory_limit_diagnostic_only": True,
        "physical_gate_counts_by_agent": {
            int(agent): int(gate.sum().item())
            for agent, gate in zip(background, physical_agent_gates)
        },
        "naturalness_gate_count": int(naturalness_gate.sum().item()),
        "proposal_sample0_naturalness_gate": (
            proposal_sample0_naturalness_gate
        ),
        "proposal_sample0_physical_gate": proposal_sample0_physical_gate,
        "proposal_sample0_max_agent_q99_tail_fraction": float(
            torch.amax(proposal_sample0_tail_fraction).detach().cpu()
        ),
        "proposal_sample0_q99_tail_fraction_by_agent": {
            int(agent): float(
                proposal_sample0_tail_fraction[offset].detach().cpu()
            )
            for offset, agent in enumerate(background)
        },
        "naturalness_gate_counts_by_agent": {
            int(agent): int((per_agent_tail_fraction[:, offset]
                             <= float(max_agent_tail_fraction)).sum().item())
            for offset, agent in enumerate(background)
        },
        "hard_gate_fallback": fallback,
        "fallback_preserved_policy_baseline": bool(fallback),
        "proposal_sample_executed": bool(not fallback),
        "all_non_ego_same_joint_sample": True,
        "non_ego_agent_count": len(background),
        "target_agent_indices": target_background,
        "target_attribution_locked": target_agent_indices is not None,
        "selected_target_min_clearance_m": float(
            baseline_target if fallback else target_min[selected].detach().cpu()
        ),
        "baseline_target_min_clearance_m": baseline_target,
        "selected_target_risk_improvement_m": float(
            0.0 if fallback
            else baseline_target - float(target_min[selected].detach().cpu())
        ),
        "selected_target_max_consecutive_risk_steps": int(
            baseline_max_running if fallback else max_running[selected].detach().cpu()
        ),
        "selected_target_max_consecutive_exposure_steps": int(
            0 if fallback else max_exposure_running[selected].detach().cpu()
        ),
        "selected_target_risk_dose_m_steps": float(
            0.0 if fallback else target_risk_dose[selected].detach().cpu()
        ),
        "best_admissible_target_risk_dose_m_steps": float(
            torch.amax(target_risk_dose[admissible]).detach().cpu()
            if bool(torch.any(admissible).item()) else 0.0
        ),
        "best_admissible_max_consecutive_exposure_steps": int(
            torch.amax(max_exposure_running[admissible]).detach().cpu()
            if bool(torch.any(admissible).item()) else 0
        ),
        "target_min_persistence_steps": persistence_steps,
        "selected_reference_min_clearance_m": float(
            baseline_reference if fallback else reference_min[selected].detach().cpu()
        ),
        "selected_min_collateral_clearance_m": float(
            baseline_min_collateral
            if fallback else min_collateral[selected].detach().cpu()
        ),
        # The admissibility comparison is against the actually executable
        # policy baseline, not correlated proposal mode 0.  Keep the legacy
        # key for old audit readers, but expose the correct name explicitly.
        "selected_collateral_nonworsening_vs_policy_baseline": bool(
            True if fallback else collateral_nonworsening[selected].item()
        ),
        "selected_collateral_nonworsening_vs_mode0": bool(
            True if fallback else collateral_nonworsening[selected].item()
        ),
        "selected_mean_displacement_from_policy_baseline_m": float(
            0.0 if fallback else mean_displacement[selected].detach().cpu()
        ),
        "selected_max_displacement_from_policy_baseline_m": float(
            0.0 if fallback else maximum_displacement[selected].detach().cpu()
        ),
        # A fallback executes the policy baseline and executes no correlated
        # proposal.  Do not mislabel the unexecuted proposal-0 tail score as
        # an executed-trajectory measurement.  Full-rollout HDF5 remains the
        # authority for baseline naturalness after fallback.
        "proposal_sample_max_agent_q99_tail_fraction": float(
            torch.amax(per_agent_tail_fraction[selected]).detach().cpu()
        ),
        "executed_sample_max_agent_q99_tail_fraction": (
            None if fallback else float(
                torch.amax(per_agent_tail_fraction[selected]).detach().cpu()
            )
        ),
        "selected_max_agent_q99_tail_fraction": (
            None if fallback else float(
                torch.amax(per_agent_tail_fraction[selected]).detach().cpu()
            )
        ),
        "kinematic_horizon_steps": int(executed_horizon),
        "boundary_kinematics_used": bool(
            history_pos is not None and history_yaw is not None
        ),
        "trajectory_modified": False,
        "projection": False,
        "row_mixing": False,
        "proposal_only": True,
        "full_horizon_verifier_required": True,
    }
    if agent_centric:
        positions, yaws = positions[0], yaws[0]
    return positions, yaws, audit


def select_marginal_joint_tail_action(
    sample_positions,
    sample_yaws,
    baseline_positions,
    baseline_yaws,
    ego_positions,
    ego_yaws,
    world_from_agent,
    extents,
    ego_index,
    receiver_index,
    collateral_margin_m=0.25,
    reference_safety_margin_m=0.0,
    min_risk_improvement_m=0.05,
    max_target_clearance_m=None,
    min_baseline_target_clearance_m=None,
    dt=0.1,
    max_speed_mps=20.0,
    max_acc_mps2=4.0,
    max_jerk_mps3=8.37,
    max_yaw_rate_rps=0.95,
    max_lateral_acc_mps2=4.0,
    kinematic_horizon_steps=5,
    naturalness_profile=None,
    history_positions=None,
    history_yaws=None,
    history_availabilities=None,
    target_ego_positions=None,
    target_ego_yaws=None,
    minimize_causal_change=False,
):
    """Compose a joint-tail scene from untouched per-agent FM samples.

    Only the receiver's sample index may change.  Every candidate trajectory is
    an unmodified model output; candidates that introduce a receiver collision
    with any non-ego actor are rejected.  The objective is ego/receiver minimum
    oriented-box separating-axis clearance in world coordinates.
    """
    agent_centric_layout = sample_positions.ndim == 4
    if agent_centric_layout:
        # NaiveAgentCentricToSceneCentricWrapper exposes [M,K,T,D].  Convert
        # at this single boundary and restore the original layout on return.
        sample_positions = sample_positions.permute(1, 0, 2, 3).unsqueeze(0)
        sample_yaws = sample_yaws.permute(1, 0, 2, 3).unsqueeze(0)
        baseline_positions = baseline_positions.unsqueeze(0)
        baseline_yaws = baseline_yaws.unsqueeze(0)
    if sample_positions.ndim != 5 or sample_yaws.ndim != 5:
        raise ValueError(
            "expected samples [B,K,M,T,D] or agent-centric [M,K,T,D]; "
            "got positions={} yaws={}".format(
                tuple(sample_positions.shape), tuple(sample_yaws.shape)
            )
        )
    if sample_positions.shape[:4] != sample_yaws.shape[:4]:
        raise ValueError("sample position/yaw shapes disagree")
    if sample_positions.shape[0] != 1 or baseline_positions.shape[0] != 1:
        raise ValueError("marginal joint-tail screen currently requires batch size one")
    _, num_samples, num_agents, horizon, _ = sample_positions.shape
    ego_index = int(ego_index)
    receiver_index = int(receiver_index)
    if not (0 <= ego_index < num_agents and 0 <= receiver_index < num_agents):
        raise IndexError("ego/receiver index exceeds modeled agent count")
    if ego_index == receiver_index:
        raise ValueError("receiver must be a non-ego agent")

    device = sample_positions.device
    dtype = sample_positions.dtype
    transforms = torch.as_tensor(world_from_agent, device=device, dtype=dtype)
    if transforms.ndim == 4:
        if transforms.shape[0] != 1:
            raise ValueError("unexpected world_from_agent batch size")
        transforms = transforms[0]
    if transforms.shape != (num_agents, 3, 3):
        raise ValueError("expected one 3x3 transform per scene agent")
    extent = torch.as_tensor(extents, device=device, dtype=dtype)
    if extent.ndim == 4:
        if extent.shape[0] != 1 or extent.shape[1] != num_agents:
            raise ValueError("unexpected batched extent shape")
        extent = extent[0, :, -1]
    if extent.ndim == 3:
        if extent.shape[0] == 1 and extent.shape[1] == num_agents:
            extent = extent[0]
        elif extent.shape[0] == num_agents:
            extent = extent[:, -1]
        else:
            raise ValueError("unexpected extent shape")
    if extent.ndim != 2 or extent.shape[0] != num_agents:
        raise ValueError("expected one extent per scene agent")

    target_local = sample_positions[0, :, receiver_index]
    reference_ego_local = ego_positions[0, :horizon]
    reference_ego_local_yaw = ego_yaws[0, :horizon]
    if target_ego_positions is None:
        objective_ego_local = reference_ego_local
        objective_ego_local_yaw = reference_ego_local_yaw
    else:
        if target_ego_yaws is None:
            raise ValueError("target_ego_yaws are required with target_ego_positions")
        objective_ego_local = target_ego_positions[0, :horizon]
        objective_ego_local_yaw = target_ego_yaws[0, :horizon]
    effective_horizon = min(
        horizon,
        reference_ego_local.shape[0],
        objective_ego_local.shape[0],
    )
    target_local = target_local[:, :effective_horizon]
    reference_ego_local = reference_ego_local[:effective_horizon]
    reference_ego_local_yaw = reference_ego_local_yaw[:effective_horizon]
    objective_ego_local = objective_ego_local[:effective_horizon]
    objective_ego_local_yaw = objective_ego_local_yaw[:effective_horizon]

    def local_to_world(points, transform):
        ones = torch.ones(*points.shape[:-1], 1, device=device, dtype=dtype)
        homogeneous = torch.cat((points, ones), dim=-1)
        return torch.matmul(homogeneous, transform.transpose(-1, -2))[..., :2]

    def local_yaw_to_world(local_yaw, transform):
        offset = torch.atan2(transform[1, 0], transform[0, 0])
        return local_yaw.squeeze(-1) + offset

    def signed_oriented_box_gap(
        center_a, yaw_a, half_extent_a, center_b, yaw_b, half_extent_b
    ):
        """SAT signed gap: positive separated, non-positive overlapping."""
        center_a, center_b = torch.broadcast_tensors(center_a, center_b)
        yaw_a, yaw_b = torch.broadcast_tensors(yaw_a, yaw_b)
        forward_a = torch.stack((torch.cos(yaw_a), torch.sin(yaw_a)), dim=-1)
        lateral_a = torch.stack((-torch.sin(yaw_a), torch.cos(yaw_a)), dim=-1)
        forward_b = torch.stack((torch.cos(yaw_b), torch.sin(yaw_b)), dim=-1)
        lateral_b = torch.stack((-torch.sin(yaw_b), torch.cos(yaw_b)), dim=-1)
        axes = torch.stack(
            (forward_a, lateral_a, forward_b, lateral_b), dim=-2
        )
        delta = center_a - center_b
        center_projection = torch.abs(
            torch.sum(delta.unsqueeze(-2) * axes, dim=-1)
        )
        radius_a = (
            torch.abs(torch.sum(axes * forward_a.unsqueeze(-2), dim=-1))
            * half_extent_a[0]
            + torch.abs(torch.sum(axes * lateral_a.unsqueeze(-2), dim=-1))
            * half_extent_a[1]
        )
        radius_b = (
            torch.abs(torch.sum(axes * forward_b.unsqueeze(-2), dim=-1))
            * half_extent_b[0]
            + torch.abs(torch.sum(axes * lateral_b.unsqueeze(-2), dim=-1))
            * half_extent_b[1]
        )
        return torch.amax(center_projection - radius_a - radius_b, dim=-1)

    target_world = local_to_world(target_local, transforms[receiver_index])
    objective_ego_world = local_to_world(
        objective_ego_local, transforms[ego_index]
    )
    reference_ego_world = local_to_world(
        reference_ego_local, transforms[ego_index]
    )
    target_world_yaw = local_yaw_to_world(
        sample_yaws[0, :, receiver_index, :effective_horizon],
        transforms[receiver_index],
    )
    objective_ego_world_yaw = local_yaw_to_world(
        objective_ego_local_yaw, transforms[ego_index]
    )
    reference_ego_world_yaw = local_yaw_to_world(
        reference_ego_local_yaw, transforms[ego_index]
    )
    target_ego_clearance = signed_oriented_box_gap(
        target_world,
        target_world_yaw,
        0.5 * extent[receiver_index, :2],
        objective_ego_world[None],
        objective_ego_world_yaw[None],
        0.5 * extent[ego_index, :2],
    )
    risk = torch.amin(target_ego_clearance, dim=-1)
    reference_ego_clearance = signed_oriented_box_gap(
        target_world,
        target_world_yaw,
        0.5 * extent[receiver_index, :2],
        reference_ego_world[None],
        reference_ego_world_yaw[None],
        0.5 * extent[ego_index, :2],
    )
    reference_risk = torch.amin(reference_ego_clearance, dim=-1)
    # Every option is already a byte-for-byte native FM sample.  Adversarial
    # severity, reference-planner safety, collateral clearance, dynamics and
    # empirical naturalness are hard constraints below.  Among feasible
    # options, minimize the causal motion change from factual mode 0 instead
    # of pulling the receiver as deeply as possible into a collision.
    displacement_from_factual = torch.linalg.vector_norm(
        target_world - target_world[0:1], dim=-1
    )
    mean_displacement_from_factual = torch.mean(
        displacement_from_factual, dim=-1
    )
    max_displacement_from_factual = torch.amax(
        displacement_from_factual, dim=-1
    )

    executed_horizon = min(effective_horizon, int(kinematic_horizon_steps))
    if executed_horizon < 2:
        raise ValueError("kinematic_horizon_steps must expose at least two positions")
    executed_world = target_world[:, :executed_horizon]
    executed_world_yaw = target_world_yaw[:, :executed_horizon]

    def scene_agent_tensor(value, expected_last_dim=None):
        if value is None:
            return None
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim >= 1 and tensor.shape[0] == 1 and (
            tensor.ndim == 4 or (tensor.ndim == 3 and expected_last_dim is None)
        ):
            tensor = tensor[0]
        if expected_last_dim is not None and tensor.shape[-1] != expected_last_dim:
            raise ValueError("unexpected history tensor shape {}".format(tuple(tensor.shape)))
        return tensor

    history_pos = scene_agent_tensor(history_positions, expected_last_dim=2)
    history_yaw = scene_agent_tensor(history_yaws, expected_last_dim=1)
    history_avail = scene_agent_tensor(history_availabilities)
    boundary_kinematics_used = False
    history_count = 0
    if history_pos is not None and history_yaw is not None:
        if history_pos.ndim != 3 or history_pos.shape[0] != num_agents:
            raise ValueError("expected history positions [M,H,2]")
        if history_yaw.ndim != 3 or history_yaw.shape[:2] != history_pos.shape[:2]:
            raise ValueError("history position/yaw shapes disagree")
        available = torch.ones(
            history_pos.shape[1], device=device, dtype=torch.bool
        )
        if history_avail is not None:
            if history_avail.ndim != 2 or history_avail.shape[0] != num_agents:
                raise ValueError("expected history availabilities [M,H]")
            available = history_avail[receiver_index].bool()
        available_index = torch.nonzero(available, as_tuple=False).flatten()
        if available_index.numel() >= 3:
            available_index = available_index[-3:]
            receiver_history_local = history_pos[receiver_index].index_select(
                0, available_index
            )
            receiver_history_yaw = history_yaw[receiver_index].index_select(
                0, available_index
            )
            receiver_history_world = local_to_world(
                receiver_history_local, transforms[receiver_index]
            )
            receiver_history_world_yaw = local_yaw_to_world(
                receiver_history_yaw, transforms[receiver_index]
            )
            history_count = int(receiver_history_world.shape[0])
            executed_world = torch.cat((
                receiver_history_world.unsqueeze(0).expand(num_samples, -1, -1),
                executed_world,
            ), dim=1)
            executed_world_yaw = torch.cat((
                receiver_history_world_yaw.unsqueeze(0).expand(num_samples, -1),
                executed_world_yaw,
            ), dim=1)
            boundary_kinematics_used = True

    speed_all = torch.linalg.vector_norm(
        torch.diff(executed_world, dim=1) / float(dt), dim=-1
    )
    acceleration_all = torch.diff(speed_all, dim=1) / float(dt)
    jerk_all = torch.diff(acceleration_all, dim=1) / float(dt)
    yaw_delta = torch.diff(executed_world_yaw, dim=1)
    yaw_rate_all = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)) / float(dt)
    # Ignore past-only quantities while retaining every derivative that spans
    # the current-state -> future-action boundary.
    speed = speed_all[:, max(0, history_count - 1):]
    acceleration = acceleration_all[:, max(0, history_count - 2):]
    jerk = jerk_all[:, max(0, history_count - 3):]
    yaw_rate = yaw_rate_all[:, max(0, history_count - 1):]
    lateral_acc = speed * torch.abs(yaw_rate)
    # The CCDiff runtime uses an older PyTorch that does not accept a tuple of
    # dimensions in Tensor.all().  Reduce coordinates, then time explicitly.
    valid = torch.isfinite(target_world).all(dim=-1).all(dim=-1)
    valid &= torch.isfinite(target_world_yaw).all(dim=-1)
    valid &= reference_risk > float(reference_safety_margin_m)
    if max_target_clearance_m is not None:
        valid &= risk <= float(max_target_clearance_m)
    if speed.shape[1]:
        valid &= torch.amax(speed, dim=-1) <= float(max_speed_mps)
    if acceleration.shape[1]:
        valid &= torch.amax(torch.abs(acceleration), dim=-1) <= float(max_acc_mps2)
    if jerk.shape[1]:
        valid &= torch.amax(torch.abs(jerk), dim=-1) <= float(max_jerk_mps3)
    if yaw_rate.shape[1]:
        valid &= torch.amax(torch.abs(yaw_rate), dim=-1) <= float(max_yaw_rate_rps)
        valid &= torch.amax(lateral_acc, dim=-1) <= float(max_lateral_acc_mps2)
    kinematic_valid_count = int(valid.sum().item())

    naturalness_violation_counts = {}
    naturalness_valid = torch.ones(
        num_samples, device=device, dtype=torch.bool
    )
    if naturalness_profile is not None:
        required_metrics = (
            "speed_mps",
            "abs_acc_mps2",
            "abs_jerk_mps3",
            "abs_yaw_rate_rps",
            "abs_lateral_acc_mps2",
        )
        upper_bounds = naturalness_profile.get(
            "speed_bin_upper_bounds_mps"
        )
        q99 = naturalness_profile.get("q99")
        if upper_bounds is None or q99 is None:
            raise ValueError("invalid marginal-tail naturalness profile")
        if any(name not in q99 for name in required_metrics):
            raise ValueError("naturalness profile is missing q99 metrics")
        bounds = torch.as_tensor(
            upper_bounds, device=device, dtype=dtype
        )

        def q99_violation_count(metric, speed_context, metric_name):
            if metric.shape[1] == 0:
                return torch.zeros(
                    num_samples, device=device, dtype=torch.long
                )
            context = speed_context[:, -metric.shape[1]:].contiguous()
            # right=True exactly matches the offline bins [0,2), [2,5), ...
            bin_index = torch.bucketize(context, bounds, right=True)
            threshold = torch.as_tensor(
                q99[metric_name], device=device, dtype=dtype
            )
            if threshold.numel() != bounds.numel() + 1:
                raise ValueError(
                    "naturalness q99/bin count mismatch for {}".format(
                        metric_name
                    )
                )
            selected_threshold = threshold.index_select(
                0, bin_index.reshape(-1)
            ).reshape_as(metric)
            violations = torch.abs(metric) > selected_threshold
            return torch.sum(violations, dim=-1)

        metric_inputs = {
            "speed_mps": (speed, speed),
            "abs_acc_mps2": (acceleration, speed),
            "abs_jerk_mps3": (jerk, speed),
            "abs_yaw_rate_rps": (yaw_rate, speed),
            "abs_lateral_acc_mps2": (lateral_acc, speed),
        }
        for metric_name in required_metrics:
            counts = q99_violation_count(
                *metric_inputs[metric_name], metric_name
            )
            naturalness_violation_counts[metric_name] = counts
            naturalness_valid &= counts <= counts[0]
        naturalness_valid_count = int(
            (valid & naturalness_valid).sum().item()
        )
        valid &= naturalness_valid
    else:
        naturalness_valid_count = kinematic_valid_count

    def per_sample_max(values):
        if values.shape[1] == 0:
            return [0.0] * int(num_samples)
        return [float(value) for value in torch.amax(torch.abs(values), dim=-1).detach().cpu()]
    for other in range(num_agents):
        if other in {ego_index, receiver_index}:
            continue
        other_local = baseline_positions[0, other, :effective_horizon]
        other_world = local_to_world(other_local, transforms[other])
        other_world_yaw = local_yaw_to_world(
            baseline_yaws[0, other, :effective_horizon], transforms[other]
        )
        clearance = signed_oriented_box_gap(
            target_world,
            target_world_yaw,
            0.5 * extent[receiver_index, :2],
            other_world[None],
            other_world_yaw[None],
            0.5 * extent[other, :2],
        )
        valid &= torch.amin(clearance, dim=-1) > float(collateral_margin_m)
    finite = torch.isfinite(risk)
    admissible = valid & finite
    baseline_target_safe_gate_pass = bool(
        min_baseline_target_clearance_m is None
        or (
            torch.isfinite(risk[0]).item()
            and float(risk[0].detach().cpu())
            >= float(min_baseline_target_clearance_m)
        )
    )
    hard_gate_fallback = (
        not bool(torch.any(admissible).item())
        or not baseline_target_safe_gate_pass
    )
    if hard_gate_fallback:
        # Preserve the factual mode only as an explicitly attested fallback;
        # downstream acceptance must reject this event as not hard-gated.
        selected = 0
    else:
        objective = (
            mean_displacement_from_factual
            if minimize_causal_change else risk
        )
        masked_cost = torch.where(
            admissible, objective, torch.full_like(objective, torch.inf),
        )
        selected = int(torch.argmin(masked_cost).item())
    baseline_risk = float(risk[0].detach().cpu())
    selected_risk = float(risk[selected].detach().cpu())
    baseline_reference_risk = float(reference_risk[0].detach().cpu())
    selected_reference_risk = float(reference_risk[selected].detach().cpu())
    baseline_hard_valid = bool(admissible[0].item())
    enforce_improvement = bool(
        min_baseline_target_clearance_m is not None or baseline_hard_valid
    )
    if (
        enforce_improvement
        and selected_risk > baseline_risk - float(min_risk_improvement_m)
    ):
        selected = 0
        selected_risk = baseline_risk
        selected_reference_risk = baseline_reference_risk

    positions = baseline_positions.clone()
    yaws = baseline_yaws.clone()
    positions[0, receiver_index] = sample_positions[0, selected, receiver_index]
    yaws[0, receiver_index] = sample_yaws[0, selected, receiver_index]
    audit = {
        "selected_sample_index": selected,
        "num_samples": int(num_samples),
        "valid_sample_count": int(valid.sum().item()),
        "kinematic_valid_sample_count": kinematic_valid_count,
        "naturalness_profile_enabled": naturalness_profile is not None,
        "naturalness_valid_sample_count": naturalness_valid_count,
        "sample_naturalness_q99_violation_counts": {
            name: [int(value) for value in counts.detach().cpu()]
            for name, counts in naturalness_violation_counts.items()
        },
        "baseline_hard_valid": baseline_hard_valid,
        "hard_gate_fallback": hard_gate_fallback,
        "kinematic_horizon_steps": executed_horizon,
        "boundary_kinematics_used": boundary_kinematics_used,
        "sample_max_speed_mps": per_sample_max(speed),
        "sample_max_acc_mps2": per_sample_max(acceleration),
        "sample_max_jerk_mps3": per_sample_max(jerk),
        "sample_max_yaw_rate_rps": per_sample_max(yaw_rate),
        "sample_max_lateral_acc_mps2": per_sample_max(lateral_acc),
        "sample_mean_displacement_from_factual_m": [
            float(value)
            for value in mean_displacement_from_factual.detach().cpu()
        ],
        "sample_max_displacement_from_factual_m": [
            float(value)
            for value in max_displacement_from_factual.detach().cpu()
        ],
        "selected_mean_displacement_from_factual_m": float(
            mean_displacement_from_factual[selected].detach().cpu()
        ),
        "selected_max_displacement_from_factual_m": float(
            max_displacement_from_factual[selected].detach().cpu()
        ),
        "constrained_objective": (
            "min_mean_displacement_from_factual"
            if minimize_causal_change else "min_target_clearance"
        ),
        "baseline_min_clearance_m": baseline_risk,
        "selected_min_clearance_m": selected_risk,
        "baseline_target_min_clearance_m": baseline_risk,
        "selected_target_min_clearance_m": selected_risk,
        "baseline_reference_min_clearance_m": baseline_reference_risk,
        "selected_reference_min_clearance_m": selected_reference_risk,
        "dual_planner_objective": target_ego_positions is not None,
        "receiver_index": receiver_index,
        "collateral_margin_m": float(collateral_margin_m),
        "reference_safety_margin_m": float(reference_safety_margin_m),
        "min_risk_improvement_m": float(min_risk_improvement_m),
        "max_target_clearance_m": (
            None if max_target_clearance_m is None
            else float(max_target_clearance_m)
        ),
        "min_baseline_target_clearance_m": (
            None
            if min_baseline_target_clearance_m is None
            else float(min_baseline_target_clearance_m)
        ),
        "baseline_target_safe_gate_pass": baseline_target_safe_gate_pass,
        "selected_target_severity_gate_pass": bool(
            max_target_clearance_m is None
            or selected_risk <= float(max_target_clearance_m)
        ),
        "trajectory_modified": False,
    }
    if agent_centric_layout:
        positions = positions[0]
        yaws = yaws[0]
    return positions, yaws, audit


def select_multi_agent_marginal_joint_tail_action(
    sample_positions,
    sample_yaws,
    baseline_positions,
    baseline_yaws,
    ego_positions,
    ego_yaws,
    world_from_agent,
    extents,
    ego_index,
    receiver_indices,
    collateral_margin_m=0.25,
    reference_safety_margin_m=0.0,
    min_risk_improvement_m=0.05,
    max_target_clearance_m=None,
    min_baseline_target_clearance_m=None,
    dt=0.1,
    max_speed_mps=20.0,
    max_acc_mps2=4.0,
    max_jerk_mps3=8.37,
    max_yaw_rate_rps=0.95,
    max_lateral_acc_mps2=4.0,
    kinematic_horizon_steps=5,
    naturalness_profile=None,
    history_positions=None,
    history_yaws=None,
    history_availabilities=None,
    target_ego_positions=None,
    target_ego_yaws=None,
    minimize_causal_change=False,
):
    """Greedily compose native marginal modes for several receivers.

    Each selected receiver trajectory remains byte-for-byte one model sample.
    Later receivers are checked against already selected earlier receivers, so
    the composition cannot silently add receiver-to-receiver collateral.
    """
    positions = baseline_positions.clone()
    yaws = baseline_yaws.clone()
    audits = []
    seen = set()
    for receiver_index in receiver_indices:
        receiver_index = int(receiver_index)
        if receiver_index in seen:
            raise ValueError("receiver indices must be unique")
        seen.add(receiver_index)
        positions, yaws, audit = select_marginal_joint_tail_action(
            sample_positions=sample_positions,
            sample_yaws=sample_yaws,
            baseline_positions=positions,
            baseline_yaws=yaws,
            ego_positions=ego_positions,
            ego_yaws=ego_yaws,
            world_from_agent=world_from_agent,
            extents=extents,
            ego_index=ego_index,
            receiver_index=receiver_index,
            collateral_margin_m=collateral_margin_m,
            reference_safety_margin_m=reference_safety_margin_m,
            min_risk_improvement_m=min_risk_improvement_m,
            max_target_clearance_m=max_target_clearance_m,
            min_baseline_target_clearance_m=(
                min_baseline_target_clearance_m
            ),
            dt=dt,
            max_speed_mps=max_speed_mps,
            max_acc_mps2=max_acc_mps2,
            max_jerk_mps3=max_jerk_mps3,
            max_yaw_rate_rps=max_yaw_rate_rps,
            max_lateral_acc_mps2=max_lateral_acc_mps2,
            kinematic_horizon_steps=kinematic_horizon_steps,
            naturalness_profile=naturalness_profile,
            history_positions=history_positions,
            history_yaws=history_yaws,
            history_availabilities=history_availabilities,
            target_ego_positions=target_ego_positions,
            target_ego_yaws=target_ego_yaws,
            minimize_causal_change=minimize_causal_change,
        )
        audits.append(audit)
    return positions, yaws, audits


def select_sparse_marginal_joint_tail_action(
    sample_positions,
    sample_yaws,
    baseline_positions,
    baseline_yaws,
    ego_positions,
    ego_yaws,
    world_from_agent,
    extents,
    ego_index,
    receiver_indices,
    collateral_margin_m=0.25,
    reference_safety_margin_m=0.0,
    min_risk_improvement_m=0.05,
    max_target_clearance_m=None,
    min_baseline_target_clearance_m=None,
    dt=0.1,
    max_speed_mps=20.0,
    max_acc_mps2=4.0,
    max_jerk_mps3=8.37,
    max_yaw_rate_rps=0.95,
    max_lateral_acc_mps2=4.0,
    kinematic_horizon_steps=5,
    naturalness_profile=None,
    history_positions=None,
    history_yaws=None,
    history_availabilities=None,
    target_ego_positions=None,
    target_ego_yaws=None,
    locked_receiver_index=None,
    minimize_causal_change=False,
):
    """Choose at most one receiver/native mode from a preregistered pool.

    The fixed-receiver screen can miss a dangerous native marginal merely
    because the wrong actor was named before sampling.  This selector applies
    the *same* objective and hard gates to every eligible non-ego receiver,
    then copies only the globally lowest-clearance admissible native mode.  A
    no-op is a valid outcome; it is preferable to forcing an invalid mode.
    """
    receiver_indices = [int(index) for index in receiver_indices]
    if not receiver_indices:
        raise ValueError("sparse marginal-tail search requires receivers")
    if len(set(receiver_indices)) != len(receiver_indices):
        raise ValueError("receiver indices must be unique")
    if locked_receiver_index is not None:
        locked_receiver_index = int(locked_receiver_index)
        if locked_receiver_index not in receiver_indices:
            raise ValueError("locked receiver is outside the eligible pool")
        receiver_indices = [locked_receiver_index]

    candidates = []
    fallback_receiver_count = 0
    baseline_target_unsafe_receiver_count = 0
    for receiver_index in receiver_indices:
        candidate_positions, candidate_yaws, audit = (
            select_marginal_joint_tail_action(
                sample_positions=sample_positions,
                sample_yaws=sample_yaws,
                baseline_positions=baseline_positions,
                baseline_yaws=baseline_yaws,
                ego_positions=ego_positions,
                ego_yaws=ego_yaws,
                world_from_agent=world_from_agent,
                extents=extents,
                ego_index=ego_index,
                receiver_index=receiver_index,
                collateral_margin_m=collateral_margin_m,
                reference_safety_margin_m=reference_safety_margin_m,
                min_risk_improvement_m=min_risk_improvement_m,
                max_target_clearance_m=max_target_clearance_m,
                min_baseline_target_clearance_m=(
                    min_baseline_target_clearance_m
                ),
                dt=dt,
                max_speed_mps=max_speed_mps,
                max_acc_mps2=max_acc_mps2,
                max_jerk_mps3=max_jerk_mps3,
                max_yaw_rate_rps=max_yaw_rate_rps,
                max_lateral_acc_mps2=max_lateral_acc_mps2,
                kinematic_horizon_steps=kinematic_horizon_steps,
                naturalness_profile=naturalness_profile,
                history_positions=history_positions,
                history_yaws=history_yaws,
                history_availabilities=history_availabilities,
                target_ego_positions=target_ego_positions,
                target_ego_yaws=target_ego_yaws,
                minimize_causal_change=minimize_causal_change,
            )
        )
        fallback_receiver_count += int(audit["hard_gate_fallback"])
        baseline_target_unsafe_receiver_count += int(
            not audit.get("baseline_target_safe_gate_pass", True)
        )
        risk_improved = (
            audit["selected_target_min_clearance_m"]
            <= audit["baseline_target_min_clearance_m"]
            - float(min_risk_improvement_m)
        )
        if (
            not audit["hard_gate_fallback"]
            and audit["selected_sample_index"] != 0
            and risk_improved
        ):
            primary_objective = (
                float(audit["selected_mean_displacement_from_factual_m"])
                if minimize_causal_change
                else float(audit["selected_target_min_clearance_m"])
            )
            candidates.append((
                primary_objective,
                float(audit["selected_target_min_clearance_m"]),
                receiver_index,
                candidate_positions,
                candidate_yaws,
                audit,
            ))

    if candidates:
        _, _, _, positions, yaws, audit = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        audit = dict(audit)
        selected_receiver_index = int(audit["receiver_index"])
        sparse_noop = False
    else:
        positions = baseline_positions.clone()
        yaws = baseline_yaws.clone()
        # Keep the no-op attestation compact and numerically serializable.
        audit = {
            "receiver_index": -1,
            "selected_sample_index": 0,
            "num_samples": int(
                sample_positions.shape[1]
                if sample_positions.ndim == 5
                else sample_positions.shape[1]
            ),
            "valid_sample_count": 0,
            "kinematic_valid_sample_count": 0,
            "baseline_min_clearance_m": float("inf"),
            "selected_min_clearance_m": float("inf"),
            "baseline_target_min_clearance_m": float("inf"),
            "selected_target_min_clearance_m": float("inf"),
            "baseline_reference_min_clearance_m": float("inf"),
            "selected_reference_min_clearance_m": float("inf"),
            "dual_planner_objective": target_ego_positions is not None,
            "hard_gate_fallback": False,
            "baseline_hard_valid": True,
            "baseline_target_safe_gate_pass": True,
            "boundary_kinematics_used": history_positions is not None,
            "trajectory_modified": False,
            "selected_mean_displacement_from_factual_m": 0.0,
            "selected_max_displacement_from_factual_m": 0.0,
            "constrained_objective": (
                "min_mean_displacement_from_factual"
                if minimize_causal_change else "min_target_clearance"
            ),
        }
        selected_receiver_index = -1
        sparse_noop = True

    audit.update({
        "auto_receiver_search": True,
        "receiver_lock_active": locked_receiver_index is not None,
        "locked_receiver_index": (
            -1 if locked_receiver_index is None else locked_receiver_index
        ),
        "evaluated_receiver_count": len(receiver_indices),
        "selectable_receiver_count": len(candidates),
        "fallback_receiver_count": fallback_receiver_count,
        "baseline_target_unsafe_receiver_count": (
            baseline_target_unsafe_receiver_count
        ),
        "selected_receiver_index": selected_receiver_index,
        "sparse_noop": sparse_noop,
    })
    return positions, yaws, audit

class PolicyWrapper(object):
    """A convenient wrapper for specifying run-time keyword arguments"""

    def __init__(self, model, get_action_kwargs=None, get_plan_kwargs=None):
        self.model = model
        self.device = model.device
        self.action_kwargs = get_action_kwargs
        self.plan_kwargs = get_plan_kwargs

    def eval(self):
        self.model.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        return self.model.get_action(obs, **self.action_kwargs, **kwargs)

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        return self.model.get_plan(obs, **self.plan_kwargs, **kwargs)

    @classmethod
    def wrap_controller(cls, model, **kwargs):
        return cls(model=model, get_action_kwargs=kwargs)

    @classmethod
    def wrap_planner(cls, model, **kwargs):
        return cls(model=model, get_plan_kwargs=kwargs)

class RefineWrapper(object):
    """A wrapper that feeds coarse motion plan to a optimization-based planner for refinement"""
    def __init__(self, initial_planner, refiner, device):
        """
        Args:
            planner: a policy that generates a coarse motion plan
            refiner: a policy (optimization based) that takes the coarse motion plan and refine it
            device: device for torch
        """
        self.initial_planner = initial_planner
        self.refiner = refiner
        self.device = device
    def eval(self):
        self.initial_planner.eval()
        self.refiner.eval()
    def get_action(self, obs, **kwargs):
        coarse_plan,_ = self.initial_planner.get_action(obs,**kwargs)
        action, action_info = self.refiner.get_action(obs,coarse_plan = coarse_plan)
        return action, {"coarse_plan":coarse_plan.to_dict(), **action_info}
    

class Pos2YawWrapper(object):
    """A wrapper that computes action yaw from action positions"""
    def __init__(self, policy, dt, yaw_correction_speed):
        """

        Args:
            policy: policy to be wrapped
            dt:
            speed_filter:
        """
        self.device = policy.device
        self.policy = policy
        self._dt = dt
        self._yaw_correction_speed = yaw_correction_speed

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs):
        action, action_info = self.policy.get_action(obs, **kwargs)
        curr_pos = torch.zeros_like(action.positions[..., [0], :])
        pos_seq = torch.cat((curr_pos, action.positions), dim=-2)
        yaws = yaw_from_pos(pos_seq, dt=self._dt, yaw_correction_speed=self._yaw_correction_speed)
        action.yaws = yaws
        return action, action_info


class RolloutWrapper(object):
    """A wrapper policy that can (optionally) control both ego and other agents in a scene"""

    def __init__(self, ego_policy=None, agents_policy=None, pass_agent_obs=True):
        self.device = ego_policy.device if agents_policy is None else agents_policy.device
        self.ego_policy = ego_policy
        self.agents_policy = agents_policy
        self.pass_agent_obs = pass_agent_obs

    def eval(self):
        self.ego_policy.eval()
        self.agents_policy.eval()

    def get_action(self, obs, step_index) -> RolloutAction:
        ego_action = None
        ego_action_info = None
        agents_action = None
        agents_action_info = None
        if self.ego_policy is not None:
            assert obs["ego"] is not None
            with torch.no_grad():
                if self.pass_agent_obs:
                    ego_action, ego_action_info = self.ego_policy.get_action(
                        obs["ego"], step_index = step_index,agent_obs = obs["agents"])
                else:
                    ego_action, ego_action_info = self.ego_policy.get_action(
                        obs["ego"], step_index = step_index)
        if self.agents_policy is not None:
            assert obs["agents"] is not None
            with torch.no_grad():
                agents_action, agents_action_info = self.agents_policy.get_action(
                    obs["agents"], step_index = step_index)
        return RolloutAction(ego_action, ego_action_info, agents_action, agents_action_info)


class PlannerAwareRolloutWrapper(RolloutWrapper):
    """Couple an ego planner to a separately generated traffic future.

    The agents policy is evaluated first so its candidate future can be passed
    to the ego planner. K=0 planners receive the value but deliberately ignore
    it; K=1 planners must explicitly advertise candidate-future support.
    """

    def __init__(
        self,
        ego_policy,
        agents_policy,
        strategy_depth=0,
        pass_agent_obs=True,
    ):
        super().__init__(
            ego_policy=ego_policy,
            agents_policy=agents_policy,
            pass_agent_obs=pass_agent_obs,
        )
        self.strategy_depth = int(strategy_depth)

    def get_action(self, obs, step_index) -> RolloutAction:
        assert obs["ego"] is not None
        assert obs["agents"] is not None

        with torch.no_grad():
            agents_action, agents_action_info = self.agents_policy.get_action(
                obs["agents"],
                step_index=step_index,
            )
            ego_action, ego_action_info = self.ego_policy.get_action(
                obs["ego"],
                step_index=step_index,
                agent_obs=obs["agents"] if self.pass_agent_obs else None,
                candidate_agents_future=agents_action,
                strategy_depth=self.strategy_depth,
            )

        if ego_action_info is None:
            ego_action_info = {}
        ego_action_info = dict(ego_action_info)
        ego_action_info["planner_strategy_depth"] = self.strategy_depth
        ego_action_info["candidate_future_supplied"] = True
        ego_action_info["candidate_future_used"] = bool(
            self.strategy_depth > 0
            and getattr(self.ego_policy, "supports_candidate_future", False)
        )
        return RolloutAction(
            ego_action,
            ego_action_info,
            agents_action,
            agents_action_info,
        )


class PlannerOverrideRolloutWrapper(object):
    """Generate one unified scene action, then replace only each ego row.

    CCDiff's social graph, controllable-set indices, and guidance all assume a
    unified scene batch.  Splitting ego from traffic before sampling changes
    that contract.  This wrapper preserves the original batch through the
    traffic model and performs the ego-planner override immediately before the
    environment consumes the action.
    """

    def __init__(
        self,
        ego_policy,
        agents_policy,
        strategy_depth=0,
        marginal_tail_config=None,
        marginal_tail_target_ego_policy=None,
        marginal_tail_attack_ego_policy=None,
        safesim_guidance=False,
        planner_aware_hcs=False,
        planner_aware_hcs_config=None,
    ):
        self.device = agents_policy.device
        self.ego_policy = ego_policy
        # guided_rollout expects this public attribute for controllable sets.
        self.agents_policy = agents_policy
        self.strategy_depth = int(strategy_depth)
        self.marginal_tail_config = marginal_tail_config
        self.marginal_tail_target_ego_policy = marginal_tail_target_ego_policy
        self.marginal_tail_attack_ego_policy = marginal_tail_attack_ego_policy
        self.safesim_guidance = bool(safesim_guidance)
        self.planner_aware_hcs = bool(planner_aware_hcs)
        self.planner_aware_hcs_config = dict(planner_aware_hcs_config or {})
        self._planner_aware_hcs_selector = None
        self._planner_aware_hcs_last_event = None
        self._planner_aware_hcs_locked_target_local = None
        self._planner_aware_hcs_locked_target_track_id = None
        self._planner_aware_hcs_locked_coalition_locals = None
        self._planner_aware_hcs_locked_coalition_track_ids = None
        self._planner_aware_hcs_semantic_plan = None
        if self.planner_aware_hcs:
            from ccdiff.models.planner_aware_hcs import PlannerAwareHCS

            self._planner_aware_hcs_selector = PlannerAwareHCS(
                dt=float(self.planner_aware_hcs_config.get("dt", 0.1)),
                min_target_speed_mps=0.5,
                min_initial_distance_m=2.0,
                max_predicted_gap_m=10.0,
                min_bystander_gap_m=1.0,
                min_conflict_step=2,
            )
            semantic_control = str(
                self.planner_aware_hcs_config.get(
                    "semantic_control", "actual"
                )
            )
            if semantic_control not in {"actual", "matched_shuffled"}:
                raise ValueError(
                    "planner-aware HCS semantic_control must be actual or "
                    "matched_shuffled"
                )
            semantic_plan_path = self.planner_aware_hcs_config.get(
                "semantic_plan_json"
            )
            if semantic_control == "actual" and semantic_plan_path is not None:
                raise ValueError(
                    "actual semantic control must use the actual planner future"
                )
            if semantic_control == "matched_shuffled":
                if semantic_plan_path is None:
                    raise ValueError(
                        "matched_shuffled semantic control requires a frozen plan"
                    )
                with open(semantic_plan_path, "r") as stream:
                    semantic_plan = json.load(stream)
                if semantic_plan.get("schema") != "m2c_b3a_r2_semantic_plan_v1":
                    raise ValueError("unsupported B3a semantic-plan schema")
                if semantic_plan.get("planner") != "strive":
                    raise ValueError("semantic donor must come from STRIVE")
                if int(semantic_plan.get("seed", -1)) != 1:
                    raise ValueError("semantic donor must use fixed seed1")
                positions = np.asarray(
                    semantic_plan.get("positions"), dtype=np.float32
                )
                yaws = np.asarray(semantic_plan.get("yaws"), dtype=np.float32)
                if yaws.ndim == 1:
                    yaws = yaws[:, None]
                if (
                    positions.ndim != 2
                    or positions.shape[-1] != 2
                    or yaws.shape != (positions.shape[0], 1)
                    or positions.shape[0] < 2
                    or not np.isfinite(positions).all()
                    or not np.isfinite(yaws).all()
                ):
                    raise ValueError(
                        "semantic donor plan must be finite [T,2]/[T,1]"
                    )
                digest = hashlib.sha256(
                    positions.tobytes(order="C") + yaws.tobytes(order="C")
                ).hexdigest()
                if digest != semantic_plan.get("plan_sha256"):
                    raise ValueError("semantic donor plan SHA256 mismatch")
                semantic_plan["positions"] = positions
                semantic_plan["yaws"] = yaws
                self._planner_aware_hcs_semantic_plan = semantic_plan
        self._marginal_tail_locked_receiver_index = None

    def _planner_aware_hcs_semantic_action(self, ego_action, step_index):
        """Return event semantics while leaving the executed planner untouched.

        The B3a shuffled control reuses a frozen future from another STRIVE
        seed1 episode solely for HCS event construction.  It never runs a
        second planner and never replaces the ego row that is executed by the
        environment.  Both actual and shuffled arms still guide only the
        complete FM velocity field downstream.
        """
        control = str(
            self.planner_aware_hcs_config.get("semantic_control", "actual")
        )
        if control == "actual":
            print(
                "PLANNER_AWARE_HCS_SEMANTIC_CONTROL_ATTESTATION "
                + json.dumps({
                    "control": "actual",
                    "actual_planner": "strive",
                    "second_planner_executed": False,
                    "rollout_step": int(step_index),
                }, sort_keys=True),
                flush=True,
            )
            return ego_action
        if control != "matched_shuffled" or self._planner_aware_hcs_semantic_plan is None:
            raise RuntimeError("matched shuffled semantic plan is unavailable")

        plan = self._planner_aware_hcs_semantic_plan
        target_horizon = int(ego_action.positions.shape[1])
        if torch.is_tensor(ego_action.positions):
            positions = torch.as_tensor(
                plan["positions"],
                device=ego_action.positions.device,
                dtype=ego_action.positions.dtype,
            )[None]
            yaws = torch.as_tensor(
                plan["yaws"],
                device=ego_action.yaws.device,
                dtype=ego_action.yaws.dtype,
            )[None]
            positions = self._fit_horizon(positions, target_horizon)
            yaws = self._fit_horizon(yaws, target_horizon)
            positions = positions.expand(ego_action.positions.shape[0], -1, -1)
            yaws = yaws.expand(ego_action.yaws.shape[0], -1, -1)
        else:
            positions = plan["positions"][None]
            yaws = plan["yaws"][None]
            positions = self._fit_horizon(positions, target_horizon)
            yaws = self._fit_horizon(yaws, target_horizon)
            positions = np.repeat(
                positions, ego_action.positions.shape[0], axis=0
            )
            yaws = np.repeat(yaws, ego_action.yaws.shape[0], axis=0)
        print(
            "PLANNER_AWARE_HCS_SEMANTIC_CONTROL_ATTESTATION "
            + json.dumps({
                "control": "matched_shuffled",
                "actual_planner": "strive",
                "donor_planner": "strive",
                "donor_seed": 1,
                "donor_source_hdf5": plan["source_hdf5"],
                "donor_source_frame": int(plan["source_frame"]),
                "plan_sha256": plan["plan_sha256"],
                "second_planner_executed": False,
                "rollout_step": int(step_index),
            }, sort_keys=True),
            flush=True,
        )
        return Action(positions=positions, yaws=yaws)

    def _clear_planner_aware_hcs_guidance(self, conditioned_obs):
        model = getattr(self.agents_policy, "model", None)
        if model is None or not hasattr(model, "clear_guidance"):
            raise RuntimeError(
                "planner-aware HCS requires a traffic policy with clear_guidance"
            )
        # An empty guidance configuration still instantiates a guidance object
        # in CCDiff.  That changes the FM sampling branch before/after the
        # scheduled intervention and can invalidate paired-prefix identity.
        # Clear both pieces of state explicitly so inactive sampling is exactly
        # the native path.
        model.clear_guidance()
        if not hasattr(model, "set_flow_velocity_guidance"):
            raise RuntimeError(
                "planner-aware HCS requires flow-velocity guidance support"
            )
        model.set_flow_velocity_guidance(None)

    def _planner_aware_hcs_step_active(self, step_index):
        start_step = int(self.planner_aware_hcs_config.get("start_step", 20))
        end_step = self.planner_aware_hcs_config.get("end_step")
        end_step = None if end_step is None else int(end_step)
        return bool(
            int(step_index) >= start_step
            and (end_step is None or int(step_index) < end_step)
        )

    def _planner_aware_hcs_event_to_action_info(self, value):
        """Make HCS metadata safe for ``RolloutAction.to_numpy``.

        Planner-aware HCS attestations are ordinary JSON-like dictionaries,
        but the rollout action contract only accepts tensors, ndarrays,
        strings, and ``None`` at its leaves.  Keep the full human-readable
        attestation in stdout and tensorize every numeric leaf before the
        event is attached to ``agents_info``.
        """
        if isinstance(value, dict):
            return {
                key: self._planner_aware_hcs_event_to_action_info(item)
                for key, item in value.items()
            }
        if isinstance(value, bool):
            return torch.tensor(value, dtype=torch.bool, device=self.device)
        if isinstance(value, (int, np.integer)):
            return torch.tensor(value, dtype=torch.long, device=self.device)
        if isinstance(value, (float, np.floating)):
            return torch.tensor(value, dtype=torch.float32, device=self.device)
        if isinstance(value, (list, tuple)):
            if all(isinstance(item, bool) for item in value):
                return torch.tensor(value, dtype=torch.bool, device=self.device)
            if all(
                isinstance(item, (int, np.integer))
                and not isinstance(item, bool)
                for item in value
            ):
                return torch.tensor(value, dtype=torch.long, device=self.device)
            if all(
                isinstance(item, (int, float, np.integer, np.floating))
                and not isinstance(item, bool)
                for item in value
            ):
                return torch.tensor(value, dtype=torch.float32, device=self.device)
            return type(value)(
                self._planner_aware_hcs_event_to_action_info(item)
                for item in value
            )
        if torch.is_tensor(value) or isinstance(value, (np.ndarray, str)):
            return value
        if value is None:
            return None
        raise TypeError(
            "unsupported planner-aware HCS action-info type {}".format(
                type(value)
            )
        )

    def _configure_planner_aware_hcs_guidance(
        self, conditioned_obs, ego_action, step_index
    ):
        """Select one causal event and install guidance before FM sampling."""
        if self._planner_aware_hcs_selector is None:
            raise RuntimeError("planner-aware HCS selector is not initialized")
        step_index = int(step_index)
        start_step = int(self.planner_aware_hcs_config.get("start_step", 20))
        end_step = self.planner_aware_hcs_config.get("end_step")
        end_step = None if end_step is None else int(end_step)
        if step_index == 0:
            self._planner_aware_hcs_locked_target_local = None
            self._planner_aware_hcs_locked_target_track_id = None
            self._planner_aware_hcs_locked_coalition_locals = None
            self._planner_aware_hcs_locked_coalition_track_ids = None
            self._planner_aware_hcs_last_event = None
        active = self._planner_aware_hcs_step_active(step_index)
        if not active:
            self._clear_planner_aware_hcs_guidance(conditioned_obs)
            print(
                "PLANNER_AWARE_HCS_SCHEDULE_ATTESTATION "
                "step={} start_step={} end_step={} active=False".format(
                    step_index, start_step, end_step
                ),
                flush=True,
            )
            return None

        candidate_local_indices = None
        if self._planner_aware_hcs_locked_target_track_id is not None:
            if "track_id" not in conditioned_obs:
                raise KeyError(
                    "planner-aware HCS target locking requires track_id"
                )
            scene_index = torch.as_tensor(
                conditioned_obs["scene_index"]
            ).reshape(-1)
            unique_scene = torch.unique(scene_index)
            if unique_scene.numel() != 1:
                raise ValueError(
                    "planner-aware HCS requires one scene per batch"
                )
            scene_agents = torch.nonzero(
                scene_index == unique_scene[0], as_tuple=True
            )[0]
            track_ids = torch.as_tensor(conditioned_obs["track_id"])
            if track_ids.ndim > 1:
                track_ids = track_ids[:, 0]
            track_ids = track_ids.reshape(-1)
            local_matches = torch.nonzero(
                track_ids[scene_agents]
                == int(self._planner_aware_hcs_locked_target_track_id),
                as_tuple=True,
            )[0]
            if local_matches.numel() != 1:
                self._clear_planner_aware_hcs_guidance(conditioned_obs)
                print(
                    "PLANNER_AWARE_HCS_LOCKED_TRACK_UNAVAILABLE_ATTESTATION "
                    "step={} target_track_id={} match_count={}".format(
                        step_index,
                        self._planner_aware_hcs_locked_target_track_id,
                        int(local_matches.numel()),
                    ),
                    flush=True,
                )
                return None
            resolved_local = int(local_matches[0])
            self._planner_aware_hcs_locked_target_local = resolved_local
            candidate_local_indices = [resolved_local]
        try:
            event = self._planner_aware_hcs_selector.select(
                conditioned_obs,
                ego_action.positions,
                reference_agent_idx=int(
                    self.planner_aware_hcs_config.get("reference_agent_idx", 0)
                ),
                candidate_local_indices=candidate_local_indices,
            )
        except Exception as error:
            from ccdiff.models.planner_aware_hcs import (
                PlannerAwareHCSSelectionError,
            )

            if not isinstance(error, PlannerAwareHCSSelectionError):
                raise
            self._clear_planner_aware_hcs_guidance(conditioned_obs)
            print(
                "PLANNER_AWARE_HCS_NO_CANDIDATE_ATTESTATION "
                "step={} start_step={} end_step={} locked_target={} error={}".format(
                    step_index,
                    start_step,
                    end_step,
                    self._planner_aware_hcs_locked_target_local,
                    str(error),
                ),
                flush=True,
            )
            return None
        if self._planner_aware_hcs_locked_target_local is None:
            self._planner_aware_hcs_locked_target_local = int(
                event.target_local_index
            )
        if self._planner_aware_hcs_locked_target_track_id is None:
            if "track_id" not in conditioned_obs:
                raise KeyError(
                    "planner-aware HCS target locking requires track_id"
                )
            track_ids = torch.as_tensor(conditioned_obs["track_id"])
            if track_ids.ndim > 1:
                track_ids = track_ids[:, 0]
            self._planner_aware_hcs_locked_target_track_id = int(
                track_ids.reshape(-1)[int(event.target_global_index)]
            )
        coalition_enabled = bool(
            self.planner_aware_hcs_config.get("coalition_enabled", False)
        )
        coalition_event = None
        if coalition_enabled:
            coalition_candidate_indices = None
            if self._planner_aware_hcs_locked_coalition_track_ids is not None:
                if "track_id" not in conditioned_obs:
                    raise KeyError(
                        "planner-aware coalition locking requires track_id"
                    )
                scene_index = torch.as_tensor(
                    conditioned_obs["scene_index"]
                ).reshape(-1)
                unique_scene = torch.unique(scene_index)
                if unique_scene.numel() != 1:
                    raise ValueError(
                        "planner-aware coalition requires one scene per batch"
                    )
                scene_agents = torch.nonzero(
                    scene_index == unique_scene[0], as_tuple=True
                )[0]
                track_ids = torch.as_tensor(conditioned_obs["track_id"])
                if track_ids.ndim > 1:
                    track_ids = track_ids[:, 0]
                track_ids = track_ids.reshape(-1)
                coalition_candidate_indices = []
                for locked_track_id in (
                    self._planner_aware_hcs_locked_coalition_track_ids
                ):
                    matches = torch.nonzero(
                        track_ids[scene_agents] == int(locked_track_id),
                        as_tuple=True,
                    )[0]
                    if matches.numel() != 1:
                        self._clear_planner_aware_hcs_guidance(conditioned_obs)
                        print(
                            "PLANNER_AWARE_COALITION_LOCKED_TRACK_UNAVAILABLE_ATTESTATION "
                            "step={} blocker_track_id={} match_count={}".format(
                                step_index, locked_track_id,
                                int(matches.numel()),
                            ),
                            flush=True,
                        )
                        return None
                    coalition_candidate_indices.append(int(matches[0]))
                self._planner_aware_hcs_locked_coalition_locals = tuple(
                    coalition_candidate_indices
                )
            try:
                coalition_event = (
                    self._planner_aware_hcs_selector.select_escape_coalition(
                        conditioned_obs,
                        ego_action.positions,
                        event,
                        candidate_local_indices=coalition_candidate_indices,
                        min_blockers=int(
                            self.planner_aware_hcs_config.get(
                                "coalition_min_blockers", 1
                            )
                        ),
                        max_blockers=int(
                            self.planner_aware_hcs_config.get(
                                "coalition_max_blockers", 2
                            )
                        ),
                        corridor_offset_m=float(
                            self.planner_aware_hcs_config.get(
                                "coalition_corridor_offset_m", 3.2
                            )
                        ),
                        max_reach_error_m=float(
                            self.planner_aware_hcs_config.get(
                                "coalition_max_reach_error_m", 12.0
                            )
                        ),
                    )
                )
            except Exception as error:
                from ccdiff.models.planner_aware_hcs import (
                    PlannerAwareHCSSelectionError,
                )
                if not isinstance(error, PlannerAwareHCSSelectionError):
                    raise
                self._clear_planner_aware_hcs_guidance(conditioned_obs)
                print(
                    "PLANNER_AWARE_COALITION_NO_CANDIDATE_ATTESTATION "
                    "step={} locked_blockers={} error={}".format(
                        step_index,
                        self._planner_aware_hcs_locked_coalition_track_ids,
                        str(error),
                    ),
                    flush=True,
                )
                return None
            if self._planner_aware_hcs_locked_coalition_track_ids is None:
                track_ids = torch.as_tensor(conditioned_obs["track_id"])
                if track_ids.ndim > 1:
                    track_ids = track_ids[:, 0]
                track_ids = track_ids.reshape(-1)
                self._planner_aware_hcs_locked_coalition_track_ids = tuple(
                    int(track_ids[int(member.global_index)])
                    for member in coalition_event.members
                )
            self._planner_aware_hcs_locked_coalition_locals = tuple(
                int(member.local_index) for member in coalition_event.members
            )

        if coalition_event is None:
            config = [[
            {
                "name": "safesim_target_pair",
                "weight": 1.0,
                "params": {
                    "target_ind": int(event.target_local_index),
                    "ref_ind": int(event.reference_local_index),
                    # HCS scores the complete planner/FM future.  Truncating
                    # this to 20 confused Euler integration count K=20 with
                    # the 52-step trajectory horizon and clamped late causal
                    # conflicts onto an unreachable early endpoint.
                    "loss_timesteps": int(event.conflict_step) + 1,
                    "sigma": 1.0,
                    "heading_weight": 1.0,
                    "buff_dist": 1.5,
                    "cause_weight": float(
                        self.planner_aware_hcs_config.get("cause_weight", 1.0)
                    ),
                    "collateral_weight": float(
                        self.planner_aware_hcs_config.get(
                            "collateral_weight", 1.0
                        )
                    ),
                    "desired_clearance": float(
                        self.planner_aware_hcs_config.get(
                            "desired_clearance", 0.0
                        )
                    ),
                    "severity_mode": str(
                        self.planner_aware_hcs_config.get(
                            "severity_mode", "collision"
                        )
                    ),
                    "softmin_temperature": 0.5,
                    "collateral_margin": 1.0,
                    "collateral_barrier_weight": 5.0,
                    "collision_type": str(event.collision_type),
                    "conflict_step": int(event.conflict_step),
                    "conflict_point_world": tuple(
                        event.conflict_point_world
                    ),
                    "arrival_weight": float(
                        self.planner_aware_hcs_config.get(
                            "arrival_weight", 0.5
                        )
                    ),
                    "typed_heading_weight": 0.2,
                    "marginal_weight": 0.5,
                    "dt": float(
                        self.planner_aware_hcs_config.get("dt", 0.1)
                    ),
                    "max_speed_mps": 25.0,
                    "max_abs_acc_mps2": 4.0,
                    "max_abs_jerk_mps3": 8.0,
                    "max_abs_yaw_rate_rps": 1.0,
                },
                "agents": [int(event.target_local_index)],
            },
            {
                "name": "map_collision",
                "weight": 1.0,
                "params": {
                    "num_points_lw": (10, 10),
                    "decay_rate": 0.9,
                },
                "agents": [int(event.target_local_index)],
            },
            ]]
            event_metadata = event.to_dict()
            event_metadata.update({
                "hard_projection": False,
                "row_mixing": False,
                "trajectory_coordinate_edit": False,
            })
        else:
            controlled_locals = list(coalition_event.controlled_local_indices)
            config = [[
                {
                    "name": "planner_response_coalition",
                    "weight": 1.0,
                    "params": {
                        "target_ind": int(event.target_local_index),
                        "ref_ind": int(event.reference_local_index),
                        "blocker_inds": [
                            int(member.local_index)
                            for member in coalition_event.members
                        ],
                        "blocker_points_world": [
                            tuple(member.corridor_point_world)
                            for member in coalition_event.members
                        ],
                        "collision_type": str(event.collision_type),
                        "conflict_step": int(event.conflict_step),
                        "loss_timesteps": int(event.conflict_step) + 1,
                        "primary_weight": float(
                            self.planner_aware_hcs_config.get(
                                "coalition_primary_weight", 1.0
                            )
                        ),
                        "blocker_weight": float(
                            self.planner_aware_hcs_config.get(
                                "coalition_blocker_weight", 0.5
                            )
                        ),
                        "collateral_weight": float(
                            self.planner_aware_hcs_config.get(
                                "coalition_collateral_weight", 2.0
                            )
                        ),
                        "marginal_weight": float(
                            self.planner_aware_hcs_config.get(
                                "coalition_marginal_weight", 1.0
                            )
                        ),
                        "blocker_phase_weight": float(
                            self.planner_aware_hcs_config.get(
                                "coalition_blocker_phase_weight", 0.25
                            )
                        ),
                        "dt": float(
                            self.planner_aware_hcs_config.get("dt", 0.1)
                        ),
                    },
                    "agents": controlled_locals,
                },
                {
                    "name": "map_collision",
                    "weight": 1.0,
                    "params": {
                        "num_points_lw": (10, 10),
                        "decay_rate": 0.9,
                    },
                    "agents": controlled_locals,
                },
            ]]
            event_metadata = coalition_event.to_dict()
            event_metadata.update({
                "target_local_index": int(event.target_local_index),
                "target_global_index": int(event.target_global_index),
                "reference_local_index": int(event.reference_local_index),
                "reference_global_index": int(event.reference_global_index),
                "collision_type": str(event.collision_type),
                "conflict_step": int(event.conflict_step),
                "conflict_time_s": float(event.conflict_time_s),
                "conflict_point_world": tuple(event.conflict_point_world),
                "hard_projection": False,
                "row_mixing": False,
                "trajectory_coordinate_edit": False,
                "coalition_track_ids": tuple(
                    self._planner_aware_hcs_locked_coalition_track_ids
                ),
            })
        model = getattr(self.agents_policy, "model", None)
        if model is None or not hasattr(model, "set_guidance"):
            raise RuntimeError(
                "planner-aware HCS requires a traffic policy with set_guidance"
            )
        if not hasattr(model, "set_flow_velocity_guidance"):
            raise RuntimeError(
                "planner-aware HCS requires flow-velocity guidance support"
            )
        model.set_guidance(config, conditioned_obs)
        model.set_flow_velocity_guidance({
            "trust_max_rms": float(
                self.planner_aware_hcs_config.get(
                    "velocity_trust_max_rms", 0.05
                )
            ),
            "trust_max_linf": float(
                self.planner_aware_hcs_config.get(
                    "velocity_trust_max_linf", 0.20
                )
            ),
        })
        self._planner_aware_hcs_last_event = event_metadata
        self._planner_aware_hcs_last_event.update({
            "rollout_step": step_index,
            "start_step": start_step,
            "end_step": end_step,
            "semantic_control": str(
                self.planner_aware_hcs_config.get(
                    "semantic_control", "actual"
                )
            ),
            "second_planner_executed": False,
            "target_locked": True,
            "target_track_id": int(
                self._planner_aware_hcs_locked_target_track_id
            ),
        })
        print(
            "PLANNER_AWARE_HCS_ATTESTATION "
            + json.dumps(self._planner_aware_hcs_last_event, sort_keys=True),
            flush=True,
        )
        return self._planner_aware_hcs_last_event

    def eval(self):
        self.ego_policy.eval()
        if self.marginal_tail_target_ego_policy is not None:
            self.marginal_tail_target_ego_policy.eval()
        if self.marginal_tail_attack_ego_policy is not None:
            self.marginal_tail_attack_ego_policy.eval()
        self.agents_policy.eval()
        return self

    @staticmethod
    def _normalize_name(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _ego_indices(self, agents_obs):
        names = agents_obs.get("agent_name")
        if names is not None:
            if torch.is_tensor(names):
                names = names.detach().cpu().numpy()
            names = np.asarray(names).reshape(-1).tolist()
            indices = [
                index
                for index, name in enumerate(names)
                if self._normalize_name(name) == "ego"
            ]
            if indices:
                return indices

        # trajdata scene batches are scene-major and place ego first.
        scene_index = agents_obs.get("scene_index")
        if scene_index is None:
            return [0]
        if torch.is_tensor(scene_index):
            scene_index = scene_index.detach().cpu().numpy()
        scene_index = np.asarray(scene_index).reshape(-1)
        _, first_indices = np.unique(scene_index, return_index=True)
        return sorted(first_indices.tolist())

    def _slice_batch(self, value, indices, batch_size):
        if isinstance(value, dict):
            return {
                key: self._slice_batch(item, indices, batch_size)
                for key, item in value.items()
            }
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            index = torch.as_tensor(
                indices, device=value.device, dtype=torch.long
            )
            return value.index_select(0, index)
        if (
            isinstance(value, np.ndarray)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            return value[np.asarray(indices, dtype=np.int64)]
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            sliced = [value[index] for index in indices]
            return tuple(sliced) if isinstance(value, tuple) else sliced
        return value

    @staticmethod
    def _fit_horizon(value, target_horizon):
        if value.shape[1] >= target_horizon:
            return value[:, :target_horizon]
        missing = target_horizon - value.shape[1]
        if torch.is_tensor(value):
            tail = value[:, -1:].expand(
                -1, missing, *value.shape[2:]
            )
            return torch.cat((value, tail), dim=1)
        tail = np.repeat(value[:, -1:], missing, axis=1)
        return np.concatenate((value, tail), axis=1)

    @staticmethod
    def _joint_selector_planner_roles(
        mode,
        executed_positions,
        executed_yaws,
        secondary_positions,
        secondary_yaws,
    ):
        """Return (safe reference, attacked target) planner futures.

        Legacy constrained-native runs compute the ordinary ego policy as the
        safety reference and the secondary policy as the attacked target.  In
        planner-aligned mode the order is deliberately reversed: the ordinary
        planner is attacked and executed, while the secondary planner is the
        safety-only reference.  Keeping this mapping at one named boundary
        prevents the most consequential silent role inversion in selection.
        """
        if mode == "planner_aligned_native_joint":
            return (
                secondary_positions,
                secondary_yaws,
                executed_positions,
                executed_yaws,
            )
        return (
            executed_positions,
            executed_yaws,
            secondary_positions,
            secondary_yaws,
        )

    def _attach_planner_reference(
        self,
        agents_obs,
        ego_indices,
        ego_action,
        prefix="planner_reference",
    ):
        """Attach one frozen ego plan to the unified scene observation."""
        if prefix not in {
            "planner_reference",
            "planner_safety_reference",
        }:
            raise ValueError("unknown planner-reference prefix")
        conditioned = dict(agents_obs)
        target_positions = agents_obs["target_positions"]
        target_yaws = agents_obs.get("target_yaws")
        if target_yaws is None:
            target_yaws = (
                torch.zeros_like(target_positions[..., :1])
                if torch.is_tensor(target_positions)
                else np.zeros_like(target_positions[..., :1])
            )
        if torch.is_tensor(target_positions):
            reference_positions = torch.zeros_like(target_positions)
            reference_yaws = torch.zeros_like(target_yaws)
            reference_mask = torch.zeros(
                target_positions.shape[0], dtype=torch.bool,
                device=target_positions.device,
            )
            index = torch.as_tensor(
                ego_indices, dtype=torch.long, device=target_positions.device
            )
            ego_positions = torch.as_tensor(
                ego_action.positions, dtype=target_positions.dtype,
                device=target_positions.device,
            )
            ego_yaws = torch.as_tensor(
                ego_action.yaws, dtype=target_yaws.dtype,
                device=target_yaws.device,
            )
        else:
            reference_positions = np.zeros_like(target_positions)
            reference_yaws = np.zeros_like(target_yaws)
            reference_mask = np.zeros(target_positions.shape[0], dtype=bool)
            index = np.asarray(ego_indices, dtype=np.int64)
            ego_positions = TensorUtils.to_numpy(ego_action.positions)
            ego_yaws = TensorUtils.to_numpy(ego_action.yaws)
        ego_positions = self._fit_horizon(
            ego_positions, target_positions.shape[1]
        )
        ego_yaws = self._fit_horizon(ego_yaws, target_yaws.shape[1])
        reference_positions[index] = ego_positions
        reference_yaws[index] = ego_yaws
        reference_mask[index] = True
        conditioned[prefix + "_positions"] = reference_positions
        conditioned[prefix + "_yaws"] = reference_yaws
        conditioned[prefix + "_mask"] = reference_mask
        return conditioned

    def get_action(self, obs, step_index) -> RolloutAction:
        agents_obs = obs["agents"]
        if agents_obs is None:
            raise ValueError("Unified planner rollout requires agents obs")
        if int(step_index) == 0:
            self._marginal_tail_locked_receiver_index = None
        with torch.no_grad():
            batch_size = agents_obs["target_positions"].shape[0]
            ego_indices = self._ego_indices(agents_obs)
            ego_obs = self._slice_batch(agents_obs, ego_indices, batch_size)
            config = self.marginal_tail_config
            conditional_fm = (
                config is not None
                and config.get("mode") == "conditional_fm_source"
            )
            planner_aligned_joint = (
                config is not None
                and config.get("mode") == "planner_aligned_native_joint"
            )
            planner_aligned_active = (
                planner_aligned_joint
                and int(step_index) >= int(config["start_step"])
                and (
                    config.get("end_step") is None
                    or int(step_index) < int(config["end_step"])
                )
            )
            selector_attack_action = None
            if conditional_fm or planner_aligned_active:
                if self.strategy_depth != 0:
                    raise ValueError(
                        "planner-aligned FM requires frozen K=0 planners"
                    )
                if self.marginal_tail_target_ego_policy is None:
                    raise RuntimeError(
                        "planner-aligned FM requires distinct attacked/reference planners"
                    )
                # The ordinary ego policy is always the row actually executed.
                # By default it is also the attacked proposal planner.  A
                # separately supplied attack policy decouples those roles so
                # the exact same adversarial environment policy can be tested
                # under a genuinely executed reference planner.
                ego_action, ego_info = self.ego_policy.get_action(
                    ego_obs,
                    step_index=step_index,
                    candidate_agents_future=None,
                    strategy_depth=0,
                )
                reference_action, reference_info = (
                    self.marginal_tail_target_ego_policy.get_action(
                        ego_obs,
                        step_index=step_index,
                        candidate_agents_future=None,
                        strategy_depth=0,
                    )
                )
                if self.marginal_tail_attack_ego_policy is None:
                    selector_attack_action = ego_action
                else:
                    selector_attack_action, _ = (
                        self.marginal_tail_attack_ego_policy.get_action(
                            ego_obs,
                            step_index=step_index,
                            candidate_agents_future=None,
                            strategy_depth=0,
                        )
                    )
                conditioned_obs = self._attach_planner_reference(
                    agents_obs,
                    ego_indices,
                    selector_attack_action,
                    prefix="planner_reference",
                )
                conditioned_obs = self._attach_planner_reference(
                    conditioned_obs,
                    ego_indices,
                    reference_action,
                    prefix="planner_safety_reference",
                )
                agents_action, agents_info = self.agents_policy.get_action(
                    conditioned_obs,
                    step_index=step_index,
                )
                # Existing audit/output plumbing stores the secondary planner
                # in target_ego_*.  In this mode it is explicitly the safety
                # reference, never the executed row.
                target_ego_action = reference_action
                target_ego_info = reference_info
            elif self.safesim_guidance:
                if self.strategy_depth != 0:
                    raise ValueError(
                        "SAFE-SIM guidance requires a frozen K=0 ego plan"
                    )
                ego_action, ego_info = self.ego_policy.get_action(
                    ego_obs,
                    step_index=step_index,
                    candidate_agents_future=None,
                    strategy_depth=0,
                )
                planner_aware_hcs_event = None
                if self.planner_aware_hcs:
                    # Before/after the intervention window, traffic must see
                    # the exact native observation.  Attaching even a masked
                    # planner future changes the model input and breaks the
                    # paired prefix from the first sampling call.
                    if self._planner_aware_hcs_step_active(step_index):
                        semantic_action = (
                            self._planner_aware_hcs_semantic_action(
                                ego_action, step_index
                            )
                        )
                        conditioned_obs = self._attach_planner_reference(
                            agents_obs, ego_indices, semantic_action
                        )
                    else:
                        semantic_action = ego_action
                        conditioned_obs = agents_obs
                    planner_aware_hcs_event = (
                        self._configure_planner_aware_hcs_guidance(
                            conditioned_obs, semantic_action, step_index
                        )
                    )
                else:
                    conditioned_obs = self._attach_planner_reference(
                        agents_obs, ego_indices, ego_action
                    )
                agents_action, agents_info = self.agents_policy.get_action(
                    conditioned_obs,
                    step_index=step_index,
                )
                if planner_aware_hcs_event is not None:
                    agents_info = dict(agents_info or {})
                    agents_info["planner_aware_hcs_event"] = (
                        self._planner_aware_hcs_event_to_action_info(
                            planner_aware_hcs_event
                        )
                    )
            else:
                agents_action, agents_info = self.agents_policy.get_action(
                    agents_obs,
                    step_index=step_index,
                )
                candidate = (
                    agents_action if self.strategy_depth > 0 else None
                )
                ego_action, ego_info = self.ego_policy.get_action(
                    ego_obs,
                    step_index=step_index,
                    candidate_agents_future=candidate,
                    strategy_depth=self.strategy_depth,
                )
            batch_size = agents_action.positions.shape[0]
            if not (conditional_fm or planner_aligned_active):
                target_ego_action = None
                target_ego_info = None
            if (
                not (conditional_fm or planner_aligned_active)
                and self.marginal_tail_target_ego_policy is not None
                and not planner_aligned_joint
            ):
                target_ego_action, target_ego_info = (
                    self.marginal_tail_target_ego_policy.get_action(
                        ego_obs,
                        step_index=step_index,
                        candidate_agents_future=candidate,
                        strategy_depth=self.strategy_depth,
                    )
                )

        if torch.is_tensor(agents_action.positions):
            positions = agents_action.positions.clone()
            yaws = agents_action.yaws.clone()
            ego_positions = torch.as_tensor(
                ego_action.positions,
                device=positions.device,
                dtype=positions.dtype,
            )
            ego_yaws = torch.as_tensor(
                ego_action.yaws,
                device=yaws.device,
                dtype=yaws.dtype,
            )
            index = torch.as_tensor(
                ego_indices, device=positions.device, dtype=torch.long
            )
            if target_ego_action is not None:
                target_ego_positions = torch.as_tensor(
                    target_ego_action.positions,
                    device=positions.device,
                    dtype=positions.dtype,
                )
                target_ego_yaws = torch.as_tensor(
                    target_ego_action.yaws,
                    device=yaws.device,
                    dtype=yaws.dtype,
                )
            else:
                target_ego_positions = None
                target_ego_yaws = None
            if selector_attack_action is not None:
                selector_attack_positions = torch.as_tensor(
                    selector_attack_action.positions,
                    device=positions.device,
                    dtype=positions.dtype,
                )
                selector_attack_yaws = torch.as_tensor(
                    selector_attack_action.yaws,
                    device=yaws.device,
                    dtype=yaws.dtype,
                )
            else:
                selector_attack_positions = None
                selector_attack_yaws = None
        else:
            positions = np.array(agents_action.positions, copy=True)
            yaws = np.array(agents_action.yaws, copy=True)
            ego_positions = TensorUtils.to_numpy(ego_action.positions)
            ego_yaws = TensorUtils.to_numpy(ego_action.yaws)
            index = np.asarray(ego_indices, dtype=np.int64)
            if target_ego_action is not None:
                target_ego_positions = TensorUtils.to_numpy(
                    target_ego_action.positions
                )
                target_ego_yaws = TensorUtils.to_numpy(target_ego_action.yaws)
            else:
                target_ego_positions = None
                target_ego_yaws = None
            if selector_attack_action is not None:
                selector_attack_positions = TensorUtils.to_numpy(
                    selector_attack_action.positions
                )
                selector_attack_yaws = TensorUtils.to_numpy(
                    selector_attack_action.yaws
                )
            else:
                selector_attack_positions = None
                selector_attack_yaws = None

        ego_positions = self._fit_horizon(
            ego_positions, positions.shape[1]
        )
        ego_yaws = self._fit_horizon(ego_yaws, yaws.shape[1])
        if target_ego_positions is not None:
            target_ego_positions = self._fit_horizon(
                target_ego_positions, positions.shape[1]
            )
            target_ego_yaws = self._fit_horizon(
                target_ego_yaws, yaws.shape[1]
            )
        if selector_attack_positions is not None:
            selector_attack_positions = self._fit_horizon(
                selector_attack_positions, positions.shape[1]
            )
            selector_attack_yaws = self._fit_horizon(
                selector_attack_yaws, yaws.shape[1]
            )

        marginal_tail_audit = None
        config = self.marginal_tail_config
        if config is not None and config.get("mode") in {
            "constrained_native_joint", "planner_aligned_native_joint"
        }:
            active = (
                int(step_index) >= int(config["start_step"])
                and (
                    config.get("end_step") is None
                    or int(step_index) < int(config["end_step"])
                )
            )
            if active:
                if target_ego_positions is None or target_ego_yaws is None:
                    raise RuntimeError(
                        "constrained joint risk requires target planner action"
                    )
                action_samples = agents_info.get("action_samples", {})
                sample_positions = action_samples.get("positions")
                sample_yaws = action_samples.get("yaws")
                world_from_agent = agents_obs.get("world_from_agent")
                extents = agents_obs.get(
                    "extent", agents_obs.get("history_extents")
                )
                if any(value is None for value in (
                    sample_positions, sample_yaws, world_from_agent, extents
                )):
                    raise RuntimeError(
                        "constrained joint risk requires samples/transforms/extents"
                    )
                if (
                    config.get("mode") == "planner_aligned_native_joint"
                    and selector_attack_positions is not None
                ):
                    selector_reference_positions = target_ego_positions
                    selector_reference_yaws = target_ego_yaws
                    selector_target_positions = selector_attack_positions
                    selector_target_yaws = selector_attack_yaws
                else:
                    (
                        selector_reference_positions,
                        selector_reference_yaws,
                        selector_target_positions,
                        selector_target_yaws,
                    ) = self._joint_selector_planner_roles(
                        config.get("mode"),
                        ego_positions,
                        ego_yaws,
                        target_ego_positions,
                        target_ego_yaws,
                    )
                positions, yaws, joint_audit = (
                    select_constrained_native_joint_mode_action(
                        sample_positions=sample_positions,
                        sample_yaws=sample_yaws,
                        baseline_positions=positions,
                        baseline_yaws=yaws,
                        reference_ego_positions=selector_reference_positions,
                        reference_ego_yaws=selector_reference_yaws,
                        target_ego_positions=selector_target_positions,
                        target_ego_yaws=selector_target_yaws,
                        world_from_agent=world_from_agent,
                        extents=extents,
                        ego_indices=ego_indices,
                        collateral_margin_m=config["collateral_margin_m"],
                        reference_safety_margin_m=config[
                            "reference_safety_margin_m"
                        ],
                        target_max_clearance_m=config[
                            "max_target_clearance_m"
                        ],
                        baseline_min_target_clearance_m=config.get(
                            "baseline_min_target_clearance_m"
                        ),
                        target_min_persistence_steps=config.get(
                            "target_min_persistence_steps", 1
                        ),
                        progressive_target_barrier=config.get(
                            "progressive_target_barrier", False
                        ),
                        progressive_activation_max_clearance_m=config.get(
                            "progressive_activation_max_clearance_m"
                        ),
                        progressive_risk_first=config.get(
                            "progressive_risk_first", False
                        ),
                        progressive_persistence_first=config.get(
                            "progressive_persistence_first", False
                        ),
                        progressive_dose_first=config.get(
                            "progressive_dose_first", False
                        ),
                        progressive_exposure_clearance_m=config.get(
                            "progressive_exposure_clearance_m", 2.0
                        ),
                        min_risk_improvement_m=config[
                            "min_risk_improvement_m"
                        ],
                        max_agent_tail_fraction=config[
                            "max_agent_tail_fraction"
                        ],
                        dt=config["dt"],
                        max_speed_mps=config["max_speed_mps"],
                        max_acc_mps2=config["max_acc_mps2"],
                        max_jerk_mps3=config["max_jerk_mps3"],
                        max_yaw_rate_rps=config["max_yaw_rate_rps"],
                        max_lateral_acc_mps2=config[
                            "max_lateral_acc_mps2"
                        ],
                        kinematic_horizon_steps=config[
                            "kinematic_horizon_steps"
                        ],
                        naturalness_profile=config["naturalness_profile"],
                        history_positions=agents_obs.get("history_positions"),
                        history_yaws=agents_obs.get("history_yaws"),
                        history_availabilities=agents_obs.get(
                            "history_availabilities"
                        ),
                        target_agent_indices=config.get(
                            "target_agent_indices"
                        ),
                    )
                )
                print(
                    "JOINT_CONSTRAINED_RISK_STEP step={} sample={}/{} "
                    "valid_joint_modes={} gates=finite:{}/target:{}/reference:{}/"
                    "collateral:{}/physical:{}/natural:{}/improvement:{} "
                    "fallback={} target_clearance_m={} "
                    "baseline_target_clearance_m={} persistent_risk_steps={}/{} "
                    "exposure_steps={} risk_dose_m_steps={} "
                    "target_gate_mode={} terminal_target_modes={} activation_gate={} "
                    "reference_clearance_m={} collateral_clearance_m={} "
                    "max_agent_q99_tail_fraction={} same_joint_sample=True "
                    "row_mixing=False projection=False trajectory_modified=False".format(
                        int(step_index), joint_audit["selected_sample_index"],
                        joint_audit["num_samples"],
                        joint_audit["valid_joint_mode_count"],
                        joint_audit["finite_gate_count"],
                        joint_audit["target_gate_count"],
                        joint_audit["reference_gate_count"],
                        joint_audit["collateral_gate_count"],
                        joint_audit["physical_gate_count"],
                        joint_audit["naturalness_gate_count"],
                        joint_audit["risk_improvement_gate_count"],
                        joint_audit["hard_gate_fallback"],
                        joint_audit["selected_target_min_clearance_m"],
                        joint_audit["baseline_target_min_clearance_m"],
                        joint_audit["selected_target_max_consecutive_risk_steps"],
                        joint_audit["target_min_persistence_steps"],
                        joint_audit[
                            "selected_target_max_consecutive_exposure_steps"
                        ],
                        joint_audit["selected_target_risk_dose_m_steps"],
                        joint_audit["target_gate_mode"],
                        joint_audit["terminal_target_gate_count"],
                        joint_audit["progressive_activation_gate"],
                        joint_audit["selected_reference_min_clearance_m"],
                        joint_audit["selected_min_collateral_clearance_m"],
                        joint_audit["selected_max_agent_q99_tail_fraction"],
                    )
                )
                print(
                    "JOINT_CONSTRAINED_RISK_DIAGNOSTIC step={} "
                    "physical_by_metric={} physical_by_agent={} "
                    "natural_by_agent={} target_agent_indices={} "
                    "best_target_sample={} best_target_clearance_m={} "
                    "best_target_improvement_m={} target_clearance_quantiles_m={} "
                    "non_target_hard_modes={} risk_and_non_target_hard_modes={} "
                    "best_hard_safe_sample={} best_hard_safe_clearance_m={} "
                    "best_hard_safe_improvement_m={} "
                    "sample0_natural={} sample0_physical={} "
                    "sample0_max_agent_q99_tail_fraction={} "
                    "sample0_q99_by_agent={}".format(
                        int(step_index),
                        joint_audit["physical_gate_counts_by_metric"],
                        joint_audit["physical_gate_counts_by_agent"],
                        joint_audit["naturalness_gate_counts_by_agent"],
                        joint_audit["target_agent_indices"],
                        joint_audit["best_target_sample_index"],
                        joint_audit["best_target_clearance_m"],
                        joint_audit["best_target_improvement_m"],
                        joint_audit["target_clearance_quantiles_m"],
                        joint_audit["non_target_hard_gate_count"],
                        joint_audit["risk_and_non_target_hard_gate_count"],
                        joint_audit["best_hard_safe_sample_index"],
                        joint_audit["best_hard_safe_target_clearance_m"],
                        joint_audit["best_hard_safe_target_improvement_m"],
                        joint_audit["proposal_sample0_naturalness_gate"],
                        joint_audit["proposal_sample0_physical_gate"],
                        joint_audit[
                            "proposal_sample0_max_agent_q99_tail_fraction"
                        ],
                        joint_audit[
                            "proposal_sample0_q99_tail_fraction_by_agent"
                        ],
                    )
                )
                marginal_tail_audit = {
                    "active": True,
                    "receiver_count": joint_audit["non_ego_agent_count"],
                    "nonzero_selection_count": int(
                        joint_audit["selected_sample_index"] != 0
                    ),
                    **joint_audit,
                }
            else:
                marginal_tail_audit = {
                    "active": False,
                    "receiver_count": max(0, batch_size - len(ego_indices)),
                    "nonzero_selection_count": 0,
                    "num_samples": int(config["num_samples"]),
                    "all_non_ego_same_joint_sample": True,
                    "trajectory_modified": False,
                }
            config = None
        if config is not None and config.get("mode") == "fixed_native_joint":
            active = (
                int(step_index) >= int(config["start_step"])
                and (
                    config.get("end_step") is None
                    or int(step_index) < int(config["end_step"])
                )
            )
            if active:
                action_samples = agents_info.get("action_samples", {})
                sample_positions = action_samples.get("positions")
                sample_yaws = action_samples.get("yaws")
                if sample_positions is None or sample_yaws is None:
                    raise RuntimeError(
                        "M2 action_samples are required for native joint proposals"
                    )
                positions, yaws, native_joint_audit = (
                    select_fixed_native_joint_mode_action(
                        sample_positions=sample_positions,
                        sample_yaws=sample_yaws,
                        baseline_positions=positions,
                        baseline_yaws=yaws,
                        ego_indices=ego_indices,
                        sample_index=config["fixed_joint_sample_index"],
                    )
                )
                print(
                    "NATIVE_JOINT_MODE_STEP step={} sample={}/{} "
                    "non_ego_agents={} native_exact=True proposal_only=True "
                    "full_horizon_verifier_required=True gradients=False "
                    "projection=False trajectory_modified=False".format(
                        int(step_index),
                        native_joint_audit["selected_sample_index"],
                        native_joint_audit["num_samples"],
                        native_joint_audit["non_ego_agent_count"],
                    )
                )
                marginal_tail_audit = {
                    "active": True,
                    "receiver_count": native_joint_audit["non_ego_agent_count"],
                    "nonzero_selection_count": int(
                        native_joint_audit["selected_sample_index"] != 0
                    ),
                    "num_samples": native_joint_audit["num_samples"],
                    "selected_sample_index": native_joint_audit[
                        "selected_sample_index"
                    ],
                    "native_joint_exact": True,
                    "proposal_only": True,
                    "full_horizon_verifier_required": True,
                    "trajectory_modified": False,
                }
            else:
                marginal_tail_audit = {
                    "active": False,
                    "receiver_count": max(0, batch_size - len(ego_indices)),
                    "nonzero_selection_count": 0,
                    "num_samples": int(config["num_samples"]),
                    "selected_sample_index": 0,
                    "native_joint_exact": True,
                    "proposal_only": True,
                    "full_horizon_verifier_required": True,
                    "trajectory_modified": False,
                }
            # The ordinary marginal selector is mutually exclusive with a
            # fixed full-joint proposal.  Keep the common planner override and
            # audit serialization below.
            config = None
        if (
            config is not None
            and config.get("mode") != "conditional_fm_source"
        ):
            active = (
                int(step_index) >= int(config["start_step"])
                and (
                    config.get("end_step") is None
                    or int(step_index) < int(config["end_step"])
                )
            )
            if active:
                action_samples = agents_info.get("action_samples", {})
                sample_positions = action_samples.get("positions")
                sample_yaws = action_samples.get("yaws")
                if sample_positions is None or sample_yaws is None:
                    raise RuntimeError("M2 action_samples are required for joint-tail selection")
                world_from_agent = agents_obs.get("world_from_agent")
                extents = agents_obs.get("extent", agents_obs.get("history_extents"))
                if world_from_agent is None or extents is None:
                    raise RuntimeError("joint-tail selection requires transforms and extents")
                selection_kwargs = dict(
                    sample_positions=sample_positions,
                    sample_yaws=sample_yaws,
                    baseline_positions=positions,
                    baseline_yaws=yaws,
                    ego_positions=ego_positions,
                    ego_yaws=ego_yaws,
                    world_from_agent=world_from_agent,
                    extents=extents,
                    ego_index=ego_indices[0],
                    collateral_margin_m=config["collateral_margin_m"],
                    reference_safety_margin_m=config["reference_safety_margin_m"],
                    min_risk_improvement_m=config["min_risk_improvement_m"],
                    max_target_clearance_m=config.get("max_target_clearance_m"),
                    min_baseline_target_clearance_m=config.get(
                        "min_baseline_target_clearance_m"
                    ),
                    dt=config["dt"],
                    max_speed_mps=config["max_speed_mps"],
                    max_acc_mps2=config["max_acc_mps2"],
                    max_jerk_mps3=config["max_jerk_mps3"],
                    max_yaw_rate_rps=config["max_yaw_rate_rps"],
                    max_lateral_acc_mps2=config["max_lateral_acc_mps2"],
                    kinematic_horizon_steps=config["kinematic_horizon_steps"],
                    naturalness_profile=config.get("naturalness_profile"),
                    history_positions=agents_obs.get("history_positions"),
                    history_yaws=agents_obs.get("history_yaws"),
                    history_availabilities=agents_obs.get("history_availabilities"),
                    target_ego_positions=target_ego_positions,
                    target_ego_yaws=target_ego_yaws,
                    minimize_causal_change=config.get(
                        "minimize_causal_change", False
                    ),
                )
                if config.get("auto_receivers", False):
                    receiver_indices = [
                        receiver_index
                        for receiver_index in range(batch_size)
                        if receiver_index not in set(ego_indices)
                    ]
                    positions, yaws, sparse_audit = (
                        select_sparse_marginal_joint_tail_action(
                            receiver_indices=receiver_indices,
                            locked_receiver_index=(
                                self._marginal_tail_locked_receiver_index
                                if config.get("sticky_receiver", False)
                                else None
                            ),
                            **selection_kwargs
                        )
                    )
                    if (
                        config.get("sticky_receiver", False)
                        and sparse_audit["selected_receiver_index"] >= 0
                    ):
                        selected_receiver = int(
                            sparse_audit["selected_receiver_index"]
                        )
                        if self._marginal_tail_locked_receiver_index is None:
                            self._marginal_tail_locked_receiver_index = (
                                selected_receiver
                            )
                        elif (
                            selected_receiver
                            != self._marginal_tail_locked_receiver_index
                        ):
                            raise RuntimeError(
                                "sticky marginal-tail receiver changed"
                            )
                    marginal_tail_audits = [sparse_audit]
                    print(
                        "SPARSE_NATIVE_JOINT_TAIL_STEP step={} "
                        "evaluated_receivers={} selectable_receivers={} "
                        "fallback_receivers={} baseline_unsafe_receivers={} "
                        "selected_receiver={} "
                        "sample={}/{} target_clearance_m={} "
                        "baseline_target_clearance_m={} "
                        "baseline_target_safe={} reference_clearance_m={} "
                        "sparse_noop={} "
                        "trajectory_modified=False sticky_receiver={} "
                        "locked_receiver={}".format(
                            int(step_index),
                            sparse_audit["evaluated_receiver_count"],
                            sparse_audit["selectable_receiver_count"],
                            sparse_audit["fallback_receiver_count"],
                            sparse_audit[
                                "baseline_target_unsafe_receiver_count"
                            ],
                            sparse_audit["selected_receiver_index"],
                            sparse_audit["selected_sample_index"],
                            sparse_audit["num_samples"],
                            sparse_audit["selected_target_min_clearance_m"],
                            sparse_audit["baseline_target_min_clearance_m"],
                            sparse_audit.get(
                                "baseline_target_safe_gate_pass", True
                            ),
                            sparse_audit["selected_reference_min_clearance_m"],
                            sparse_audit["sparse_noop"],
                            config.get("sticky_receiver", False),
                            (
                                -1
                                if self._marginal_tail_locked_receiver_index
                                is None
                                else self._marginal_tail_locked_receiver_index
                            ),
                        )
                    )
                else:
                    positions, yaws, marginal_tail_audits = (
                        select_multi_agent_marginal_joint_tail_action(
                            receiver_indices=config["receiver_indices"],
                            **selection_kwargs
                        )
                    )
                for marginal_receiver_audit in marginal_tail_audits:
                    if marginal_receiver_audit.get("auto_receiver_search", False):
                        continue
                    print(
                        "MARGINAL_JOINT_TAIL_STEP step={} receiver={} sample={}/{} "
                        "valid={} kinematic_valid={} baseline_clearance_m={:.6f} "
                        "selected_clearance_m={:.6f} trajectory_modified=False "
                        "baseline_reference_clearance_m={:.6f} "
                        "selected_reference_clearance_m={:.6f} "
                        "dual_planner={} "
                        "boundary_kinematics={} speed_max={} acc_max={} "
                        "hard_fallback={} baseline_hard_valid={} "
                        "jerk_max={} yaw_rate_max={} "
                        "lateral_acc_max={}".format(
                            int(step_index),
                            marginal_receiver_audit["receiver_index"],
                            marginal_receiver_audit["selected_sample_index"],
                            marginal_receiver_audit["num_samples"],
                            marginal_receiver_audit["valid_sample_count"],
                            marginal_receiver_audit["kinematic_valid_sample_count"],
                            marginal_receiver_audit["baseline_min_clearance_m"],
                            marginal_receiver_audit["selected_min_clearance_m"],
                            marginal_receiver_audit[
                                "baseline_reference_min_clearance_m"
                            ],
                            marginal_receiver_audit[
                                "selected_reference_min_clearance_m"
                            ],
                            marginal_receiver_audit["dual_planner_objective"],
                            marginal_receiver_audit["boundary_kinematics_used"],
                            [round(value, 3) for value in marginal_receiver_audit["sample_max_speed_mps"]],
                            [round(value, 3) for value in marginal_receiver_audit["sample_max_acc_mps2"]],
                            marginal_receiver_audit["hard_gate_fallback"],
                            marginal_receiver_audit["baseline_hard_valid"],
                            [round(value, 3) for value in marginal_receiver_audit["sample_max_jerk_mps3"]],
                            [round(value, 3) for value in marginal_receiver_audit["sample_max_yaw_rate_rps"]],
                            [round(value, 3) for value in marginal_receiver_audit["sample_max_lateral_acc_mps2"]],
                        )
                    )
                marginal_tail_audit = {
                    "active": True,
                    "receiver_count": len(marginal_tail_audits),
                    "nonzero_selection_count": sum(
                        row["selected_sample_index"] != 0
                        for row in marginal_tail_audits
                    ),
                    "num_samples": marginal_tail_audits[0]["num_samples"],
                    "trajectory_modified": False,
                }
            else:
                marginal_tail_audit = {
                    "active": False,
                    "receiver_count": (
                        batch_size
                        if config.get("auto_receivers", False)
                        else len(config["receiver_indices"])
                    ),
                    "nonzero_selection_count": 0,
                    "trajectory_modified": False,
                }
        positions[index] = ego_positions
        yaws[index] = ego_yaws

        agents_info = {} if agents_info is None else dict(agents_info)
        agents_info["ego_planner_positions"] = ego_positions
        agents_info["ego_planner_yaws"] = ego_yaws
        if target_ego_positions is not None:
            agents_info["marginal_tail_target_ego_positions"] = (
                target_ego_positions
            )
            agents_info["marginal_tail_target_ego_yaws"] = target_ego_yaws
        if selector_attack_positions is not None:
            agents_info["marginal_tail_attack_ego_positions"] = (
                selector_attack_positions
            )
            agents_info["marginal_tail_attack_ego_yaws"] = (
                selector_attack_yaws
            )
        info_device = positions.device if torch.is_tensor(positions) else self.device
        if marginal_tail_audit is not None:
            # RolloutAction.to_numpy cannot serialize Python scalar leaves.
            # Store every runtime attestation as a per-agent tensor instead.
            for key, value in marginal_tail_audit.items():
                if value is None:
                    continue
                if isinstance(value, (bool, int, float, np.number)):
                    dtype = torch.bool if isinstance(value, (bool, np.bool_)) else None
                    agents_info["marginal_joint_tail_{}".format(key)] = torch.full(
                        (batch_size,), value, device=info_device, dtype=dtype
                    )
        agents_info["planner_strategy_depth"] = torch.full(
            (batch_size,), self.strategy_depth, device=info_device
        )
        agents_info["candidate_future_supplied"] = torch.full(
            (batch_size,),
            self.strategy_depth > 0,
            device=info_device,
            dtype=torch.bool,
        )
        agents_info["candidate_future_used"] = torch.full(
            (batch_size,),
            self.strategy_depth > 0
            and getattr(self.ego_policy, "supports_candidate_future", False),
            device=info_device,
            dtype=torch.bool,
        )
        if ego_info:
            for key, value in ego_info.items():
                if torch.is_tensor(value) or isinstance(value, np.ndarray):
                    agents_info["ego_planner_{}".format(key)] = value
                elif isinstance(value, (bool, int, float, np.number)):
                    agents_info["ego_planner_{}".format(key)] = torch.full(
                        (len(ego_indices),), value, device=info_device
                    )
        return RolloutAction(
            agents=Action(positions=positions, yaws=yaws),
            agents_info=agents_info,
        )


class GenerativeEgoReplayRolloutWrapper(PlannerOverrideRolloutWrapper):
    """Use the unified generative model only for ego and replay other agents.

    This adapter makes the trained scene model an ego trajectory policy while
    holding the surrounding traffic to the logged future.  Results are a
    non-reactive log-replay closed-loop planner test, not a reactive world-model
    evaluation.
    """

    def __init__(self, generative_policy, transport_name):
        self.device = generative_policy.device
        self.agents_policy = generative_policy
        self.transport_name = str(transport_name)

    def eval(self):
        self.agents_policy.eval()
        return self

    def get_action(self, obs, step_index) -> RolloutAction:
        agents_obs = obs["agents"]
        if agents_obs is None:
            raise ValueError("Generative ego rollout requires unified agents obs")
        with torch.no_grad():
            generated, info = self.agents_policy.get_action(
                agents_obs, step_index=step_index
            )

        ego_indices = self._ego_indices(agents_obs)
        batch_size, horizon = generated.positions.shape[:2]
        replay_positions = torch.as_tensor(
            agents_obs["target_positions"],
            device=generated.positions.device,
            dtype=generated.positions.dtype,
        )
        replay_yaws = torch.as_tensor(
            agents_obs["target_yaws"],
            device=generated.yaws.device,
            dtype=generated.yaws.dtype,
        )
        replay_positions = torch.nan_to_num(
            self._fit_horizon(replay_positions, horizon)
        )
        replay_yaws = torch.nan_to_num(
            self._fit_horizon(replay_yaws, horizon)
        )
        positions = replay_positions.clone()
        yaws = replay_yaws.clone()
        index = torch.as_tensor(
            ego_indices, device=positions.device, dtype=torch.long
        )
        positions[index] = generated.positions[index]
        yaws[index] = generated.yaws[index]

        info = {} if info is None else dict(info)
        info["generative_ego_only"] = torch.ones(
            batch_size, device=positions.device, dtype=torch.bool
        )
        info["generative_ego_transport_is_flow"] = torch.full(
            (batch_size,),
            self.transport_name == "flow_matching",
            device=positions.device,
            dtype=torch.bool,
        )
        info["generative_ego_indices"] = index
        return RolloutAction(
            agents=Action(positions=positions, yaws=yaws),
            agents_info=info,
        )


class PerturbationWrapper(object):
    """A wrapper policy that perturbs the policy action with Ornstein Uhlenbeck noise"""

    def __init__(self, policy, noise):
        self.device = policy.device
        self.noise = noise
        self.policy = policy

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        actions, action_info = self.policy.get_action(obs, **kwargs)
        actions_dict = OrderedDict(target_positions=actions.positions,target_yaws=actions.yaws) 
        perturbed_action_dict = self.noise.perturb(TensorUtils.to_numpy(actions_dict))
        perturbed_action_dict = TensorUtils.to_torch(perturbed_action_dict,self.device)
        perturbed_actions = Action(perturbed_action_dict["target_positions"],perturbed_action_dict["target_yaws"])
        return perturbed_actions, action_info

# --------------------------------------------------------------------------------
from tbsim.utils.guidance_loss import PerturbationGuidance
from ccdiff.models.diffuser_helpers import state_grad_general_transform
class NewSamplingPolicyWrapper(object):
    def __init__(self, ego_action_sampler, guidance_config_list, transform=state_grad_general_transform, transform_params={'dt': 0.1}):
        """

        Args:
            -ego_action_sampler: a policy that generates N action samples
            -guidance_config_list: guidance config for loss estimation
            -transform: transform applied on the trajectory before estimating loss
            -transform_params: extra parameters taken by transform
        """
        self.device = ego_action_sampler.device
        self.sampler = ego_action_sampler

        self.guidance = PerturbationGuidance(transform, transform_params)
        self.guidance_config_list = guidance_config_list
        self.guidance.set_guidance(self.guidance_config_list)

    def eval(self):
        self.sampler.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        # actions of shape [B, num_samples, ...]
        _, action_info = self.sampler.get_action(obs, **kwargs)
        action_samples = action_info["action_samples"]

        if isinstance(action_samples, dict):
            action_samples = Action.from_dict(action_samples)

        ego_trajs = action_samples.trajectories
        B, N, *_ = ego_trajs.shape        
        ego_trajs_BN = TensorUtils.join_dimensions(ego_trajs, begin_axis=0, end_axis=2) # B*N, T, D
        
        _, guide_losses = self.guidance.compute_guidance_loss(ego_trajs_BN, obs, num_samp=N)

        act_idx = torch.zeros((B), dtype=torch.long) # arbitrarily use the first sample as the action if no guidance is given
        if any(self.guidance_config_list):
            from ccdiff.models.diffuser_helpers import choose_act_using_guide_loss
            act_idx = choose_act_using_guide_loss(guide_losses, self.guidance.current_guidance.guide_configs, act_idx)    
   
        ego_trajs_best = TensorUtils.map_tensor(ego_trajs, lambda x: x[torch.arange(B), act_idx])


        ego_actions = Action(
            positions=ego_trajs_best[..., :2], yaws=ego_trajs_best[..., 2:])
        action_info["action_samples"] = action_samples.to_dict()
        # print('action_info["action_samples"]["positions"]', action_info["action_samples"]["positions"].shape, action_info["action_samples"]["positions"][0, 0, 0, 0].data)
        

        if "plan_samples" in action_info and not isinstance(action_info["plan_samples"], dict):
            action_info["plan_samples"] = action_info["plan_samples"].to_dict()
        return ego_actions, action_info
    
    def update_guidance_config(self, guidance_config_list):
        self.guidance.set_guidance(guidance_config_list)


class AgentCentricToSceneCentricWrapper(object):
    '''
    Note: only support batch size 1 for now.

    1.takes in agent-centric data and convert it to scene-centric data
    2.apply policy get_action
    3.convert scene-centric actions back to agent-centric actions and return
    '''
    def __init__(self, policy):
        self.device = policy.device
        self.policy = policy
        self.model = policy.model

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        # convert agent-centric data to scene-centric data
        ego_obs, agent_obs = self.split_ego_and_agent_obs(obs)
        scene_obs = self.convert_agent_centric_obs_to_scene_centric(ego_obs, agent_obs)

        # TBD: temporary hack for debugging
        actions_scene_centric, action_info_scene_centric = self.policy.get_action(scene_obs, agent_obs_gt=obs, **kwargs)

        # convert scene-centric actions to agent-centric actions
        actions, action_info = self.convert_scene_centric_action_to_agent_centric(actions_scene_centric, action_info_scene_centric, ego_obs, agent_obs)
        
        return actions, action_info
    
    @staticmethod
    def split_ego_and_agent_obs(obs):
        # split ego and agent observation
        # Assume the first element in obs is ego (we can also use other criteria to set ego)
        # TBD: support multiple scenes
        ego_obs = dict()
        agent_obs = dict()
        for k,v in obs.items():
            if v is not None:
                if k not in ['history_pad_dir', 'extras']:
                    ego_obs[k] = v[:1]
                    agent_obs[k] = v[1:]
                elif k == 'extras':
                    ego_obs[k], agent_obs[k] = dict(), dict()
                    for sub_k, sub_v in v.items():
                        ego_obs[k][sub_k] = sub_v[:1]
                        agent_obs[k][sub_k] = sub_v[1:]
        return ego_obs, agent_obs

    @staticmethod
    def convert_agent_centric_obs_to_scene_centric(ego_obs, agent_obs, max_num_agents=None):
        # TBD: handle scene_index for multiple scenes in a batch
        num_agents = [sum(agent_obs["scene_index"]==i)+1 for i in ego_obs["scene_index"]]
        num_agents = torch.tensor(num_agents,device=ego_obs["scene_index"].device)
        hist_pos_b = list()
        hist_yaw_b = list()
        hist_speeds_b = list()
        hist_avail_b = list()
        fut_pos_b = list()
        fut_yaw_b = list()
        fut_avail_b = list()
        agent_from_center_b = list()
        center_from_agent_b = list()
        raster_from_center_b = list()
        center_from_raster_b = list()
        raster_from_world_b = list()
        maps_b = list()
        curr_speed_b = list()
        type_b = list()
        extents_b = list()
        scene_index_b = list()
        neighbor_indices_b = list()
        closest_lanes_points_b = list()
        extras = dict()

        full_fut_traj_b = list()
        full_fut_valid_b = list()
        agent_hist_b = list()

        for i,scene_idx in enumerate(ego_obs["scene_index"]):
            agent_idx = torch.where(agent_obs["scene_index"]==scene_idx)[0]
            center_from_agent = ego_obs["agent_from_world"][i].unsqueeze(0) @ agent_obs["world_from_agent"][agent_idx]
            center_from_agents = torch.cat((torch.eye(3,device=center_from_agent.device).unsqueeze(0),center_from_agent),0)
            ego_and_agent_yaw = torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0) # [M]
            hist_pos_raw = torch.cat((ego_obs["history_positions"][i:i+1],agent_obs["history_positions"][agent_idx]),0)
            hist_yaw_raw = torch.cat((ego_obs["history_yaws"][i:i+1],agent_obs["history_yaws"][agent_idx]),0)
            
            agents_hist_avail = torch.cat((ego_obs["history_availabilities"][i:i+1],agent_obs["history_availabilities"][agent_idx]),0)
            agents_hist_pos = GeoUtils.transform_points_tensor(hist_pos_raw,center_from_agents)*agents_hist_avail.unsqueeze(-1)
            agents_hist_yaw = (hist_yaw_raw+ego_and_agent_yaw[:, None, None]-ego_obs["yaw"][i:i+1])*agents_hist_avail.unsqueeze(-1)
            agents_hist_speeds = torch.cat((ego_obs["history_speeds"][i:i+1],agent_obs["history_speeds"][agent_idx]),0)*agents_hist_avail

            if 'closest_lane_point' in ego_obs['extras']:
                closest_lane_point = torch.cat((ego_obs['extras']['closest_lane_point'][i:i+1], agent_obs['extras']['closest_lane_point'][agent_idx]),0)
                M, S_seg, S_p, _ = closest_lane_point.shape
                # [M, S_seg, S_p, 3] -> [M, S_seg*S_p, 3]
                closest_lane_point = closest_lane_point.reshape(M, S_seg*S_p, -1)
                closest_lane_point_pos = closest_lane_point[..., :2] # [M, S_seg*S_p, 2]
                closest_lane_point_yaw = closest_lane_point[..., [2]] # [M, S_seg*S_p, 1]

                closest_lane_point_pos_transformed = GeoUtils.transform_points_tensor(closest_lane_point_pos,center_from_agents)
                closest_lane_point_yaw_transformed = angle_wrap(closest_lane_point_yaw+ego_and_agent_yaw[:, None, None]-ego_obs["yaw"][i:i+1])

                closest_lanes_points = torch.cat((closest_lane_point_pos_transformed, closest_lane_point_yaw_transformed), -1).reshape(M, S_seg, S_p, -1)
                
                closest_lanes_points_b.append(closest_lanes_points)
            
            # TBD: warning: currently we only do the conversion for x and y in full_fut_traj but not other fields as those are not used but those should be converted as well!!!
            if 'full_fut_valid' in ego_obs['extras'] and 'full_fut_traj' in ego_obs['extras']:
                full_fut_valid = torch.cat((ego_obs['extras']['full_fut_valid'][i:i+1],agent_obs['extras']['full_fut_valid'][agent_idx]),0)
                full_fut_valid_b.append(full_fut_valid)
                full_fut_traj = torch.cat((ego_obs['extras']['full_fut_traj'][i:i+1],agent_obs['extras']['full_fut_traj'][agent_idx]),0)

                full_fut_traj[...,:2] = GeoUtils.transform_points_tensor(full_fut_traj[...,:2],center_from_agents)*full_fut_valid.unsqueeze(-1)
                full_fut_traj_b.append(full_fut_traj)
            
            # TBD: warning: currently we only do the conversion for x, y, and yaw in agent_hist but not other fields as those are not used but those should be converted as well!!!
            agent_hist = torch.cat((ego_obs["agent_hist"][i:i+1],agent_obs["agent_hist"][agent_idx]),0)
            agent_hist[...,:2] = agents_hist_pos
            agent_hist[...,[2]] = agents_hist_yaw 
            agent_hist_b.append(agent_hist)
            
            
            hist_pos_b.append(agents_hist_pos)
            hist_yaw_b.append(agents_hist_yaw)
            hist_speeds_b.append(agents_hist_speeds)
            hist_avail_b.append(agents_hist_avail)
            if agent_obs["target_availabilities"].shape[1]<ego_obs["target_availabilities"].shape[1]:
                pad_shape=(agent_obs["target_availabilities"].shape[0],ego_obs["target_availabilities"].shape[1]-agent_obs["target_availabilities"].shape[1])
                agent_obs["target_availabilities"] = torch.cat((agent_obs["target_availabilities"],torch.zeros(pad_shape,device=agent_obs["target_availabilities"].device)),1)
            agents_fut_avail = torch.cat((ego_obs["target_availabilities"][i:i+1],agent_obs["target_availabilities"][agent_idx]),0)
            if agent_obs["target_positions"].shape[1]<ego_obs["target_positions"].shape[1]:
                pad_shape=(agent_obs["target_positions"].shape[0],ego_obs["target_positions"].shape[1]-agent_obs["target_positions"].shape[1],*agent_obs["target_positions"].shape[2:])
                agent_obs["target_positions"] = torch.cat((agent_obs["target_positions"],torch.zeros(pad_shape,device=agent_obs["target_positions"].device)),1)
                pad_shape=(agent_obs["target_yaws"].shape[0],ego_obs["target_yaws"].shape[1]-agent_obs["target_yaws"].shape[1],*agent_obs["target_yaws"].shape[2:])
                agent_obs["target_yaws"] = torch.cat((agent_obs["target_yaws"],torch.zeros(pad_shape,device=agent_obs["target_yaws"].device)),1)
            fut_pos_raw = torch.cat((ego_obs["target_positions"][i:i+1],agent_obs["target_positions"][agent_idx]),0)
            fut_yaw_raw = torch.cat((ego_obs["target_yaws"][i:i+1],agent_obs["target_yaws"][agent_idx]),0)
            agents_fut_pos = GeoUtils.transform_points_tensor(fut_pos_raw,center_from_agents)*agents_fut_avail.unsqueeze(-1)
            agents_fut_yaw = (fut_yaw_raw+torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0)[:,None,None]-ego_obs["yaw"][i])*agents_fut_avail.unsqueeze(-1)
            fut_pos_b.append(agents_fut_pos)
            fut_yaw_b.append(agents_fut_yaw)
            fut_avail_b.append(agents_fut_avail)

            curr_yaw = agents_hist_yaw[:,-1]
            curr_pos = agents_hist_pos[:,-1]
            agents_from_center = GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros_like(curr_pos))@GeoUtils.transform_matrices(torch.zeros_like(curr_yaw).flatten(),-curr_pos)
                             
            # raster_from_center = centered_raster_from_agent @ agents_from_center
            center_from_raster = center_from_agents @ ego_obs["agent_from_raster"]
            # raster_from_world = torch.cat((ego_obs["raster_from_world"][i:i+1],agent_obs["raster_from_world"][agent_idx]),0)


            agent_from_center_b.append(agents_from_center)
            center_from_agent_b.append(center_from_agents)
            # raster_from_center_b.append(raster_from_center)
            center_from_raster_b.append(center_from_raster)
            # raster_from_world_b.append(raster_from_world)

            maps = torch.cat((ego_obs["image"][i:i+1],agent_obs["image"][agent_idx]),0)
            curr_speed = torch.cat((ego_obs["curr_speed"][i:i+1],agent_obs["curr_speed"][agent_idx]),0)
            agents_type = torch.cat((ego_obs["type"][i:i+1],agent_obs["type"][agent_idx]),0)
            agents_extent = torch.cat((ego_obs["extent"][i:i+1],agent_obs["extent"][agent_idx]),0)
            scene_index = torch.cat((ego_obs["scene_index"][i:i+1],agent_obs["scene_index"][agent_idx]),0)
            neighbor_indices = torch.cat((ego_obs["all_other_agents_indices"][i:i+1],agent_obs["all_other_agents_indices"][agent_idx]),0)
            maps_b.append(maps)
            curr_speed_b.append(curr_speed)
            type_b.append(agents_type)
            extents_b.append(agents_extent)
            scene_index_b.append(scene_index)
            neighbor_indices_b.append(neighbor_indices)

        extras['closest_lane_point'] = pad_sequence(closest_lanes_points_b,batch_first=True,padding_value=0)
        extras['full_fut_traj'] = pad_sequence(full_fut_traj_b,batch_first=True,padding_value=0)
        extras['full_fut_valid'] = pad_sequence(full_fut_valid_b,batch_first=True,padding_value=0)
        agent_hist = pad_sequence(agent_hist_b,batch_first=True,padding_value=0)
        scene_obs = dict(
            num_agents=num_agents,
            image=pad_sequence(maps_b,batch_first=True,padding_value=0),
            map_names=ego_obs["map_names"],
            drivable_map=ego_obs["drivable_map"],
            target_positions=pad_sequence(fut_pos_b,batch_first=True,padding_value=0),
            target_yaws=angle_wrap(pad_sequence(fut_yaw_b,batch_first=True,padding_value=0)),
            target_availabilities=pad_sequence(fut_avail_b,batch_first=True,padding_value=0),
            history_positions=pad_sequence(hist_pos_b,batch_first=True,padding_value=0),
            history_yaws=angle_wrap(pad_sequence(hist_yaw_b,batch_first=True,padding_value=0)),
            history_speeds=pad_sequence(hist_speeds_b,batch_first=True,padding_value=0),
            history_availabilities=pad_sequence(hist_avail_b,batch_first=True,padding_value=0),
            curr_speed=pad_sequence(curr_speed_b,batch_first=True,padding_value=0),
            centroid=ego_obs["centroid"],
            yaw=angle_wrap(ego_obs["yaw"]),
            type=pad_sequence(type_b,batch_first=True,padding_value=0),
            extent=pad_sequence(extents_b,batch_first=True,padding_value=0),
            raster_from_agent=ego_obs["raster_from_agent"].squeeze(0),
            agent_from_raster=ego_obs["agent_from_raster"].squeeze(0),
            # raster_from_center=pad_sequence(raster_from_center_b,batch_first=True,padding_value=0),
            # center_from_raster=pad_sequence(center_from_raster_b,batch_first=True,padding_value=0),
            agents_from_center=pad_sequence(agent_from_center_b,batch_first=True,padding_value=0),
            center_from_agents=pad_sequence(center_from_agent_b,batch_first=True,padding_value=0),
            # raster_from_world=pad_sequence(raster_from_world_b,batch_first=True,padding_value=0),
            agent_from_world=ego_obs["agent_from_world"],
            world_from_agent=ego_obs["world_from_agent"],
            extras=extras,
            agent_hist=agent_hist,

            scene_index=pad_sequence(scene_index_b,batch_first=True,padding_value=-1),
            neighbor_indices=pad_sequence(neighbor_indices_b,batch_first=True,padding_value=0),

        )
        if max_num_agents is not None and scene_obs["num_agents"].max()>max_num_agents:
            dis = torch.norm(scene_obs["history_positions"][:,:,-1],dim=-1)
            dis = dis.masked_fill_(~scene_obs["history_availabilities"][...,-1],np.inf)
            idx = torch.argsort(dis,dim=1)[:,:max_num_agents]
            for k,v in scene_obs.items():
                if v.shape[:2]==dis.shape:
                    scene_obs[k] = TensorUtils.gather_from_start_single(scene_obs[k],idx)

        return scene_obs
    
    @staticmethod
    def convert_scene_centric_action_to_agent_centric(action, info, ego_obs, agent_obs):

        # Approach 1: Transform based on state
        pos_all = action.positions # (B, M, T, 2)
        yaw_all = action.yaws # (B, M, T, 1)
        pos_info_all = info['action_samples']['positions'] # (B, N, M, T, 2)
        yaw_info_all = info['action_samples']['yaws'] # (B, N, M, T, 1)

        pos_agent_list = []
        yaw_agent_list = []
        pos_info_agent_list = []
        yaw_info_agent_list = []


        for i,scene_idx in enumerate(ego_obs["scene_index"]):
            pos = pos_all[i] # (M, T, 2)
            yaw = yaw_all[i] # (M, T, 1)
            pos_info = pos_info_all[i] # (N, M, T, 2)
            yaw_info = yaw_info_all[i] # (N, M, T, 1)

            N, M, T, _ = pos_info.shape

            agent_idx = torch.where(agent_obs["scene_index"]==scene_idx)[0]
            agent_from_center = agent_obs["agent_from_world"][agent_idx] @ ego_obs["world_from_agent"][i].unsqueeze(0)
            agents_from_center = torch.cat((torch.eye(3,device=agent_from_center.device).unsqueeze(0),agent_from_center),0) # (M, 3, 3)

            ego_and_agent_yaw = torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0)
            
            pos_agent = GeoUtils.transform_points_tensor(pos,agents_from_center)
            yaw_agent = yaw-ego_and_agent_yaw[:,None,None]+ego_obs["yaw"][i:i+1]
            
            agents_from_center_N = agents_from_center.unsqueeze(0).repeat(N, 1, 1, 1).reshape(N*M, 3, 3)
            pos_info_agent = GeoUtils.transform_points_tensor(pos_info.reshape(N*M, T, 2),agents_from_center_N)
            yaw_info_agent = yaw_info.reshape(N*M, T, 1)-ego_and_agent_yaw[:,None,None].unsqueeze(0).repeat(N, 1, 1, 1).reshape(N*M, 1, 1)+ego_obs["yaw"][i:i+1]
            
            pos_info_agent = pos_info_agent.reshape(N, M, T, 2)
            yaw_info_agent = yaw_info_agent.reshape(N, M, T, 1)

            pos_agent_list.append(pos_agent)
            yaw_agent_list.append(yaw_agent)
            pos_info_agent_list.append(pos_info_agent)
            yaw_info_agent_list.append(yaw_info_agent)

        pos_agent_list = torch.cat(pos_agent_list,0)        
        yaw_agent_list = torch.cat(yaw_agent_list,0)
        pos_info_agent_list = torch.cat(pos_info_agent_list,0)
        yaw_info_agent_list = torch.cat(yaw_info_agent_list,0)
        actions = Action(
            positions=pos_agent_list,
            yaws=yaw_agent_list
        )
        infos = dict(
            action_samples=Action(
                positions=pos_info_agent_list.permute(1, 0, 2, 3), # (N, M, T, 2) -> (M, N, T, 2)
                yaws=yaw_info_agent_list.permute(1, 0, 2, 3), # (N, M, T, 1) -> (M, N, T, 1)
            ).to_dict(),
        )
        
        return actions, infos
        

# TBD: implement this (currently a placeholder)
class SceneCentricToAgentCentricWrapper(object):
    '''
    Note: only support batch size 1 for now.
    TBD: for debugging purpose. Because of reassign_values, this wrapper does not work when using independently (i.e., w.o. using the AgentCentricToSceneCentricWrapper)

    1.takes in scene-centric data and convert it to agent-centric data
    2.apply policy get_action
    3.convert agent-centric actions back to scene-centric actions and return
    '''
    def __init__(self, policy):
        self.device = policy.device
        self.policy = policy

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        from copy import deepcopy
        agent_centric_obs2 = convert_scene_data_to_agent_coordinates(obs)

        # reassign some values
        obs = self.reassign_values(obs)
        # convert scene-centric data to agent-centric data
        ego_obs, agent_obs = self.split_ego_and_agent_obs(obs)
        agent_centric_obs = self.convert_scene_centric_obs_to_agent_centric(ego_obs, agent_obs)

        # check if double conversion is correct
        if 'agent_obs_gt' in kwargs:
            agent_obs_gt = kwargs['agent_obs_gt']
            check_consistency(major_keywords+prep_keywords, agent_centric_obs2, agent_obs_gt, agent_obs_in_scene_format=True)
            check_consistency(full_keywords, agent_centric_obs, agent_obs_gt)
                        
        actions_agent_centric, action_info_agent_centric = self.policy.get_action(agent_centric_obs, **kwargs)

        # convert agent-centric actions to scene-centric actions
        actions, action_info = self.convert_agent_centric_action_to_scene_centric(actions_agent_centric, action_info_agent_centric, ego_obs, agent_obs)

        # TBD: for debugging
        action_info['action_agent_gt'] = actions_agent_centric
        
        return actions, action_info
    @staticmethod
    def reassign_values(obs):
        new_obs = convert_scene_obs_to_agent_prep(obs)
        for k, v in new_obs.items():
            obs[k] = v
        
        # TBD: hacky picking the only first scene in the batch
        for k, v in obs.items():
            if k == 'extras':
                for sub_k, sub_v in v.items():
                    obs[k][sub_k] = sub_v[0]
            # TBD: hacky way to manually avoid scene_index
            else:
                obs[k] = obs[k][0]
        return obs

    @staticmethod
    def split_ego_and_agent_obs(obs):
        # split ego and agent observation
        # Assume the first element in obs is ego (we can also use other criteria to set ego)
        ego_obs = dict()
        agent_obs = dict()
        for k,v in obs.items():
            if v is not None:
                if k not in ['history_pad_dir', 'extras', 'num_agents']:
                    ego_obs[k] = v[:1]
                    agent_obs[k] = v[1:]
                elif k == 'extras':
                    for sub_k in v:
                        ego_obs[k], agent_obs[k] = dict(), dict()
                        ego_obs[k][sub_k] = v[sub_k][:1]
                        agent_obs[k][sub_k] = v[sub_k][1:]
        return ego_obs, agent_obs

    @staticmethod
    def convert_scene_centric_obs_to_agent_centric(ego_obs,agent_obs):
        hist_pos_b = list()
        hist_yaw_b = list()
        hist_speeds_b = list()
        hist_avail_b = list()
        fut_pos_b = list()
        fut_yaw_b = list()
        fut_avail_b = list()
        maps_b = list()
        curr_speed_b = list()
        type_b = list()
        extents_b = list()
        closest_lanes_points_b = list()
        extras = dict()

        neigh_indices_b = list()
        hist_neigh_pos_b = list()
        hist_neigh_yaw_b = list()
        hist_neigh_speeds_b = list()
        hist_neigh_avail_b = list()
        neigh_curr_speed_b = list()
        fut_neigh_pos_b = list()
        fut_neigh_yaw_b = list()
        fut_neigh_avail_b = list()
        neigh_type_b = list()
        neigh_extents_b = list()

        
        # TBD: implement the case when assuming one scene
        for i,scene_idx in enumerate(ego_obs["scene_index"]):
            agent_idx = torch.where(agent_obs["scene_index"]==scene_idx)[0]
            agent_from_center = agent_obs["agent_from_world"][agent_idx] @ ego_obs["world_from_agent"][i].unsqueeze(0)
            agents_from_center = torch.cat((torch.eye(3,device=agent_from_center.device).unsqueeze(0),agent_from_center),0) # (M, 3, 3)
            ego_and_agent_yaw = torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0) # [M]
            hist_pos_raw = torch.cat((ego_obs["history_positions"][i:i+1],agent_obs["history_positions"][agent_idx]),0)
            hist_yaw_raw = torch.cat((ego_obs["history_yaws"][i:i+1],agent_obs["history_yaws"][agent_idx]),0)
            
            agents_hist_avail = torch.cat((ego_obs["history_availabilities"][i:i+1],agent_obs["history_availabilities"][agent_idx]),0)
            agents_hist_pos = GeoUtils.transform_points_tensor(hist_pos_raw,agents_from_center)*agents_hist_avail.unsqueeze(-1)
            agents_hist_yaw = (hist_yaw_raw-ego_and_agent_yaw[:, None, None]+ego_obs["yaw"][i:i+1])*agents_hist_avail.unsqueeze(-1)
            agents_hist_speeds = torch.cat((ego_obs["history_speeds"][i:i+1],agent_obs["history_speeds"][agent_idx]),0)*agents_hist_avail

            if 'closest_lane_point' in ego_obs['extras']:
                closest_lane_point = torch.cat((ego_obs['extras']['closest_lane_point'][i:i+1], agent_obs['extras']['closest_lane_point'][agent_idx]),0)
                M, S_seg, S_p, _ = closest_lane_point.shape
                # [M, S_seg, S_p, 3] -> [M, S_seg*S_p, 3]
                closest_lane_point = closest_lane_point.reshape(M, S_seg*S_p, -1)
                closest_lane_point_pos = closest_lane_point[..., :2] # [M, S_seg*S_p, 2]
                closest_lane_point_yaw = closest_lane_point[..., [2]] # [M, S_seg*S_p, 1]

                closest_lane_point_pos_transformed = GeoUtils.transform_points_tensor(closest_lane_point_pos,agents_from_center)
                closest_lane_point_yaw_transformed = angle_wrap(closest_lane_point_yaw-ego_and_agent_yaw[:, None, None]+ego_obs["yaw"][i:i+1])

                closest_lanes_points = torch.cat((closest_lane_point_pos_transformed, closest_lane_point_yaw_transformed), -1).reshape(M, S_seg, S_p, -1)
                
                closest_lanes_points_b.append(closest_lanes_points)

            
            hist_pos_b.append(agents_hist_pos)
            hist_yaw_b.append(agents_hist_yaw)
            hist_speeds_b.append(agents_hist_speeds)
            hist_avail_b.append(agents_hist_avail)
            if agent_obs["target_availabilities"].shape[1]<ego_obs["target_availabilities"].shape[1]:
                pad_shape=(agent_obs["target_availabilities"].shape[0],ego_obs["target_availabilities"].shape[1]-agent_obs["target_availabilities"].shape[1])
                agent_obs["target_availabilities"] = torch.cat((agent_obs["target_availabilities"],torch.zeros(pad_shape,device=agent_obs["target_availabilities"].device)),1)
            agents_fut_avail = torch.cat((ego_obs["target_availabilities"][i:i+1],agent_obs["target_availabilities"][agent_idx]),0)
            if agent_obs["target_positions"].shape[1]<ego_obs["target_positions"].shape[1]:
                pad_shape=(agent_obs["target_positions"].shape[0],ego_obs["target_positions"].shape[1]-agent_obs["target_positions"].shape[1],*agent_obs["target_positions"].shape[2:])
                agent_obs["target_positions"] = torch.cat((agent_obs["target_positions"],torch.zeros(pad_shape,device=agent_obs["target_positions"].device)),1)
                pad_shape=(agent_obs["target_yaws"].shape[0],ego_obs["target_yaws"].shape[1]-agent_obs["target_yaws"].shape[1],*agent_obs["target_yaws"].shape[2:])
                agent_obs["target_yaws"] = torch.cat((agent_obs["target_yaws"],torch.zeros(pad_shape,device=agent_obs["target_yaws"].device)),1)
            fut_pos_raw = torch.cat((ego_obs["target_positions"][i:i+1],agent_obs["target_positions"][agent_idx]),0)
            fut_yaw_raw = torch.cat((ego_obs["target_yaws"][i:i+1],agent_obs["target_yaws"][agent_idx]),0)
            agents_fut_pos = GeoUtils.transform_points_tensor(fut_pos_raw,agents_from_center)*agents_fut_avail.unsqueeze(-1)
            agents_fut_yaw = (fut_yaw_raw-torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0)[:,None,None]+ego_obs["yaw"][i])*agents_fut_avail.unsqueeze(-1)
            fut_pos_b.append(agents_fut_pos)
            fut_yaw_b.append(agents_fut_yaw)
            fut_avail_b.append(agents_fut_avail)

            curr_yaw = agents_hist_yaw[:,-1]
            curr_pos = agents_hist_pos[:,-1]
            agents_from_center = GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros_like(curr_pos))@GeoUtils.transform_matrices(torch.zeros_like(curr_yaw).flatten(),-curr_pos)
                                 
            maps = torch.cat((ego_obs["image"][i:i+1],agent_obs["image"][agent_idx]),0)
            curr_speed = torch.cat((ego_obs["curr_speed"][i:i+1],agent_obs["curr_speed"][agent_idx]),0)
            agents_type = torch.cat((ego_obs["type"][i:i+1],agent_obs["type"][agent_idx]),0)
            agents_extent = torch.cat((ego_obs["extent"][i:i+1],agent_obs["extent"][agent_idx]),0)
            maps_b.append(maps)
            curr_speed_b.append(curr_speed)
            type_b.append(agents_type)
            extents_b.append(agents_extent)

            # neighbor
            num_agents_i = agent_idx.shape[0]+1
            ego_and_agent_idx = torch.cat((torch.tensor([0], device=agent_idx.device), agent_idx+1), 0)
            neighbor_indices = torch.cat((ego_obs["neighbor_indices"][i:i+1], agent_obs["neighbor_indices"][agent_idx]), 0)
            agent_from_world = torch.cat([ego_obs["agent_from_world"][i].unsqueeze(0), agent_obs["agent_from_world"][agent_idx]], 0)
            world_from_agent = torch.cat([ego_obs["world_from_agent"][i].unsqueeze(0), agent_obs["world_from_agent"][agent_idx]], 0)
            
            hist_neigh_pos_b_sub = []
            hist_neigh_yaw_b_sub = []
            hist_neigh_speeds_b_sub = []
            hist_neigh_avail_b_sub = []
            neigh_curr_speed_b_sub = []
            fut_neigh_pos_b_sub = []
            fut_neigh_yaw_b_sub = []
            fut_neigh_avail_b_sub = []
            neigh_type_b_sub = []
            neigh_extents_b_sub = []

            for ci in ego_and_agent_idx:
                neigh_inds = neighbor_indices[ci]
                chosen_neigh_inds = neigh_inds[neigh_inds>=0].long()
                # print('chosen_neigh_inds', chosen_neigh_inds)

                index_agent = lambda x: x[ci] if x is not None else None
                index_neighbors = lambda x: x[chosen_neigh_inds]
                center_from_world = index_agent(agent_from_world)
                world_from_neigh = index_neighbors(world_from_agent)

                # (Q. 3. 3)
                center_from_neigh = center_from_world.unsqueeze(0) @ world_from_neigh

                hist_neigh_pos_b_sub_ci = index_neighbors(agents_hist_pos)
                hist_neigh_yaw_b_sub_ci = index_neighbors(agents_hist_yaw)
                hist_neigh_speeds_b_sub_ci = index_neighbors(agents_hist_speeds)
                hist_neigh_avail_b_sub_ci = index_neighbors(agents_hist_avail)
                neigh_curr_speed_b_sub_ci = index_neighbors(curr_speed)
                fut_neigh_pos_b_sub_ci = index_neighbors(agents_fut_pos)
                fut_neigh_yaw_b_sub_ci = index_neighbors(agents_fut_yaw)
                fut_neigh_avail_b_sub_ci = index_neighbors(agents_fut_avail)
                neigh_type_b_sub_ci = index_neighbors(agents_type)
                neigh_extents_b_sub_ci = index_neighbors(agents_extent)



                hist_neigh_pos_b_sub_ci = GeoUtils.transform_points_tensor(hist_neigh_pos_b_sub_ci,center_from_neigh)*hist_neigh_avail_b_sub_ci.unsqueeze(-1)
                
                ego_yaw_ci, agent_yaw_ci = index_agent(ego_and_agent_yaw), index_neighbors(ego_and_agent_yaw)
                
                hist_neigh_yaw_b_sub_ci = (hist_neigh_yaw_b_sub_ci+agent_yaw_ci[:, None, None]-ego_yaw_ci)*hist_neigh_avail_b_sub_ci.unsqueeze(-1)
                hist_neigh_speeds_b_sub_ci = hist_neigh_speeds_b_sub_ci*hist_neigh_avail_b_sub_ci
                neigh_curr_speed_b_sub_ci = neigh_curr_speed_b_sub_ci
                
                fut_neigh_pos_b_sub_ci = GeoUtils.transform_points_tensor(fut_neigh_pos_b_sub_ci,center_from_neigh)*fut_neigh_avail_b_sub_ci.unsqueeze(-1)
                fut_neigh_yaw_b_sub_ci = (fut_neigh_yaw_b_sub_ci+agent_yaw_ci[:, None, None]-ego_yaw_ci)*fut_neigh_avail_b_sub_ci.unsqueeze(-1)
                neigh_type_b_sub_ci = neigh_type_b_sub_ci
                neigh_extents_b_sub_ci = neigh_extents_b_sub_ci

                hist_neigh_pos_b_sub.append(hist_neigh_pos_b_sub_ci)
                hist_neigh_yaw_b_sub.append(hist_neigh_yaw_b_sub_ci)
                hist_neigh_speeds_b_sub.append(hist_neigh_speeds_b_sub_ci)
                hist_neigh_avail_b_sub.append(hist_neigh_avail_b_sub_ci)
                neigh_curr_speed_b_sub.append(neigh_curr_speed_b_sub_ci)
                fut_neigh_pos_b_sub.append(fut_neigh_pos_b_sub_ci)
                fut_neigh_yaw_b_sub.append(fut_neigh_yaw_b_sub_ci)
                fut_neigh_avail_b_sub.append(fut_neigh_avail_b_sub_ci)
                neigh_type_b_sub.append(neigh_type_b_sub_ci)
                neigh_extents_b_sub.append(neigh_extents_b_sub_ci)

            neigh_indices_b.append(neighbor_indices)
            hist_neigh_pos_b.append(pad_sequence(hist_neigh_pos_b_sub, batch_first=True, padding_value=np.nan))
            hist_neigh_yaw_b.append(pad_sequence(hist_neigh_yaw_b_sub, batch_first=True, padding_value=np.nan))
            hist_neigh_speeds_b.append(pad_sequence(hist_neigh_speeds_b_sub, batch_first=True, padding_value=np.nan))
            hist_neigh_avail_b.append(pad_sequence(hist_neigh_avail_b_sub, batch_first=True, padding_value=0))
            neigh_curr_speed_b.append(pad_sequence(neigh_curr_speed_b_sub, batch_first=True, padding_value=np.nan))
            fut_neigh_pos_b.append(pad_sequence(fut_neigh_pos_b_sub, batch_first=True, padding_value=np.nan))
            fut_neigh_yaw_b.append(pad_sequence(fut_neigh_yaw_b_sub, batch_first=True, padding_value=np.nan))
            fut_neigh_avail_b.append(pad_sequence(fut_neigh_avail_b_sub, batch_first=True, padding_value=0))
            neigh_type_b.append(pad_sequence(neigh_type_b_sub, batch_first=True, padding_value=0))
            neigh_extents_b.append(pad_sequence(neigh_extents_b_sub, batch_first=True, padding_value=np.nan))
            # # # #

        extras['closest_lane_point'] = torch.cat(closest_lanes_points_b,0)
        agent_obs = dict(
            image=torch.cat(maps_b,0),
            target_positions=torch.cat(fut_pos_b,0),
            target_yaws=angle_wrap(torch.cat(fut_yaw_b,0)),
            target_availabilities=torch.cat(fut_avail_b,0),
            history_positions=torch.cat(hist_pos_b,0),
            history_yaws=angle_wrap(torch.cat(hist_yaw_b,0)),
            history_speeds=torch.cat(hist_speeds_b,0),
            history_availabilities=torch.cat(hist_avail_b,0),
            curr_speed=torch.cat(curr_speed_b,0),
            centroid=torch.cat([ego_obs["centroid"], agent_obs["centroid"]], dim=0),
            yaw=angle_wrap(torch.cat([ego_obs["yaw"], agent_obs["yaw"]], dim=0),
            type=torch.cat(type_b,0)),
            extent=torch.cat(extents_b,0),
            raster_from_agent=torch.cat([ego_obs["raster_from_agent"], agent_obs["raster_from_agent"]], dim=0),
            agent_from_raster=torch.cat([ego_obs["agent_from_raster"], agent_obs["agent_from_raster"]], dim=0),
            agent_from_world=torch.cat([ego_obs["agent_from_world"], agent_obs["agent_from_world"]], dim=0),
            world_from_agent=torch.cat([ego_obs["world_from_agent"], agent_obs["world_from_agent"]], dim=0),
            extras=extras,

            # [M, Q, T, 2]
            all_other_agents_indices=torch.cat(neigh_indices_b,0),
            all_other_agents_history_positions=torch.cat(hist_neigh_pos_b,0),
            # [M, Q, T, 1]
            all_other_agents_history_yaws=angle_wrap(torch.cat(hist_neigh_yaw_b,0)),
            # [M, Q, T]
            all_other_agents_history_speeds=torch.cat(hist_neigh_speeds_b,0),
            # [M, Q, T]
            all_other_agents_history_availabilities=torch.cat(hist_neigh_avail_b,0),
            all_other_agents_history_availability=torch.cat(hist_neigh_avail_b,0),
            
            # [M, Q]
            all_other_agents_curr_speed=torch.cat(neigh_curr_speed_b,0),
            
            # [M, Q, T, 2]
            all_other_agents_future_positions=torch.cat(fut_neigh_pos_b,0),
            # [M, Q, T]
            all_other_agents_future_yaws=angle_wrap(torch.cat(fut_neigh_yaw_b,0)),
            # [M, Q, T]
            all_other_agents_future_availability=torch.cat(fut_neigh_avail_b,0),
            
            # [M, Q]
            all_other_agents_types=torch.cat(neigh_type_b,0),
            # [M, Q, E]
            all_other_agents_extents=torch.cat(neigh_extents_b,0),
            # [M]
            scene_index=torch.cat([ego_obs["scene_index"], agent_obs["scene_index"]],0),
        )

        return agent_obs
    
    @staticmethod
    def convert_agent_centric_action_to_scene_centric(action, info, ego_obs, agent_obs):
        # TBD: assume one scene
        pos = action.positions # (M, T, 2)
        yaw = action.yaws # (M, T, 1)
        pos_info = info['action_samples']['positions'].permute(1, 0, 2, 3) # (N, M, T, 2)
        yaw_info = info['action_samples']['yaws'].permute(1, 0, 2, 3) # (N, M, T, 1)
        N, M, T, _ = pos_info.shape

        # (1, 3, 3) @ (M-1, 3, 3) -> (M-1, 3, 3)
        center_from_agent = ego_obs["agent_from_world"] @ agent_obs["world_from_agent"]
        # (M, 3, 3)
        center_from_agents = torch.cat((torch.eye(3,device=center_from_agent.device).unsqueeze(0),center_from_agent),0) 

        ego_and_agent_yaw = torch.cat((ego_obs["yaw"],agent_obs["yaw"]),0)
        
        pos_agent = GeoUtils.transform_points_tensor(pos,center_from_agents)
        yaw_agent = yaw+ego_and_agent_yaw[:,None,None]-ego_obs["yaw"]
        
        center_from_agents_N = center_from_agents.unsqueeze(0).repeat(N, 1, 1, 1).reshape(N*M, 3, 3)
        pos_info_agent = GeoUtils.transform_points_tensor(pos_info.reshape(N*M, T, 2),center_from_agents_N)
        yaw_info_agent = yaw_info.reshape(N*M, T, 1)+ego_and_agent_yaw[:,None,None].unsqueeze(0).repeat(N, 1, 1, 1).reshape(N*M, 1, 1)-ego_obs["yaw"]
        
        pos_info_agent = pos_info_agent.reshape(N, M, T, 2)
        yaw_info_agent = yaw_info_agent.reshape(N, M, T, 1)

        actions = Action(
            positions=pos_agent.unsqueeze(0), # (M, T, 2) -> (B, M, T, 2)
            yaws=yaw_agent.unsqueeze(0), # (M, T, 2) -> (B, M, T, 2)
        )
        infos = dict(
            action_samples=Action(
                positions=pos_info_agent.unsqueeze(0), # (N, M, T, 2) -> (B, N, M, T, 2)
                yaws=yaw_info_agent.unsqueeze(0), # (N, M, T, 1) -> (B, N, M, T, 1)
            ).to_dict(),
        )
        
        return actions, infos
        


class NaiveAgentCentricToSceneCentricWrapper(object):
    '''
    Use agent-centric coordinates for scene-centric model
    Used to debug scene-centric model with agent-centric data

    Note: only support batch size 1 for now.

    1.takes in agent-centric data and convert it to scene-centric data without coordinate transformation
    2.apply policy get_action
    3.convert scene-centric actions back to agent-centric actions without coordinate transformation and return
    '''
    def __init__(self, policy):
        self.device = policy.device
        self.policy = policy
        self.model = policy.model

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        scene_obs = self.convert_agent_centric_obs_to_scene_centric(obs)

        actions, action_info = self.policy.get_action(scene_obs, **kwargs)

        # convert scene-centric actions to agent-centric actions
        actions, action_info = self.convert_scene_centric_action_to_agent_centric(actions, action_info)
        
        return actions, action_info

    @staticmethod
    def convert_agent_centric_obs_to_scene_centric(agent_centric_obs):
        scene_obs = add_scene_dim_to_agent_data(agent_centric_obs)
        scene_obs["num_agents"] = torch.tensor([agent_centric_obs['target_positions'].shape[0]],device=agent_centric_obs['target_positions'].device)

        return scene_obs
    
    @staticmethod
    def convert_scene_centric_action_to_agent_centric(action, info):

        pos_all = action.positions # (B, M, T, 2)
        yaw_all = action.yaws # (B, M, T, 1)
        pos_info_all = info['action_samples']['positions'] # (B, N, M, T, 2)
        yaw_info_all = info['action_samples']['yaws'] # (B, N, M, T, 1)

        actions = Action(
            positions=pos_all[0],
            yaws=yaw_all[0],
        )
        infos = dict(
            action_samples=Action(
                positions=pos_info_all[0].permute(1, 0, 2, 3), # (N, M, T, 2) -> (M, N, T, 2)
                yaws=yaw_info_all[0].permute(1, 0, 2, 3), # (N, M, T, 1) -> (M, N, T, 1)
            ).to_dict(),
            attn_weights=info['attn_weights'],
        )
        
        return actions, infos
