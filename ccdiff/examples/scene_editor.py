#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

"""A script for evaluating closed-loop simulation"""
import argparse
from symbol import star_expr
import numpy as np
import json
import random
import yaml
import importlib
from collections import Counter
from pprint import pprint
import joblib

import os
import subprocess
import time
import torch

from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.utils.scene_edit_utils import guided_rollout, compute_heuristic_guidance, merge_guidance_configs
from tbsim.evaluation.env_builders import EnvUnifiedBuilder
from tbsim.utils.guidance_loss import verify_guidance_config_list

from tbsim.policies.wrappers import (
    GenerativeEgoReplayRolloutWrapper,
    RolloutWrapper,
    PlannerOverrideRolloutWrapper,
    Pos2YawWrapper,
)
from ccdiff.planners import LoggedReplayTrafficPolicy, build_ego_planner

from tbsim.utils.tensor_utils import map_ndarray
import tbsim.utils.tensor_utils as TensorUtils


def attest_expected_cuda_binding():
    """Fail closed when an evaluation process lands on the wrong physical GPU.

    Older CUDA/PyTorch builds on this host do not accept GPU UUIDs in
    CUDA_VISIBLE_DEVICES, so the launcher supplies a numeric mask plus the
    expected physical UUID.  After creating the CUDA context, this process
    asks nvidia-smi which physical UUID owns its PID and compares it exactly.
    """
    expected_uuid = os.environ.get("CRASHDOJO_EXPECTED_GPU_UUID")
    if expected_uuid is None:
        return None
    expected_uuid = expected_uuid.strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not expected_uuid.startswith("GPU-"):
        raise RuntimeError("CRASHDOJO_EXPECTED_GPU_UUID must be a physical GPU UUID")
    if visible is None or "," in visible:
        raise RuntimeError(
            "GPU-bound evaluation requires exactly one CUDA_VISIBLE_DEVICES entry"
        )
    if cublas_workspace != ":4096:8":
        raise RuntimeError(
            "GPU-bound deterministic evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU-bound evaluation requires exactly one visible CUDA device")

    # The CUDA process PID reported by nvidia-smi belongs to the host PID
    # namespace and can differ from os.getpid() inside a container.  Verify the
    # numeric CUDA mask against nvidia-smi's physical index-to-UUID map instead.
    # This establishes the physical-device binding without relying on PID
    # namespace translation.
    try:
        visible_index = int(visible)
    except ValueError as exc:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be a numeric physical GPU index on this host"
        ) from exc
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    physical_gpu_map = {}
    for row in rows.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 2:
            physical_gpu_map[int(fields[0])] = fields[1]
    observed_uuid = physical_gpu_map.get(visible_index)
    if observed_uuid is None:
        raise RuntimeError(
            "nvidia-smi did not report CUDA_VISIBLE_DEVICES index {}".format(
                visible_index
            )
        )
    if observed_uuid != expected_uuid:
        raise RuntimeError(
            "physical GPU binding mismatch: expected {} but observed {}".format(
                expected_uuid, observed_uuid
            )
        )
    attestation = {
        "pid": os.getpid(),
        "cuda_visible_devices": visible,
        "expected_gpu_uuid": expected_uuid,
        "observed_gpu_uuid": observed_uuid,
        "logical_device_count": torch.cuda.device_count(),
        "logical_device_name": torch.cuda.get_device_name(0),
        "cublas_workspace_config": cublas_workspace,
    }
    print("GPU_BINDING_ATTESTATION " + json.dumps(attestation, sort_keys=True), flush=True)
    return attestation

def read_txt(file_path): 
    # Initialize an empty list to store the data
    data = []

    # Open the file and read line by line
    with open(file_path, 'r') as f:
        for line in f:
            # Strip any extra whitespace and split the line into individual numbers
            row = list(map(int, line.strip().split()))
            # Append the row to the data list
            data.append(row)
    return data


def build_safesim_target_guidance(
    example_batch,
    target_agent_idx=None,
    reference_agent_idx=0,
    severity_mode="collision",
    desired_clearance=0.0,
    cause_weight=1.0,
    collateral_weight=1.0,
    map_weight=1.0,
):
    """Build an explicit SAFE-SIM target-pair baseline for one scene."""
    scene_index = example_batch["scene_index"]
    if torch.is_tensor(scene_index):
        scene_index = scene_index.detach().cpu().numpy()
    scene_index = np.asarray(scene_index).reshape(-1)
    unique_scenes = np.unique(scene_index)
    if len(unique_scenes) != 1:
        raise ValueError("SAFE-SIM baseline requires one scene per batch")
    scene_agents = np.flatnonzero(scene_index == unique_scenes[0])
    num_agents = len(scene_agents)
    ref_ind = int(reference_agent_idx)
    if not 0 <= ref_ind < num_agents:
        raise IndexError("reference_agent_idx exceeds scene agent count")

    if target_agent_idx is None:
        transforms = example_batch["world_from_agent"]
        speeds = example_batch["curr_speed"]
        if torch.is_tensor(transforms):
            transforms = transforms.detach().cpu().numpy()
        if torch.is_tensor(speeds):
            speeds = speeds.detach().cpu().numpy()
        positions = np.asarray(transforms)[scene_agents, :2, 2]
        speeds = np.asarray(speeds).reshape(-1)[scene_agents]
        distances = np.linalg.norm(positions - positions[ref_ind], axis=-1)
        valid = np.isfinite(distances) & np.isfinite(speeds)
        valid &= np.abs(speeds) > 0.5
        valid &= distances > 2.0
        valid[ref_ind] = False
        if not np.any(valid):
            raise ValueError("no moving non-ego target is available")
        target_ind = int(np.argmin(np.where(valid, distances, np.inf)))
        selection_source = "nearest_moving_smoke_fallback"
    else:
        target_ind = int(target_agent_idx)
        selection_source = "explicit"
    if not 0 <= target_ind < num_agents or target_ind == ref_ind:
        raise IndexError("target_agent_idx must be a non-reference agent")
    print(
        "SAFESIM_GUIDANCE_CONTRACT target_ind={} ref_ind={} severity={} "
        "desired_clearance={} selection_source={}".format(
            target_ind, ref_ind, severity_mode, desired_clearance,
            selection_source,
        )
    )
    return [[
        {
            "name": "safesim_target_pair",
            "weight": 1.0,
            "params": {
                "target_ind": target_ind,
                "ref_ind": ref_ind,
                "loss_timesteps": 20,
                "sigma": 1.0,
                "heading_weight": 1.0,
                "buff_dist": 1.5,
                "cause_weight": float(cause_weight),
                "collateral_weight": float(collateral_weight),
                "desired_clearance": float(desired_clearance),
                "severity_mode": severity_mode,
                "softmin_temperature": 0.5,
                "collateral_margin": 1.0,
                "collateral_barrier_weight": 5.0,
            },
            "agents": [target_ind],
        },
        {
            "name": "map_collision",
            "weight": float(map_weight),
            "params": {"num_points_lw": (10, 10), "decay_rate": 0.9},
            "agents": [target_ind],
        },
    ]]

def run_scene_editor(eval_cfg, save_cfg, data_to_disk, render_to_video, render_to_img, render_cfg, part_control=False, controllable_agent=-1,
                    human=False, rand_agent=False, num_steps_intervention=0,
                    ego_planner_name=None, scene_source="m2_unguided",
                    strategy_depth=0, strive_profile="final_tuned_val_1",
                    guidance_method="ccdiff", target_agent_idx=None,
                    selection_manifest=None, selection_row=None, five_class_request=None,
                    reference_agent_idx=0, severity_mode="collision",
                    desired_clearance=0.0, safesim_cause_weight=1.0,
                    safesim_collateral_weight=1.0,
                    planner_hcs_start_step=20,
                    planner_hcs_end_step=None,
                    planner_hcs_arrival_weight=0.5,
                    planner_hcs_velocity_trust_max_rms=0.05,
                    planner_hcs_velocity_trust_max_linf=0.20,
                    planner_hcs_semantic_control="actual",
                    planner_hcs_semantic_plan_json=None,
                    planner_hcs_coalition_min_blockers=1,
                    planner_hcs_coalition_max_blockers=2,
                    planner_hcs_coalition_corridor_offset_m=3.2,
                    planner_hcs_coalition_max_reach_error_m=12.0,
                    typed_collision_type=None,
                    typed_projection_activation_distance_m=25.0,
                    typed_projection_step_size=0.35,
                    typed_projection_max_delta=0.75,
                    typed_projection_temporal_only=False,
                    typed_projection_native_joint_selection=False,
                    typed_projection_final_step_only=False,
                    typed_projection_min_collision_step=5,
                    typed_projection_max_collision_step=15,
                    typed_projection_phase_shift_steps=0,
                    typed_projection_joint_continuity=False,
                    typed_projection_joint_continuity_risk_slack=1.0,
                    typed_projection_joint_continuity_max_error_m=2.0,
                    typed_projection_trust_region=False,
                    typed_projection_trust_max_rms=0.20,
                    typed_projection_trust_max_linf=0.75,
                    typed_projection_trust_max_energy_increase=0.05,
                    typed_projection_trust_min_improvement=0.001,
                    typed_projection_trust_tangent=False,
                    typed_projection_trust_tangent_rank=4,
                    typed_projection_latent_shooting=False,
                    typed_projection_latent_iterations=3,
                    typed_projection_latent_lr=0.1,
                    typed_projection_latent_max_delta=0.35,
                    typed_projection_latent_prior_weight=0.05,
                    typed_projection_latent_bank=False,
                    typed_projection_latent_bank_max_delta=0.25,
                    typed_projection_latent_bank_prior_weight=0.10,
                    typed_projection_latent_bank_min_improvement=0.01,
                    typed_projection_latent_bank_max_nll_increase=0.02,
                    interaction_intervention="none", edge_receiver_idx=None,
                    edge_sender_idx=None, edge_cut_pairs=None,
                    edge_bidirectional=False, edge_cut_start_step=0,
                    edge_cut_end_step=None,
                    semantic_assertive_receiver_idx=None,
                    semantic_assertive_token_scale=1.0,
                    semantic_assertive_start_step=0,
                    semantic_assertive_end_step=None,
                    semantic_assertive_overlay=False,
                    full_visibility_cut_start_step=0,
                    full_visibility_cut_end_step=None, edge_delay_pairs=None,
                    edge_delay_steps=None, edge_delay_start_step=0,
                    edge_delay_end_step=None, future_intent_hold_pairs=None,
                    future_intent_hold_start_step=0,
                    future_intent_hold_end_step=None,
                    yield_belief_pairs=None, yield_belief_start_step=0,
                    yield_belief_end_step=None, reaction_receiver_idx=None,
                    reaction_delay_steps=None, reaction_delay_start_step=0,
                    reaction_delay_end_step=None, reaction_accel_bias_mps2=0.0,
                    native_joint_sample_idx=None,
                    native_joint_start_step=0,
                    native_joint_end_step=None,
                    native_joint_num_samples=20,
                    marginal_tail_receiver_idx=None,
                    marginal_tail_receiver_indices=None,
                    marginal_tail_auto_receivers=False,
                    marginal_tail_sticky_receiver=False,
                    marginal_tail_minimize_causal_change=False,
                    marginal_tail_start_step=0,
                    marginal_tail_end_step=None,
                    marginal_tail_collateral_margin_m=0.25,
                    marginal_tail_reference_safety_margin_m=0.25,
                    marginal_tail_min_improvement_m=0.05,
                    marginal_tail_target_max_clearance_m=None,
                    marginal_tail_progressive_target_barrier=False,
                    marginal_tail_progressive_activation_max_clearance_m=8.0,
                    marginal_tail_progressive_risk_first=False,
                    marginal_tail_progressive_persistence_first=False,
                    marginal_tail_progressive_dose_first=False,
                    marginal_tail_progressive_exposure_clearance_m=2.0,
                    marginal_tail_baseline_min_clearance_m=None,
                    marginal_tail_target_min_persistence_steps=1,
                    marginal_tail_max_speed_mps=20.0,
                    marginal_tail_max_acc_mps2=4.0,
                    marginal_tail_max_jerk_mps3=8.37,
                    marginal_tail_max_yaw_rate_rps=0.95,
                    marginal_tail_max_lateral_acc_mps2=4.0,
                    marginal_tail_kinematic_horizon_steps=5,
                    marginal_tail_max_agent_tail_fraction=0.01,
                    marginal_tail_naturalness_profile_path=None,
                    marginal_tail_target_planner_name=None,
                    marginal_tail_attack_planner_name=None,
                    deterministic_pairing=False,
                    generative_ego_only=False,
                    frozen_traffic_h5=None,
                    frozen_traffic_frame_offset=0,
                    correlated_flow_prior_pairs=None,
                    correlated_flow_prior_rho=None,
                    correlated_flow_prior_lag_steps=0,
                    correlated_flow_prior_action_dims=None,
                    correlated_flow_prior_start_step=0,
                    correlated_flow_prior_end_step=None,
                    persistent_copula_persistence=None,
                    persistent_copula_temporal_rho=0.0,
                    persistent_copula_temporal_action_dims=None,
                    pcn_joint_flow_prior_beta=None,
                    pcn_joint_flow_prior_start_step=0,
                    pcn_joint_flow_prior_end_step=None,
                    conditional_flow_num_steps=1,
                    conditional_flow_source_disabled=False,
                    conditional_flow_target_mean_shift=None,
                    conditional_flow_target_mean_shift_max_abs=0.25,
                    conditional_flow_target_mean_shift_max_kl=0.02,
                    conditional_flow_target_mean_shift_basis="constant_ramp",
                    conditional_flow_target_mean_shift_pulse_width=0.15,
                    conditional_flow_target_mean_shift_online_cpa=False,
                    conditional_flow_target_mean_shift_max_cpa_distance_m=30.0,
                    conditional_flow_target_velocity_residual=False,
                    conditional_flow_target_velocity_residual_arrival_aligned=False,
                    conditional_flow_target_velocity_residual_arrival_max_lateral_distance_m=4.0,
                    conditional_flow_normalize_gradient=False,
                    conditional_flow_step_size=0.05,
                    conditional_flow_max_source_delta_rms=0.05,
                    conditional_flow_max_actor_delta_rms=0.05,
                    conditional_flow_max_energy_increase=0.01,
                    conditional_flow_persistence_steps=3,
                    conditional_flow_temperature_m=0.25,
                    conditional_flow_reference_weight=4.0,
                    conditional_flow_collateral_weight=4.0,
                    conditional_flow_source_proximity_weight=0.25,
                    conditional_flow_optimization_scope="all_background",
                    conditional_flow_optimization_start_fraction=0.0,
                    conditional_flow_global_start_step=0,
                    conditional_flow_global_end_step=None,
                    conditional_flow_global_conflict_step=None,
                    conditional_flow_conflict_anchor_radius_steps=2):
    assert eval_cfg.env in ["nusc", "trajdata"], "Currently only nusc and trajdata environments are supported"
        
    set_global_batch_type("trajdata")
    if eval_cfg.env == "nusc":
        set_global_trajdata_batch_env("nusc_trainval")
    elif eval_cfg.env == "trajdata":
        # assumes all used trajdata datasets use share same map layers
        set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])

    # print(eval_cfg)

    selection_record = None
    if selection_manifest is not None:
        from ccdiff.evaluation.selection_manifest import load_selection, resolve_scene, SelectionBinding
        selection_record = load_selection(selection_manifest, selection_row)
        if eval_cfg.eval_class != "CCFlowMatching" or eval_cfg.env != "trajdata":
            raise ValueError("Website selection currently requires CCFlowMatching + trajdata")
        if not part_control or (ego_planner_name is not None and five_class_request is None) or generative_ego_only:
            raise ValueError("Website selection requires partial control without ego planner override")
        if human or rand_agent or num_steps_intervention != 0:
            raise ValueError("Website selection cannot use legacy/random selection or delayed intervention")
        if eval_cfg.apply_guidance or list(eval_cfg.edits.editing_source) != ["none"]:
            raise ValueError("Selection-only integration requires --disable_guidance --editing_source none; custom conflict functions are not installed")
        if interaction_intervention != "none" or typed_collision_type is not None or guidance_method != "ccdiff":
            raise ValueError("Selection-only integration cannot enable other interaction/guidance modes")
        if scene_source != "m2_unguided":
            raise ValueError("Selection-only integration requires --scene_source m2_unguided")

    if five_class_request is not None:
        if selection_record is None or ego_planner_name is None:
            raise ValueError('five_class_request requires selection_manifest and ego_planner')
        if strategy_depth != 0:
            raise ValueError('Five-class adapter supports frozen per-replan ego plan only')
        from ccdiff.evaluation.five_class_guidance import load_request
        five_class_request = load_request(five_class_request, selection_record)

    # for reproducibility
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    if deterministic_pairing:
        torch.cuda.manual_seed_all(eval_cfg.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("DETERMINISTIC_PAIRING_ATTESTATION enabled=True warn_only_for_missing_kernels=True")
    # basic setup
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)

    if render_to_video or render_to_img:
        os.makedirs(os.path.join(eval_cfg.results_dir, "viz/"), exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))
    if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
        os.remove(eval_cfg.experience_hdf5_path)
        pass
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    # The upstream CCDiff composer reconstructs the default DDPM policy class.
    # M2 therefore needs an explicit composer that injects CCFlowMatching.
    if eval_cfg.eval_class == "CCFlowMatching":
        from ccdiff.evaluation.flow_matching_policy_composer import (
            CCFlowMatchingPolicyComposer,
        )
        composer_class = CCFlowMatchingPolicyComposer
    else:
        composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device)
    policy, exp_config = composer.get_policy()
    ego_planner = None
    marginal_tail_target_planner = None
    marginal_tail_attack_planner = None
    agents_policy = policy
    if ego_planner_name is not None:
        if part_control and five_class_request is None:
            raise ValueError("ego_planner and part_control cannot be enabled together")
        if five_class_request is not None:
            from ccdiff.evaluation.five_class_planners import build_selected_ego
            os.environ["RULE_OUTPUT"] = os.path.join(eval_cfg.results_dir, "rule_adapter")
            planner_factory = build_selected_ego
        else:
            planner_factory = build_ego_planner
        ego_planner = planner_factory(
            ego_planner_name,
            device=device,
            strive_profile=strive_profile,
        )
        if scene_source == "logged_replay":
            agents_policy = LoggedReplayTrafficPolicy(device=device)
        elif scene_source == "frozen_hdf5":
            if frozen_traffic_h5 is None:
                raise ValueError("frozen_hdf5 requires --frozen_traffic_h5")
            from ccdiff.planners.frozen_hdf5_replay import (
                FrozenHDF5TrafficPolicy,
            )
            agents_policy = FrozenHDF5TrafficPolicy(
                frozen_traffic_h5,
                device=device,
                frame_offset=frozen_traffic_frame_offset,
            )
        elif scene_source not in ["m2_unguided", "m2_guided"]:
            raise ValueError("Unknown scene_source: {}".format(scene_source))
        print(
            "PLANNER_CONTRACT planner={} scene_source={} strategy_depth={} "
            "supports_candidate_future={}".format(
                ego_planner_name,
                scene_source,
                strategy_depth,
                ego_planner.supports_candidate_future,
            )
        )
    if marginal_tail_target_planner_name is not None:
        if ego_planner is None:
            raise ValueError(
                "marginal-tail target planner requires --ego_planner"
            )
        reactive_role_decoupling = (
            interaction_intervention == "planner_aligned_joint_risk"
            and marginal_tail_attack_planner_name is not None
        )
        if (
            marginal_tail_target_planner_name == ego_planner_name
            and not reactive_role_decoupling
        ):
            raise ValueError(
                "target and reference planners must be different"
            )
        marginal_tail_target_planner = build_ego_planner(
            marginal_tail_target_planner_name,
            device=device,
            strive_profile=strive_profile,
        )
        if not reactive_role_decoupling:
            print(
                "SELECTIVE_PLANNER_CONTRACT target={} reference={} "
                "objective=target_clearance constraint=reference_safety".format(
                    marginal_tail_target_planner_name,
                    ego_planner_name,
                )
            )
    if marginal_tail_attack_planner_name is not None:
        if (
            interaction_intervention != "planner_aligned_joint_risk"
            or ego_planner is None
            or marginal_tail_target_planner is None
        ):
            raise ValueError(
                "marginal-tail attack planner is restricted to planner-aligned "
                "joint risk with executed and safety planners"
            )
        if marginal_tail_attack_planner_name == marginal_tail_target_planner_name:
            raise ValueError("proposal attack and safety planners must differ")
        marginal_tail_attack_planner = build_ego_planner(
            marginal_tail_attack_planner_name,
            device=device,
            strive_profile=strive_profile,
        )
        print(
            "REACTIVE_DUAL_PLANNER_CONTRACT executed={} proposal_attack={} "
            "proposal_safety={} both_planners_executed_in_separate_rollouts=True".format(
                ego_planner_name,
                marginal_tail_attack_planner_name,
                marginal_tail_target_planner_name,
            )
        )
    if part_control and five_class_request is None: 
        composer_class = getattr(policy_composers, 'CCDiffHierarchicalPolicy')
        composer = composer_class(eval_cfg, device)
        control_policy, control_exp_config = composer.get_policy()
        if selection_record is None:
            exp_config = control_exp_config  # Preserve legacy behavior; selection uses FM timing/config.
        # control_policy.add_policy(policy)


    if five_class_request is not None:
        class FixedControlIdentity:
            def set_controllable_set(self, indices):
                self.controllable_set = list(indices)
        control_policy = FixedControlIdentity()

    # determines cfg for rasterizing agents
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    # print(exp_config.algo)
    # ----------------------------------------------------------------------------------
    policy_model = None
    print('policy', policy)
    if hasattr(policy, 'model'):
        policy_model = policy.model

    marginal_tail_config = None
    assertive_overlay_active = bool(semantic_assertive_overlay)
    if assertive_overlay_active and interaction_intervention != "planner_aligned_joint_risk":
        raise ValueError(
            "semantic_assertive_overlay is restricted to planner_aligned_joint_risk"
        )
    if interaction_intervention == "semantic_assertive_token" or assertive_overlay_active:
        if policy_model is None or eval_cfg.eval_class != "CCFlowMatching":
            raise ValueError(
                "semantic_assertive_token requires an M2 flow-matching policy"
            )
        if eval_cfg.apply_guidance:
            raise ValueError(
                "semantic_assertive_token is conditioning-only; disable guidance"
            )
        if semantic_assertive_receiver_idx is None:
            raise ValueError("semantic assertive receiver index is required")
        policy_model.set_semantic_assertive_receivers(
            [semantic_assertive_receiver_idx],
            scale=semantic_assertive_token_scale,
        )
        policy_model.set_semantic_assertive_receiver_schedule(
            start_step=semantic_assertive_start_step,
            end_step=semantic_assertive_end_step,
        )
        if assertive_overlay_active:
            print(
                "SEMANTIC_ASSERTIVE_OVERLAY_ATTESTATION receiver={} scale={} "
                "start={} end={} output_edit=False projection=False "
                "native_joint_selector=True".format(
                    semantic_assertive_receiver_idx,
                    semantic_assertive_token_scale,
                    semantic_assertive_start_step,
                    semantic_assertive_end_step,
                )
            )
    if interaction_intervention == "semantic_assertive_token":
        pass
    elif interaction_intervention in {"edge_cut", "full_visibility_cut"}:
        if policy_model is None:
            raise ValueError("visibility cut requires an M2 traffic policy")
        if eval_cfg.apply_guidance:
            raise ValueError("visibility cut is no-gradient; disable guidance")
        parsed_pairs = []
        if edge_cut_pairs:
            for item in edge_cut_pairs.split(","):
                receiver, sender = item.split(":", maxsplit=1)
                parsed_pairs.append((int(receiver), int(sender)))
        elif edge_receiver_idx is not None and edge_sender_idx is not None:
            parsed_pairs.append((edge_receiver_idx, edge_sender_idx))
        else:
            raise ValueError("visibility cut requires receiver/sender indices or --edge_cut_pairs")
        if interaction_intervention == "edge_cut":
            policy_model.set_interaction_edge_cuts(
                parsed_pairs, bidirectional=edge_bidirectional
            )
            policy_model.set_interaction_edge_cut_schedule(
                start_step=edge_cut_start_step, end_step=edge_cut_end_step
            )
        else:
            policy_model.set_interaction_full_visibility_cuts(
                parsed_pairs, bidirectional=edge_bidirectional
            )
            policy_model.set_interaction_full_visibility_cut_schedule(
                start_step=full_visibility_cut_start_step,
                end_step=full_visibility_cut_end_step,
            )
    elif interaction_intervention == "correlated_flow_prior":
        if policy_model is None or eval_cfg.eval_class != "CCFlowMatching":
            raise ValueError(
                "correlated_flow_prior requires an M2 flow-matching policy"
            )
        if eval_cfg.apply_guidance:
            raise ValueError(
                "correlated_flow_prior is a no-gradient prior intervention"
            )
        if correlated_flow_prior_pairs is None or correlated_flow_prior_rho is None:
            raise ValueError(
                "correlated_flow_prior requires --correlated_flow_prior_pairs "
                "and --correlated_flow_prior_rho"
            )
        parsed_pairs = []
        for item in correlated_flow_prior_pairs.split(","):
            left, right = item.split(":", maxsplit=1)
            parsed_pairs.append((int(left), int(right)))
        policy_model.set_marginal_preserving_correlated_flow_prior(
            parsed_pairs,
            correlated_flow_prior_rho,
            lag_steps=correlated_flow_prior_lag_steps,
            action_dims=(
                None
                if correlated_flow_prior_action_dims is None
                else [
                    int(token)
                    for token in correlated_flow_prior_action_dims.split(",")
                ]
            ),
        )
        policy_model.set_marginal_preserving_correlated_flow_prior_schedule(
            start_step=correlated_flow_prior_start_step,
            end_step=correlated_flow_prior_end_step,
        )
    elif interaction_intervention == "edge_delay":
        if policy_model is None:
            raise ValueError("edge_delay requires an M2 traffic policy")
        if eval_cfg.apply_guidance:
            raise ValueError("edge_delay is a no-gradient intervention; disable guidance")
        parsed_delays = []
        if edge_delay_pairs:
            for item in edge_delay_pairs.split(","):
                receiver, sender, delay = item.split(":", maxsplit=2)
                parsed_delays.append((int(receiver), int(sender), int(delay)))
        elif (
            edge_receiver_idx is not None
            and edge_sender_idx is not None
            and edge_delay_steps is not None
        ):
            parsed_delays.append(
                (edge_receiver_idx, edge_sender_idx, edge_delay_steps)
            )
        else:
            raise ValueError(
                "edge_delay requires receiver/sender/delay or --edge_delay_pairs"
            )
        policy_model.set_interaction_edge_delays(
            parsed_delays, bidirectional=edge_bidirectional
        )
        policy_model.set_interaction_edge_delay_schedule(
            start_step=edge_delay_start_step, end_step=edge_delay_end_step
        )
    elif interaction_intervention == "future_intent_hold":
        if policy_model is None or eval_cfg.eval_class != "CCFlowMatching":
            raise ValueError("future_intent_hold requires an M2 flow-matching policy")
        if eval_cfg.apply_guidance:
            raise ValueError(
                "future_intent_hold is an internal no-gradient intervention"
            )
        if future_intent_hold_pairs is None:
            raise ValueError(
                "future_intent_hold requires --future_intent_hold_pairs"
            )
        parsed_pairs = []
        for item in future_intent_hold_pairs.split(","):
            receiver, sender = item.split(":", maxsplit=1)
            parsed_pairs.append((int(receiver), int(sender)))
        policy_model.set_interaction_future_intent_holds(
            parsed_pairs, bidirectional=edge_bidirectional
        )
        policy_model.set_interaction_future_intent_hold_schedule(
            start_step=future_intent_hold_start_step,
            end_step=future_intent_hold_end_step,
        )
    elif interaction_intervention == "yield_belief":
        if policy_model is None or eval_cfg.eval_class != "CCFlowMatching":
            raise ValueError("yield_belief requires an M2 flow-matching policy")
        if eval_cfg.apply_guidance:
            raise ValueError("yield_belief is an internal no-gradient intervention")
        if yield_belief_pairs is None:
            raise ValueError("yield_belief requires --yield_belief_pairs")
        parsed_beliefs = []
        for item in yield_belief_pairs.split(","):
            receiver, sender, deceleration = item.split(":", maxsplit=2)
            parsed_beliefs.append(
                (int(receiver), int(sender), float(deceleration))
            )
        policy_model.set_interaction_yield_beliefs(
            parsed_beliefs, bidirectional=edge_bidirectional
        )
        policy_model.set_interaction_yield_belief_schedule(
            start_step=yield_belief_start_step,
            end_step=yield_belief_end_step,
        )
    elif interaction_intervention == "reaction_delay":
        if policy_model is None or eval_cfg.eval_class != "CCFlowMatching":
            raise ValueError("reaction_delay requires an M2 flow-matching policy")
        if eval_cfg.apply_guidance:
            raise ValueError(
                "reaction_delay is a semantic no-gradient intervention; disable guidance"
            )
        if reaction_receiver_idx is None or reaction_delay_steps is None:
            raise ValueError(
                "reaction_delay requires --reaction_receiver_idx and "
                "--reaction_delay_steps"
            )
        policy_model.set_semantic_reaction_delay(
            receiver_idx=reaction_receiver_idx,
            delay_steps=reaction_delay_steps,
            start_step=reaction_delay_start_step,
            end_step=reaction_delay_end_step,
            accel_bias_mps2=reaction_accel_bias_mps2,
        )
    elif interaction_intervention == "native_joint_mode":
        if policy_model is None or eval_cfg.eval_class != "CCFlowMatching":
            raise ValueError("native_joint_mode requires an M2 flow-matching policy")
        if eval_cfg.apply_guidance:
            raise ValueError(
                "native_joint_mode enumerates native samples; disable guidance"
            )
        if ego_planner is None:
            raise ValueError("native_joint_mode requires --ego_planner")
        if native_joint_sample_idx is None:
            raise ValueError("native_joint_mode requires --native_joint_sample_idx")
        if not 0 <= int(native_joint_sample_idx) < int(native_joint_num_samples):
            raise ValueError("native joint sample index must be within the K bank")
        if native_joint_start_step < 0:
            raise ValueError("native_joint_start_step must be non-negative")
        if (
            native_joint_end_step is not None
            and native_joint_end_step <= native_joint_start_step
        ):
            raise ValueError("native_joint_end_step must exceed start_step")
        marginal_tail_config = {
            "mode": "fixed_native_joint",
            "fixed_joint_sample_index": int(native_joint_sample_idx),
            "num_samples": int(native_joint_num_samples),
            "start_step": int(native_joint_start_step),
            "end_step": (
                None
                if native_joint_end_step is None
                else int(native_joint_end_step)
            ),
        }
        print(
            "NATIVE_JOINT_MODE_ATTESTATION transport=flow_matching "
            "selection_pool=native_joint_action_samples sample={}/{} "
            "all_non_ego_from_same_joint_mode=True gradients=False "
            "projection=False trajectory_modified=False proposal_only=True "
            "full_horizon_verifier_required=True start={} end={}".format(
                native_joint_sample_idx,
                native_joint_num_samples,
                native_joint_start_step,
                native_joint_end_step,
            )
        )
    elif interaction_intervention == "conditional_fm_source":
        if policy_model is None or eval_cfg.eval_class != "CCFlowMatching":
            raise ValueError("conditional_fm_source requires M2 flow matching")
        if eval_cfg.apply_guidance:
            raise ValueError("Conditional-FM forbids legacy gradient guidance")
        if ego_planner is None or marginal_tail_target_planner is None:
            raise ValueError(
                "Conditional-FM requires distinct attacked and safety planners"
            )
        if target_agent_idx is None or int(target_agent_idx) == 0:
            raise ValueError("Conditional-FM requires a non-ego target_agent_idx")
        if int(eval_cfg.policy.num_action_samples) != 20:
            raise ValueError("frozen M2 Conditional-FM contract requires K20")
        from ccdiff.models.conditional_flow_control import (
            ConditionalFlowSourceConfig,
        )
        from ccdiff.models.conditional_fm_sampler_adapter import (
            PlannerSeparationObjectiveConfig,
        )
        source_config = ConditionalFlowSourceConfig(
            enabled=not bool(conditional_flow_source_disabled),
            num_steps=int(conditional_flow_num_steps),
            step_size=float(conditional_flow_step_size),
            max_source_delta_rms=float(
                conditional_flow_max_source_delta_rms
            ),
            max_per_actor_source_delta_rms=float(
                conditional_flow_max_actor_delta_rms
            ),
            max_source_energy_increase=float(
                conditional_flow_max_energy_increase
            ),
            normalize_masked_gradient_rms=bool(
                conditional_flow_normalize_gradient
            ),
            optimization_scope=str(conditional_flow_optimization_scope),
            optimization_start_fraction=float(
                conditional_flow_optimization_start_fraction
            ),
            target_mean_shift_coefficients=(
                None
                if conditional_flow_target_mean_shift is None
                else tuple(
                    float(value)
                    for value in conditional_flow_target_mean_shift.split(",")
                )
            ),
            target_mean_shift_max_abs_coefficient=float(
                conditional_flow_target_mean_shift_max_abs
            ),
            target_mean_shift_max_kl_per_coordinate=float(
                conditional_flow_target_mean_shift_max_kl
            ),
            target_mean_shift_basis_kind=str(
                conditional_flow_target_mean_shift_basis
            ),
            target_mean_shift_pulse_width_fraction=float(
                conditional_flow_target_mean_shift_pulse_width
            ),
            target_mean_shift_online_cpa=bool(
                conditional_flow_target_mean_shift_online_cpa
            ),
            target_mean_shift_max_cpa_distance_m=float(
                conditional_flow_target_mean_shift_max_cpa_distance_m
            ),
            target_velocity_residual=bool(
                conditional_flow_target_velocity_residual
            ),
            target_velocity_residual_arrival_aligned=bool(
                conditional_flow_target_velocity_residual_arrival_aligned
            ),
            target_velocity_residual_arrival_max_lateral_distance_m=float(
                conditional_flow_target_velocity_residual_arrival_max_lateral_distance_m
            ),
        )
        objective_config = PlannerSeparationObjectiveConfig(
            target_actor_idx=int(target_agent_idx),
            executed_planner_actor_idx=0,
            selected_sample_idx=0,
            persistence_steps=int(conditional_flow_persistence_steps),
            temperature_m=float(conditional_flow_temperature_m),
            reference_safety_margin_m=float(
                marginal_tail_reference_safety_margin_m
            ),
            collateral_margin_m=float(marginal_tail_collateral_margin_m),
            reference_barrier_weight=float(
                conditional_flow_reference_weight
            ),
            collateral_barrier_weight=float(
                conditional_flow_collateral_weight
            ),
            source_proximity_weight=float(
                conditional_flow_source_proximity_weight
            ),
            global_conflict_step=conditional_flow_global_conflict_step,
            conflict_anchor_radius_steps=int(
                conditional_flow_conflict_anchor_radius_steps
            ),
        )
        policy_model.set_conditional_flow_source_control_schedule(
            source_config,
            objective_config,
            start_step=int(conditional_flow_global_start_step),
            end_step=conditional_flow_global_end_step,
        )
        # Optional human-error semantics are applied to the same directed
        # target<-executed-planner future edge that is now built from the real
        # planner plan.  They alter conditioning only; the complete joint FM
        # source and native 20-step transport remain the sole trajectory
        # generator.  Exactly one semantic family may be paired with source
        # control in a preregistered arm.
        semantic_modes = int(future_intent_hold_pairs is not None) + int(
            yield_belief_pairs is not None
        )
        if semantic_modes > 1:
            raise ValueError(
                "Conditional-FM permits at most one human-error semantic mode"
            )
        semantic_mode = "actual_planner_future"
        if future_intent_hold_pairs is not None:
            parsed_pairs = []
            for item in future_intent_hold_pairs.split(","):
                receiver_idx, sender_idx = item.split(":")
                parsed_pairs.append((int(receiver_idx), int(sender_idx)))
            policy_model.set_interaction_future_intent_holds(parsed_pairs)
            policy_model.set_interaction_future_intent_hold_schedule(
                start_step=future_intent_hold_start_step,
                end_step=future_intent_hold_end_step,
            )
            semantic_mode = "stale_future_intent"
        elif yield_belief_pairs is not None:
            parsed_beliefs = []
            for item in yield_belief_pairs.split(","):
                receiver_idx, sender_idx, deceleration = item.split(":")
                parsed_beliefs.append(
                    (int(receiver_idx), int(sender_idx), float(deceleration))
                )
            policy_model.set_interaction_yield_beliefs(parsed_beliefs)
            policy_model.set_interaction_yield_belief_schedule(
                start_step=yield_belief_start_step,
                end_step=yield_belief_end_step,
            )
            semantic_mode = "assumed_planner_yield"
        marginal_tail_config = {
            "mode": "conditional_fm_source",
            "num_samples": 20,
            "target_agent_indices": [int(target_agent_idx)],
        }
        print(
            "PLANNER_FUTURE_CONDITION_ATTESTATION "
            "sender=executed_attacked_planner target_receiver={} "
            "semantic_mode={} output_edit=False row_mixing=False "
            "endpoint_projection=False".format(
                target_agent_idx, semantic_mode
            )
        )
        print(
            "CONDITIONAL_FM_CONTRACT transport=frozen_flow_matching "
            "ode_steps=20 native_k=20 optimized_complete_joint_modes=1 "
            "attacked_planner={} safety_planner={} target_agent={} "
            "source_enabled={} normalize_gradient={} source_steps={} source_step_size={} max_source_rms={} "
            "max_actor_rms={} max_energy_increase={} row_mixing=False "
            "endpoint_projection=False legacy_guidance=False "
            "global_start_step={} global_end_step={} "
            "global_conflict_step={} conflict_anchor_radius_steps={} "
            "external_hard_acceptance_required=True".format(
                ego_planner_name,
                marginal_tail_target_planner_name,
                target_agent_idx,
                not bool(conditional_flow_source_disabled),
                bool(conditional_flow_normalize_gradient),
                conditional_flow_num_steps,
                conditional_flow_step_size,
                conditional_flow_max_source_delta_rms,
                conditional_flow_max_actor_delta_rms,
                conditional_flow_max_energy_increase,
                conditional_flow_global_start_step,
                conditional_flow_global_end_step,
                conditional_flow_global_conflict_step,
                conditional_flow_conflict_anchor_radius_steps,
            )
        )
    elif interaction_intervention in {
        "joint_constrained_risk", "planner_aligned_joint_risk",
        "correlated_joint_risk",
        "future_intent_joint_risk", "yield_belief_joint_risk",
        "pcn_joint_risk", "persistent_copula_joint_risk"
    }:
        if policy_model is None or eval_cfg.eval_class != "CCFlowMatching":
            raise ValueError("joint_constrained_risk requires M2 flow matching")
        if eval_cfg.apply_guidance:
            raise ValueError("joint_constrained_risk is native-sample selection")
        if ego_planner is None or marginal_tail_target_planner is None:
            raise ValueError(
                "joint_constrained_risk requires distinct target/reference planners"
            )
        if (
            interaction_intervention == "planner_aligned_joint_risk"
            and (target_agent_idx is None or int(target_agent_idx) == 0)
        ):
            raise ValueError(
                "planner_aligned_joint_risk requires a non-ego target_agent_idx"
            )
        naturalness_profile = None
        if marginal_tail_naturalness_profile_path is not None:
            with open(marginal_tail_naturalness_profile_path, "r") as handle:
                naturalness_profile = json.load(handle)
        if naturalness_profile is None:
            raise ValueError("joint_constrained_risk requires a frozen q99 profile")
        target_agent_indices = None
        if interaction_intervention == "pcn_joint_risk":
            if pcn_joint_flow_prior_beta is None:
                raise ValueError("pcn_joint_risk requires --pcn_joint_flow_prior_beta")
            if marginal_tail_receiver_indices is not None:
                target_agent_indices = sorted({
                    int(token)
                    for token in marginal_tail_receiver_indices.split(",")
                })
            elif marginal_tail_receiver_idx is not None:
                target_agent_indices = [int(marginal_tail_receiver_idx)]
            else:
                raise ValueError(
                    "pcn_joint_risk requires an explicit target via "
                    "--marginal_tail_receiver_idx or --marginal_tail_receiver_indices"
                )
            policy_model.set_pcn_joint_flow_prior(pcn_joint_flow_prior_beta)
            policy_model.set_pcn_joint_flow_prior_schedule(
                start_step=pcn_joint_flow_prior_start_step,
                end_step=pcn_joint_flow_prior_end_step,
            )
        elif interaction_intervention == "persistent_copula_joint_risk":
            if persistent_copula_persistence is None:
                raise ValueError(
                    "persistent_copula_joint_risk requires persistence"
                )
            if (
                correlated_flow_prior_pairs is None
                or correlated_flow_prior_rho is None
            ):
                raise ValueError(
                    "persistent_copula_joint_risk requires actor pairs and rho"
                )
            parsed_pairs = []
            for item in correlated_flow_prior_pairs.split(","):
                receiver, anchor = item.split(":", maxsplit=1)
                parsed_pairs.append((int(receiver), int(anchor)))
            action_dims = (
                None
                if correlated_flow_prior_action_dims is None
                else [
                    int(token)
                    for token in correlated_flow_prior_action_dims.split(",")
                ]
            )
            temporal_action_dims = (
                None
                if persistent_copula_temporal_action_dims is None
                else [
                    int(token)
                    for token in persistent_copula_temporal_action_dims.split(",")
                ]
            )
            policy_model.set_persistent_copula_flow_prior(
                persistence=float(persistent_copula_persistence),
                execute_steps=int(eval_cfg.n_step_action),
                actor_pairs=parsed_pairs,
                copula_rho=float(correlated_flow_prior_rho),
                action_dims=action_dims,
                proposal_start_index=1,
                temporal_rho=float(persistent_copula_temporal_rho),
                temporal_action_dims=temporal_action_dims,
            )
            intervention_receivers = sorted({
                int(receiver) for receiver, _ in parsed_pairs
            })
            if marginal_tail_receiver_indices is not None:
                target_agent_indices = sorted({
                    int(token)
                    for token in marginal_tail_receiver_indices.split(",")
                })
            elif marginal_tail_receiver_idx is not None:
                target_agent_indices = [int(marginal_tail_receiver_idx)]
            else:
                target_agent_indices = intervention_receivers
            if (
                not target_agent_indices
                or not set(target_agent_indices).issubset(intervention_receivers)
            ):
                raise ValueError(
                    "persistent-copula failure targets must be a non-empty "
                    "subset of prior receivers"
                )
        elif interaction_intervention == "correlated_joint_risk":
            if (
                correlated_flow_prior_pairs is None
                or correlated_flow_prior_rho is None
            ):
                raise ValueError(
                    "correlated_joint_risk requires a correlated FM-prior pair "
                    "and rho"
                )
            parsed_pairs = []
            for item in correlated_flow_prior_pairs.split(","):
                receiver, anchor = item.split(":", maxsplit=1)
                parsed_pairs.append((int(receiver), int(anchor)))
            policy_model.set_marginal_preserving_correlated_flow_prior(
                parsed_pairs,
                correlated_flow_prior_rho,
                lag_steps=correlated_flow_prior_lag_steps,
                action_dims=(
                    None
                    if correlated_flow_prior_action_dims is None
                    else [
                        int(token)
                        for token in correlated_flow_prior_action_dims.split(",")
                    ]
                ),
            )
            policy_model.set_marginal_preserving_correlated_flow_prior_schedule(
                start_step=correlated_flow_prior_start_step,
                end_step=correlated_flow_prior_end_step,
            )
            intervention_receivers = sorted({
                int(receiver) for receiver, _ in parsed_pairs
            })
            if marginal_tail_receiver_indices is not None:
                target_agent_indices = sorted({
                    int(token)
                    for token in marginal_tail_receiver_indices.split(",")
                })
            elif marginal_tail_receiver_idx is not None:
                target_agent_indices = [int(marginal_tail_receiver_idx)]
            else:
                target_agent_indices = intervention_receivers
            if (
                not target_agent_indices
                or not set(target_agent_indices).issubset(intervention_receivers)
            ):
                raise ValueError(
                    "correlated_joint_risk failure targets must be a non-empty "
                    "subset of the correlated-prior receivers"
                )
            print(
                "CORRELATED_JOINT_TARGET_CONTRACT intervention_receivers={} "
                "failure_targets={} support_receivers={}".format(
                    intervention_receivers,
                    target_agent_indices,
                    sorted(set(intervention_receivers) - set(target_agent_indices)),
                )
            )
        elif interaction_intervention == "future_intent_joint_risk":
            if future_intent_hold_pairs is None:
                raise ValueError(
                    "future_intent_joint_risk requires --future_intent_hold_pairs"
                )
            parsed_pairs = []
            for item in future_intent_hold_pairs.split(","):
                receiver, sender = item.split(":", maxsplit=1)
                parsed_pairs.append((int(receiver), int(sender)))
            policy_model.set_interaction_future_intent_holds(
                parsed_pairs, bidirectional=edge_bidirectional
            )
            policy_model.set_interaction_future_intent_hold_schedule(
                start_step=future_intent_hold_start_step,
                end_step=future_intent_hold_end_step,
            )
        elif interaction_intervention == "yield_belief_joint_risk":
            if yield_belief_pairs is None:
                raise ValueError(
                    "yield_belief_joint_risk requires --yield_belief_pairs"
                )
            parsed_beliefs = []
            for item in yield_belief_pairs.split(","):
                receiver, sender, deceleration = item.split(":", maxsplit=2)
                parsed_beliefs.append(
                    (int(receiver), int(sender), float(deceleration))
                )
            policy_model.set_interaction_yield_beliefs(
                parsed_beliefs, bidirectional=edge_bidirectional
            )
            policy_model.set_interaction_yield_belief_schedule(
                start_step=yield_belief_start_step,
                end_step=yield_belief_end_step,
            )
        if interaction_intervention == "planner_aligned_joint_risk":
            target_agent_indices = [int(target_agent_idx)]
        marginal_tail_config = {
            "mode": (
                "planner_aligned_native_joint"
                if interaction_intervention in {
                    "planner_aligned_joint_risk",
                    "persistent_copula_joint_risk",
                }
                else "constrained_native_joint"
            ),
            "start_step": int(marginal_tail_start_step),
            "end_step": (
                None if marginal_tail_end_step is None
                else int(marginal_tail_end_step)
            ),
            "collateral_margin_m": float(marginal_tail_collateral_margin_m),
            "reference_safety_margin_m": float(
                marginal_tail_reference_safety_margin_m
            ),
            "min_risk_improvement_m": float(marginal_tail_min_improvement_m),
            "max_target_clearance_m": float(
                1.0 if marginal_tail_target_max_clearance_m is None
                else marginal_tail_target_max_clearance_m
            ),
            "baseline_min_target_clearance_m": (
                None if marginal_tail_baseline_min_clearance_m is None
                else float(marginal_tail_baseline_min_clearance_m)
            ),
            "target_min_persistence_steps": int(
                marginal_tail_target_min_persistence_steps
            ),
            "progressive_target_barrier": bool(
                marginal_tail_progressive_target_barrier
            ),
            "progressive_activation_max_clearance_m": float(
                marginal_tail_progressive_activation_max_clearance_m
            ),
            "progressive_risk_first": bool(
                marginal_tail_progressive_risk_first
            ),
            "progressive_persistence_first": bool(
                marginal_tail_progressive_persistence_first
            ),
            "progressive_dose_first": bool(
                marginal_tail_progressive_dose_first
            ),
            "progressive_exposure_clearance_m": float(
                marginal_tail_progressive_exposure_clearance_m
            ),
            "max_agent_tail_fraction": float(
                marginal_tail_max_agent_tail_fraction
            ),
            "dt": float(exp_config.algo.step_time),
            "max_speed_mps": float(marginal_tail_max_speed_mps),
            "max_acc_mps2": float(marginal_tail_max_acc_mps2),
            "max_jerk_mps3": float(marginal_tail_max_jerk_mps3),
            "max_yaw_rate_rps": float(marginal_tail_max_yaw_rate_rps),
            "max_lateral_acc_mps2": float(
                marginal_tail_max_lateral_acc_mps2
            ),
            "kinematic_horizon_steps": int(
                marginal_tail_kinematic_horizon_steps
            ),
            "naturalness_profile": naturalness_profile,
            "num_samples": int(eval_cfg.policy.num_action_samples),
            "executed_planner_name": ego_planner_name,
            "proposal_attack_planner_name": (
                ego_planner_name
                if marginal_tail_attack_planner_name is None
                else marginal_tail_attack_planner_name
            ),
            "proposal_safety_planner_name": marginal_tail_target_planner_name,
            "target_agent_indices": (
                target_agent_indices
                if interaction_intervention in {
                    "planner_aligned_joint_risk", "correlated_joint_risk",
                    "pcn_joint_risk", "persistent_copula_joint_risk"
                }
                else None
            ),
        }
        if interaction_intervention == "planner_aligned_joint_risk":
            print(
                "PLANNER_ALIGNED_JOINT_RISK_ATTESTATION "
                "executed_planner={} proposal_attack_planner={} "
                "proposal_safety_planner={} role_decoupled={} "
                "target_agent={} planner_future_conditioned_before_sampling=True "
                "selection=complete_native_K20_joint_mode row_mixing=False "
                "gradients=False projection=False trajectory_modified=False".format(
                    ego_planner_name,
                    (
                        ego_planner_name
                        if marginal_tail_attack_planner_name is None
                        else marginal_tail_attack_planner_name
                    ),
                    marginal_tail_target_planner_name,
                    marginal_tail_attack_planner_name is not None,
                    target_agent_idx,
                )
            )
        print(
            "JOINT_CONSTRAINED_RISK_ATTESTATION transport=flow_matching "
            "same_joint_sample=True row_mixing=False gradients=False "
            "projection=False trajectory_modified=False target_planner={} "
            "reference_planner={} target_clearance_m={} baseline_target_clearance_m={} "
            "target_agent_indices={} "
            "target_min_persistence_steps={} "
            "reference_clearance_m={} collateral_clearance_m={} "
            "max_agent_q99_tail_fraction={} objective={} "
            "online_target_gate={} progressive_activation_max_clearance_m={} "
            "progressive_objective={} "
            "proposal_only=True full_horizon_verifier_required=True".format(
                marginal_tail_target_planner_name,
                ego_planner_name,
                marginal_tail_config["max_target_clearance_m"],
                marginal_tail_config["baseline_min_target_clearance_m"],
                marginal_tail_config["target_agent_indices"],
                marginal_tail_config["target_min_persistence_steps"],
                marginal_tail_config["reference_safety_margin_m"],
                marginal_tail_config["collateral_margin_m"],
                marginal_tail_config["max_agent_tail_fraction"],
                (
                    "min_displacement_from_policy_baseline"
                    if interaction_intervention == "correlated_joint_risk"
                    else "min_displacement_from_policy_baseline"
                ),
                (
                    "progressive_clearance_barrier"
                    if marginal_tail_config["progressive_target_barrier"]
                    else "terminal_persistent_severity"
                ),
                marginal_tail_config[
                    "progressive_activation_max_clearance_m"
                ],
                (
                    "terminal_then_min_displacement_else_max_safe_dose"
                    if marginal_tail_config["progressive_dose_first"]
                    else "terminal_then_min_displacement_else_max_safe_persistence"
                    if marginal_tail_config["progressive_persistence_first"]
                    else "terminal_then_min_displacement_else_max_safe_risk"
                    if marginal_tail_config["progressive_risk_first"]
                    else "minimum_displacement"
                ),
            )
        )
        if interaction_intervention == "correlated_joint_risk":
            print(
                "CORRELATED_JOINT_RISK_ATTESTATION "
                "source_prior=marginal_preserving_correlated_gaussian "
                "individual_marginals=standard_normal "
                "selection=complete_native_joint_mode same_joint_sample=True "
                "row_mixing=False gradients=False projection=False "
                "trajectory_modified=False"
            )
        elif interaction_intervention == "pcn_joint_risk":
            print(
                "PCN_JOINT_RISK_ATTESTATION "
                "source_prior=original_standard_gaussian_per_complete_mode "
                "proposal_kernel=pcn_neighbourhood_of_factual_latent "
                "selection=complete_native_joint_mode same_joint_sample=True "
                "row_mixing=False gradients=False projection=False "
                "trajectory_modified=False"
            )
        elif interaction_intervention == "persistent_copula_joint_risk":
            print(
                "PERSISTENT_COPULA_JOINT_RISK_ATTESTATION "
                "executed_attacked_planner={} safety_reference_planner={} "
                "sample0=independent_policy_baseline "
                "proposal_process=stationary_standard_gaussian "
                "cross_replan_persistence={} actor_pairs={} rho={} "
                "selection=complete_native_joint_mode same_joint_sample=True "
                "row_mixing=False gradients=False projection=False "
                "trajectory_modified=False full80_external_verifier_required=True".format(
                    ego_planner_name,
                    marginal_tail_target_planner_name,
                    persistent_copula_persistence,
                    parsed_pairs,
                    correlated_flow_prior_rho,
                )
            )
        elif interaction_intervention == "future_intent_joint_risk":
            print(
                "FUTURE_INTENT_JOINT_RISK_ATTESTATION "
                "source_shift=future_intent_hold "
                "selection=complete_native_joint_mode same_joint_sample=True "
                "row_mixing=False gradients=False projection=False "
                "trajectory_modified=False"
            )
        elif interaction_intervention == "yield_belief_joint_risk":
            print(
                "YIELD_BELIEF_JOINT_RISK_ATTESTATION "
                "source_shift=directed_yield_belief "
                "selection=complete_native_joint_mode same_joint_sample=True "
                "row_mixing=False gradients=False projection=False "
                "trajectory_modified=False"
            )
    elif interaction_intervention == "marginal_tail":
        if policy_model is None or eval_cfg.eval_class != "CCFlowMatching":
            raise ValueError("marginal_tail requires an M2 flow-matching policy")
        if eval_cfg.apply_guidance:
            raise ValueError(
                "marginal_tail selects native samples; disable gradient guidance"
            )
        if ego_planner is None:
            raise ValueError("marginal_tail requires --ego_planner")
        if (
            marginal_tail_target_planner_name is not None
            and marginal_tail_target_planner is None
        ):
            raise ValueError("failed to build marginal-tail target planner")
        if (
            marginal_tail_receiver_idx is not None
            and marginal_tail_receiver_indices is not None
        ):
            raise ValueError(
                "use only one of marginal_tail_receiver_idx/receiver_indices"
            )
        if marginal_tail_auto_receivers and (
            marginal_tail_receiver_idx is not None
            or marginal_tail_receiver_indices is not None
        ):
            raise ValueError(
                "auto receiver search cannot be combined with a fixed receiver"
            )
        if marginal_tail_receiver_indices is not None:
            parsed_receivers = [
                int(token)
                for token in marginal_tail_receiver_indices.split(",")
                if token.strip()
            ]
        elif marginal_tail_receiver_idx is not None:
            parsed_receivers = [int(marginal_tail_receiver_idx)]
        else:
            parsed_receivers = []
        if not parsed_receivers and not marginal_tail_auto_receivers:
            raise ValueError(
                "marginal_tail requires a receiver index/list or auto search"
            )
        if len(set(parsed_receivers)) != len(parsed_receivers):
            raise ValueError("marginal-tail receiver indices must be unique")
        if marginal_tail_start_step < 0:
            raise ValueError("marginal_tail_start_step must be non-negative")
        if (
            marginal_tail_end_step is not None
            and marginal_tail_end_step <= marginal_tail_start_step
        ):
            raise ValueError("marginal_tail_end_step must exceed start_step")
        if marginal_tail_collateral_margin_m < 0:
            raise ValueError("marginal_tail_collateral_margin_m must be non-negative")
        if marginal_tail_reference_safety_margin_m < 0:
            raise ValueError(
                "marginal_tail_reference_safety_margin_m must be non-negative"
            )
        if marginal_tail_min_improvement_m < 0:
            raise ValueError("marginal_tail_min_improvement_m must be non-negative")
        if marginal_tail_target_min_persistence_steps < 1:
            raise ValueError(
                "marginal_tail_target_min_persistence_steps must be positive"
            )
        if (
            marginal_tail_target_max_clearance_m is not None
            and marginal_tail_target_max_clearance_m <= 0
        ):
            raise ValueError(
                "marginal_tail_target_max_clearance_m must be positive"
            )
        if (
            marginal_tail_baseline_min_clearance_m is not None
            and marginal_tail_baseline_min_clearance_m <= 0
        ):
            raise ValueError(
                "marginal_tail_baseline_min_clearance_m must be positive"
            )
        if (
            marginal_tail_baseline_min_clearance_m is not None
            and marginal_tail_target_max_clearance_m is not None
            and marginal_tail_baseline_min_clearance_m
            <= marginal_tail_target_max_clearance_m
        ):
            raise ValueError(
                "baseline clearance gate must exceed candidate clearance gate"
            )
        kinematic_limits = {
            "max_speed_mps": marginal_tail_max_speed_mps,
            "max_acc_mps2": marginal_tail_max_acc_mps2,
            "max_jerk_mps3": marginal_tail_max_jerk_mps3,
            "max_yaw_rate_rps": marginal_tail_max_yaw_rate_rps,
            "max_lateral_acc_mps2": marginal_tail_max_lateral_acc_mps2,
        }
        if any(float(value) <= 0 for value in kinematic_limits.values()):
            raise ValueError("marginal-tail kinematic limits must be positive")
        if marginal_tail_kinematic_horizon_steps < 2:
            raise ValueError(
                "marginal_tail_kinematic_horizon_steps must be at least two"
            )
        naturalness_profile = None
        if marginal_tail_naturalness_profile_path is not None:
            if not os.path.isfile(marginal_tail_naturalness_profile_path):
                raise ValueError(
                    "missing marginal-tail naturalness profile {}".format(
                        marginal_tail_naturalness_profile_path
                    )
                )
            with open(marginal_tail_naturalness_profile_path, "r") as handle:
                naturalness_profile = json.load(handle)
            if (
                naturalness_profile.get("speed_bin_upper_bounds_mps") is None
                or naturalness_profile.get("q99") is None
            ):
                raise ValueError("invalid marginal-tail naturalness profile")
        marginal_tail_config = {
            "receiver_indices": parsed_receivers,
            "auto_receivers": bool(marginal_tail_auto_receivers),
            "sticky_receiver": bool(marginal_tail_sticky_receiver),
            "minimize_causal_change": bool(
                marginal_tail_minimize_causal_change
            ),
            "start_step": int(marginal_tail_start_step),
            "end_step": (
                None if marginal_tail_end_step is None
                else int(marginal_tail_end_step)
            ),
            "collateral_margin_m": float(marginal_tail_collateral_margin_m),
            "reference_safety_margin_m": float(
                marginal_tail_reference_safety_margin_m
            ),
            "min_risk_improvement_m": float(marginal_tail_min_improvement_m),
            "max_target_clearance_m": (
                None if marginal_tail_target_max_clearance_m is None
                else float(marginal_tail_target_max_clearance_m)
            ),
            "min_baseline_target_clearance_m": (
                None
                if marginal_tail_baseline_min_clearance_m is None
                else float(marginal_tail_baseline_min_clearance_m)
            ),
            "dt": float(exp_config.algo.step_time),
            **{key: float(value) for key, value in kinematic_limits.items()},
            "kinematic_horizon_steps": int(marginal_tail_kinematic_horizon_steps),
            "naturalness_profile": naturalness_profile,
        }
        print(
            "MARGINAL_JOINT_TAIL_ATTESTATION transport=flow_matching "
            "solver_steps=20 selection_pool=native_action_samples "
            "trajectory_modified=False gradients=False projection=False "
            "receivers={} sticky_receiver={} objective={} start={} end={} collateral_margin_m={} "
            "reference_safety_margin_m={} "
            "target_planner={} reference_planner={} "
            "min_improvement_m={} target_max_clearance_m={} "
            "baseline_min_clearance_m={} "
            "max_speed_mps={} max_acc_mps2={} "
            "max_jerk_mps3={} max_yaw_rate_rps={} "
            "max_lateral_acc_mps2={} kinematic_horizon_steps={} "
            "naturalness_profile={} naturalness_gate={}".format(
                "all_non_ego" if marginal_tail_auto_receivers else parsed_receivers,
                marginal_tail_sticky_receiver,
                (
                    "min_mean_displacement_from_factual"
                    if marginal_tail_minimize_causal_change
                    else "min_target_clearance"
                ),
                marginal_tail_start_step,
                marginal_tail_end_step,
                marginal_tail_collateral_margin_m,
                marginal_tail_reference_safety_margin_m,
                marginal_tail_target_planner_name,
                ego_planner_name,
                marginal_tail_min_improvement_m,
                marginal_tail_target_max_clearance_m,
                marginal_tail_baseline_min_clearance_m,
                marginal_tail_max_speed_mps,
                marginal_tail_max_acc_mps2,
                marginal_tail_max_jerk_mps3,
                marginal_tail_max_yaw_rate_rps,
                marginal_tail_max_lateral_acc_mps2,
                marginal_tail_kinematic_horizon_steps,
                marginal_tail_naturalness_profile_path,
                (
                    "speed_conditioned_q99_nonworsening"
                    if naturalness_profile is not None else "disabled"
                ),
            )
        )

    fm_sample_calls_before = 0
    if eval_cfg.eval_class == "CCFlowMatching":
        audit_state = getattr(policy_model, "_fm_evaluator_audit_state", None)
        if audit_state is None:
            raise RuntimeError(
                "CCFlowMatching evaluator did not install its sampler audit"
            )
        if audit_state["transport"] != "flow_matching":
            raise RuntimeError("M2 evaluator transport attestation is invalid")

    # Set evaluation time sampling/optimization parameters
    if eval_cfg.apply_guidance:
        if eval_cfg.eval_class in ['CCDiff', 'CCFlowMatching', 'Diffuser', 'TrafficSim', 'BC', 'HierarchicalSampleNew']:
            # policy_model.nets["policy"].controllable_agent = controllable_agent
            # policy_model.nets["policy"].current_perturbation_guidance.controlled_agent = controllable_agent
            # print('set agent number to: ', controllable_agent)
            # policy_model.nets["policy"].reset_guidance()
            policy_model.set_guidance_optimization_params(eval_cfg.guidance_optimization_params)
            
        if eval_cfg.eval_class in ['CCDiff', 'CCFlowMatching', 'Diffuser']:
            # policy_model.nets["policy"].controllable_agent = controllable_agent
            # policy_model.nets["policy"].current_perturbation_guidance.controlled_agent = controllable_agent
            # print('set agent number to: ', controllable_agent)
            # policy_model.nets["policy"].reset_guidance()
            policy_model.set_diffusion_specific_params(eval_cfg.diffusion_specific_params)
    # ----------------------------------------------------------------------------------

    # create env
    if eval_cfg.env == "trajdata":
        env_builder = EnvUnifiedBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
        env = env_builder.get_env(
            # CCDiff is scene-centric.  Keep one unified observation and only
            # replace ego in the final action tensor.
            split_ego=False,
            parse_obs=True,
            planner_override=(ego_planner is not None or generative_ego_only),
        )
    else:
        raise NotImplementedError("{} is not a valid env".format(eval_cfg.env))

    # eval loop
    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "GroundTruthOpenLoop", "GroundTruthNaN", "CCDiffHierarchicalPolicy" "ReplayAction"]

    heuristic_config = None
    use_ui = False
    if "ui" in eval_cfg.edits.editing_source:
        # TODO if using UI, initialize UI
        print("Using ONLY user interface to get scene edits...")
        use_ui = True
        raise NotImplementedError('UI')
    elif "heuristic" in eval_cfg.edits.editing_source:
        # verify heuristic args are valid
        if eval_cfg.edits.heuristic_config is not None:
            heuristic_config = eval_cfg.edits.heuristic_config
        else:
            heuristic_config = []

    render_rasterizer = None
    if render_to_video or render_to_img:
        from tbsim.utils.scene_edit_utils import get_trajdata_renderer
        # initialize rasterizer once for all scenes
        render_rasterizer = get_trajdata_renderer(eval_cfg.trajdata_source_test,
                                                  eval_cfg.trajdata_data_dirs,
                                                  future_sec=eval_cfg.future_sec,
                                                  history_sec=eval_cfg.history_sec,
                                                  raster_size=render_cfg['size'],
                                                  px_per_m=render_cfg['px_per_m'],
                                                  rebuild_maps=False,
                                                  cache_location='~/.unified_data_cache')

    result_stats = None
    scene_i = 0
    eval_scenes = eval_cfg.eval_scenes
    selection_binding = None
    if selection_record is not None:
        eval_scenes = [resolve_scene(env.dataset, selection_record)]
        selection_binding = SelectionBinding(
            env, selection_record, eval_scenes[0], control_policy,
            policy_model.nets["policy"], eval_cfg.n_step_action,
            os.path.join(eval_cfg.results_dir, "selection_binding.json"),
            history_num_frames=exp_config.algo.history_num_frames)
    if part_control and selection_record is None: 
        controllable_idx_dict = joblib.load("data/10_control_idx_test_init.pkl")
        scene_idx_mapping = joblib.load("data/0_scene_idx.pkl")
        if human: 
            controllable_idx_human = read_txt("data/0_human_label.txt")
            print(controllable_idx_human)
    while scene_i < eval_cfg.num_scenes_to_evaluate:
        scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch
        # check to make sure all the scenes are valid at starting step
        scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=(
            [selection_record["start_step_10hz"]] if selection_record is not None else None))
        if part_control and selection_record is None:
            max_controllable_size = min(controllable_agent, env.current_num_agents)
            if human: 
                control_idx = controllable_idx_human[scene_indices[0]][:max_controllable_size]
            elif rand_agent:
                control_idx = [idx for idx in range(max_controllable_size)]
            else:  # selected agents
                control_idx = controllable_idx_dict[scene_idx_mapping[scene_indices[0]]][:max_controllable_size]
            
            if len(control_idx) < max_controllable_size: 
                additional_idx = list(np.random.choice([i for i in range(env.current_num_agents) if i not in control_idx], 
                                size=(max_controllable_size-len(control_idx))))
                control_idx = control_idx + additional_idx

            mark_agents = [(i, "*", "red") for i in control_idx]
            control_policy.set_controllable_set(control_idx)
            if eval_cfg.eval_class in ['CCDiff', 'CCFlowMatching']:
                policy_model.nets['policy'].set_guidance_dim(control_idx, eval_cfg.n_step_action)
                print("set guidance dim: ", control_idx, 'timestep: ', eval_cfg.n_step_action)
            else: 
                policy_model.nets['policy'].set_guidance_dim(control_idx)
                print("set guidance dim: ", control_idx,)

        else: 
            mark_agents = []
        scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
        if len(scene_indices) == 0:
            print('no valid scenes in this batch, skipping...')
            torch.cuda.empty_cache()
            continue
        

        # if requested, split each scene up into multiple simulations
        start_frame_index = [[exp_config.algo.history_num_frames+1]] * len(scene_indices)
        if selection_record is not None:
            start_frame_index = [[selection_record["start_step_10hz"]]]
            target_agent_idx = selection_binding.target_index
            reference_agent_idx = selection_binding.ego_index
            mark_agents = [(target_agent_idx, "*", "red")]
        if eval_cfg.num_sim_per_scene > 1:
            start_frame_index = []
            for si in range(len(scene_indices)):
                cur_scene = env._current_scenes[si].scene
                sframe = exp_config.algo.history_num_frames+1
                # want to make sure there's GT for the full rollout
                eframe = cur_scene.length_timesteps - eval_cfg.num_simulation_steps
                scene_frame_inds = np.linspace(sframe, eframe, num=eval_cfg.num_sim_per_scene, dtype=int).tolist()
                start_frame_index.append(scene_frame_inds)

        # how many sims to run for the current batch of scenes
        print('Starting frames in current scenes:', start_frame_index)
        for ei in range(eval_cfg.num_sim_per_scene):
            guidance_config = None   # for the current batch of scenes
            constraint_config = None # for the current batch of scenes
            
            cur_start_frames = [scene_start[ei] for scene_start in start_frame_index]
            # double check all scenes are valid at the current start step
            scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=cur_start_frames)
            sim_scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
            sim_start_frames = [sframe for sframe, sval in zip(cur_start_frames, scenes_valid) if sval]
            if len(sim_scene_indices) == 0:
                torch.cuda.empty_cache()
                continue

            if not use_ui:
                # getting edits from either the config file or on-the-fly heuristics
                if "config" in eval_cfg.edits.editing_source:
                    guidance_config = eval_cfg.edits.guidance_config
                    constraint_config  = eval_cfg.edits.constraint_config
                if "heuristic" in eval_cfg.edits.editing_source:
                    # reset so that we can get an example batch to initialize guidance more efficiently
                    env.reset(scene_indices=scene_indices, start_frame_index=sim_start_frames)
                    ex_obs = env.get_observation()
                    if obs_to_torch:
                        device = policy.device if device is None else device
                        ex_obs = TensorUtils.to_torch(ex_obs, device=device, ignore_if_unspecified=True)

                    if typed_collision_type is not None:
                        if eval_cfg.eval_class != "CCFlowMatching":
                            raise ValueError("typed projection is M2 FlowMatching-only")
                        if target_agent_idx is None or ego_planner_name is None:
                            raise ValueError(
                                "typed projection requires explicit target and ego planner"
                            )
                        projection_config = {
                            "target_idx": int(target_agent_idx),
                            "reference_idx": int(reference_agent_idx),
                            "collision_type": str(typed_collision_type),
                            "activation_distance_m": float(
                                typed_projection_activation_distance_m
                            ),
                            "projection_step_size": float(
                                typed_projection_step_size
                            ),
                            "max_normalized_delta": float(
                                typed_projection_max_delta
                            ),
                            "temporal_only": bool(
                                typed_projection_temporal_only
                            ),
                            "native_joint_selection": bool(
                                typed_projection_native_joint_selection
                            ),
                            "final_step_only": bool(
                                typed_projection_final_step_only
                            ),
                            "damping": 1e-3,
                            "min_collision_step": int(
                                typed_projection_min_collision_step
                            ),
                            "max_collision_step": int(
                                typed_projection_max_collision_step
                            ),
                            "phase_shift_steps": int(
                                typed_projection_phase_shift_steps
                            ),
                            "joint_continuity_enabled": bool(
                                typed_projection_joint_continuity
                            ),
                            "joint_continuity_shift_steps": int(
                                eval_cfg.n_step_action
                            ),
                            "joint_continuity_risk_slack": float(
                                typed_projection_joint_continuity_risk_slack
                            ),
                            "joint_continuity_max_mean_error_m": float(
                                typed_projection_joint_continuity_max_error_m
                            ),
                            "latent_shooting_enabled": bool(
                                typed_projection_latent_shooting
                            ),
                            "latent_shooting_iterations": int(
                                typed_projection_latent_iterations
                            ),
                            "latent_shooting_lr": float(
                                typed_projection_latent_lr
                            ),
                            "latent_shooting_max_delta": float(
                                typed_projection_latent_max_delta
                            ),
                            "latent_shooting_prior_weight": float(
                                typed_projection_latent_prior_weight
                            ),
                            "latent_bank_enabled": bool(
                                typed_projection_latent_bank
                            ),
                            "latent_bank_max_delta": float(
                                typed_projection_latent_bank_max_delta
                            ),
                            "latent_bank_prior_weight": float(
                                typed_projection_latent_bank_prior_weight
                            ),
                            "latent_bank_min_improvement": float(
                                typed_projection_latent_bank_min_improvement
                            ),
                            "latent_bank_max_nll_increase": float(
                                typed_projection_latent_bank_max_nll_increase
                            ),
                            "collateral_margin_m": 0.5,
                            "trust_region_enabled": bool(
                                typed_projection_trust_region
                            ),
                            "trust_max_action_rms": float(
                                typed_projection_trust_max_rms
                            ),
                            "trust_max_action_linf": float(
                                typed_projection_trust_max_linf
                            ),
                            "trust_max_action_energy_increase": float(
                                typed_projection_trust_max_energy_increase
                            ),
                            "trust_min_residual_improvement": float(
                                typed_projection_trust_min_improvement
                            ),
                            "trust_tangent_enabled": bool(
                                typed_projection_trust_tangent
                            ),
                            "trust_tangent_rank": int(
                                typed_projection_trust_tangent_rank
                            ),
                        }
                        policy_model.set_typed_collision_projection(
                            projection_config
                        )

                    # build heuristic guidance configs for these scenes
                    if guidance_method == "safesim":
                        heuristic_guidance_cfg = build_safesim_target_guidance(
                            ex_obs["agents"],
                            target_agent_idx=target_agent_idx,
                            reference_agent_idx=reference_agent_idx,
                            severity_mode=severity_mode,
                            desired_clearance=desired_clearance,
                            cause_weight=safesim_cause_weight,
                            collateral_weight=safesim_collateral_weight,
                            map_weight=1.0,
                        )
                    elif guidance_method in {
                        "planner_hcs_safesim", "planner_hcs_coalition"
                    }:
                        # The actual ego plan does not exist until the rollout
                        # wrapper calls the single tested planner.  Install an
                        # empty placeholder here; the wrapper replaces it with
                        # the causal HCS-selected target immediately before
                        # each joint FM sample.
                        if target_agent_idx is not None:
                            raise ValueError(
                                "planner-aware HCS forbids manual target_agent_idx"
                            )
                        heuristic_guidance_cfg = [
                            [] for _ in sim_scene_indices
                        ]
                    elif guidance_method == "typed_projection":
                        # The hard projector is installed on the FM policy
                        # above; no competing endpoint-gradient loss is used.
                        heuristic_guidance_cfg = [
                            [] for _ in sim_scene_indices
                        ]
                    else:
                        heuristic_guidance_cfg = compute_heuristic_guidance(
                            heuristic_config,
                            env,
                            sim_scene_indices,
                            sim_start_frames,
                            example_batch=ex_obs['agents'],
                        )
                                                                        
                    if (
                        guidance_method not in {
                            "typed_projection",
                            "planner_hcs_safesim", "planner_hcs_coalition",
                        }
                        and len(heuristic_config) > 0
                    ):
                        # we asked to apply some guidance, but if heuristic determined there was no valid
                        #       guidance to apply (e.g. no social groups), we should skip these scenes.
                        valid_scene_inds = []
                        for sci, sc_cfg in enumerate(heuristic_guidance_cfg):
                            if len(sc_cfg) > 0:
                                valid_scene_inds.append(sci)

                        # collect only valid scenes under the given heuristic config
                        heuristic_guidance_cfg = [heuristic_guidance_cfg[vi] for vi in valid_scene_inds]
                        sim_scene_indices = [sim_scene_indices[vi] for vi in valid_scene_inds]
                        sim_start_frames = [sim_start_frames[vi] for vi in valid_scene_inds]
                        # skip if no valid...
                        if len(sim_scene_indices) == 0:
                            print('No scenes with valid heuristic configs in this sim, skipping...')
                            torch.cuda.empty_cache()
                            continue

                    # add to the current guidance config
                    guidance_config = merge_guidance_configs(guidance_config, heuristic_guidance_cfg)
            else:
                # TODO get guidance from the UI
                # TODO for UI, get edits from user. loop continuously until the user presses
                #       "play" or something like that then we roll out.
                raise NotImplementedError()
        if len(sim_scene_indices) == 0:
            print('No scenes with valid heuristic configs in this scene, skipping...')
            torch.cuda.empty_cache()
            continue

        # An unguided rollout legitimately has no edit configuration. Normalize
        # that case to one empty configuration per scene so the shared rollout
        # and result-writing path remains identical to guided evaluation.
        if guidance_config is None:
            guidance_config = [[] for _ in sim_scene_indices]
        if constraint_config is None:
            constraint_config = []

        # remove agents from agent_collision guidance if they are in chosen gptcollision pair
        for si in range(len(guidance_config)):
            if len(guidance_config[si]) > 0:
                agent_collision_heur_ind = None
                gpt_collision_heur_ind = None
                for i, cur_heur in enumerate(guidance_config[si]):
                    if cur_heur['name'] == 'agent_collision':
                        agent_collision_heur_ind = i
                    elif cur_heur['name'] == 'gptcollision':
                        gpt_collision_heur_ind = i
                if agent_collision_heur_ind is not None and gpt_collision_heur_ind is not None:
                        ind1 = guidance_config[si][gpt_collision_heur_ind]['params']['target_ind']
                        ind2 = guidance_config[si][gpt_collision_heur_ind]['params']['ref_ind']
                        excluded_agents = [ind1, ind2]
                        guidance_config[si][agent_collision_heur_ind]['params']['excluded_agents'] = excluded_agents
                        print('excluded_agents', excluded_agents)

        # ----------------------------------------------------------------------------------
        # Sampling Wrapper leveraging most existing policy composer sampling interfaces
        from tbsim.policies.wrappers import NewSamplingPolicyWrapper
        if eval_cfg.eval_class in ['TrafficSim', 'HierarchicalSampleNew']:
            if scene_i == eval_cfg.num_scenes_per_batch or not isinstance(policy, NewSamplingPolicyWrapper):
                policy = NewSamplingPolicyWrapper(policy, guidance_config)
            else:
                policy.update_guidance_config(guidance_config)
        # ----------------------------------------------------------------------------------

        if eval_cfg.policy.pos_to_yaw:
            policy = Pos2YawWrapper(
                policy,
                dt=exp_config.algo.step_time,
                yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
            )
        # right now assume control of full scene
        if five_class_request is not None:
            from ccdiff.evaluation.selected_planner_rollout import SelectedPlannerRollout
            rollout_policy = SelectedPlannerRollout(policy, ego_planner, policy_model,
                selection_binding, five_class_request, eval_cfg.results_dir, exp_config.algo.step_time)
        elif generative_ego_only:
            if ego_planner is not None or part_control:
                raise ValueError(
                    "generative_ego_only cannot be combined with an external "
                    "ego planner or part_control"
                )
            if eval_cfg.apply_guidance:
                raise ValueError(
                    "generative_ego_only is an unguided planner contract"
                )
            rollout_policy = GenerativeEgoReplayRolloutWrapper(
                generative_policy=policy,
                transport_name="flow_matching",
            )
            print(
                "GENERATIVE_EGO_PLANNER_ATTESTATION "
                "transport=flow_matching traffic=logged_replay "
                "reactive_traffic=False"
            )
        elif ego_planner is not None:
            rollout_policy = PlannerOverrideRolloutWrapper(
                ego_policy=ego_planner,
                agents_policy=agents_policy,
                strategy_depth=strategy_depth,
                marginal_tail_config=marginal_tail_config,
                marginal_tail_target_ego_policy=marginal_tail_target_planner,
                marginal_tail_attack_ego_policy=marginal_tail_attack_planner,
                safesim_guidance=(
                    guidance_method in {
                        "safesim", "planner_hcs_safesim",
                        "planner_hcs_coalition",
                        "typed_projection"
                    }
                ),
                planner_aware_hcs=(
                    guidance_method in {
                        "planner_hcs_safesim", "planner_hcs_coalition"
                    }
                ),
                planner_aware_hcs_config={
                    "dt": float(exp_config.algo.step_time),
                    "reference_agent_idx": int(reference_agent_idx),
                    "severity_mode": str(severity_mode),
                    "desired_clearance": float(desired_clearance),
                    "cause_weight": float(safesim_cause_weight),
                    "collateral_weight": float(
                        safesim_collateral_weight
                    ),
                    "start_step": int(planner_hcs_start_step),
                    "end_step": (
                        None if planner_hcs_end_step is None
                        else int(planner_hcs_end_step)
                    ),
                    "arrival_weight": float(planner_hcs_arrival_weight),
                    "velocity_trust_max_rms": float(
                        planner_hcs_velocity_trust_max_rms
                    ),
                    "velocity_trust_max_linf": float(
                        planner_hcs_velocity_trust_max_linf
                    ),
                    "semantic_control": str(
                        planner_hcs_semantic_control
                    ),
                    "semantic_plan_json": (
                        None if planner_hcs_semantic_plan_json is None
                        else str(planner_hcs_semantic_plan_json)
                    ),
                    "coalition_enabled": bool(
                        guidance_method == "planner_hcs_coalition"
                    ),
                    "coalition_min_blockers": int(
                        planner_hcs_coalition_min_blockers
                    ),
                    "coalition_max_blockers": int(
                        planner_hcs_coalition_max_blockers
                    ),
                    "coalition_corridor_offset_m": float(
                        planner_hcs_coalition_corridor_offset_m
                    ),
                    "coalition_max_reach_error_m": float(
                        planner_hcs_coalition_max_reach_error_m
                    ),
                },
            )
        elif part_control:
            control_policy.add_policy(policy)
            rollout_policy = RolloutWrapper(agents_policy=control_policy)
            # policy = RolloutWrapper(agents_policy=policy)
        else: 
            rollout_policy = RolloutWrapper(agents_policy=policy)
        stats, info, renderings = guided_rollout(
            env,
            rollout_policy,
            policy_model,
            n_step_action=eval_cfg.n_step_action,
            guidance_config=guidance_config,
            constraint_config=constraint_config,
            render=False, # render after the fact
            scene_indices=scene_indices,
            obs_to_torch=obs_to_torch,
            horizon=eval_cfg.num_simulation_steps,
            start_frames=sim_start_frames,
            eval_class=eval_cfg.eval_class,
            apply_guidance=eval_cfg.apply_guidance, 
            num_steps_intervention=num_steps_intervention
        )

        if five_class_request is not None:
            rollout_policy.finalize()

        if (
            eval_cfg.eval_class == "CCFlowMatching"
            and scene_source not in ["logged_replay", "frozen_hdf5"]
        ):
            audit_state = policy_model._fm_evaluator_audit_state
            fm_sample_calls_after = audit_state["sample_calls"]
            if fm_sample_calls_after <= fm_sample_calls_before:
                raise RuntimeError(
                    "Closed-loop rollout completed without calling the audited "
                    "CCFlowMatching.p_sample_loop"
                )
            print(
                "FM_EVALUATOR_AUDIT sample_calls={} total_nfe={}".format(
                    fm_sample_calls_after,
                    audit_state["total_model_evaluations"],
                )
            )
            fm_sample_calls_before = fm_sample_calls_after

        print(info["scene_index"])
        print(sim_start_frames)
        pprint(stats)

        # aggregate stats from the same class of guidance within each scene
        #       this helps parse_scene_edit_results
        guide_agg_dict = {}
        pop_list = []
        for k,v in stats.items():
            if k.split('_')[0] == 'guide':
                guide_name = '_'.join(k.split('_')[:-1])
                guide_scene_tag = k.split('_')[-1][:2]
                canon_name = guide_name + '_%sg0' % (guide_scene_tag)
                if canon_name not in guide_agg_dict:
                    guide_agg_dict[canon_name] = []
                guide_agg_dict[canon_name].append(v)
                # remove from stats
                pop_list.append(k)
        for k in pop_list:
            stats.pop(k, None)
        # average over all of the same guide stats in each scene
        for k,v in guide_agg_dict.items():
            scene_stats = np.stack(v, axis=0) # guide_per_scenes x num_scenes (all are nan except 1)
            stats[k] = np.mean(scene_stats, axis=0)

        # aggregate metrics stats
        if result_stats is None:
            result_stats = stats
            result_stats["scene_index"] = np.array(info["scene_index"])
        else:
            for k in stats:
                if k not in result_stats:
                    result_stats[k] = stats[k]
                else:
                    result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
            result_stats["scene_index"] = np.concatenate([result_stats["scene_index"], np.array(info["scene_index"])])

        # write stats to disk
        with open(os.path.join(eval_cfg.results_dir, "stats.json"), "w+") as fp:
            stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
            json.dump(stats_to_write, fp)
        if "matrix_ttc" in info.keys() and "matrix_dist" in info.keys():
            os.makedirs(os.path.join(eval_cfg.results_dir, "matrix"), exist_ok=True)
            np.save(os.path.join(eval_cfg.results_dir, "matrix", "ttc_{}.npy".format(info["scene_index"])), info["matrix_ttc"])
            np.save(os.path.join(eval_cfg.results_dir, "matrix", "dist_{}.npy".format(info["scene_index"])), info["matrix_dist"])

        if render_to_video or render_to_img:
            # high quality
            from tbsim.utils.scene_edit_utils import visualize_guided_rollout
            scene_cnt = 0
            for si, scene_buffer in zip(info["scene_index"], info["buffer"]):
                viz_dir = os.path.join(eval_cfg.results_dir, "viz/")
                invalid_guidance = guidance_config is None or len(guidance_config) == 0
                invalid_constraint = constraint_config is None or len(constraint_config) == 0
                visualize_guided_rollout(viz_dir, render_rasterizer, si, scene_buffer,
                                            guidance_config=None if invalid_guidance else guidance_config[scene_cnt],
                                            constraint_config=None if invalid_constraint else constraint_config[scene_cnt],
                                            fps=(1.0 / exp_config.algo.step_time),
                                            n_step_action=eval_cfg.n_step_action,
                                            viz_diffusion_steps=False,
                                            first_frame_only=render_to_img,
                                            sim_num=sim_start_frames[scene_cnt],
                                            save_every_n_frames=render_cfg['save_every_n_frames'],
                                            draw_mode=render_cfg['draw_mode'],
                                            mark_agents=mark_agents)
                scene_cnt += 1

        if data_to_disk and "buffer" in info:
            dump_episode_buffer(
                info["buffer"],
                info["scene_index"],
                sim_start_frames,
                h5_path=eval_cfg.experience_hdf5_path
            )
        torch.cuda.empty_cache()


def dump_episode_buffer(buffer, scene_index, start_frames, h5_path):
    import h5py
    with h5py.File(h5_path, "a") as h5_file:
        for ei, si, scene_buffer in zip(start_frames, scene_index, buffer):
            for mk in scene_buffer:
                h5key = "/{}_{}/{}".format(si, ei, mk)
                value = np.asarray(scene_buffer[mk])
                # The existing parser expects map names as UTF-8 bytes.
                if value.dtype.kind in {"U", "O"}:
                    value = value.astype("S")
                h5_file.create_dataset(h5key, data=value)
    print("scene {} written to {}".format(scene_index, h5_path))


if __name__ == "__main__":
    GPU_BINDING_ATTESTATION = attest_expected_cuda_binding()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["nusc", "trajdata"],
        help="Which env to run editing in",
        required=True
    )

    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None
    )

    parser.add_argument(
        "--metric_ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location for the learned metric",
        default=None
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--policy_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--policy_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ------ for BITS ------
    parser.add_argument(
        "--planner_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--planner_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )

    parser.add_argument(
        "--predictor_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--predictor_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ----------------------

    parser.add_argument(
        "--results_root_dir",
        type=str,
        default=None,
        help="Directory to save results and videos"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Root directory of the dataset"
    )

    parser.add_argument(
        "--eval_scenes_file",
        type=str,
        default=None,
        help="JSON file containing a scene-index list or a scene_indices field"
    )

    parser.add_argument(
        "--num_scenes_to_evaluate",
        type=int,
        default=None,
        help="Evaluate this many entries from the fixed scene list"
    )

    parser.add_argument(
        "--num_simulation_steps",
        type=int,
        default=None,
        help="Closed-loop rollout horizon in 10 Hz frames"
    )

    parser.add_argument(
        "--num_scenes_per_batch",
        type=int,
        default=None,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="whether to render videos"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--registered_name",
        type=str,
        default='trajdata_nusc_diff',
    )

    parser.add_argument(
        "--render_img",
        action="store_true",
        default=False,
        help="whether to only render the first frame of rollout"
    )

    parser.add_argument(
        "--render_size",
        type=int,
        default=400,
        help="width and height of the rendered image size in pixels"
    )

    parser.add_argument(
        "--render_px_per_m",
        type=float,
        default=2.0,
        help="resolution of rendering"
    )

    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=5,
        help="saving videos while skipping every n frames"
    )

    parser.add_argument(
        "--draw_mode",
        type=str,
        default='action',
        help="['action', 'entire_traj', 'map']"
    )
    
    #
    # Editing options
    #
    parser.add_argument(
        "--editing_source",
        type=str,
        choices=["config", "heuristic", "ui", "none"],
        nargs="+",
        help="Which edits to use. config is directly from the configuration file. heuristic will \
              set edits automatically based on heuristics. UI will use interactive interface. \
              config and heuristic may be used together. If none, does not use edits."
    )
    parser.add_argument(
        "--controllable_agent",
        type=int,
        default=1, 
    )
    parser.add_argument(
        "--part_control",
        action="store_true",
    )
    parser.add_argument(
        "--human",
        action="store_true",
    )
    parser.add_argument(
        "--rand_agent",
        action="store_true",
    )
    parser.add_argument(
        "--n_step_action",
        type=int,
        default=5, 
    )

    parser.add_argument(
        "--w_collision",
        type=float,
        default=50.0, 
    )
    parser.add_argument(
        "--w_offroad",
        type=float,
        default=1.0, 
    )

    parser.add_argument(
        "--guidance_method",
        choices=[
            "ccdiff", "safesim", "planner_hcs_safesim",
            "typed_projection",
        ],
        default="ccdiff",
        help="Original all-agent CCDiff loss or explicit SAFE-SIM target pair",
    )
    parser.add_argument("--target_agent_idx", type=int, default=None)
    parser.add_argument("--reference_agent_idx", type=int, default=0)
    parser.add_argument(
        "--severity_mode",
        choices=["near_miss", "emergency", "collision"],
        default="collision",
    )
    parser.add_argument("--desired_clearance", type=float, default=0.0)
    parser.add_argument("--safesim_cause_weight", type=float, default=1.0)
    parser.add_argument("--safesim_collateral_weight", type=float, default=1.0)
    parser.add_argument("--planner_hcs_start_step", type=int, default=20)
    parser.add_argument("--planner_hcs_end_step", type=int, default=None)
    parser.add_argument("--planner_hcs_arrival_weight", type=float, default=0.5)
    parser.add_argument(
        "--planner_hcs_velocity_trust_max_rms", type=float, default=0.05
    )
    parser.add_argument(
        "--planner_hcs_velocity_trust_max_linf", type=float, default=0.20
    )
    parser.add_argument(
        "--planner_hcs_semantic_control",
        choices=["actual", "matched_shuffled"],
        default="actual",
    )
    parser.add_argument(
        "--planner_hcs_semantic_plan_json",
        default=None,
        help=(
            "Frozen same-P* seed1 donor future used only by the B3a "
            "matched-shuffled semantic control"
        ),
    )
    parser.add_argument(
        "--planner_hcs_coalition_min_blockers", type=int, default=1
    )
    parser.add_argument(
        "--planner_hcs_coalition_max_blockers", type=int, default=2
    )
    parser.add_argument(
        "--planner_hcs_coalition_corridor_offset_m", type=float, default=3.2
    )
    parser.add_argument(
        "--planner_hcs_coalition_max_reach_error_m", type=float, default=12.0
    )
    parser.add_argument(
        "--typed_collision_type",
        choices=["rear_end", "side", "cut_in", "head_on"],
        default=None,
        help="Inference-only typed action projection",
    )
    parser.add_argument(
        "--typed_projection_activation_distance_m", type=float, default=25.0
    )
    parser.add_argument(
        "--typed_projection_step_size", type=float, default=0.35
    )
    parser.add_argument(
        "--typed_projection_max_delta", type=float, default=0.75
    )
    parser.add_argument(
        "--typed_projection_temporal_only", action="store_true",
        help="Preserve FM yaw-rate exactly; project longitudinal timing only",
    )
    parser.add_argument(
        "--typed_projection_native_joint_selection", action="store_true",
        help="Promote one complete K-mode joint sample to sample-0 before projecting only its target",
    )
    parser.add_argument(
        "--typed_projection_final_step_only", action="store_true",
        help="Leave FM ODE native until its final endpoint, then apply one projection",
    )
    parser.add_argument(
        "--typed_projection_min_collision_step", type=int, default=5,
        help="Earliest future step eligible for the typed collision event",
    )
    parser.add_argument(
        "--typed_projection_max_collision_step", type=int, default=15,
        help="Latest future step eligible for the typed collision event",
    )
    parser.add_argument(
        "--typed_projection_phase_shift_steps", type=int, default=0,
        help="Signed path phase shift: positive makes target reach its native future point early",
    )
    parser.add_argument(
        "--typed_projection_joint_continuity", action="store_true",
        help="Choose risky complete K modes subject to cross-replan world-trajectory continuity",
    )
    parser.add_argument(
        "--typed_projection_joint_continuity_risk_slack", type=float, default=1.0,
        help="Risk-score band within which continuity breaks ties",
    )
    parser.add_argument(
        "--typed_projection_joint_continuity_max_error_m", type=float, default=2.0,
        help="Reject a promoted joint mode above this mean overlap error",
    )
    parser.add_argument(
        "--typed_projection_trust_region", action="store_true",
        help="Accept per-flow-step typed GN only inside FM action trust gates",
    )
    parser.add_argument(
        "--typed_projection_trust_max_rms", type=float, default=0.20,
    )
    parser.add_argument(
        "--typed_projection_trust_max_linf", type=float, default=0.75,
    )
    parser.add_argument(
        "--typed_projection_trust_max_energy_increase", type=float, default=0.05,
    )
    parser.add_argument(
        "--typed_projection_trust_min_improvement", type=float, default=0.001,
    )
    parser.add_argument(
        "--typed_projection_trust_tangent", action="store_true",
        help="Project typed GN into the low-rank span of native K-mode FM endpoints",
    )
    parser.add_argument(
        "--typed_projection_trust_tangent_rank", type=int, default=4,
    )
    parser.add_argument(
        "--typed_projection_latent_shooting", action="store_true",
        help="Optimize only target longitudinal initial FM latent through the full ODE",
    )
    parser.add_argument("--typed_projection_latent_iterations", type=int, default=3)
    parser.add_argument("--typed_projection_latent_lr", type=float, default=0.1)
    parser.add_argument("--typed_projection_latent_max_delta", type=float, default=0.35)
    parser.add_argument("--typed_projection_latent_prior_weight", type=float, default=0.05)
    parser.add_argument(
        "--typed_projection_latent_bank", action="store_true",
        help="Search an antithetic likelihood-budgeted FM source bank once",
    )
    parser.add_argument("--typed_projection_latent_bank_max_delta", type=float, default=0.25)
    parser.add_argument("--typed_projection_latent_bank_prior_weight", type=float, default=0.10)
    parser.add_argument("--typed_projection_latent_bank_min_improvement", type=float, default=0.01)
    parser.add_argument("--typed_projection_latent_bank_max_nll_increase", type=float, default=0.02)

    parser.add_argument(
        "--disable_guidance",
        action="store_true",
        help="Run the identical policy/scene protocol without guidance gradients"
    )
    parser.add_argument(
        "--eval_num_action_samples",
        type=int,
        default=None,
        help=(
            "Explicit native FM sample-bank size for an unmodified M2 rollout. "
            "Omit to preserve the legacy K5 planner default."
        ),
    )
    parser.add_argument(
        "--interaction_intervention",
        choices=["none", "semantic_assertive_token", "edge_cut", "full_visibility_cut", "edge_delay", "future_intent_hold", "yield_belief", "reaction_delay", "correlated_flow_prior", "native_joint_mode", "conditional_fm_source", "marginal_tail", "joint_constrained_risk", "planner_aligned_joint_risk", "correlated_joint_risk", "future_intent_joint_risk", "yield_belief_joint_risk", "pcn_joint_risk", "persistent_copula_joint_risk"],
        default="none",
    )
    parser.add_argument(
        "--semantic_num_action_samples",
        type=int,
        default=None,
        help=(
            "Explicit native FM sample-bank size for conditioning-only "
            "semantic interventions. Omit to preserve the legacy K5 default."
        ),
    )
    parser.add_argument("--edge_receiver_idx", type=int, default=None)
    parser.add_argument("--edge_sender_idx", type=int, default=None)
    parser.add_argument("--edge_cut_pairs", type=str, default=None)
    parser.add_argument("--edge_bidirectional", action="store_true")
    parser.add_argument("--edge_cut_start_step", type=int, default=0)
    parser.add_argument("--edge_cut_end_step", type=int, default=None)
    parser.add_argument("--semantic_assertive_receiver_idx", type=int, default=None)
    parser.add_argument("--semantic_assertive_token_scale", type=float, default=1.0)
    parser.add_argument("--semantic_assertive_start_step", type=int, default=0)
    parser.add_argument("--semantic_assertive_end_step", type=int, default=None)
    parser.add_argument(
        "--semantic_assertive_overlay",
        action="store_true",
        help=(
            "Overlay the trained M2-e assertive conditioning token on the "
            "planner-aligned native-K selector; trajectories remain untouched."
        ),
    )
    parser.add_argument("--full_visibility_cut_start_step", type=int, default=0)
    parser.add_argument("--full_visibility_cut_end_step", type=int, default=None)
    parser.add_argument(
        "--edge_delay_pairs",
        type=str,
        default=None,
        help="Comma-separated receiver:sender:delay_steps entries",
    )
    parser.add_argument("--edge_delay_steps", type=int, default=None)
    parser.add_argument("--edge_delay_start_step", type=int, default=0)
    parser.add_argument("--edge_delay_end_step", type=int, default=None)
    parser.add_argument("--future_intent_hold_pairs", type=str, default=None)
    parser.add_argument("--future_intent_hold_start_step", type=int, default=0)
    parser.add_argument("--future_intent_hold_end_step", type=int, default=None)
    parser.add_argument(
        "--yield_belief_pairs", type=str, default=None,
        help="comma-separated receiver:sender:deceleration_mps2 beliefs",
    )
    parser.add_argument("--yield_belief_start_step", type=int, default=0)
    parser.add_argument("--yield_belief_end_step", type=int, default=None)
    parser.add_argument("--reaction_receiver_idx", type=int, default=None)
    parser.add_argument("--reaction_delay_steps", type=int, default=None)
    parser.add_argument("--reaction_delay_start_step", type=int, default=0)
    parser.add_argument("--reaction_delay_end_step", type=int, default=None)
    parser.add_argument("--reaction_accel_bias_mps2", type=float, default=0.0)
    parser.add_argument(
        "--correlated_flow_prior_pairs",
        type=str,
        default=None,
        help="Comma-separated disjoint agent-index pairs, e.g. 23:17",
    )
    parser.add_argument("--correlated_flow_prior_rho", type=float, default=None)
    parser.add_argument("--correlated_flow_prior_lag_steps", type=int, default=0)
    parser.add_argument(
        "--correlated_flow_prior_action_dims", type=str, default=None,
        help="Comma-separated action dims: 0=acceleration, 1=yaw velocity",
    )
    parser.add_argument("--correlated_flow_prior_start_step", type=int, default=0)
    parser.add_argument("--correlated_flow_prior_end_step", type=int, default=None)
    parser.add_argument(
        "--persistent_copula_persistence", type=float, default=None,
        help=(
            "AR(1)-style overlap correlation for proposal FM sources across "
            "replanning calls; only valid with persistent_copula_joint_risk"
        ),
    )
    parser.add_argument(
        "--persistent_copula_temporal_rho", type=float, default=0.0,
        help=(
            "Within-horizon AR(1) Gaussian-copula correlation for proposal "
            "FM sources; preserves every one-step N(0,1) marginal"
        ),
    )
    parser.add_argument(
        "--persistent_copula_temporal_action_dims", type=str, default=None,
        help=(
            "Comma-separated action dims receiving within-horizon source "
            "correlation; default is all action dims"
        ),
    )
    parser.add_argument("--pcn_joint_flow_prior_beta", type=float, default=None)
    parser.add_argument("--pcn_joint_flow_prior_start_step", type=int, default=0)
    parser.add_argument("--pcn_joint_flow_prior_end_step", type=int, default=None)
    parser.add_argument("--native_joint_sample_idx", type=int, default=None)
    parser.add_argument("--native_joint_start_step", type=int, default=0)
    parser.add_argument("--native_joint_end_step", type=int, default=None)
    parser.add_argument("--native_joint_num_samples", type=int, default=20)
    parser.add_argument("--conditional_flow_num_steps", type=int, default=1)
    parser.add_argument(
        "--conditional_flow_source_disabled", action="store_true",
        help="Run the exact native K20 source through the same dual-planner path",
    )
    parser.add_argument(
        "--conditional_flow_target_mean_shift",
        type=str,
        default=None,
        help=(
            "Four comma-separated source coefficients: constant/ramp for "
            "acceleration, then constant/ramp for yaw velocity"
        ),
    )
    parser.add_argument(
        "--conditional_flow_target_mean_shift_max_abs", type=float, default=0.25
    )
    parser.add_argument(
        "--conditional_flow_target_mean_shift_max_kl", type=float, default=0.02
    )
    parser.add_argument(
        "--conditional_flow_target_mean_shift_basis",
        choices=("constant_ramp", "gaussian_pulse"),
        default="constant_ramp",
    )
    parser.add_argument(
        "--conditional_flow_target_mean_shift_pulse_width", type=float, default=0.15
    )
    parser.add_argument(
        "--conditional_flow_target_mean_shift_online_cpa", action="store_true"
    )
    parser.add_argument(
        "--conditional_flow_target_mean_shift_max_cpa_distance_m",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--conditional_flow_target_velocity_residual",
        action="store_true",
        help=(
            "Interpret the four target coefficients as a trust-bounded "
            "residual on the frozen FM vector field instead of a source shift"
        ),
    )
    parser.add_argument(
        "--conditional_flow_target_velocity_residual_arrival_aligned",
        action="store_true",
        help=(
            "Interpret acceleration coefficients as unsigned magnitudes and "
            "derive their sign and pulse time from the executed planner path "
            "and the target current-state arrival-time error"
        ),
    )
    parser.add_argument(
        "--conditional_flow_target_velocity_residual_arrival_max_lateral_distance_m",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--conditional_flow_normalize_gradient", action="store_true",
        help="Interpret step size as active-source RMS dose",
    )
    parser.add_argument("--conditional_flow_step_size", type=float, default=0.05)
    parser.add_argument(
        "--conditional_flow_max_source_delta_rms", type=float, default=0.05
    )
    parser.add_argument(
        "--conditional_flow_max_actor_delta_rms", type=float, default=0.05
    )
    parser.add_argument(
        "--conditional_flow_max_energy_increase", type=float, default=0.01
    )
    parser.add_argument(
        "--conditional_flow_persistence_steps", type=int, default=3
    )
    parser.add_argument(
        "--conditional_flow_temperature_m", type=float, default=0.25
    )
    parser.add_argument(
        "--conditional_flow_reference_weight", type=float, default=4.0
    )
    parser.add_argument(
        "--conditional_flow_collateral_weight", type=float, default=4.0
    )
    parser.add_argument(
        "--conditional_flow_source_proximity_weight", type=float, default=0.25
    )
    parser.add_argument(
        "--conditional_flow_optimization_scope",
        choices=("all_background", "target_only"),
        default="all_background",
    )
    parser.add_argument(
        "--conditional_flow_optimization_start_fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--conditional_flow_global_start_step", type=int, default=0
    )
    parser.add_argument(
        "--conditional_flow_global_end_step", type=int, default=None
    )
    parser.add_argument(
        "--conditional_flow_global_conflict_step", type=int, default=None
    )
    parser.add_argument(
        "--conditional_flow_conflict_anchor_radius_steps", type=int, default=2
    )
    parser.add_argument("--marginal_tail_receiver_idx", type=int, default=None)
    parser.add_argument("--marginal_tail_receiver_indices", type=str, default=None)
    parser.add_argument(
        "--marginal_tail_auto_receivers",
        action="store_true",
        help=(
            "Evaluate every modeled non-ego receiver under identical native-"
            "sample gates and select at most one receiver/mode per replan"
        ),
    )
    parser.add_argument(
        "--marginal_tail_sticky_receiver",
        action="store_true",
        help=(
            "After the first non-factual selection, restrict every later "
            "replan in the episode to that same receiver"
        ),
    )
    parser.add_argument(
        "--marginal_tail_minimize_causal_change",
        action="store_true",
        help=(
            "After applying severity/safety/naturalness hard gates, select "
            "the native mode with minimum mean displacement from factual mode 0"
        ),
    )
    parser.add_argument("--marginal_tail_start_step", type=int, default=0)
    parser.add_argument("--marginal_tail_end_step", type=int, default=None)
    parser.add_argument(
        "--marginal_tail_collateral_margin_m", type=float, default=0.25
    )
    parser.add_argument(
        "--marginal_tail_reference_safety_margin_m", type=float, default=0.25
    )
    parser.add_argument(
        "--marginal_tail_min_improvement_m", type=float, default=0.05
    )
    parser.add_argument(
        "--marginal_tail_target_max_clearance_m",
        type=float,
        default=None,
        help=(
            "Hard target-planner clearance gate; 2 m corresponds to the "
            "preregistered clearance definition of at least near-miss"
        ),
    )
    parser.add_argument(
        "--marginal_tail_progressive_target_barrier",
        action="store_true",
        help=(
            "Online proposal gate requires target-clearance progress relative "
            "to mode0 within each short replan, rather than terminal severity "
            "in every replan. The final "
            "full-rollout persistent-failure verifier remains mandatory."
        ),
    )
    parser.add_argument(
        "--marginal_tail_progressive_activation_max_clearance_m",
        type=float,
        default=8.0,
        help=(
            "Progressive native-mode selection is disabled when mode0 target "
            "clearance exceeds this factual-only risk activation radius."
        ),
    )
    parser.add_argument(
        "--marginal_tail_progressive_risk_first",
        action="store_true",
        help=(
            "For progressive complete-joint selection, maximize safe target-"
            "risk progress until a terminal-severity native mode exists; then "
            "minimize displacement only among terminal-success modes."
        ),
    )
    parser.add_argument(
        "--marginal_tail_progressive_persistence_first",
        action="store_true",
        help=(
            "Before terminal severity exists, lexicographically maximize the "
            "longest consecutive exposure below the semantic clearance band, "
            "then minimize clearance and use displacement only as a tie-break."
        ),
    )
    parser.add_argument(
        "--marginal_tail_progressive_exposure_clearance_m",
        type=float, default=2.0,
        help=(
            "Semantic clearance band used by persistence-first complete-joint "
            "selection; 2m is the preregistered near-miss definition."
        ),
    )
    parser.add_argument(
        "--marginal_tail_progressive_dose_first",
        action="store_true",
        help=(
            "Before terminal severity, rank hard-safe complete native modes "
            "by integrated target clearance deficit within the exposure band."
        ),
    )
    parser.add_argument(
        "--marginal_tail_baseline_min_clearance_m",
        type=float,
        default=None,
        help=(
            "Require factual native mode 0 to remain at least this far from "
            "the target-planner rollout before any intervention is allowed"
        ),
    )
    parser.add_argument(
        "--marginal_tail_target_min_persistence_steps", type=int, default=1,
        help=(
            "Require target risk below the clearance threshold for this many "
            "consecutive proposal frames"
        ),
    )
    parser.add_argument("--marginal_tail_max_speed_mps", type=float, default=20.0)
    parser.add_argument("--marginal_tail_max_acc_mps2", type=float, default=4.0)
    parser.add_argument("--marginal_tail_max_jerk_mps3", type=float, default=8.37)
    parser.add_argument("--marginal_tail_max_yaw_rate_rps", type=float, default=0.95)
    parser.add_argument(
        "--marginal_tail_max_lateral_acc_mps2", type=float, default=4.0
    )
    parser.add_argument(
        "--marginal_tail_kinematic_horizon_steps", type=int, default=5
    )
    parser.add_argument(
        "--marginal_tail_max_agent_tail_fraction", type=float, default=0.01
    )
    parser.add_argument(
        "--marginal_tail_naturalness_profile",
        type=str,
        default=None,
        help=(
            "JSON profile of logged-data speed-conditioned q99 thresholds; "
            "each selected native mode must not add violations over mode zero"
        ),
    )
    parser.add_argument(
        "--marginal_tail_num_native_samples", type=int, default=None,
        help="Inference-only native FM candidate bank size for marginal-tail selection",
    )
    parser.add_argument(
        "--marginal_tail_target_planner",
        type=str,
        choices=["idm", "strive"],
        default=None,
        help=(
            "Optional target planner for selective adversariality; the "
            "regular --ego_planner remains the executed safety reference"
        ),
    )
    parser.add_argument(
        "--marginal_tail_attack_planner",
        type=str,
        choices=["idm", "strive"],
        default=None,
        help=(
            "Proposal-time attacked planner for planner_aligned_joint_risk. "
            "When set, --ego_planner is only the actually executed planner, "
            "allowing the same fixed attack/safety proposal roles to be "
            "evaluated in separate reactive rollouts."
        ),
    )
    parser.add_argument("--deterministic_pairing", action="store_true")

    parser.add_argument(
        "--ego_planner",
        type=str,
        choices=["idm", "strive", "pdm_closed", "diffusion_planner", "flow_planner"],
        default=None,
        help="Split ego from traffic agents and control it with this planner",
    )
    parser.add_argument(
        "--generative_ego_only",
        action="store_true",
        help=(
            "Use the unified flow-matching output only for ego and replace "
            "every non-ego action with logged replay."
        ),
    )
    parser.add_argument(
        "--scene_source",
        type=str,
        choices=["logged_replay", "m2_unguided", "m2_guided", "frozen_hdf5"],
        default="m2_unguided",
        help="Source of non-ego traffic futures for planner evaluation",
    )
    parser.add_argument("--frozen_traffic_h5", type=str, default=None)
    parser.add_argument("--frozen_traffic_frame_offset", type=int, default=0)
    parser.add_argument(
        "--strategy_depth",
        type=int,
        choices=[0, 1],
        default=0,
        help="0 ignores candidate futures; 1 requires a candidate-aware planner",
    )
    parser.add_argument(
        "--strive_profile",
        type=str,
        choices=["default", "final_tuned_val_1"],
        default="final_tuned_val_1",
    )

    parser.add_argument(
        "--num_steps_intervention",
        type=int,
        default=0,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument("--five_class_request", type=str, default=None, help="Explicit fixed-identity guide request JSON (legacy guidance stays disabled)")
    parser.add_argument("--selection_manifest", type=str, default=None,
                        help="Website JSONL selection; exact scene, time and native vehicle identity")
    parser.add_argument("--selection_row", type=int, default=None,
                        help="Explicit zero-based record index in selection_manifest")
    args = parser.parse_args()

    cfg = SceneEditingConfig(registered_name=args.registered_name)

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key

    if args.planner_ckpt_dir is not None:
        cfg.ckpt.planner.ckpt_dir = args.planner_ckpt_dir
        cfg.ckpt.planner.ckpt_key = args.planner_ckpt_key

    if args.predictor_ckpt_dir is not None:
        cfg.ckpt.predictor.ckpt_dir = args.predictor_ckpt_dir
        cfg.ckpt.predictor.ckpt_key = args.predictor_ckpt_key

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.seed is not None:
        cfg.seed = args.seed
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)
    
    # add eval_class into the results_dir
    # cfg.results_dir = os.path.join(cfg.results_dir, cfg.eval_class)

    if args.env is not None:
        cfg.env = args.env
    else:
        assert cfg.env is not None

    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]
    if "ui" in cfg.edits.editing_source:
        # can only handle one scene with UI
        cfg.num_scenes_per_batch = 1

    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]

    cfg.pop("nusc")
    cfg.pop("trajdata")

    # Apply runtime data/evaluation overrides after copying the env-specific
    # block; applying them earlier would be silently overwritten above.
    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path
        if "nusc_trainval" in cfg.trajdata_data_dirs:
            cfg.trajdata_data_dirs["nusc_trainval"] = args.dataset_path

    if args.eval_scenes_file is not None:
        with open(args.eval_scenes_file, "r") as f:
            scene_manifest = json.load(f)
        if isinstance(scene_manifest, dict):
            scene_manifest = scene_manifest["scene_indices"]
        if not isinstance(scene_manifest, list) or len(scene_manifest) == 0:
            raise ValueError("eval_scenes_file must contain a non-empty scene-index list")
        cfg.eval_scenes = [int(scene_index) for scene_index in scene_manifest]
        cfg.num_scenes_to_evaluate = len(cfg.eval_scenes)

    if args.num_scenes_to_evaluate is not None:
        if args.num_scenes_to_evaluate <= 0:
            raise ValueError("num_scenes_to_evaluate must be positive")
        if args.num_scenes_to_evaluate > len(cfg.eval_scenes):
            raise ValueError("num_scenes_to_evaluate exceeds the fixed scene list")
        cfg.num_scenes_to_evaluate = args.num_scenes_to_evaluate
        cfg.eval_scenes = cfg.eval_scenes[:args.num_scenes_to_evaluate]

    if args.num_simulation_steps is not None:
        if args.num_simulation_steps <= 0:
            raise ValueError("num_simulation_steps must be positive")
        cfg.num_simulation_steps = args.num_simulation_steps

    if args.disable_guidance or args.scene_source in ["logged_replay", "m2_unguided", "frozen_hdf5"]:
        cfg.apply_guidance = False
        cfg.edits.editing_source = ["none"]
        cfg.policy.guide_as_filter_only = True
    elif args.scene_source == "m2_guided":
        cfg.apply_guidance = True

    if args.eval_num_action_samples is not None:
        if args.eval_class != "CCFlowMatching":
            raise ValueError("eval_num_action_samples requires CCFlowMatching")
        if args.scene_source != "m2_unguided":
            raise ValueError("eval_num_action_samples requires m2_unguided")
        if args.interaction_intervention != "none":
            raise ValueError(
                "eval_num_action_samples is only for the unmodified M2 control"
            )
        if args.eval_num_action_samples < 2:
            raise ValueError("eval_num_action_samples must be at least two")
        cfg.policy.num_action_samples = int(args.eval_num_action_samples)

    if args.guidance_method in {
        "safesim", "planner_hcs_safesim", "planner_hcs_coalition"
    }:
        if not cfg.apply_guidance or args.scene_source != "m2_guided":
            raise ValueError("SAFE-SIM baseline requires m2_guided")
        if args.ego_planner is None or args.strategy_depth != 0:
            raise ValueError("SAFE-SIM baseline requires a frozen K=0 ego planner")
        if (
            args.guidance_method in {
                "planner_hcs_safesim", "planner_hcs_coalition"
            }
            and args.target_agent_idx is not None
        ):
            raise ValueError(
                "planner-aware HCS selects the target causally; "
                "target_agent_idx must be omitted"
            )
        if args.planner_hcs_start_step < 0:
            raise ValueError("planner_hcs_start_step must be non-negative")
        if (
            args.planner_hcs_end_step is not None
            and args.planner_hcs_end_step <= args.planner_hcs_start_step
        ):
            raise ValueError(
                "planner_hcs_end_step must exceed planner_hcs_start_step"
            )
        if args.planner_hcs_arrival_weight < 0.0:
            raise ValueError("planner_hcs_arrival_weight must be non-negative")
        if (
            args.planner_hcs_velocity_trust_max_rms <= 0.0
            or args.planner_hcs_velocity_trust_max_linf <= 0.0
        ):
            raise ValueError(
                "planner HCS velocity trust bounds must be positive"
            )
        if args.planner_hcs_semantic_control == "actual":
            if args.planner_hcs_semantic_plan_json is not None:
                raise ValueError(
                    "actual HCS semantics cannot accept a donor plan"
                )
        else:
            if args.guidance_method != "planner_hcs_safesim":
                raise ValueError(
                    "matched-shuffled semantics is a single-target B3a control"
                )
            if args.planner_hcs_semantic_plan_json is None:
                raise ValueError(
                    "matched-shuffled HCS semantics requires a donor plan"
                )
        if args.guidance_method == "planner_hcs_coalition":
            if not (
                1 <= args.planner_hcs_coalition_min_blockers
                <= args.planner_hcs_coalition_max_blockers <= 2
            ):
                raise ValueError(
                    "coalition blockers must satisfy 1 <= min <= max <= 2"
                )
            if (
                args.planner_hcs_coalition_corridor_offset_m <= 0.0
                or args.planner_hcs_coalition_max_reach_error_m <= 0.0
            ):
                raise ValueError("coalition reach geometry must be positive")
        cfg.policy.num_action_samples = 20
        cfg.guidance_optimization_params["lr"] = 0.2

    if args.typed_collision_type is not None:
        if args.guidance_method != "typed_projection":
            raise ValueError(
                "typed_collision_type requires typed_projection"
            )
        if args.scene_source != "m2_guided":
            raise ValueError("typed projection requires m2_guided")
        if args.target_agent_idx is None or args.ego_planner is None:
            raise ValueError("typed projection requires target_agent_idx and ego_planner")
        cfg.apply_guidance = True
        cfg.policy.num_action_samples = 20

    if args.interaction_intervention in ["semantic_assertive_token", "edge_cut", "full_visibility_cut", "edge_delay", "future_intent_hold", "yield_belief", "correlated_flow_prior", "native_joint_mode", "conditional_fm_source", "marginal_tail", "joint_constrained_risk", "planner_aligned_joint_risk", "correlated_joint_risk", "future_intent_joint_risk", "yield_belief_joint_risk", "pcn_joint_risk", "persistent_copula_joint_risk"] and cfg.apply_guidance:
        raise ValueError("semantic edge interventions require guidance to be disabled")

    if args.semantic_num_action_samples is not None:
        semantic_interventions = {
            "semantic_assertive_token", "edge_cut", "full_visibility_cut", "edge_delay",
            "future_intent_hold", "yield_belief", "reaction_delay",
        }
        if args.interaction_intervention not in semantic_interventions:
            raise ValueError(
                "semantic_num_action_samples requires a conditioning-only "
                "semantic intervention"
            )
        if args.semantic_num_action_samples < 2:
            raise ValueError("semantic_num_action_samples must be at least two")
        cfg.policy.num_action_samples = int(args.semantic_num_action_samples)

    if args.interaction_intervention == "native_joint_mode":
        if args.native_joint_num_samples < 2:
            raise ValueError("native joint sample bank must be at least two")
        if args.native_joint_sample_idx is None:
            raise ValueError("native_joint_mode requires native_joint_sample_idx")
        if not 0 <= args.native_joint_sample_idx < args.native_joint_num_samples:
            raise ValueError("native_joint_sample_idx exceeds native joint sample bank")
        cfg.policy.num_action_samples = int(args.native_joint_num_samples)

    if args.interaction_intervention == "conditional_fm_source":
        if args.scene_source != "m2_unguided":
            raise ValueError("Conditional-FM source control requires m2_unguided")
        if args.ego_planner is None or args.marginal_tail_target_planner is None:
            raise ValueError(
                "Conditional-FM requires --ego_planner and "
                "--marginal_tail_target_planner"
            )
        if args.target_agent_idx is None or args.target_agent_idx == 0:
            raise ValueError("Conditional-FM requires non-ego target_agent_idx")
        if args.ego_planner == args.marginal_tail_target_planner:
            raise ValueError("attacked and safety planners must differ")
        if args.conditional_flow_num_steps < 1:
            raise ValueError("conditional_flow_num_steps must be positive")
        if not 0.0 <= args.conditional_flow_optimization_start_fraction < 1.0:
            raise ValueError(
                "conditional_flow_optimization_start_fraction must be in [0,1)"
            )
        if args.conditional_flow_global_start_step < 0:
            raise ValueError("conditional_flow_global_start_step must be non-negative")
        if (
            args.conditional_flow_global_end_step is not None
            and args.conditional_flow_global_end_step
            <= args.conditional_flow_global_start_step
        ):
            raise ValueError(
                "conditional_flow_global_end_step must exceed global start step"
            )
        if (
            args.conditional_flow_global_conflict_step is not None
            and args.conditional_flow_global_conflict_step < 0
        ):
            raise ValueError(
                "conditional_flow_global_conflict_step must be non-negative"
            )
        if args.conditional_flow_conflict_anchor_radius_steps < 0:
            raise ValueError(
                "conditional_flow_conflict_anchor_radius_steps must be non-negative"
            )
        cfg.policy.num_action_samples = 20

    if args.marginal_tail_num_native_samples is not None:
        if args.interaction_intervention not in {
            "marginal_tail", "joint_constrained_risk", "planner_aligned_joint_risk", "correlated_joint_risk", "future_intent_joint_risk", "yield_belief_joint_risk", "pcn_joint_risk", "persistent_copula_joint_risk"
        }:
            raise ValueError(
                "marginal_tail_num_native_samples requires marginal_tail"
            )
        if args.marginal_tail_num_native_samples < 2:
            raise ValueError("marginal-tail native sample bank must be at least two")
        cfg.policy.num_action_samples = int(args.marginal_tail_num_native_samples)

    if args.marginal_tail_attack_planner is not None:
        if args.interaction_intervention != "planner_aligned_joint_risk":
            raise ValueError(
                "marginal_tail_attack_planner requires planner_aligned_joint_risk"
            )
        if args.ego_planner is None or args.marginal_tail_target_planner is None:
            raise ValueError(
                "reactive planner-role decoupling requires executed, attack, "
                "and safety planners"
            )
        if args.marginal_tail_attack_planner == args.marginal_tail_target_planner:
            raise ValueError("proposal attack and safety planners must differ")
        if cfg.policy.num_action_samples != 20:
            raise ValueError("reactive dual-planner contract is frozen at exact K20")

    if args.interaction_intervention == "persistent_copula_joint_risk":
        if cfg.num_scenes_to_evaluate != 1 or cfg.num_scenes_per_batch != 1:
            raise ValueError(
                "persistent_copula_joint_risk currently requires one scene per process"
            )
        if cfg.policy.num_action_samples != 20:
            raise ValueError("persistent_copula_joint_risk is frozen at exact K20")

    if args.ego_planner is not None or args.generative_ego_only:
        # Closed-loop planner evaluation needs the executed trajectory, not
        # every M1 candidate. Avoid duplicating large candidate tensors in HDF5.
        cfg.save_action_samples = False
        planner_label = (
            "flow_ego" if args.generative_ego_only else args.ego_planner
        )
        cfg.name = "{}_{}_{}".format(args.scene_source, planner_label, cfg.name)
        if args.results_root_dir is not None:
            cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
        else:
            cfg.results_dir = os.path.join(os.path.dirname(cfg.results_dir), cfg.name)
        cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    if args.metric_ckpt_yaml is not None:
        with open(args.metric_ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    render_cfg = {
        'size' : args.render_size,
        'px_per_m' : args.render_px_per_m,
        'save_every_n_frames': args.save_every_n_frames,
        'draw_mode': args.draw_mode,
    }
    
    cfg["edits"]["heuristic_config"][0].update({"weight": args.w_collision})
    cfg["edits"]["heuristic_config"][1].update({"weight": args.w_offroad})

    cfg.update({"n_step_action": args.n_step_action})
    cfg.update({
        "ego_planner": args.ego_planner,
        "scene_source": args.scene_source,
        "guidance_method": args.guidance_method,
        "target_agent_idx": args.target_agent_idx,
        "reference_agent_idx": args.reference_agent_idx,
        "severity_mode": args.severity_mode,
        "desired_clearance": args.desired_clearance,
        "safesim_cause_weight": args.safesim_cause_weight,
        "safesim_collateral_weight": args.safesim_collateral_weight,
        "typed_collision_type": args.typed_collision_type,
        "typed_projection_activation_distance_m": args.typed_projection_activation_distance_m,
        "typed_projection_step_size": args.typed_projection_step_size,
        "typed_projection_max_delta": args.typed_projection_max_delta,
        "typed_projection_temporal_only": args.typed_projection_temporal_only,
        "typed_projection_native_joint_selection": args.typed_projection_native_joint_selection,
        "typed_projection_final_step_only": args.typed_projection_final_step_only,
        "typed_projection_min_collision_step": args.typed_projection_min_collision_step,
        "typed_projection_max_collision_step": args.typed_projection_max_collision_step,
        "typed_projection_phase_shift_steps": args.typed_projection_phase_shift_steps,
        "typed_projection_joint_continuity": args.typed_projection_joint_continuity,
        "typed_projection_joint_continuity_risk_slack": args.typed_projection_joint_continuity_risk_slack,
        "typed_projection_joint_continuity_max_error_m": args.typed_projection_joint_continuity_max_error_m,
        "typed_projection_trust_region": args.typed_projection_trust_region,
        "typed_projection_trust_max_rms": args.typed_projection_trust_max_rms,
        "typed_projection_trust_max_linf": args.typed_projection_trust_max_linf,
        "typed_projection_trust_max_energy_increase": args.typed_projection_trust_max_energy_increase,
        "typed_projection_trust_min_improvement": args.typed_projection_trust_min_improvement,
        "typed_projection_trust_tangent": args.typed_projection_trust_tangent,
        "typed_projection_trust_tangent_rank": args.typed_projection_trust_tangent_rank,
        "typed_projection_latent_shooting": args.typed_projection_latent_shooting,
        "typed_projection_latent_iterations": args.typed_projection_latent_iterations,
        "typed_projection_latent_lr": args.typed_projection_latent_lr,
        "typed_projection_latent_max_delta": args.typed_projection_latent_max_delta,
        "typed_projection_latent_prior_weight": args.typed_projection_latent_prior_weight,
        "typed_projection_latent_bank": args.typed_projection_latent_bank,
        "typed_projection_latent_bank_max_delta": args.typed_projection_latent_bank_max_delta,
        "typed_projection_latent_bank_prior_weight": args.typed_projection_latent_bank_prior_weight,
        "typed_projection_latent_bank_min_improvement": args.typed_projection_latent_bank_min_improvement,
        "typed_projection_latent_bank_max_nll_increase": args.typed_projection_latent_bank_max_nll_increase,
        "strategy_depth": args.strategy_depth,
        "strive_profile": args.strive_profile,
        "interaction_intervention": args.interaction_intervention,
        "eval_num_action_samples": args.eval_num_action_samples,
        "semantic_num_action_samples": args.semantic_num_action_samples,
        "edge_receiver_idx": args.edge_receiver_idx,
        "edge_sender_idx": args.edge_sender_idx,
        "edge_cut_pairs": args.edge_cut_pairs,
        "edge_bidirectional": args.edge_bidirectional,
        "edge_cut_start_step": args.edge_cut_start_step,
        "edge_cut_end_step": args.edge_cut_end_step,
        "semantic_assertive_receiver_idx": args.semantic_assertive_receiver_idx,
        "semantic_assertive_token_scale": args.semantic_assertive_token_scale,
        "semantic_assertive_start_step": args.semantic_assertive_start_step,
        "semantic_assertive_end_step": args.semantic_assertive_end_step,
        "semantic_assertive_overlay": args.semantic_assertive_overlay,
        "full_visibility_cut_start_step": args.full_visibility_cut_start_step,
        "full_visibility_cut_end_step": args.full_visibility_cut_end_step,
        "edge_delay_pairs": args.edge_delay_pairs,
        "edge_delay_steps": args.edge_delay_steps,
        "edge_delay_start_step": args.edge_delay_start_step,
        "edge_delay_end_step": args.edge_delay_end_step,
        "future_intent_hold_pairs": args.future_intent_hold_pairs,
        "future_intent_hold_start_step": args.future_intent_hold_start_step,
        "future_intent_hold_end_step": args.future_intent_hold_end_step,
        "yield_belief_pairs": args.yield_belief_pairs,
        "yield_belief_start_step": args.yield_belief_start_step,
        "yield_belief_end_step": args.yield_belief_end_step,
        "reaction_receiver_idx": args.reaction_receiver_idx,
        "reaction_delay_steps": args.reaction_delay_steps,
        "reaction_delay_start_step": args.reaction_delay_start_step,
        "reaction_delay_end_step": args.reaction_delay_end_step,
        "reaction_accel_bias_mps2": args.reaction_accel_bias_mps2,
        "correlated_flow_prior_pairs": args.correlated_flow_prior_pairs,
        "correlated_flow_prior_rho": args.correlated_flow_prior_rho,
        "correlated_flow_prior_lag_steps": args.correlated_flow_prior_lag_steps,
        "correlated_flow_prior_action_dims": args.correlated_flow_prior_action_dims,
        "correlated_flow_prior_start_step": args.correlated_flow_prior_start_step,
        "correlated_flow_prior_end_step": args.correlated_flow_prior_end_step,
        "persistent_copula_persistence": args.persistent_copula_persistence,
        "persistent_copula_temporal_rho": args.persistent_copula_temporal_rho,
        "persistent_copula_temporal_action_dims": (
            args.persistent_copula_temporal_action_dims
        ),
        "pcn_joint_flow_prior_beta": args.pcn_joint_flow_prior_beta,
        "pcn_joint_flow_prior_start_step": args.pcn_joint_flow_prior_start_step,
        "pcn_joint_flow_prior_end_step": args.pcn_joint_flow_prior_end_step,
        "native_joint_sample_idx": args.native_joint_sample_idx,
        "native_joint_start_step": args.native_joint_start_step,
        "native_joint_end_step": args.native_joint_end_step,
        "native_joint_num_samples": args.native_joint_num_samples,
        "conditional_flow_num_steps": args.conditional_flow_num_steps,
        "conditional_flow_source_disabled": args.conditional_flow_source_disabled,
        "conditional_flow_target_mean_shift": args.conditional_flow_target_mean_shift,
        "conditional_flow_target_mean_shift_max_abs": args.conditional_flow_target_mean_shift_max_abs,
        "conditional_flow_target_mean_shift_max_kl": args.conditional_flow_target_mean_shift_max_kl,
        "conditional_flow_target_mean_shift_basis": args.conditional_flow_target_mean_shift_basis,
        "conditional_flow_target_mean_shift_pulse_width": args.conditional_flow_target_mean_shift_pulse_width,
        "conditional_flow_target_mean_shift_online_cpa": args.conditional_flow_target_mean_shift_online_cpa,
        "conditional_flow_target_mean_shift_max_cpa_distance_m": args.conditional_flow_target_mean_shift_max_cpa_distance_m,
        "conditional_flow_target_velocity_residual": args.conditional_flow_target_velocity_residual,
        "conditional_flow_target_velocity_residual_arrival_aligned": args.conditional_flow_target_velocity_residual_arrival_aligned,
        "conditional_flow_target_velocity_residual_arrival_max_lateral_distance_m": args.conditional_flow_target_velocity_residual_arrival_max_lateral_distance_m,
        "conditional_flow_normalize_gradient": args.conditional_flow_normalize_gradient,
        "conditional_flow_step_size": args.conditional_flow_step_size,
        "conditional_flow_max_source_delta_rms": args.conditional_flow_max_source_delta_rms,
        "conditional_flow_max_actor_delta_rms": args.conditional_flow_max_actor_delta_rms,
        "conditional_flow_max_energy_increase": args.conditional_flow_max_energy_increase,
        "conditional_flow_persistence_steps": args.conditional_flow_persistence_steps,
        "conditional_flow_temperature_m": args.conditional_flow_temperature_m,
        "conditional_flow_reference_weight": args.conditional_flow_reference_weight,
        "conditional_flow_collateral_weight": args.conditional_flow_collateral_weight,
        "conditional_flow_source_proximity_weight": args.conditional_flow_source_proximity_weight,
        "conditional_flow_optimization_scope": args.conditional_flow_optimization_scope,
        "conditional_flow_optimization_start_fraction": args.conditional_flow_optimization_start_fraction,
        "conditional_flow_global_start_step": args.conditional_flow_global_start_step,
        "conditional_flow_global_end_step": args.conditional_flow_global_end_step,
        "conditional_flow_global_conflict_step": args.conditional_flow_global_conflict_step,
        "conditional_flow_conflict_anchor_radius_steps": args.conditional_flow_conflict_anchor_radius_steps,
        "marginal_tail_receiver_idx": args.marginal_tail_receiver_idx,
        "marginal_tail_receiver_indices": args.marginal_tail_receiver_indices,
        "marginal_tail_auto_receivers": args.marginal_tail_auto_receivers,
        "marginal_tail_sticky_receiver": args.marginal_tail_sticky_receiver,
        "marginal_tail_minimize_causal_change": args.marginal_tail_minimize_causal_change,
        "marginal_tail_start_step": args.marginal_tail_start_step,
        "marginal_tail_end_step": args.marginal_tail_end_step,
        "marginal_tail_collateral_margin_m": args.marginal_tail_collateral_margin_m,
        "marginal_tail_reference_safety_margin_m": args.marginal_tail_reference_safety_margin_m,
        "marginal_tail_min_improvement_m": args.marginal_tail_min_improvement_m,
        "marginal_tail_target_max_clearance_m": args.marginal_tail_target_max_clearance_m,
        "marginal_tail_progressive_target_barrier": args.marginal_tail_progressive_target_barrier,
        "marginal_tail_progressive_activation_max_clearance_m": args.marginal_tail_progressive_activation_max_clearance_m,
        "marginal_tail_progressive_risk_first": args.marginal_tail_progressive_risk_first,
        "marginal_tail_progressive_persistence_first": args.marginal_tail_progressive_persistence_first,
        "marginal_tail_progressive_dose_first": args.marginal_tail_progressive_dose_first,
        "marginal_tail_progressive_exposure_clearance_m": args.marginal_tail_progressive_exposure_clearance_m,
        "marginal_tail_baseline_min_clearance_m": args.marginal_tail_baseline_min_clearance_m,
        "marginal_tail_target_min_persistence_steps": args.marginal_tail_target_min_persistence_steps,
        "marginal_tail_max_speed_mps": args.marginal_tail_max_speed_mps,
        "marginal_tail_max_acc_mps2": args.marginal_tail_max_acc_mps2,
        "marginal_tail_max_jerk_mps3": args.marginal_tail_max_jerk_mps3,
        "marginal_tail_max_yaw_rate_rps": args.marginal_tail_max_yaw_rate_rps,
        "marginal_tail_max_lateral_acc_mps2": args.marginal_tail_max_lateral_acc_mps2,
        "marginal_tail_kinematic_horizon_steps": args.marginal_tail_kinematic_horizon_steps,
        "marginal_tail_max_agent_tail_fraction": args.marginal_tail_max_agent_tail_fraction,
        "marginal_tail_naturalness_profile": args.marginal_tail_naturalness_profile,
        "marginal_tail_num_native_samples": args.marginal_tail_num_native_samples,
        "marginal_tail_target_planner": args.marginal_tail_target_planner,
        "marginal_tail_attack_planner": args.marginal_tail_attack_planner,
        "deterministic_pairing": args.deterministic_pairing,
        "planner_hcs_start_step": args.planner_hcs_start_step,
        "planner_hcs_end_step": args.planner_hcs_end_step,
        "planner_hcs_arrival_weight": args.planner_hcs_arrival_weight,
        "planner_hcs_velocity_trust_max_rms": (
            args.planner_hcs_velocity_trust_max_rms
        ),
        "planner_hcs_velocity_trust_max_linf": (
            args.planner_hcs_velocity_trust_max_linf
        ),
        "planner_hcs_semantic_control": args.planner_hcs_semantic_control,
        "planner_hcs_semantic_plan_json": (
            args.planner_hcs_semantic_plan_json
        ),
        "planner_hcs_coalition_min_blockers": (
            args.planner_hcs_coalition_min_blockers
        ),
        "planner_hcs_coalition_max_blockers": (
            args.planner_hcs_coalition_max_blockers
        ),
        "planner_hcs_coalition_corridor_offset_m": (
            args.planner_hcs_coalition_corridor_offset_m
        ),
        "planner_hcs_coalition_max_reach_error_m": (
            args.planner_hcs_coalition_max_reach_error_m
        ),
        "generative_ego_only": args.generative_ego_only,
        "frozen_traffic_h5": args.frozen_traffic_h5,
        "frozen_traffic_frame_offset": args.frozen_traffic_frame_offset,
    })
    if args.selection_manifest is not None:
        args.part_control = True
        cfg.num_scenes_per_batch = 1
        cfg.num_scenes_to_evaluate = 1
        cfg.num_sim_per_scene = 1
        cfg.update({"selection_manifest": args.selection_manifest, "selection_row": args.selection_row})
    elif args.selection_row is not None:
        raise ValueError("--selection_row requires --selection_manifest")
    cfg.lock()
    run_scene_editor(
        cfg,
        save_cfg=True,
        data_to_disk=True,
        render_to_video=args.render,
        render_to_img=args.render_img,
        render_cfg=render_cfg,
        part_control=args.part_control, 
        controllable_agent=args.controllable_agent,
        human=args.human, 
        rand_agent=args.rand_agent, 
        num_steps_intervention=args.num_steps_intervention,
        ego_planner_name=args.ego_planner,
        scene_source=args.scene_source,
        strategy_depth=args.strategy_depth,
        strive_profile=args.strive_profile,
        guidance_method=args.guidance_method,
        selection_manifest=args.selection_manifest,
        selection_row=args.selection_row,
        five_class_request=args.five_class_request,
        target_agent_idx=args.target_agent_idx,
        reference_agent_idx=args.reference_agent_idx,
        severity_mode=args.severity_mode,
        desired_clearance=args.desired_clearance,
        safesim_cause_weight=args.safesim_cause_weight,
        safesim_collateral_weight=args.safesim_collateral_weight,
        planner_hcs_start_step=args.planner_hcs_start_step,
        planner_hcs_end_step=args.planner_hcs_end_step,
        planner_hcs_arrival_weight=args.planner_hcs_arrival_weight,
        planner_hcs_velocity_trust_max_rms=(
            args.planner_hcs_velocity_trust_max_rms
        ),
        planner_hcs_velocity_trust_max_linf=(
            args.planner_hcs_velocity_trust_max_linf
        ),
        planner_hcs_semantic_control=args.planner_hcs_semantic_control,
        planner_hcs_semantic_plan_json=(
            args.planner_hcs_semantic_plan_json
        ),
        planner_hcs_coalition_min_blockers=(
            args.planner_hcs_coalition_min_blockers
        ),
        planner_hcs_coalition_max_blockers=(
            args.planner_hcs_coalition_max_blockers
        ),
        planner_hcs_coalition_corridor_offset_m=(
            args.planner_hcs_coalition_corridor_offset_m
        ),
        planner_hcs_coalition_max_reach_error_m=(
            args.planner_hcs_coalition_max_reach_error_m
        ),
        typed_collision_type=args.typed_collision_type,
        typed_projection_activation_distance_m=args.typed_projection_activation_distance_m,
        typed_projection_step_size=args.typed_projection_step_size,
        typed_projection_max_delta=args.typed_projection_max_delta,
        typed_projection_temporal_only=args.typed_projection_temporal_only,
        typed_projection_native_joint_selection=args.typed_projection_native_joint_selection,
        typed_projection_final_step_only=args.typed_projection_final_step_only,
        typed_projection_min_collision_step=args.typed_projection_min_collision_step,
        typed_projection_max_collision_step=args.typed_projection_max_collision_step,
        typed_projection_phase_shift_steps=args.typed_projection_phase_shift_steps,
        typed_projection_joint_continuity=args.typed_projection_joint_continuity,
        typed_projection_joint_continuity_risk_slack=args.typed_projection_joint_continuity_risk_slack,
        typed_projection_joint_continuity_max_error_m=args.typed_projection_joint_continuity_max_error_m,
        typed_projection_trust_region=args.typed_projection_trust_region,
        typed_projection_trust_max_rms=args.typed_projection_trust_max_rms,
        typed_projection_trust_max_linf=args.typed_projection_trust_max_linf,
        typed_projection_trust_max_energy_increase=args.typed_projection_trust_max_energy_increase,
        typed_projection_trust_min_improvement=args.typed_projection_trust_min_improvement,
        typed_projection_trust_tangent=args.typed_projection_trust_tangent,
        typed_projection_trust_tangent_rank=args.typed_projection_trust_tangent_rank,
        typed_projection_latent_shooting=args.typed_projection_latent_shooting,
        typed_projection_latent_iterations=args.typed_projection_latent_iterations,
        typed_projection_latent_lr=args.typed_projection_latent_lr,
        typed_projection_latent_max_delta=args.typed_projection_latent_max_delta,
        typed_projection_latent_prior_weight=args.typed_projection_latent_prior_weight,
        typed_projection_latent_bank=args.typed_projection_latent_bank,
        typed_projection_latent_bank_max_delta=args.typed_projection_latent_bank_max_delta,
        typed_projection_latent_bank_prior_weight=args.typed_projection_latent_bank_prior_weight,
        typed_projection_latent_bank_min_improvement=args.typed_projection_latent_bank_min_improvement,
        typed_projection_latent_bank_max_nll_increase=args.typed_projection_latent_bank_max_nll_increase,
        interaction_intervention=args.interaction_intervention,
        edge_receiver_idx=args.edge_receiver_idx,
        edge_sender_idx=args.edge_sender_idx,
        edge_cut_pairs=args.edge_cut_pairs,
        edge_bidirectional=args.edge_bidirectional,
        edge_cut_start_step=args.edge_cut_start_step,
        edge_cut_end_step=args.edge_cut_end_step,
        semantic_assertive_receiver_idx=args.semantic_assertive_receiver_idx,
        semantic_assertive_token_scale=args.semantic_assertive_token_scale,
        semantic_assertive_start_step=args.semantic_assertive_start_step,
        semantic_assertive_end_step=args.semantic_assertive_end_step,
        semantic_assertive_overlay=args.semantic_assertive_overlay,
        full_visibility_cut_start_step=args.full_visibility_cut_start_step,
        full_visibility_cut_end_step=args.full_visibility_cut_end_step,
        edge_delay_pairs=args.edge_delay_pairs,
        edge_delay_steps=args.edge_delay_steps,
        edge_delay_start_step=args.edge_delay_start_step,
        edge_delay_end_step=args.edge_delay_end_step,
        future_intent_hold_pairs=args.future_intent_hold_pairs,
        future_intent_hold_start_step=args.future_intent_hold_start_step,
        future_intent_hold_end_step=args.future_intent_hold_end_step,
        yield_belief_pairs=args.yield_belief_pairs,
        yield_belief_start_step=args.yield_belief_start_step,
        yield_belief_end_step=args.yield_belief_end_step,
        reaction_receiver_idx=args.reaction_receiver_idx,
        reaction_delay_steps=args.reaction_delay_steps,
        reaction_delay_start_step=args.reaction_delay_start_step,
        reaction_delay_end_step=args.reaction_delay_end_step,
        reaction_accel_bias_mps2=args.reaction_accel_bias_mps2,
        correlated_flow_prior_pairs=args.correlated_flow_prior_pairs,
        correlated_flow_prior_rho=args.correlated_flow_prior_rho,
        correlated_flow_prior_lag_steps=args.correlated_flow_prior_lag_steps,
        correlated_flow_prior_action_dims=args.correlated_flow_prior_action_dims,
        correlated_flow_prior_start_step=args.correlated_flow_prior_start_step,
        correlated_flow_prior_end_step=args.correlated_flow_prior_end_step,
        persistent_copula_persistence=args.persistent_copula_persistence,
        persistent_copula_temporal_rho=args.persistent_copula_temporal_rho,
        persistent_copula_temporal_action_dims=(
            args.persistent_copula_temporal_action_dims
        ),
        pcn_joint_flow_prior_beta=args.pcn_joint_flow_prior_beta,
        pcn_joint_flow_prior_start_step=args.pcn_joint_flow_prior_start_step,
        pcn_joint_flow_prior_end_step=args.pcn_joint_flow_prior_end_step,
        native_joint_sample_idx=args.native_joint_sample_idx,
        native_joint_start_step=args.native_joint_start_step,
        native_joint_end_step=args.native_joint_end_step,
        native_joint_num_samples=args.native_joint_num_samples,
        conditional_flow_num_steps=args.conditional_flow_num_steps,
        conditional_flow_source_disabled=args.conditional_flow_source_disabled,
        conditional_flow_target_mean_shift=args.conditional_flow_target_mean_shift,
        conditional_flow_target_mean_shift_max_abs=args.conditional_flow_target_mean_shift_max_abs,
        conditional_flow_target_mean_shift_max_kl=args.conditional_flow_target_mean_shift_max_kl,
        conditional_flow_target_mean_shift_basis=args.conditional_flow_target_mean_shift_basis,
        conditional_flow_target_mean_shift_pulse_width=args.conditional_flow_target_mean_shift_pulse_width,
        conditional_flow_target_mean_shift_online_cpa=args.conditional_flow_target_mean_shift_online_cpa,
        conditional_flow_target_mean_shift_max_cpa_distance_m=args.conditional_flow_target_mean_shift_max_cpa_distance_m,
        conditional_flow_target_velocity_residual=args.conditional_flow_target_velocity_residual,
        conditional_flow_target_velocity_residual_arrival_aligned=args.conditional_flow_target_velocity_residual_arrival_aligned,
        conditional_flow_target_velocity_residual_arrival_max_lateral_distance_m=args.conditional_flow_target_velocity_residual_arrival_max_lateral_distance_m,
        conditional_flow_normalize_gradient=args.conditional_flow_normalize_gradient,
        conditional_flow_step_size=args.conditional_flow_step_size,
        conditional_flow_max_source_delta_rms=args.conditional_flow_max_source_delta_rms,
        conditional_flow_max_actor_delta_rms=args.conditional_flow_max_actor_delta_rms,
        conditional_flow_max_energy_increase=args.conditional_flow_max_energy_increase,
        conditional_flow_persistence_steps=args.conditional_flow_persistence_steps,
        conditional_flow_temperature_m=args.conditional_flow_temperature_m,
        conditional_flow_reference_weight=args.conditional_flow_reference_weight,
        conditional_flow_collateral_weight=args.conditional_flow_collateral_weight,
        conditional_flow_source_proximity_weight=args.conditional_flow_source_proximity_weight,
        conditional_flow_optimization_scope=args.conditional_flow_optimization_scope,
        conditional_flow_optimization_start_fraction=args.conditional_flow_optimization_start_fraction,
        conditional_flow_global_start_step=args.conditional_flow_global_start_step,
        conditional_flow_global_end_step=args.conditional_flow_global_end_step,
        conditional_flow_global_conflict_step=args.conditional_flow_global_conflict_step,
        conditional_flow_conflict_anchor_radius_steps=args.conditional_flow_conflict_anchor_radius_steps,
        marginal_tail_receiver_idx=args.marginal_tail_receiver_idx,
        marginal_tail_receiver_indices=args.marginal_tail_receiver_indices,
        marginal_tail_auto_receivers=args.marginal_tail_auto_receivers,
        marginal_tail_sticky_receiver=args.marginal_tail_sticky_receiver,
        marginal_tail_minimize_causal_change=args.marginal_tail_minimize_causal_change,
        marginal_tail_start_step=args.marginal_tail_start_step,
        marginal_tail_end_step=args.marginal_tail_end_step,
        marginal_tail_collateral_margin_m=args.marginal_tail_collateral_margin_m,
        marginal_tail_reference_safety_margin_m=args.marginal_tail_reference_safety_margin_m,
        marginal_tail_min_improvement_m=args.marginal_tail_min_improvement_m,
        marginal_tail_target_max_clearance_m=args.marginal_tail_target_max_clearance_m,
        marginal_tail_progressive_target_barrier=args.marginal_tail_progressive_target_barrier,
        marginal_tail_progressive_activation_max_clearance_m=args.marginal_tail_progressive_activation_max_clearance_m,
        marginal_tail_progressive_risk_first=args.marginal_tail_progressive_risk_first,
        marginal_tail_progressive_persistence_first=args.marginal_tail_progressive_persistence_first,
        marginal_tail_progressive_dose_first=args.marginal_tail_progressive_dose_first,
        marginal_tail_progressive_exposure_clearance_m=args.marginal_tail_progressive_exposure_clearance_m,
        marginal_tail_baseline_min_clearance_m=args.marginal_tail_baseline_min_clearance_m,
        marginal_tail_target_min_persistence_steps=args.marginal_tail_target_min_persistence_steps,
        marginal_tail_max_speed_mps=args.marginal_tail_max_speed_mps,
        marginal_tail_max_acc_mps2=args.marginal_tail_max_acc_mps2,
        marginal_tail_max_jerk_mps3=args.marginal_tail_max_jerk_mps3,
        marginal_tail_max_yaw_rate_rps=args.marginal_tail_max_yaw_rate_rps,
        marginal_tail_max_lateral_acc_mps2=args.marginal_tail_max_lateral_acc_mps2,
        marginal_tail_kinematic_horizon_steps=args.marginal_tail_kinematic_horizon_steps,
        marginal_tail_max_agent_tail_fraction=args.marginal_tail_max_agent_tail_fraction,
        marginal_tail_naturalness_profile_path=args.marginal_tail_naturalness_profile,
        marginal_tail_target_planner_name=args.marginal_tail_target_planner,
        marginal_tail_attack_planner_name=args.marginal_tail_attack_planner,
        deterministic_pairing=args.deterministic_pairing,
        generative_ego_only=args.generative_ego_only,
        frozen_traffic_h5=args.frozen_traffic_h5,
        frozen_traffic_frame_offset=args.frozen_traffic_frame_offset,
    )
