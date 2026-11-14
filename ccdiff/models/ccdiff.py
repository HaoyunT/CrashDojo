#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

from typing import Dict, List
from collections import OrderedDict
from copy import deepcopy
import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt

import tbsim.utils.tensor_utils as TensorUtils
import ccdiff.models.base_models as base_models
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.geometry_utils as GeoUtils


import numpy as np
from tbsim.utils.diffuser_utils.progress import Progress, Silent
from .diffuser_helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
    query_feature_grid,
    convert_state_to_state_and_action,
    unicyle_forward_dynamics,
    AgentHistoryEncoder,
    NeighborHistoryEncoder,
    SimpleNeighborHistoryEncoder,
    MapEncoder,
    angle_wrap_torch,
)
from .scenetemporal import (
    SceneTransformer,
    apply_directed_future_intent_hold,
    apply_directed_yield_belief,
)
from .temporal import TemporalMapUnet
import tbsim.dynamics as dynamics
from tbsim.utils.guidance_loss import verify_guidance_config_list, verify_constraint_config, apply_constraints, DiffuserGuidance
from ccdiff.utils.guidance_loss import PerturbationGuidance, PartialPerturbationGuidance
from trajdata.utils.arr_utils import angle_wrap
from tbsim.utils.trajdata_utils import extract_data_batch_for_guidance
from .conditional_flow_control import optimize_complete_joint_source
from .target_source_mean_shift import (
    TargetSourceMeanShiftConfig,
    apply_target_source_mean_shift,
    estimate_online_planner_target_cpa,
)
from .target_velocity_residual import (
    TargetVelocityResidualConfig,
    apply_target_velocity_residual,
)
from .conditional_fm_sampler_adapter import (
    PlannerSeparationObjectiveConfig,
    planner_separation_source_objective,
    selected_joint_background_mask,
    verify_noisy_edge_gradient_contract,
)
from .persistent_copula_prior import (
    PersistentCopulaConfig,
    PersistentCopulaSource,
)


def inject_planner_reference_future_states(
    future_positions,
    future_speeds,
    future_yaws,
    planner_positions,
    planner_yaws,
    planner_mask,
    dt=0.1,
):
    """Replace only planner actor futures before directed-edge construction.

    ``future_*`` contains one model state per FM sample in each actor's own
    local frame. ``planner_*`` contains the actual ego-plan rows in the same
    local-frame convention, but only once per scene.  Repeating those rows
    across K before calling ``get_neighbor_relative_states`` makes every
    receiver condition on the trajectory that the rollout wrapper will really
    execute.  No sampled background row or output trajectory is modified.

    Returns cloned tensors and a repeated boolean sender mask.  The helper is
    deliberately independent of the model so the frame/batch contract can be
    regression-tested without loading a checkpoint.
    """
    if future_positions.ndim != 4 or future_positions.shape[-1] != 2:
        raise ValueError("future positions must have shape [B*K,M,T,2]")
    if future_speeds.shape != future_positions.shape[:-1]:
        raise ValueError("future speeds must have shape [B*K,M,T]")
    if future_yaws.shape != future_positions.shape[:-1] + (1,):
        raise ValueError("future yaws must have shape [B*K,M,T,1]")
    if planner_positions.ndim != 4 or planner_positions.shape[-1] != 2:
        raise ValueError("planner positions must have shape [B,M,T,2]")
    if planner_yaws.shape != planner_positions.shape[:-1] + (1,):
        raise ValueError("planner yaws must have shape [B,M,T,1]")
    if planner_mask.shape != planner_positions.shape[:2]:
        raise ValueError("planner mask must have shape [B,M]")
    if planner_positions.shape[1:3] != future_positions.shape[1:3]:
        raise ValueError("planner and FM actor/horizon dimensions disagree")
    batch_size = int(planner_positions.shape[0])
    sample_batch = int(future_positions.shape[0])
    if batch_size <= 0 or sample_batch % batch_size != 0:
        raise ValueError("FM sample batch is not divisible by scene batch")
    if float(dt) <= 0.0:
        raise ValueError("planner future dt must be positive")

    planner_mask = planner_mask.bool()
    if not torch.all(planner_mask.sum(dim=1) == 1):
        raise ValueError("each scene must provide exactly one executed planner row")
    repeats = sample_batch // batch_size
    repeated_mask = planner_mask.repeat_interleave(repeats, dim=0)
    repeated_positions = planner_positions.repeat_interleave(repeats, dim=0)
    repeated_yaws = planner_yaws.repeat_interleave(repeats, dim=0)

    conditioned_positions = torch.where(
        repeated_mask[..., None, None], repeated_positions, future_positions
    )
    conditioned_yaws = torch.where(
        repeated_mask[..., None, None], repeated_yaws, future_yaws
    )

    # Planner positions are future displacements in the planner actor's local
    # frame.  The first speed uses the current local origin; later speeds use
    # finite differences.  Only masked planner rows replace model speeds.
    origin = torch.zeros_like(repeated_positions[:, :, :1])
    deltas = torch.diff(
        torch.cat((origin, repeated_positions), dim=2), dim=2
    )
    repeated_speeds = torch.linalg.vector_norm(deltas, dim=-1) / float(dt)
    conditioned_speeds = torch.where(
        repeated_mask[..., None], repeated_speeds, future_speeds
    )
    return (
        conditioned_positions,
        conditioned_speeds,
        conditioned_yaws,
        repeated_mask,
    )

class CCDiffuserModel(nn.Module):
    """CCDiff model for closed-loop scenario generation.
    
    data usage from data_batch
    # "history_positions" (coord), "history_yaws" (coord), "history_speeds", "extent", "history_availabilities"
    # "curr_speed"
    # "target_positions" (coord), "target_yaws" (coord), "target_availabilities"
    
    # If modeling map
    # ("image" and "maps") or "closest_lane_point" (coord)

    # If modeling neighbors:
    # "all_other_agents_history_positions" (coord), "all_other_agents_history_yaws" (coord), "all_other_agents_history_speeds", "all_other_agents_extents", "all_other_agents_history_availabilities"
    """

    def __init__(
            self,
            map_encoder_model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            map_grid_feature_dim: int,
            
            diffuser_model_arch: str,
            horizon: int,

            observation_dim: int, 
            action_dim: int,

            output_dim: int,

            # curr_state_feature_dim = 64,
            rasterized_map: bool = True,
            use_map_feat_global: bool = True,
            use_map_feat_grid: bool = False,
            rasterized_hist: bool = True,
            hist_num_frames: bool = 31,
            hist_feature_dim: bool = 128,

            n_timesteps: int = 100,
            
            loss_type: str = 'l2', 
            clip_denoised: bool = False, 
            
            predict_epsilon: bool = True,
            action_weight: float = 1.0, 
            loss_discount: float = 1.0, 
            loss_weights = None,
            loss_decay_rates = {},

            dynamics_type = None,
            dynamics_kwargs = {},

            action_loss_only = False,

            diffuser_input_mode: str = 'state_and_action',
            use_reconstructed_state: bool = False,

            use_conditioning: bool = True,
            cond_fill_value: float = -1.0,

            # norm info is ([add_coeffs, div_coeffs])
            diffuser_norm_info=([-17.5, 0, 0, 0, 0, 0],[22.5, 10, 40, 3.14, 500, 31.4]),
            agent_hist_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
            neighbor_hist_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
            neighbor_fut_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),

            agent_hist_embed_method: str = 'mlp',
            neigh_hist_embed_method=None,
            map_embed_method: str = 'transformer',
            interaction_edge_speed_repr: str = 'abs_speed',
            mask_social_interaction: bool = False,
            mask_edge: bool = False,
            neighbor_inds=[0,1,2,3],
            edge_attr_separation=[],
            motion_dist: str = 'ttc', 
            social_attn_radius: int = 50,
            use_last_hist_step: bool = False,
            use_noisy_fut_edge: bool = False,
            semantic_visibility_dropout_p: float = 0.0,
            semantic_visibility_dropout_radius: float = 30.0,
            use_const_speed_edge: bool = False,
            normalize_rel_states: bool = True,
            all_interactive_social: bool = False,
            mask_time: bool = True,
            layer_num_per_edge_decoder: int = 1,
            attn_combination: str = 'sum',
            single_cond_feat: bool = False,
            disable_control_on_stationary: bool = False,
            coordinate: str = 'agent_centric',
            controllable_agent=-1,
            dt=0.1,
    ) -> None:
        super().__init__()

        self.global_t = 0

        # dimension parameters
        self.dt = dt
        self.horizon = horizon
        self.hist_num_frames = hist_num_frames
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.output_dim = output_dim

        self.hist_feature_dim = hist_feature_dim

        # ------ architectures ------
        # ['TemporalMapUnet', 'SceneTransformer']
        self.diffuser_model_arch = diffuser_model_arch
        # decide how the context is integrated
        self.agent_hist_embed_method = agent_hist_embed_method
        self.neigh_hist_embed_method = neigh_hist_embed_method
        self.map_embed_method = map_embed_method
        self.interaction_edge_speed_repr = interaction_edge_speed_repr
        self.normalize_rel_states = normalize_rel_states
        # this field is used only when SceneTransformer is used
        self.mask_social_interaction = mask_social_interaction
        # this field is used only when agent_hist_embed_method == 'mlp' or map_embed_method == 'cnn'
        self.single_cond_feat = single_cond_feat
        self.mask_edge = mask_edge
        self.neighbor_inds = neighbor_inds
        self.edge_attr_separation = edge_attr_separation
        
        self.motion_dist = motion_dist
        self.social_attn_radius = social_attn_radius
        self.use_last_hist_step = use_last_hist_step
        self.use_noisy_fut_edge = use_noisy_fut_edge
        self.semantic_visibility_dropout_p = float(
            semantic_visibility_dropout_p
        )
        self.semantic_visibility_dropout_radius = float(
            semantic_visibility_dropout_radius
        )
        if not 0.0 <= self.semantic_visibility_dropout_p <= 1.0:
            raise ValueError(
                "semantic_visibility_dropout_p must be in [0, 1]"
            )
        if self.semantic_visibility_dropout_radius <= 0.0:
            raise ValueError(
                "semantic_visibility_dropout_radius must be positive"
            )
        self.use_const_speed_edge = use_const_speed_edge
        self.all_interactive_social = all_interactive_social
        self.mask_time = mask_time
        self.layer_num_per_edge_decoder = layer_num_per_edge_decoder
        self.attn_combination = attn_combination

        self.disable_control_on_stationary = disable_control_on_stationary
        self.stationary_mask = None
        self.coordinate = coordinate

        assert not self.single_cond_feat or (self.single_cond_feat and (self.agent_hist_embed_method == 'mlp' or self.map_embed_method == 'cnn'))
        if self.single_cond_feat:
            cond_in_feat_size = 0

        # this applies to map and past NEIGHBOR conditioning only
        #       curr state or past ego trajecotry are always given
        self.use_conditioning = use_conditioning
        # for test-time classifier-free guidance, if desired
        self.cond_fill_value = cond_fill_value 

        self.rasterized_map = rasterized_map
        self.rasterized_hist = rasterized_hist

        self.agent_hist_norm_info = agent_hist_norm_info
        self.neighbor_hist_norm_info = neighbor_hist_norm_info
        self.neighbor_fut_norm_info = neighbor_fut_norm_info

        # agent history encoding (mlp)
        if self.agent_hist_embed_method == 'mlp':
            self.agent_hist_encoder = None
            if not self.rasterized_hist:
                # ego history is ALWAYS used as conditioning
                self.agent_hist_encoder = AgentHistoryEncoder(hist_num_frames,
                                                            out_dim=self.hist_feature_dim,
                                                            use_norm=True,
                                                            norm_info=agent_hist_norm_info)
            if self.single_cond_feat:
                cond_in_feat_size += self.hist_feature_dim

        # neighbor history encoding (mlp)
        if self.neigh_hist_embed_method == 'mlp':
            if self.use_conditioning and not self.rasterized_hist:
                self.neighbor_hist_encoder = NeighborHistoryEncoder(hist_num_frames, out_dim=self.hist_feature_dim, use_norm=True, norm_info=self.neighbor_hist_norm_info)
                if self.single_cond_feat:
                    cond_in_feat_size += self.hist_feature_dim
        # neighbor info is encoded through edge and fut input
        elif self.neigh_hist_embed_method == 'interaction_edge_and_input':
            # for encoding neighbor future
            # the input dimension assumes relvel
            self.neighbor_fut_encoder = SimpleNeighborHistoryEncoder(input_dim=9, out_dim=32, use_norm=True)



        # map encoding (cnn)
        if self.map_embed_method in ['cnn', 'cnn_local_patch']:
            self.map_encoder = None
            self.use_map_feat_global = use_map_feat_global
            self.use_map_feat_grid = use_map_feat_grid
            self.input_image_shape = input_image_shape
            if self.use_conditioning and (self.rasterized_map or self.rasterized_hist):
                self.map_encoder = MapEncoder(
                    model_arch=map_encoder_model_arch,
                    input_image_shape=input_image_shape,
                    global_feature_dim=map_feature_dim if self.use_map_feat_global else None,
                    grid_feature_dim=map_grid_feature_dim if self.use_map_feat_grid else None,
                )
                if self.single_cond_feat:
                    cond_in_feat_size += map_feature_dim

        # if apply mlp to process all the conditions
        if self.single_cond_feat:
            # hacky way to get the output feature size
            if self.diffuser_model_arch == 'SceneTransformer':
               cond_out_feat_size = 128
            elif self.diffuser_model_arch == 'TemporalMapUnet':
                cond_out_feat_size = 256
            else:
                raise NotImplementedError
            
            combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, cond_out_feat_size, cond_out_feat_size)
            self.process_cond_mlp = base_models.MLP(cond_in_feat_size,
                                                    cond_out_feat_size,
                                                    combine_layer_dims,
                                                    normalization=True)


        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self._create_dynamics()
        
        # ----- diffuser -----

        # TBD: include in algo_config
        self.use_target_availabilities = True

        self.action_loss_only = action_loss_only
        
        # x, y, vel, yaw, acc, yawvel
        assert len(diffuser_norm_info) == 2
        norm_add_coeffs = diffuser_norm_info[0]
        norm_div_coeffs = diffuser_norm_info[1]
        assert len(norm_add_coeffs) == 6
        assert len(norm_div_coeffs) == 6
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32')       

        print('self.add_coeffs', self.add_coeffs)
        print('self.div_coeffs', self.div_coeffs)


        self.use_reconstructed_state = use_reconstructed_state
        self.diffuser_input_mode = diffuser_input_mode
        
        if diffuser_input_mode == 'state':
            self.default_chosen_inds = [0, 1, 3]
        elif diffuser_input_mode == 'action':
            self.default_chosen_inds = [4, 5]
        elif diffuser_input_mode in ['state_and_action', 'state_and_action_no_dyn']:
            self.default_chosen_inds = [0, 1, 2, 3, 4, 5]
        else:
            raise

        if self.diffuser_model_arch == "SceneTransformer":
            # transition_in_dim = self.transition_dim
            # if self.use_map_feat_grid and self.map_encoder is not None:
            #     # will be appending map features to each step of trajectory
            #     transition_in_dim += map_grid_feature_dim
            self.model = SceneTransformer(
                                    d_k=128,
                                    d_edge=self.hist_feature_dim, 
                                    L_enc=2, 
                                    dropout=0.0, 
                                    k_attr=7,
                                    neighbor_inds=self.neighbor_inds,
                                    edge_attr_separation=self.edge_attr_separation,
                                    map_attr=2, 
                                    num_heads=16, 
                                    L_dec=2,
                                    tx_hidden_size=384, 
                                    use_map_lanes=True,

                                    transition_dim=self.transition_dim,
                                    output_dim=output_dim,
                                    agent_hist_embed_method=self.agent_hist_embed_method,
                                    neigh_hist_embed_method=self.neigh_hist_embed_method,
                                    map_embed_method=self.map_embed_method,
                                    interaction_edge_speed_repr=self.interaction_edge_speed_repr,
                                    single_cond_feat=self.single_cond_feat,
                                    mask_social_interaction=self.mask_social_interaction,
                                    mask_edge=self.mask_edge,
                                    social_attn_radius=self.social_attn_radius,
                                    all_interactive_social=self.all_interactive_social,
                                    mask_time=self.mask_time,
                                    layer_num_per_edge_decoder=self.layer_num_per_edge_decoder,
                                    attn_combination=self.attn_combination,
                                    )
        elif self.diffuser_model_arch == "TemporalMapUnet":
            self.model = TemporalMapUnet(horizon=horizon,
                                      transition_dim=self.transition_dim,
                                      cond_dim=256,
                                      output_dim=self.output_dim,
                                      dim=32,
                                      dim_mults=(2,4,8),
                                      diffuser_building_block='concat')
        else:
            print('unknown diffuser_model_arch:', self.diffuser_model_arch)
            raise
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # calculations for class-free guidance
        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(alphas_cumprod / (1.0 - alphas_cumprod))
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(1. - alphas_cumprod)
        # self.register_buffer('sqrt_alphas_over_one_minus_alphas_cumprod', torch.sqrt(alphas_cumprod / (1.0 - alphas_cumprod)))
        # self.register_buffer('sqrt_recip_one_minus_alphas_cumprod', 1.0 / torch.sqrt(1. - alphas_cumprod))

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        recon_loss_weights = self.get_loss_weights(action_weight, loss_discount)
        self.loss_fn = Losses[loss_type](recon_loss_weights, self.action_dim, loss_decay_rates)
        self.loss_decay_rates = loss_decay_rates

        # for guided sampling
        # This is 1 by default
        self.stride = 1
        self.apply_guidance_output = False
        self.guidance_optimization_params = None
        self.flow_velocity_guidance_config = None
        if self.diffuser_input_mode in ['action', 'state_and_action']:
            self.transform_params = {'scaled_input':True, 'scaled_output':True}
            self.transform = self.state_action_grad_inner_transform
        elif self.diffuser_input_mode in ['state_and_action_no_dyn']:
            self.transform_params = {'scaled_input':True, 'scaled_output':True}
            self.transform = self.state_action_no_dyn_grad_inner_transform
        else:
            self.transform_params = {}
            self.transform = self.state_grad_inner_transform
        
        self.current_constraints = None
        # Directed test-time perception interventions. Entries are
        # (receiver_index, sender_index, bidirectional) in padded scene order.
        self.interaction_edge_cut = None
        # A stronger visibility intervention that masks the same directed edge
        # in the history encoder, edge-aware decoder, and any later ordinary
        # social decoder.  It remains a conditioning intervention: no output
        # trajectory coordinates are edited or projected.
        self.interaction_full_visibility_cut = None
        # Directed test-time perception delays. Entries are
        # (receiver_index, sender_index, delay_steps, bidirectional).
        self.interaction_edge_delay = None
        # Directed semantic intervention: keep real history but replace the
        # selected receiver<-sender future interaction belief with the final
        # observed relative state throughout denoising.
        self.interaction_future_intent_hold = None
        # Directed semantic intervention: the receiver keeps real history but
        # internally assumes the sender will smoothly decelerate to yield.
        # This changes relation conditioning only, never sampled/executed paths.
        self.interaction_yield_belief = None
        # M2-e behavior token. It changes only SceneTransformer conditioning;
        # no source sample, action, state, or trajectory coordinate is edited.
        self.semantic_assertive_receiver_indices = None
        self.semantic_assertive_token_scale = 1.0
        # Optional test-time copula on the FM source prior.  It changes only
        # cross-agent dependence: every individual action coordinate remains
        # standard normal, and the learned vector field/output trajectory are
        # never projected or edited.
        self.marginal_preserving_correlated_flow_prior = None
        # Optional preconditioned-Crank--Nicolson neighbourhood bank in the
        # *complete joint* FM source.  Sample 0 is the ordinary independent
        # policy draw.  Every proposal has the exact original N(0, I) source
        # law, but is correlated with sample 0, which enables local search
        # without changing any single-agent marginal or projecting outputs.
        self.pcn_joint_flow_prior_beta = None
        # Stateful FM source proposal. Sample 0 remains the byte-identical
        # independent policy baseline. Proposal modes keep a stationary
        # standard-Gaussian marginal while their overlapping future source is
        # correlated across receding-horizon calls; an optional Gaussian
        # copula changes only cross-agent dependence.
        self.persistent_copula_flow_prior = None
        self._persistent_copula_replan_call = 0
        self._persistent_copula_last_shape = None
        self.controllable_agent = controllable_agent
        # wrapper for optimization using current_guidance
        self.current_perturbation_guidance = PartialPerturbationGuidance(self.transform, self.transform_params, self.scale_traj, self.descale_traj, available_idx=None)
        # --------------------------------------------------------------------------------------
        # print("after settting: ", self.current_perturbation_guidance.available_idx, self.current_perturbation_guidance)

    #------------------------------------------ guidance utils ------------------------------------------#
    def set_interaction_edge_cut(self, receiver_idx, sender_idx, bidirectional=False):
        self.set_interaction_edge_cuts(
            [(receiver_idx, sender_idx)], bidirectional=bidirectional
        )

    def set_interaction_edge_cuts(self, edge_pairs, bidirectional=False):
        parsed = []
        for receiver_idx, sender_idx in edge_pairs:
            receiver_idx = int(receiver_idx)
            sender_idx = int(sender_idx)
            if receiver_idx < 0 or sender_idx < 0 or receiver_idx == sender_idx:
                raise ValueError("edge receiver/sender must be distinct non-negative indices")
            parsed.append((receiver_idx, sender_idx, bool(bidirectional)))
        if not parsed:
            raise ValueError("at least one directed interaction edge is required")
        self.interaction_edge_cut = parsed

    def clear_interaction_edge_cut(self):
        self.interaction_edge_cut = None

    def set_interaction_full_visibility_cuts(
        self, edge_pairs, bidirectional=False
    ):
        parsed = []
        for receiver_idx, sender_idx in edge_pairs:
            receiver_idx = int(receiver_idx)
            sender_idx = int(sender_idx)
            if receiver_idx < 0 or sender_idx < 0 or receiver_idx == sender_idx:
                raise ValueError(
                    "visibility receiver/sender must be distinct non-negative indices"
                )
            parsed.append((receiver_idx, sender_idx, bool(bidirectional)))
        if not parsed:
            raise ValueError("at least one full-visibility edge is required")
        self.interaction_full_visibility_cut = parsed

    def clear_interaction_full_visibility_cut(self):
        self.interaction_full_visibility_cut = None

    def set_interaction_edge_delays(self, edge_delays, bidirectional=False):
        parsed = []
        for receiver_idx, sender_idx, delay_steps in edge_delays:
            receiver_idx = int(receiver_idx)
            sender_idx = int(sender_idx)
            delay_steps = int(delay_steps)
            if receiver_idx < 0 or sender_idx < 0 or receiver_idx == sender_idx:
                raise ValueError("edge receiver/sender must be distinct non-negative indices")
            if delay_steps <= 0:
                raise ValueError("edge delay must be a positive number of model steps")
            parsed.append(
                (receiver_idx, sender_idx, delay_steps, bool(bidirectional))
            )
        if not parsed:
            raise ValueError("at least one directed interaction edge delay is required")
        self.interaction_edge_delay = parsed

    def clear_interaction_edge_delay(self):
        self.interaction_edge_delay = None

    def set_interaction_future_intent_holds(
        self, edge_pairs, bidirectional=False
    ):
        parsed = []
        for receiver_idx, sender_idx in edge_pairs:
            receiver_idx = int(receiver_idx)
            sender_idx = int(sender_idx)
            if receiver_idx < 0 or sender_idx < 0 or receiver_idx == sender_idx:
                raise ValueError(
                    "future-intent receiver/sender must be distinct non-negative indices"
                )
            parsed.append((receiver_idx, sender_idx, bool(bidirectional)))
        if not parsed:
            raise ValueError("at least one future-intent edge is required")
        self.interaction_future_intent_hold = parsed

    def clear_interaction_future_intent_hold(self):
        self.interaction_future_intent_hold = None

    def set_interaction_yield_beliefs(self, edge_beliefs, bidirectional=False):
        parsed = []
        for receiver_idx, sender_idx, deceleration_mps2 in edge_beliefs:
            receiver_idx = int(receiver_idx)
            sender_idx = int(sender_idx)
            deceleration_mps2 = float(deceleration_mps2)
            if receiver_idx < 0 or sender_idx < 0 or receiver_idx == sender_idx:
                raise ValueError(
                    "yield-belief receiver/sender must be distinct non-negative indices"
                )
            if deceleration_mps2 <= 0.0:
                raise ValueError("yield-belief deceleration must be positive")
            parsed.append(
                (receiver_idx, sender_idx, deceleration_mps2, bool(bidirectional))
            )
        if not parsed:
            raise ValueError("at least one directed yield belief is required")
        self.interaction_yield_belief = parsed

    def clear_interaction_yield_belief(self):
        self.interaction_yield_belief = None

    def set_semantic_assertive_receivers(self, receiver_indices, scale=1.0):
        parsed = sorted({int(index) for index in receiver_indices})
        if not parsed or parsed[0] < 0:
            raise ValueError("assertive receivers must be non-negative indices")
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("assertive token scale must be finite and positive")
        self.semantic_assertive_receiver_indices = parsed
        self.semantic_assertive_token_scale = scale

    def clear_semantic_assertive_receivers(self):
        self.semantic_assertive_receiver_indices = None

    def set_marginal_preserving_correlated_flow_prior(
        self, agent_pairs, rho, lag_steps=0, action_dims=None
    ):
        rho = float(rho)
        lag_steps = int(lag_steps)
        if not -1.0 < rho < 1.0:
            raise ValueError("correlated flow-prior rho must be strictly inside (-1, 1)")
        if lag_steps < 0:
            raise ValueError("correlated flow-prior lag must be non-negative")
        parsed = [(int(left), int(right)) for left, right in agent_pairs]
        if not parsed:
            raise ValueError("at least one correlated flow-prior pair is required")
        flattened = [index for pair in parsed for index in pair]
        if any(index < 0 for index in flattened):
            raise ValueError("correlated flow-prior indices must be non-negative")
        if any(left == right for left, right in parsed):
            raise ValueError("correlated flow-prior agents must be distinct")
        receivers = [left for left, _ in parsed]
        anchors = [right for _, right in parsed]
        if (
            len(set(receivers)) != len(receivers)
            or set(receivers).intersection(anchors)
        ):
            raise ValueError(
                "correlated flow-prior receivers must be unique and cannot "
                "also serve as anchors; repeated anchors are allowed for an "
                "exact shared-factor Gaussian-copula contract"
            )
        if action_dims is None:
            parsed_dims = list(range(int(self.action_dim)))
        else:
            parsed_dims = sorted({int(dim) for dim in action_dims})
        if not parsed_dims or any(
            dim < 0 or dim >= int(self.action_dim) for dim in parsed_dims
        ):
            raise ValueError("correlated flow-prior action dims are invalid")
        self.marginal_preserving_correlated_flow_prior = {
            "agent_pairs": parsed,
            "rho": rho,
            "lag_steps": lag_steps,
            "action_dims": parsed_dims,
        }

    def clear_marginal_preserving_correlated_flow_prior(self):
        self.marginal_preserving_correlated_flow_prior = None

    def set_pcn_joint_flow_prior(self, beta):
        beta = float(beta)
        if not 0.0 < beta <= 1.0:
            raise ValueError("pCN flow-prior beta must be in (0, 1]")
        self.pcn_joint_flow_prior_beta = beta

    def clear_pcn_joint_flow_prior(self):
        self.pcn_joint_flow_prior_beta = None

    def set_persistent_copula_flow_prior(
        self, persistence, execute_steps, actor_pairs=(), copula_rho=0.0,
        action_dims=None, proposal_start_index=1, temporal_rho=0.0,
        temporal_action_dims=None,
    ):
        config = PersistentCopulaConfig(
            persistence=float(persistence),
            execute_steps=int(execute_steps),
            actor_pairs=tuple(
                (int(receiver), int(anchor))
                for receiver, anchor in actor_pairs
            ),
            copula_rho=float(copula_rho),
            action_dims=(
                None
                if action_dims is None
                else tuple(int(dim) for dim in action_dims)
            ),
            proposal_start_index=int(proposal_start_index),
            temporal_rho=float(temporal_rho),
            temporal_action_dims=(
                None
                if temporal_action_dims is None
                else tuple(int(dim) for dim in temporal_action_dims)
            ),
        )
        self.persistent_copula_flow_prior = {
            "config": config,
            "sampler": PersistentCopulaSource(config),
        }
        self._persistent_copula_replan_call = 0
        self._persistent_copula_last_shape = None

    def clear_persistent_copula_flow_prior(self):
        self.persistent_copula_flow_prior = None
        self._persistent_copula_replan_call = 0
        self._persistent_copula_last_shape = None

    def persistent_copula_flow_prior_attestation(self):
        prior = self.persistent_copula_flow_prior
        if prior is None:
            return None
        config = prior["config"]
        payload = {
            "persistence": float(config.persistence),
            "execute_steps": int(config.execute_steps),
            "actor_pairs": [list(pair) for pair in config.actor_pairs],
            "copula_rho": float(config.copula_rho),
            "action_dims": (
                None
                if config.action_dims is None
                else list(config.action_dims)
            ),
            "proposal_start_index": int(config.proposal_start_index),
            "temporal_rho": float(config.temporal_rho),
            "temporal_action_dims": (
                None
                if config.temporal_action_dims is None
                else list(config.temporal_action_dims)
            ),
            "replan_calls": int(self._persistent_copula_replan_call),
            "source_contract": (
                "sample0 independent standard Gaussian; proposal process has "
                "stationary standard-Gaussian single-coordinate marginals"
            ),
        }
        if self._persistent_copula_last_shape is not None:
            payload["last_shape"] = list(self._persistent_copula_last_shape)
            payload["analytic_kl"] = prior["sampler"].normalized_kl_contract(
                self._persistent_copula_last_shape,
                replan_calls=max(1, self._persistent_copula_replan_call),
            )
        return payload

    @staticmethod
    def sample_pcn_joint_prior(
        action_shape, beta, device=None, dtype=None
    ):
        """Draw a local pCN bank with the exact original joint Gaussian law.

        For independent ``z0, xi ~ N(0, I)``, each proposal is
        ``sqrt(1-beta**2) * z0 + beta * xi``.  Thus every complete proposal is
        itself N(0, I), including independence across agents/time/action
        coordinates.  Only correlations *between proposal modes* change.
        Sample 0 is left byte-identical to the ordinary FM policy draw, so a
        rejected bank has an exact factual fallback under a fixed RNG seed.
        """
        if len(action_shape) != 5:
            raise ValueError("pCN FM prior expects shape (B,K,N,T,D)")
        beta = float(beta)
        if not 0.0 < beta <= 1.0:
            raise ValueError("pCN flow-prior beta must be in (0, 1]")
        noise = torch.randn(action_shape, device=device, dtype=dtype)
        if int(action_shape[1]) > 1:
            factual = noise[:, :1].clone()
            innovations = noise[:, 1:].clone()
            factual_scale = float(np.sqrt(1.0 - beta * beta))
            noise[:, 1:] = factual_scale * factual + beta * innovations
        return noise

    @staticmethod
    def sample_marginal_preserving_correlated_prior(
        action_shape, agent_pairs, rho, lag_steps=0, action_dims=None,
        device=None, dtype=None
    ):
        """Draw a lagged Gaussian copula with exact per-agent N(0, I) marginals.

        ``action_shape`` is the unflattened FM source shape ``(B,K,N,T,D)``.
        Receivers are unique and cannot also be anchors. Anchors may repeat,
        which represents a shared latent cause without changing any agent's
        marginal distribution. Replacing one receiver with
        ``rho*z_anchor[t-lag] + sqrt(1-rho^2)*z_member[t]`` is
        order-independent. The unavailable prefix remains independent. Since
        every receiver uses its original independent residual and anchors are
        never overwritten, each source vector remains white standard Gaussian;
        repeated anchors only add cross-agent dependence (rho**2 between two
        receivers sharing one anchor at aligned coordinates).
        """
        if len(action_shape) != 5:
            raise ValueError("correlated FM prior expects shape (B,K,N,T,D)")
        rho = float(rho)
        lag_steps = int(lag_steps)
        if not -1.0 < rho < 1.0:
            raise ValueError("correlated flow-prior rho must be strictly inside (-1, 1)")
        if lag_steps < 0 or lag_steps >= int(action_shape[3]):
            raise ValueError(
                "correlated flow-prior lag must be in [0, prediction_horizon)"
            )
        pairs = [(int(left), int(right)) for left, right in agent_pairs]
        flattened = [index for pair in pairs for index in pair]
        receivers = [left for left, _ in pairs]
        anchors = [right for _, right in pairs]
        if (
            not pairs
            or len(set(receivers)) != len(receivers)
            or set(receivers).intersection(anchors)
        ):
            raise ValueError(
                "correlated flow-prior receivers must be unique and cannot "
                "also serve as anchors"
            )
        num_agents = int(action_shape[2])
        if any(index < 0 or index >= num_agents for index in flattened):
            raise IndexError("correlated flow-prior pair exceeds the agent dimension")
        num_action_dims = int(action_shape[4])
        if action_dims is None:
            dims = list(range(num_action_dims))
        else:
            dims = sorted({int(dim) for dim in action_dims})
        if not dims or any(dim < 0 or dim >= num_action_dims for dim in dims):
            raise ValueError("correlated flow-prior action dims are invalid")
        noise = torch.randn(action_shape, device=device, dtype=dtype)
        # Reserve sample 0 as the exact independent FM policy baseline.  With
        # a fixed RNG seed this row is byte-identical to the first mode drawn
        # by the interaction=none sampler; only proposal modes 1..K-1 receive
        # the copula intervention.  This makes hard-gate fallback a genuine
        # factual action instead of correlated proposal 0.
        proposal_slice = slice(1, None)
        independent_scale = float(np.sqrt(1.0 - rho * rho))
        for receiver, anchor in pairs:
            receiver_noise = noise[:, :, receiver].clone()
            anchor_noise = noise[:, :, anchor]
            for action_dim in dims:
                if lag_steps == 0:
                    noise[:, proposal_slice, receiver, :, action_dim] = (
                        rho * anchor_noise[:, proposal_slice, :, action_dim]
                        + independent_scale
                        * receiver_noise[:, proposal_slice, :, action_dim]
                    )
                else:
                    noise[:, proposal_slice, receiver, lag_steps:, action_dim] = (
                        rho * anchor_noise[
                            :, proposal_slice, :-lag_steps, action_dim
                        ]
                        + independent_scale
                        * receiver_noise[
                            :, proposal_slice, lag_steps:, action_dim
                        ]
                    )
        return noise

    def set_guidance_dim(self, idx, ts=None): 
        '''
        Decide which frame to perturb first
        '''
        self.current_perturbation_guidance.set_perturbation_idx(idx)
        if ts is not None: 
            self.current_perturbation_guidance.set_controllable_ts(ts)
    
    def reset_guidance(self): 
        # print("reset guidance")
        self.current_perturbation_guidance = PartialPerturbationGuidance(self.transform, self.transform_params, self.scale_traj, self.descale_traj, available_idx=None)

    def set_guidance(self, guidance_config_list, example_batch=None):
        '''
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        '''
        if guidance_config_list is not None:
            if len(guidance_config_list) > 0 and verify_guidance_config_list(guidance_config_list):
                print('Instantiating test-time guidance with configs:')
                print(guidance_config_list)
                self.current_perturbation_guidance.set_guidance(guidance_config_list, example_batch)

    def set_constraints(self, constraint_config):
        '''
        Instantiates test-time hard constraints using the config (dict) passed in.
        '''
        if constraint_config is not None and len(constraint_config) > 0:
            verify_constraint_config(constraint_config)
            print('Instantiating test-time constraints with config:')
            print(constraint_config)
            self.current_constraints = constraint_config
    
    def update_guidance(self, **kwargs):
        if self.current_perturbation_guidance.current_guidance is not None:
            self.current_perturbation_guidance.update(**kwargs)

    def clear_guidance(self):
        self.current_perturbation_guidance.clear_guidance()

    def set_guidance_optimization_params(self, guidance_optimization_params):
        self.guidance_optimization_params = guidance_optimization_params

    def set_flow_velocity_guidance(self, config=None):
        """Select velocity-field guidance for FM without endpoint editing."""
        if config is None:
            self.flow_velocity_guidance_config = None
            return
        config = dict(config)
        max_rms = float(config.get("trust_max_rms", 0.05))
        max_linf = float(config.get("trust_max_linf", 0.20))
        if max_rms <= 0.0 or max_linf <= 0.0:
            raise ValueError("flow-velocity trust bounds must be positive")
        self.flow_velocity_guidance_config = {
            "trust_max_rms": max_rms,
            "trust_max_linf": max_linf,
        }

    def set_diffusion_specific_params(self, diffusion_specific_params):
        self.apply_guidance_intermediate = diffusion_specific_params['apply_guidance_intermediate']
        self.apply_guidance_output = diffusion_specific_params['apply_guidance_output']
        self.final_step_opt_params = diffusion_specific_params['final_step_opt_params']
        self.stride = diffusion_specific_params['stride']

    #------------------------------------------ utility ------------------------------------------#
    def _create_dynamics(self):
        if self._dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        else:
            self.dyn = None

    def prepare_scene_agent_hist(self, pos, yaw, speed, extent, avail, norm_info, scale=True, speed_repr='abs_speed'):
        '''
        Input:
        - pos : (B, M, (Q), T, 2)
        - yaw : (B, M, (Q), T, 1)
        - speed : (B, M, (Q), T)
        - extent: (B, M, (Q), 3)
        - avail: (B, M, (Q), T)
        - norm_info: [2, 5]
        - speed_repr: 'abs_speed' or 'rel_vel'
        Output:
        - hist_in: [B, M, (Q), T, 8] (x,y,cos,sin,v,l,w,avail)
        - info (motion kinematic graph): 
            {matrix_dist, matrix_ttc}
        '''
        differentiable_conditioning = bool(getattr(
            self, "conditional_flow_differentiable_conditioning", False
        ))
        M = pos.shape[1]
        T = pos.shape[-2]
        # print(pos)
        # assert 1==2
        info = {}
        if speed_repr == 'rel_vel_per_step':
            # (B, M, M, T, 1) -> (B, T, 1, M) -> (B, M, T, 1) -> (B, M, 1, T, 1) -> (B, M, M, T, 1)
            yaw_self = torch.diagonal(yaw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(yaw)
            # (B, M, M, T, 1) -> (B*M*M*T)
            yaw_self_ = yaw_self.reshape(-1)
            # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2) -> (B*M*M*T, 2)
            pos_self_ = torch.diagonal(pos, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(pos).reshape(-1, 2)

            # (B, M1, M2, T, 2) -> (B*M1*M2*T, 2)
            pos_ = pos.view(-1, 2)
            
            # self_from_self_per_timestep
            # (B*M1*M2*T, 3, 3)
            i_from_i_per_time = torch.stack(
                [
                    torch.stack([torch.cos(yaw_self_), -torch.sin(yaw_self_), pos_self_[..., 0]], dim=-1),
                    torch.stack([torch.sin(yaw_self_), torch.cos(yaw_self_), pos_self_[..., 1]], dim=-1),
                    torch.stack([0.0*torch.ones_like(yaw_self_), 0.0*torch.ones_like(yaw_self_), 1.0*torch.ones_like(yaw_self_)], dim=-1)
                ], dim=-2
            )
            i_per_time_from_i = torch.linalg.inv(i_from_i_per_time)
            
            # transform coord
            pos_transformed = torch.einsum("...jk,...k->...j", i_per_time_from_i[..., :-1, :-1], pos_)
            pos_transformed += i_per_time_from_i[..., :-1, -1]
            # (B*M1*M2*T, 2) -> (B, M1, M2, T, 2)
            pos = pos_transformed.view(pos.shape)

            # transform angle
            yaw = angle_wrap_torch(yaw - yaw_self)


        # convert yaw to heading vec
        hvec = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1) # (B, M, (Q), T, 2)
        # only need length, width for pred
        lw = extent[..., :2].unsqueeze(-2).expand(pos.shape) # (B, M, (Q), T, 2)

        # Get time to collision
        if speed_repr in ['rel_vel', 'rel_vel_per_step', 'rel_vel_new', 'rel_vel_new_new']:
            d_th = 20
            t_to_col_th = 20

            # print('pos.shape', pos.shape)
            # print('pos[0,0]', pos[0,0])
            # raise

            # estimate relative distance to other agents
            if speed_repr == 'rel_vel_new_new':
                # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2)
                pos_self = torch.diagonal(pos, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(pos)
                pos_for_edge = (
                    pos if differentiable_conditioning
                    else pos.detach().clone()
                )
                pos_diff = torch.abs(pos_for_edge - pos_self)
            else:
                pos_diff = pos
            # (B, M, (Q), T, 2) -> (B, M, (Q), T, 1)
            rel_d = torch.norm(pos_diff, dim=-1).unsqueeze(-1)
            
            # rel_d_lw also consider the lw of both agents (half of the mean of each lw)
            # (B, M, (Q), T, 2) -> (B, M, (Q), T, 1)
            lw_avg_half = (torch.mean(lw, dim=-1) / 2).unsqueeze(-1)
            # (B, M, (Q), T, 1) -> (B, M, T, 1)
            ego_lw_avg_half = lw_avg_half[...,torch.arange(M), torch.arange(M), :, :]
            # (B, M, (Q), T, 1)
            lw_avg_half_sum = lw_avg_half + ego_lw_avg_half.unsqueeze(2).expand_as(lw_avg_half)
            # (B, M, (Q), T, 1)
            rel_d_lw = rel_d - lw_avg_half_sum
            # normalize rel_d and rel_d_lw
            rel_d = torch.clip(rel_d, min=0, max=d_th)
            rel_d = (d_th - rel_d) / d_th
            rel_d_lw = torch.clip(rel_d_lw, min=0, max=d_th)
            rel_d_lw = (d_th - rel_d_lw) / d_th

            B, M, M, T = speed.shape
            # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M, 1, T) -> (B, M, M, T)
            ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1).unsqueeze(-2).expand(B, M, M, T).clone()
            # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2)
            ego_lw = torch.diagonal(lw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand(B, M, M, T, 2).clone()
            if differentiable_conditioning:
                ego_vx = torch.nan_to_num(ego_vx, nan=0.0)
            else:
                ego_vx[torch.isnan(ego_vx)] = 0.0
            
            if speed_repr == 'rel_vel_new_new':
                # (B, M, M, T, 1) -> (B, T, 1, M) -> (B, M, T, 1) -> (B, M, 1, T, 1) -> (B, M, M, T, 1)
                yaw_self = torch.diagonal(yaw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(yaw)
                hvec_self = torch.cat([torch.cos(yaw_self), torch.sin(yaw_self)], dim=-1)
                vx = ego_vx * hvec_self[...,0] - speed * hvec[...,0]
                vy = ego_vx * hvec_self[...,1] - speed * hvec[...,1]
            else:
                vx = ego_vx - speed * hvec[...,0]
                vy = 0 - speed * hvec[...,1]

            x_dist = pos_diff[...,0] - (ego_lw[...,0]/2) - (lw[...,0]/2)
            y_dist = pos_diff[...,1] - (ego_lw[...,1]/2) - (lw[...,1]/2)
            if differentiable_conditioning:
                # The native detached branch permits +/-inf here and removes it
                # through the following overwrite/clamp.  Autograd nevertheless
                # produces NaNs from the inactive divide-by-zero branch.  Use a
                # value-equivalent functional zero-velocity branch so the
                # Conditional-FM Jacobian remains finite.
                safe_vx = torch.where(vx == 0, torch.ones_like(vx), vx)
                safe_vy = torch.where(vy == 0, torch.ones_like(vy), vy)
                x_t_to_col = x_dist / safe_vx
                y_t_to_col = y_dist / safe_vy
                x_t_to_col = torch.where(
                    vx == 0,
                    torch.where(
                        x_dist > 0,
                        torch.full_like(x_dist, t_to_col_th),
                        torch.zeros_like(x_dist),
                    ),
                    x_t_to_col,
                )
                y_t_to_col = torch.where(
                    vy == 0,
                    torch.where(
                        y_dist > 0,
                        torch.full_like(y_dist, t_to_col_th),
                        torch.zeros_like(y_dist),
                    ),
                    y_t_to_col,
                )
            else:
                x_t_to_col = x_dist / vx
                y_t_to_col = y_dist / vy
            # if collision has not happened and moving in opposite direction, set t_to_col to t_to_col_th
            if differentiable_conditioning:
                x_t_to_col = torch.where(
                    (x_dist > 0) & (x_t_to_col < 0),
                    torch.full_like(x_t_to_col, t_to_col_th),
                    x_t_to_col,
                )
                y_t_to_col = torch.where(
                    (y_dist > 0) & (y_t_to_col < 0),
                    torch.full_like(y_t_to_col, t_to_col_th),
                    y_t_to_col,
                )
                # if collision already happened, set time to collision to 0
                x_t_to_col = torch.where(
                    x_dist < 0, torch.zeros_like(x_t_to_col), x_t_to_col
                )
                y_t_to_col = torch.where(
                    y_dist < 0, torch.zeros_like(y_t_to_col), y_t_to_col
                )
            else:
                x_t_to_col[(x_dist>0) & (x_t_to_col<0)] = t_to_col_th
                y_t_to_col[(y_dist>0) & (y_t_to_col<0)] = t_to_col_th
                # if collision already happened, set t_to_col to 0
                x_t_to_col[x_dist<0] = 0
                y_t_to_col[y_dist<0] = 0
            # both directions need to be met for collision to happen
            rel_t_to_col = torch.max(torch.cat([x_t_to_col.unsqueeze(-1), y_t_to_col.unsqueeze(-1)], dim=-1), dim=-1)[0]
            rel_t_to_col = torch.clip(rel_t_to_col, min=0, max=t_to_col_th)
            # normalize rel_t_to_col
            rel_t_to_col = (t_to_col_th - rel_t_to_col.unsqueeze(-1)) / t_to_col_th

            matrix_ttc = rel_t_to_col[0, :, :, :, 0].detach().cpu().numpy() # normalized ttc
            matrix_dist = rel_d[0, :, :, :, 0].detach().cpu().numpy() # meters
            info = {
                "matrix_ttc": matrix_ttc, 
                "matrix_dist": matrix_dist
            }
        # normalize everything
        #  note: don't normalize hvec since already unit vector
        add_coeffs = torch.tensor(norm_info[0]).to(pos.device)
        div_coeffs = torch.tensor(norm_info[1]).to(pos.device)

        if len(pos.shape) == 4:
            add_coeffs_expand = add_coeffs[None, None, None, :]
            div_coeffs_expand = div_coeffs[None, None, None, :]
        else:
            add_coeffs_expand = add_coeffs[None, None, None, None, :]
            div_coeffs_expand = div_coeffs[None, None, None, None, :]

        pos_original = (
            pos if differentiable_conditioning else pos.detach().clone()
        )
        if scale:
            pos = (pos + add_coeffs_expand[...,:2]) / div_coeffs_expand[...,:2]
            speed = (speed.unsqueeze(-1) + add_coeffs[2]) / div_coeffs[2]
            lw = (lw + add_coeffs_expand[...,3:]) / div_coeffs_expand[...,3:]
        else:
            speed = speed.unsqueeze(-1)
        
        if speed_repr in ['rel_vel', 'rel_vel_new', 'rel_vel_per_step', 'rel_vel_new_new']:
            speed = speed.squeeze(-1)
            B, M, M, T = speed.shape
            if speed_repr == 'rel_vel_new_new':
                # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M) -> (B, M, 1, 1) -> (B, M, M, T)
                ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1)[...,0].unsqueeze(-1).unsqueeze(-1).expand(B, M, M, T).clone()
                if differentiable_conditioning:
                    ego_vx = torch.nan_to_num(ego_vx, nan=0.0)
                else:
                    ego_vx[torch.isnan(ego_vx)] = 0.0
                vx = speed * hvec[...,0] - ego_vx
                vy = speed * hvec[...,1]
            else:
                # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M, 1, T) -> (B, M, M, T)
                ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1).unsqueeze(-2).expand(B, M, M, T).clone()
                ego_vx[torch.isnan(ego_vx)] = 0.0
                vx = speed * hvec[...,0] - ego_vx
                vy = speed * hvec[...,1]
            vvec = torch.cat([vx.unsqueeze(-1), vy.unsqueeze(-1)], dim=-1) # (B, M, M, T, 2)

            # also need to zero out the symmetric entries as we apply anothe matrix transformation
            if speed_repr in ['rel_vel_per_step', 'rel_vel_new', 'rel_vel_new_new']:
                # (B, M1, M2, T) -> (B, M2, M1, T)
                avail_perm = avail.permute(0, 2, 1, 3)
                avail = avail * avail_perm

                hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, pos_original, avail.unsqueeze(-1)], dim=-1)
            elif speed_repr in ['rel_vel_new', 'rel_vel_new_new']:
                hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, pos_original, avail.unsqueeze(-1)], dim=-1)
            else:
                hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, avail.unsqueeze(-1)], dim=-1)
        elif speed_repr == 'abs_speed':
            # combine to get full input
            hist_in = torch.cat([pos, hvec, speed, lw, avail.unsqueeze(-1)], dim=-1)
        else:
            raise ValueError('Unknown speed representation: {}'.format(speed_repr))
        # zero out values we don't have data for
        if differentiable_conditioning:
            hist_in = torch.where(
                avail.unsqueeze(-1), hist_in, torch.zeros_like(hist_in)
            )
        else:
            hist_in[~avail] = 0.0
        if torch.isnan(hist_in).any():
            hist_in = torch.where(torch.isnan(hist_in), torch.zeros_like(hist_in), hist_in)

            # log nan values
            print('torch.where(torch.isnan(hist_in))', torch.where(torch.isnan(hist_in)))
            # with open('nan_log.txt', 'a') as f:
            #     f.write('torch.where(torch.isnan(hist_in)): '+str(torch.where(torch.isnan(hist_in))))
            #     f.write('torch.where(torch.isnan(pos)): '+str(torch.where(torch.isnan(pos))))
            #     f.write('torch.where(torch.isnan(rel_d)): '+str(torch.where(torch.isnan(rel_d))))
            #     f.write('torch.where(torch.isnan(lw_avg_half_sum)): '+str(torch.where(torch.isnan(lw_avg_half_sum))))
                    
        # print("hist isnan: ", hist_in.isnan().any())
        
        return hist_in, info

    def get_aux_info(self, data_batch, include_class_free_cond=False):
        # if torch.isnan(data_batch["history_positions"]).any():
        #     print('0 torch.where(torch.isnan(data_batch["history_positions"]))', torch.where(torch.isnan(data_batch["history_positions"])))
        #     raise ValueError('NaN in data_batch["history_positions"]')

        # current ego state (B, 4)
        # Note: [x, y, vel, yaw] (unicycle), always need this for rolling out actions
        if self._dynamics_type is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())
        else:
            curr_states = None
        

        if self.single_cond_feat:
            cond_feat_in_list = []
            non_cond_feat_in_list = []
        # print('statistics: ', data_batch["history_availabilities"].sum(dim=-1).float().mean(), data_batch["target_availabilities"].sum(dim=-1).float().mean())
        # 1.agents_hist (B, M, T, K_vehicle)
        agents_hist, _ = self.prepare_scene_agent_hist(data_batch["history_positions"], data_batch["history_yaws"], data_batch["history_speeds"], data_batch["extent"], data_batch["history_availabilities"], self.agent_hist_norm_info)
        if self.agent_hist_embed_method == 'mlp':
            # encode agents_hist with MLP (only support vectorized version)
            B, M, _, _ = agents_hist.shape
            # (B, M, T, 8) -> (B*M, K_d)
            agent_hist_feat = self.agent_hist_encoder.scene_forward(agents_hist)
            # (B*M, K_d) -> (B, M, K_d) -> (B, M, 1, K_d)
            agent_hist_feat = agent_hist_feat.reshape((B, M, -1)).unsqueeze(-2)
            if self.single_cond_feat:
                cond_feat_in_list.append(agent_hist_feat)
        elif self.agent_hist_embed_method in ['transformer', 'concat']:
            # do nothing and directly pass the vector forms (only support vectorized version)
            # (B, M, T, K_vehicle)
            agent_hist_feat = agents_hist
        else:
            raise NotImplementedError

        # 2.neighbor trajectory encoding (B, M, Q, T, K_vehicle)
        if self.neigh_hist_embed_method == 'mlp':
            neighbor_hist, _ = self.prepare_scene_agent_hist(data_batch["all_other_agents_history_positions"], data_batch["all_other_agents_history_yaws"], data_batch["all_other_agents_history_speeds"], data_batch["all_other_agents_extents"], data_batch["all_other_agents_history_availabilities"], self.neighbor_hist_norm_info)

            # (B, M, Q, T, K_vehicle) -> (B*M, K_d)
            neighbor_hist_feat = self.neighbor_hist_encoder.scene_forward(neighbor_hist, data_batch["all_other_agents_history_availabilities"])
            # (B*M, K_d) -> (B*M, K_d) -> (B, M, 1, K_d)
            neighbor_hist_feat = neighbor_hist_feat.reshape((B, M, -1)).unsqueeze(-2)
            if self.single_cond_feat:
                cond_feat_in_list.append(neighbor_hist_feat)
            
        elif self.neigh_hist_embed_method in ['interaction_edge', 'interaction_edge_and_input']:
            # (B, M, M, T_hist, K_vehicle), (B, M, M, T_hist, K_vehicle)
            neighbor_hist_feat, neighbor_hist_feat_non_cond = self.get_neighbor_history_relative_states(data_batch, include_class_free_cond)

        elif self.neigh_hist_embed_method is None:
            neighbor_hist_feat, neighbor_hist_feat_non_cond = None, None
        else:
            raise NotImplementedError(f'{self.neigh_hist_embed_method}')


        # 3.map encoding
        if self.map_embed_method in ['cnn', 'cnn_local_patch']:
            # encode map with CNN (only support rasterized version)
            # rasterized map (B, M, C, H, W)
            image_batch = data_batch["image"]
            B, M, C, H, W = image_batch.shape
            # Padded agents can carry NaN raster values. Never pass those slots
            # through the CNN: BatchNorm would contaminate running stats, and
            # their map features would propagate NaNs into the denoiser.
            image_batch = image_batch.reshape(B*M, C, H, W)
            valid_map_agents = data_batch["history_availabilities"].bool().any(-1).reshape(-1)
            valid_indices = torch.nonzero(valid_map_agents, as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                raise ValueError("Map encoder received a batch with no valid agents")
            valid_images = image_batch.index_select(0, valid_indices)
            if not torch.isfinite(valid_images).all():
                raise ValueError("Non-finite raster found for a valid agent")
            valid_global_feat, valid_grid_feat = self.map_encoder(valid_images)
            map_global_feat = valid_global_feat.new_zeros((B*M, valid_global_feat.shape[-1]))
            map_global_feat = map_global_feat.index_copy(0, valid_indices, valid_global_feat)
            if valid_grid_feat is not None:
                map_grid_feat = valid_grid_feat.new_zeros((B*M,) + tuple(valid_grid_feat.shape[1:]))
                map_grid_feat = map_grid_feat.index_copy(0, valid_indices, valid_grid_feat)
            else:
                map_grid_feat = None

            if map_global_feat is not None:
                # (B*M, K_d) -> (B, M, 1, K_d)
                map_feat = map_global_feat.reshape(B, M, -1).unsqueeze(-2)
            else:
                map_feat = None

            if self.single_cond_feat:
                cond_feat_in_list.append(map_feat)

            if self.use_map_feat_grid and self.map_encoder is not None:
                raster_from_agent = data_batch["raster_from_agent"]
                map_grid_feat = map_grid_feat.reshape(B, M, *map_grid_feat.shape[1:])

        elif self.map_embed_method == 'transformer':
            # do nothing and directly pass the vector forms (only support vector version)
            # vectorized map (B, M, S_seg, S_p, K_map)
            map_feat = data_batch["extras"]["closest_lane_point"]
            # print('map_feat.shape', map_feat.shape)
        else:
            raise NotImplementedError


        if self.single_cond_feat:
            cond_feat_in = torch.cat(cond_feat_in_list, dim=-1)
            B, M, _, _ = cond_feat_in.shape
            # (B, M, 1, K_d*3) -> (B, M, K_d)
            cond_feat = self.process_cond_mlp(cond_feat_in.reshape(B*M, -1)).reshape(B, M, -1)
        else:
            cond_feat = None


        aux_info = {
            'curr_states': curr_states,
            'agent_hist_feat': agent_hist_feat,
            'map_feat': map_feat,
            'cond_feat': cond_feat,
            'neighbor_hist_feat': neighbor_hist_feat,
        }

        if self.semantic_assertive_receiver_indices is not None:
            batch_size, num_agents = data_batch["history_positions"].shape[:2]
            receiver_mask = torch.zeros(
                (batch_size, num_agents),
                dtype=torch.float32,
                device=data_batch["history_positions"].device,
            )
            for receiver_idx in self.semantic_assertive_receiver_indices:
                if receiver_idx >= num_agents:
                    raise IndexError(
                        "assertive receiver {} exceeds padded agent count {}".format(
                            receiver_idx, num_agents
                        )
                    )
                receiver_mask[:, receiver_idx] = self.semantic_assertive_token_scale
            aux_info["semantic_assertive_receiver_mask"] = receiver_mask

        if self.interaction_edge_cut is not None:
            batch_size, num_agents = data_batch["history_positions"].shape[:2]
            for receiver_idx, sender_idx, _ in self.interaction_edge_cut:
                if receiver_idx >= num_agents or sender_idx >= num_agents:
                    raise IndexError(
                        "edge cut ({}, {}) exceeds padded agent count {}".format(
                            receiver_idx, sender_idx, num_agents
                        )
                    )
            edge_cut_mask = torch.zeros(
                (batch_size, num_agents, num_agents),
                dtype=torch.bool,
                device=data_batch["history_positions"].device,
            )
            for receiver_idx, sender_idx, bidirectional in self.interaction_edge_cut:
                edge_cut_mask[:, receiver_idx, sender_idx] = True
                if bidirectional:
                    edge_cut_mask[:, sender_idx, receiver_idx] = True
            aux_info["interaction_edge_cut_mask"] = edge_cut_mask

        if self.interaction_full_visibility_cut is not None:
            batch_size, num_agents = data_batch["history_positions"].shape[:2]
            full_visibility_mask = torch.zeros(
                (batch_size, num_agents, num_agents),
                dtype=torch.bool,
                device=data_batch["history_positions"].device,
            )
            for receiver_idx, sender_idx, bidirectional in (
                self.interaction_full_visibility_cut
            ):
                if receiver_idx >= num_agents or sender_idx >= num_agents:
                    raise IndexError(
                        "full visibility cut ({}, {}) exceeds padded agent count {}".format(
                            receiver_idx, sender_idx, num_agents
                        )
                    )
                full_visibility_mask[:, receiver_idx, sender_idx] = True
                if bidirectional:
                    full_visibility_mask[:, sender_idx, receiver_idx] = True
            aux_info[
                "interaction_full_visibility_cut_mask"
            ] = full_visibility_mask

        if self.interaction_edge_delay is not None:
            batch_size, num_agents = data_batch["history_positions"].shape[:2]
            edge_delay_steps = torch.zeros(
                (batch_size, num_agents, num_agents),
                dtype=torch.long,
                device=data_batch["history_positions"].device,
            )
            for receiver_idx, sender_idx, delay_steps, bidirectional in self.interaction_edge_delay:
                if receiver_idx >= num_agents or sender_idx >= num_agents:
                    raise IndexError(
                        "edge delay ({}, {}) exceeds padded agent count {}".format(
                            receiver_idx, sender_idx, num_agents
                        )
                    )
                edge_delay_steps[:, receiver_idx, sender_idx] = delay_steps
                if bidirectional:
                    edge_delay_steps[:, sender_idx, receiver_idx] = delay_steps
            aux_info["interaction_edge_delay_steps"] = edge_delay_steps

        if self.interaction_future_intent_hold is not None:
            batch_size, num_agents = data_batch["history_positions"].shape[:2]
            intent_hold_mask = torch.zeros(
                (batch_size, num_agents, num_agents), dtype=torch.bool,
                device=data_batch["history_positions"].device,
            )
            for receiver_idx, sender_idx, bidirectional in (
                self.interaction_future_intent_hold
            ):
                if receiver_idx >= num_agents or sender_idx >= num_agents:
                    raise IndexError(
                        "future-intent hold ({}, {}) exceeds padded agent count {}".format(
                            receiver_idx, sender_idx, num_agents
                        )
                    )
                intent_hold_mask[:, receiver_idx, sender_idx] = True
                if bidirectional:
                    intent_hold_mask[:, sender_idx, receiver_idx] = True
            aux_info["interaction_future_intent_hold_mask"] = intent_hold_mask

        if self.interaction_yield_belief is not None:
            batch_size, num_agents = data_batch["history_positions"].shape[:2]
            yield_deceleration = torch.zeros(
                (batch_size, num_agents, num_agents),
                dtype=data_batch["history_positions"].dtype,
                device=data_batch["history_positions"].device,
            )
            for receiver_idx, sender_idx, deceleration_mps2, bidirectional in (
                self.interaction_yield_belief
            ):
                if receiver_idx >= num_agents or sender_idx >= num_agents:
                    raise IndexError(
                        "yield belief ({}, {}) exceeds padded agent count {}".format(
                            receiver_idx, sender_idx, num_agents
                        )
                    )
                yield_deceleration[:, receiver_idx, sender_idx] = deceleration_mps2
                if bidirectional:
                    yield_deceleration[:, sender_idx, receiver_idx] = deceleration_mps2
            aux_info["interaction_yield_belief_deceleration_mps2"] = (
                yield_deceleration
            )

        # do not have non_cond_curr_states since it is not used in the conditioning
        if self.map_embed_method in ['cnn', 'cnn_local_patch']:
            if self.use_map_feat_grid and self.map_encoder is not None:
                aux_info['map_grid_feat'] = map_grid_feat
                aux_info['raster_from_agent'] = raster_from_agent

        return aux_info
    
    def get_neighbor_relative_states(self, relative_positions, relative_speeds, relative_yaws, data_batch_agent_from_world, data_batch_world_from_agent, data_batch_yaw, data_batch_extent):
        BN, M, _, _ = relative_positions.shape
        B = data_batch_agent_from_world.shape[0]
        N = int(BN // B)

        # [M, M]
        nb_idx = torch.arange(M).unsqueeze(0).repeat(M, 1)

        all_other_agents_relative_positions_list = []
        all_other_agents_relative_yaws_list = []
        all_other_agents_relative_speeds_list = []
        all_other_agents_extent_list = []

        # get relative states
        for k in range(BN):
            i = int(k // N)
            agent_from_world = data_batch_agent_from_world[i]
            world_from_agent = data_batch_world_from_agent[i]

            all_other_agents_relative_positions_list_sub = []
            all_other_agents_relative_yaws_list_sub = []
            all_other_agents_relative_speeds_list_sub = []
            all_other_agents_extent_list_sub = []

            for j in range(M):
                chosen_neigh_inds = nb_idx[j][nb_idx[j]>=0].tolist()

                # (Q. 3. 3)
                center_from_world = agent_from_world[j]
                world_from_neigh = world_from_agent[chosen_neigh_inds]
                center_from_neigh = center_from_world.unsqueeze(0) @ world_from_neigh

                fut_neigh_pos_b_sub = relative_positions[k][chosen_neigh_inds]
                fut_neigh_yaw_b_sub = relative_yaws[k][chosen_neigh_inds]

                all_other_agents_relative_positions_list_sub.append(GeoUtils.transform_points_tensor(fut_neigh_pos_b_sub,center_from_neigh))
                all_other_agents_relative_yaws_list_sub.append(fut_neigh_yaw_b_sub+data_batch_yaw[i][chosen_neigh_inds][:,None,None]-data_batch_yaw[i][j])
                all_other_agents_relative_speeds_list_sub.append(relative_speeds[k][chosen_neigh_inds])
                all_other_agents_extent_list_sub.append(data_batch_extent[i][chosen_neigh_inds])

            all_other_agents_relative_positions_list.append(pad_sequence(all_other_agents_relative_positions_list_sub, batch_first=True, padding_value=np.nan))
            all_other_agents_relative_yaws_list.append(pad_sequence(all_other_agents_relative_yaws_list_sub, batch_first=True, padding_value=np.nan))
            all_other_agents_relative_speeds_list.append(pad_sequence(all_other_agents_relative_speeds_list_sub, batch_first=True, padding_value=np.nan))
            all_other_agents_extent_list.append(pad_sequence(all_other_agents_extent_list_sub, batch_first=True, padding_value=0))

        max_second_dim = max(a.size(1) for a in all_other_agents_relative_positions_list)

        all_other_agents_relative_positions = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_positions_list], dim=0)
        all_other_agents_relative_yaws = angle_wrap(torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_yaws_list], dim=0))
        all_other_agents_relative_speeds = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_speeds_list], dim=0)
        all_other_agents_extents = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_extent_list], dim=0)

        return all_other_agents_relative_positions, all_other_agents_relative_yaws, all_other_agents_relative_speeds, all_other_agents_extents

    def get_neighbor_future_relative_states(self, x: torch.tensor, aux_info: dict, data_batch: dict) -> dict:
        '''
        -param x: (B*N, M, T, transition_dim)
        -return: aux_info with 
            'neighbor_fut_feat': (B*N, M, M, T_fut, K_neigh)
        '''
        
        T_fut = data_batch['target_positions'].shape[-2]

        if self.use_last_hist_step:
            # (B*N, M, M, T_hist, K_neigh) -> (B*N, M, M, K_neigh) -> (B*N, M, M, 1, K_neigh) -> (B*N, M, M, T_fut, K_neigh) (x,y,cos,sin,speed,l,w,avail) or (x,y,cos,sin,vx,vy,l,w,avail)
            neighbor_fut = aux_info['neighbor_hist_feat'][...,-1,:].unsqueeze(-2).repeat(1,1,1,T_fut,1)
            # Set relative velocity to 0 as we assume they are relatively stationary
            # if self.interaction_edge_speed_repr == 'rel_vel':
            #     neighbor_fut[...,4:6] = 0.0
        elif self.use_noisy_fut_edge:
            # TBD: support other transition_dim
            assert x.shape[-1] == 6, f'currently this mode only support x.shape[-1] being 6, but got {x.shape[-1]}'
            # descale to agent coords (x,y,speed,yaw,accel,yawvel)
            x_descaled = self.descale_traj(x)
            future_positions = x_descaled[...,:2]
            future_speeds = x_descaled[...,2]
            future_yaws = x_descaled[...,3:4]

            planner_future_mask = None
            planner_positions = data_batch.get("planner_reference_positions")
            planner_yaws = data_batch.get("planner_reference_yaws")
            planner_mask = data_batch.get("planner_reference_mask")
            planner_fields = (planner_positions, planner_yaws, planner_mask)
            if any(value is not None for value in planner_fields):
                if any(value is None for value in planner_fields):
                    raise ValueError(
                        "planner future conditioning requires positions/yaws/mask"
                    )
                (
                    future_positions,
                    future_speeds,
                    future_yaws,
                    planner_future_mask,
                ) = inject_planner_reference_future_states(
                    future_positions,
                    future_speeds,
                    future_yaws,
                    planner_positions,
                    planner_yaws,
                    planner_mask,
                    dt=0.1,
                )
                aux_info["planner_future_conditioned_sender_mask"] = (
                    planner_future_mask
                )

            all_other_agents_future_positions, all_other_agents_future_yaws, all_other_agents_future_speeds, all_other_agents_extents = self.get_neighbor_relative_states(future_positions, future_speeds, future_yaws, data_batch['agent_from_world'], data_batch['world_from_agent'], data_batch['yaw'], data_batch["extent"])

            future_batch = all_other_agents_future_positions.shape[0]
            yield_history_features = aux_info["neighbor_hist_feat"]
            if yield_history_features.shape[0] != future_batch:
                raise ValueError(
                    "yield-belief canonical history was not repeated with samples"
                )
            if yield_history_features.shape[:3] != (
                all_other_agents_future_positions.shape[:3]
            ):
                raise ValueError(
                    "yield-belief canonical directed edge shapes disagree"
                )
            if yield_history_features.shape[-1] < 3:
                raise ValueError("yield-belief history feature width is invalid")
            # Under rel_vel_new_new, prepare_scene_agent_hist appends raw
            # relative x/y immediately before availability.  This canonical
            # padded tensor is the only safe receiver<-sender history boundary.
            last_observed_relative_positions = yield_history_features[
                ..., -1, -3:-1
            ]
            initial_future_speeds = all_other_agents_future_speeds[..., 0]
            yield_deceleration = aux_info.get(
                "interaction_yield_belief_deceleration_mps2"
            )
            if yield_deceleration is None:
                yield_deceleration = torch.zeros(
                    all_other_agents_future_positions.shape[:3],
                    dtype=all_other_agents_future_positions.dtype,
                    device=all_other_agents_future_positions.device,
                )
            elif yield_deceleration.shape[0] != future_batch:
                if future_batch % yield_deceleration.shape[0] != 0:
                    raise ValueError(
                        "yield-belief sample/mask batch dimensions disagree"
                    )
                yield_deceleration = yield_deceleration.repeat_interleave(
                    future_batch // yield_deceleration.shape[0], dim=0
                )
            all_other_agents_future_positions, all_other_agents_future_speeds = (
                apply_directed_yield_belief(
                    all_other_agents_future_positions,
                    all_other_agents_future_speeds,
                    last_observed_relative_positions,
                    initial_future_speeds,
                    yield_deceleration,
                    dt=0.1,
                )
            )

            # future availabilities depend on the current time step availability
            # (B*N, M, M, T_hist, k_vehicle) -> (B*N, M, M) -> (B*N, M, M, 1) -> (B*N, M, M, T_fut)
            all_other_agents_future_availabilities = aux_info['neighbor_hist_feat'][...,-1,-1].bool().unsqueeze(-1).repeat(1,1,1,T_fut)

            neighbor_fut, info_mask = self.prepare_scene_agent_hist(all_other_agents_future_positions, all_other_agents_future_yaws, all_other_agents_future_speeds, all_other_agents_extents, all_other_agents_future_availabilities, self.neighbor_fut_norm_info, scale=self.normalize_rel_states, speed_repr=self.interaction_edge_speed_repr)
        elif self.use_const_speed_edge or x is None:
            # (B*N, M, M, T_hist, K_neigh) -> (B*N, M, M, K_neigh) -> (B*N, M, M, 1, K_neigh) -> (B*N, M, M, T_fut, K_neigh) (x,y,cos,sin,speed,l,w,avail) or (x,y,cos,sin,vx,vy,l,w,avail)
            neighbor_fut = aux_info['neighbor_hist_feat'][...,-1,:].unsqueeze(-2).repeat(1,1,1,T_fut,1)
            if self.interaction_edge_speed_repr in ['rel_vel', 'rel_vel_new', 'rel_vel_per_step', 'rel_vel_new_new']:
                dt = 0.1
                # Get a time weight with shape (T_fut) from 0.1 to 0.1*T_fut
                time = torch.arange(1, T_fut+1, dtype=torch.float32, device=neighbor_fut.device) * dt
                # (T_fut) -> (1, 1, 1, T_fut)
                time = time.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                neighbor_fut[...,:2] = neighbor_fut[...,:2] + neighbor_fut[...,4:6] * time
            else:
                raise NotImplementedError

        
        neighbor_fut = apply_directed_future_intent_hold(
            neighbor_fut,
            aux_info["neighbor_hist_feat"],
            aux_info.get("interaction_future_intent_hold_mask"),
        )
        aux_info['neighbor_fut_feat'] = neighbor_fut
        aux_info['neighbor_feat'] = torch.cat([aux_info['neighbor_hist_feat'], aux_info['neighbor_fut_feat']], dim=-2)

        if 'agent_hist_feat_non_cond' in aux_info:
            if self.use_last_hist_step:
                # TBD: if cond_fill_value is not zero, this needs to be changed
                neighbor_fut_non_cond = torch.zeros_like(neighbor_fut)
            elif self.use_noisy_fut_edge:
                all_other_agents_future_availabilities_non_cond = torch.zeros_like(all_other_agents_future_availabilities, dtype=torch.bool, device=all_other_agents_future_availabilities.device)
                neighbor_fut_non_cond, info_mask = self.prepare_scene_agent_hist(all_other_agents_future_positions, all_other_agents_future_yaws, all_other_agents_future_speeds, all_other_agents_extents, all_other_agents_future_availabilities_non_cond, self.neighbor_fut_norm_info, scale=self.normalize_rel_states, speed_repr=self.interaction_edge_speed_repr)
            elif self.use_const_speed_edge:
                pass
            
            aux_info['neighbor_fut_feat_non_cond'] = neighbor_fut_non_cond
            aux_info['neighbor_feat_non_cond'] = torch.cat([aux_info['neighbor_hist_feat_non_cond'], aux_info['neighbor_fut_feat_non_cond']], dim=-2)

        return aux_info
        
    def get_neighbor_history_relative_states(self, data_batch: dict, include_class_free_cond: bool) -> dict:
        '''
        get the neighbor history relative states (only need once per data_batch). We do this because fields like all_other_agents_history_positions in data_batch may not include all the agents controlled and may include agents not controlled.

        - output neighbor_hist: (B, M, M, T_hist, K_vehicle)
        - output neighbor_hist_non_cond: (B, M, M, T_hist, K_vehicle)
        '''
        M = data_batch['history_positions'].shape[1]

        all_other_agents_history_positions, all_other_agents_history_yaws, all_other_agents_history_speeds, all_other_agents_extents = self.get_neighbor_relative_states(data_batch['history_positions'], data_batch['history_speeds'], data_batch['history_yaws'], data_batch['agent_from_world'], data_batch['world_from_agent'], data_batch['yaw'], data_batch["extent"])

        # (B, M, T_hist) -> (B, 1, M, T_hist) -> (B, M, M, T_hist)
        all_other_agents_history_availabilities = data_batch["history_availabilities"].unsqueeze(1).repeat(1,M,1,1)

        neighbor_hist, info_mask = self.prepare_scene_agent_hist(all_other_agents_history_positions, all_other_agents_history_yaws, all_other_agents_history_speeds, all_other_agents_extents, all_other_agents_history_availabilities, self.neighbor_fut_norm_info, scale=self.normalize_rel_states, speed_repr=self.interaction_edge_speed_repr)

        # # (B, M, M, T_hist, K_vehicle) -> (B, M, M, T_hist, K_neigh)
        # neighbor_hist_feat = self.neighbor_hist_encoder(neighbor_hist)

        neighbor_hist_non_cond = None
        
        return neighbor_hist, neighbor_hist_non_cond

    def query_neigh_feats(self, neighbor_fut):
        '''
        -params neighbor_fut : (B*N/B, M, M, T_fut, K_neigh)
        -output neighbor_fut_feat : (B*N/B, M, T_fut, K_d)
        '''
        # (B*N, M, M, T_fut, K_neigh)
        BN, M, _, T, _ = neighbor_fut.shape

        # (B*N, M, M, T_fut, K_neigh) -> (B*N, M, T_fut, M, K_neigh)
        neighbor_fut = neighbor_fut.permute(0, 1, 3, 2, 4)

        # --- Sort neighbors according to distances ---
        
        # Assign large values to invalid neighbors
        invalid_mask = neighbor_fut[..., -1] == 0
        expanded_mask = invalid_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 2)
        neighbor_fut[..., :2].masked_fill_(expanded_mask, torch.inf)
        
        # sort neighbor indices
        norms = torch.norm(neighbor_fut[..., :2], dim=-1)
        _, sorted_indices = torch.sort(norms, dim=-1) 
        # assign 0 to neighbors far away
        neighbor_fut[norms > self.social_attn_radius] = 0
        neighbor_fut = torch.gather(neighbor_fut, 3, sorted_indices.unsqueeze(-1).expand_as(neighbor_fut))
        # # remove self, (B*N, M, T_fut, M, K_neigh) -> (B*N, M, T_fut, M-1, K_neigh)
        # neighbor_fut = neighbor_fut[...,1:,:]

        # --- Encoder neighbors using MLP ---
        # (B*N, M, T_fut, M, K_neigh) -> (B*N*M*T_fut, M, K_neigh)
        neighbor_fut = neighbor_fut.view(BN*M*T, *neighbor_fut.shape[-2:])
        # (B*N*M*T_fut, M, K_neigh) -> (B*N*M*T_fut, K_d)
        neighbor_fut_feat = self.neighbor_fut_encoder(neighbor_fut)
        # (B*N*M*T_fut, K_d) -> (B*N, M, T_fut, K_d)
        neighbor_fut_feat = neighbor_fut_feat.view(BN, M, T, -1)
        return neighbor_fut_feat
    
    def query_map_feats(self, x, map_grid_feat, raster_from_agent):
        '''
        - x : (B, M, T, D)
        - map_grid_feat : (B, M, C, H, W)
        - raster_from_agent: (B, M, 3, 3)

        - output feats_out : (B, M, T, C)
        '''

        B, M, T, _ = x.shape
        x = x.reshape((B*M, *x.shape[2:]))
        map_grid_feat = map_grid_feat.reshape((B*M, *map_grid_feat.shape[2:]))
        raster_from_agent = raster_from_agent.reshape((B*M, *raster_from_agent.shape[2:]))

        Hfeat, Wfeat = map_grid_feat.shape[-2:]

        # unscale to agent coords
        pos_traj = self.descale_traj(x.detach())[...,:2]
        # convert to raster frame
        raster_pos_traj = GeoUtils.transform_points_tensor(pos_traj, raster_from_agent)

        # scale to the feature map size
        _, H, W = self.input_image_shape
        xscale = Wfeat / W
        yscale = Hfeat / H
        raster_pos_traj[...,0] = raster_pos_traj[...,0] * xscale
        raster_pos_traj[...,1] = raster_pos_traj[...,1] * yscale

        # interpolate into feature grid
        feats_out = query_feature_grid(
                            raster_pos_traj,
                            map_grid_feat
                            )
        feats_out = feats_out.reshape((B, M, T, -1))
        return feats_out

    def get_state_and_action_from_data_batch(self, data_batch, chosen_inds=[]):
        '''
        Extract state and(or) action from the data_batch from data_batch
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            data_batch: dict
        Output:
            x: (batch_size, [num_agents], num_steps, len(chosen_inds)).
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        # NOTE: for predicted agent, history and future with always be fully available
        traj_state = torch.cat(
                (data_batch["target_positions"][..., :self.horizon, :], data_batch["target_yaws"][..., :self.horizon, :]), dim=-1)
        traj_state_and_action = convert_state_to_state_and_action(traj_state, data_batch["curr_speed"], self.dt)

        return traj_state_and_action[..., chosen_inds]
    
    def convert_action_to_state_and_action(self, x_out, curr_states, scaled_input=True, descaled_output=False):
        '''
        Apply dynamics on input action trajectory to get state+action trajectory
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            x_out: (B, (M), T, 2). scaled action trajectory
            curr_states: (B, (M), 4). current state
        Output:
            x_out: (B, (M), T, 6). scaled state+action trajectory
        '''

        if scaled_input:
            x_out = self.descale_traj(x_out, [4, 5])
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=curr_states,
            actions=x_out,
            step_time=self.dt,
            mode='parallel',
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)
        if scaled_input and not descaled_output:
            x_out_all = self.scale_traj(x_out_all, [0, 1, 2, 3, 4, 5])

        return x_out_all

    def scale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        scale the trajectory from original scale to standard normal distribution
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            - target_traj_orig: (B, (M), T, D)
        Output:
            - target_traj: (B, (M), T, D)
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = (target_traj_orig + dx_add) / dx_div

        return target_traj

    def descale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        scale back the trajectory from standard normal distribution to original scale
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            - target_traj_orig: (B, (M), T, D)
        Output:
            - target_traj: (B, (M), T, D)
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device) 

        target_traj = target_traj_orig * dx_div - dx_add

        return target_traj

    
    def forward(self, data_batch: Dict[str, torch.Tensor], plan=None, num_samp=1,
                    return_diffusion=False,
                    return_guidance_losses=False,
                    class_free_guide_w=0.0,
                    apply_guidance=True,
                    guide_clean=False,
                    global_t=0, stationary_mask=None, mode='testing') -> Dict[str, torch.Tensor]:
        # update stationary_mask at the beginnig of each rollout
        if global_t == 0:
            self.stationary_mask = stationary_mask
        self.global_t = global_t
        use_class_free_guide = class_free_guide_w != 0.0
        aux_info = self.get_aux_info(data_batch, use_class_free_guide)

        cond_samp_out = self.conditional_sample(data_batch, 
                                                horizon=None,
                                                aux_info=aux_info,
                                                return_diffusion=return_diffusion,
                                                return_guidance_losses=return_guidance_losses,
                                                num_samp=num_samp,
                                                verbose=False, class_free_guide_w=class_free_guide_w,
                                                apply_guidance=apply_guidance,
                                                guide_clean=guide_clean, mode=mode)
        traj_init = cond_samp_out['pred_traj']
        attn_weights = cond_samp_out['attn_weights']
        diff_init = guide_losses = None
        if return_diffusion:
            diff_init = cond_samp_out['diffusion']
        if return_guidance_losses:
            guide_losses = cond_samp_out['guide_losses']

        traj = self.descale_traj(traj_init)
        if diff_init is not None:
            diff_steps = self.descale_traj(diff_init)
        else:
            diff_steps = None

        if self.diffuser_input_mode in ['state_and_action', 'state_and_action_no_dyn']:
            traj_pred = traj[..., [0, 1, 3]]
        elif self.diffuser_input_mode == 'action':
            traj = self.convert_action_to_state_and_action(traj_init, aux_info['curr_states'], scaled_input=False)
            traj_pred = traj[..., [0, 1, 3]]
        elif self.diffuser_input_mode == 'state':
            pass
        else:
            raise

        pred_positions = traj_pred[..., :2]
        pred_yaws = traj_pred[..., 2:3]
        
        out_dict = {
            "predictions": {"positions": pred_positions, "yaws": pred_yaws, "trajectories": traj, "attn_weights": attn_weights},
        }
        if diff_steps is not None:
            out_dict["predictions"]["diffusion_steps"] = diff_steps # (1, B*N, M, T, transition_dim)
        if guide_losses is not None:
            out_dict["predictions"]["guide_losses"] = guide_losses
        if self.dyn is not None:
            out_dict["curr_states"] = aux_info['curr_states']
        return out_dict

    def compute_losses(self, data_batch):
        aux_info = self.get_aux_info(data_batch)
        target_traj = self.get_state_and_action_from_data_batch(data_batch)
        
        if self.use_reconstructed_state and self.diffuser_input_mode in ['state_and_action', 'state_and_action_no_dyn']:
            target_traj = self.convert_action_to_state_and_action(target_traj[..., [4, 5]], aux_info['curr_states'], scaled_input=False)        


        x = self.scale_traj(target_traj)
                
        diffusion_loss, info = self.loss(x, data_batch, aux_info=aux_info)
        losses = OrderedDict(
            diffusion_loss=diffusion_loss,
        )
        if 'collision_loss' in info:
            losses['collision_loss'] = info['collision_loss']
        if 'offroad_loss' in info:
            losses['offroad_loss'] = info['offroad_loss']
        if 'history_reconstruction_loss' in info:
            losses['history_reconstruction_loss'] = info['history_reconstruction_loss']
        
        return losses

    def get_loss_weights(self, action_weight, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        ## manually set a0 weight
        loss_weights[0, -self.action_dim:] = action_weight

        if self.action_loss_only:
            loss_weights = loss_weights[:, -2:]
        return loss_weights

    def predict_start_from_noise(self, x_t, t, noise, force_noise=False):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon or force_noise:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x_start):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return x_start
        else:
            return (
                extract(self.sqrt_recip_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                extract(self.sqrt_alphas_over_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_start
            )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, aux_info={}, class_free_guide_w=0.0, data_batch=None, x_clean_model_out=None):

        t_inp = t
        
        x_model_in = x
        
        if self.map_embed_method in ['cnn', 'cnn_local_patch']:
            if self.use_map_feat_grid and self.map_encoder is not None:
                # get features from map and append to the trajectory
                map_feat_traj = self.query_map_feats(x.detach().clone(),
                                                    aux_info['map_grid_feat'],
                                                    aux_info['raster_from_agent'])
                x_model_in = torch.cat([x_model_in, map_feat_traj], dim=-1)

        if self.neigh_hist_embed_method in ['interaction_edge', 'interaction_edge_and_input']:
            # fill in noise at the first denoising step
            if self.use_noisy_fut_edge:
                x_edge = x.detach().clone()
            else:
                if x_clean_model_out is None:
                    x_edge = None
                else:
                    x_edge = self.convert_action_to_state_and_action(x_clean_model_out.detach(), aux_info['curr_states'])
            aux_info = self.get_neighbor_future_relative_states(x_edge, aux_info, data_batch)

            if self.neigh_hist_embed_method == 'interaction_edge_and_input':
                # (B*N, M, M, T_fut, K_neigh) -> (B*N, M, T_fut, K_d)
                neighbor_fut_feat = self.query_neigh_feats(aux_info['neighbor_fut_feat'])
                # (B*N, M, T_fut, D) + (B*N, M, T_fut, K_d) -> (B*N, M, T_fut, D+K_d)
                x_model_in = torch.cat([x_model_in, neighbor_fut_feat], dim=-1)
        
        # visualzie attention weigths for the last denoising step of each rollout step
        if t_inp[0] == 0:
            aux_info['attn_name'] = ""
        else:
            aux_info['attn_name'] = ""

        model_prediction, info = self.model(x_model_in, aux_info, t_inp, motion_dist=self.motion_dist, use_cond=False)

        if self.agent_hist_embed_method == 'concat':
            model_prediction = model_prediction[..., -self.horizon:,:]

        if self.diffuser_input_mode == 'state_and_action':
            x_tmp = x[..., 4:].detach()
        else:
            x_tmp = x.detach()

        if class_free_guide_w != 0.0:
            
            model_non_cond_prediction, _ = self.model(x_model_in, aux_info, t_inp, motion_dist=self.motion_dist, use_cond=True)

            if self.agent_hist_embed_method == 'concat':
                model_non_cond_prediction = model_non_cond_prediction[..., -self.horizon:,:]
            # and combine to get actual model prediction (in noise space as in original paper)
            model_pred_noise = self.predict_noise_from_start(x_tmp, t=t, x_start=model_prediction)
            model_non_cond_pred_noise = self.predict_noise_from_start(x_tmp, t=t, x_start=model_non_cond_prediction)
            class_free_guide_noise = (1 + class_free_guide_w)*model_pred_noise - class_free_guide_w*model_non_cond_pred_noise
            model_prediction = self.predict_start_from_noise(x_tmp, t=t, noise=class_free_guide_noise, force_noise=True)

        x_recon = self.predict_start_from_noise(x_tmp, t=t, noise=model_prediction)

        
        if self.disable_control_on_stationary:
            if self.diffuser_input_mode == 'state_and_action':
                inds = [4, 5]
            else:
                inds = self.default_chosen_inds
            x_recon_stationary = x_recon[self.stationary_mask]
            x_recon_stationary = self.descale_traj(x_recon_stationary, inds)
            x_recon_stationary[...] = 0
            x_recon_stationary = self.scale_traj(x_recon_stationary, inds)
            x_recon[self.stationary_mask] = x_recon_stationary

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x_tmp, t=t)
        return model_mean, posterior_variance, posterior_log_variance, (x_recon, x_tmp, t), model_prediction, info

    def state_action_grad_inner_transform(self, x_guidance, data_batch, transform_params, **kwargs):
        bsize = kwargs.get('bsize', x_guidance.shape[0])
        num_samp = kwargs.get('num_samp', 1)

        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())
        expand_states = curr_states.unsqueeze(1).expand((bsize, num_samp, 4)).reshape((bsize*num_samp, 4))

        x_all = self.convert_action_to_state_and_action(x_guidance, expand_states, scaled_input=transform_params['scaled_input'], descaled_output=transform_params['scaled_output'])
        return x_all

    def state_action_no_dyn_grad_inner_transform(self, x_guidance, data_batch, transform_params, **kwargs):
        bsize = kwargs.get('bsize', x_guidance.shape[0])
        num_samp = kwargs.get('num_samp', 1)

        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())
        expand_states = curr_states.unsqueeze(1).expand((bsize, num_samp, 4)).reshape((bsize*num_samp, 4))

        x_all = self.convert_action_to_state_and_action(x_guidance[..., -2:], expand_states, scaled_input=transform_params['scaled_input'], descaled_output=transform_params['scaled_output'])
        return x_all
        
    def state_grad_inner_transform(self, x_guidance, data_batch, transform_params, **kwargs):
        bsize = kwargs.get('bsize', x_guidance.shape[0])
        num_samp = kwargs.get('num_samp', 1)

        x_state = self.descale_traj(x_guidance, [0, 1, 3])
        curr_speed = data_batch['curr_speed']
        expand_speed = curr_speed.unsqueeze(1).expand((bsize, num_samp)).reshape((bsize*num_samp))
        x_all = convert_state_to_state_and_action(x_state, expand_speed, self.dt)
        return x_all

    @torch.no_grad()
    def p_sample(self, x, t, data_batch, aux_info={}, num_samp=1, class_free_guide_w=0.0, apply_guidance=True, guide_clean=False, eval_final_guide_loss=False, x_clean_model_out=None, data_batch_for_guidance={}):
        # print('init perturbation: ', self.current_perturbation_guidance.available_idx, self.current_perturbation_guidance)
        b, *_, device = *x.shape, x.device
        with_func = torch.no_grad
        if self.current_perturbation_guidance.current_guidance is not None and apply_guidance and guide_clean == "video_diff":
            # will need to take grad wrt noisy
            x = x.detach()
            x.requires_grad_()
            with_func = torch.enable_grad
        with with_func():
            # get prior mean and variance for next step
            model_mean, _, model_log_variance, q_posterior_in, x_clean_model_out, info = self.p_mean_variance(x=x, t=t, aux_info=aux_info, class_free_guide_w=class_free_guide_w, data_batch=data_batch, x_clean_model_out=x_clean_model_out)
        sigma = (0.5 * model_log_variance).exp()
        
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        if self.current_perturbation_guidance.current_guidance is not None and apply_guidance and guide_clean:
            # want to guide the predicted clean traj from model, not the noisy one
            x_initial = q_posterior_in[0]
            return_grad_of = x
        else:
            x_initial = model_mean.clone().detach()
            return_grad_of = x_initial
            x_initial.requires_grad_()

        guide_losses = dict()
        x_guidance = None
        # consider intermediate and final guidance (for ablation) separately
        if apply_guidance and self.guidance_optimization_params is not None:
            if t[0] == 0:
                apply_guidance = self.apply_guidance_output
                if apply_guidance:
                    opt_params = self.final_step_opt_params
            else:
                apply_guidance = self.apply_guidance_intermediate
                if apply_guidance:
                    assert self.guidance_optimization_params['grad_steps'] > 0
                    perturb_th = self.guidance_optimization_params['perturb_th']
                    apply_guidance_output = self.apply_guidance_output
                    lr = self.guidance_optimization_params['lr']
                    
                    if perturb_th is not None:
                        # sigma scale scheduling
                        sig_scale = (torch.sigmoid(10 * t[0] / self.n_timesteps) - 1/2) * 2 
                        perturb_th = sig_scale * (4-perturb_th) + perturb_th
                        if not apply_guidance_output:
                            perturb_th = perturb_th * nonzero_mask
                    else:
                        if t[0] == 0 and not apply_guidance_output:
                            perturb_th = nonzero_mask * sigma
                        else:
                            perturb_th = sigma

                    if lr is None:
                        lr = sigma
                    opt_params = deepcopy(self.guidance_optimization_params)
                    opt_params['lr'] = lr
                    opt_params['perturb_th'] = perturb_th
            
            if apply_guidance:
                x_guidance, guide_losses = self.current_perturbation_guidance.perturb(x_initial, data_batch_for_guidance, opt_params, num_samp=num_samp, return_grad_of=return_grad_of)
        # for filtration only
        if x_guidance is None:
            if self.current_perturbation_guidance.current_guidance is not None and eval_final_guide_loss:
                _, guide_losses = self.current_perturbation_guidance.compute_guidance_loss(x_initial, data_batch_for_guidance, num_samp=num_samp)
            x_guidance = x_initial

        # add noise
        noise = torch.randn_like(x_guidance)
        noise = nonzero_mask * sigma * noise
        x_out = x_guidance + noise

        # convert action to state+action
        if self.diffuser_input_mode == 'state_and_action':
            x_out = self.convert_action_to_state_and_action(x_out, aux_info['curr_states'])
            
        return x_out, guide_losses, x_clean_model_out, info
        
    @torch.no_grad()
    def p_sample_loop(self, shape, data_batch, num_samp, 
                    aux_info={}, 
                    verbose=True, 
                    return_diffusion=False,
                    return_guidance_losses=False,
                    class_free_guide_w=0.0,
                    apply_guidance=True,
                    guide_clean=False,
                    mode='testing'):
        '''
        shape: (5), batch_size, num_samp, num_agents, horizon, self.transition_dim
        '''
        # merge B and M to be compatible with guidance loss developed for agent-centric models
        data_batch_for_guidance = {}
        if apply_guidance:
            data_batch_for_guidance = extract_data_batch_for_guidance(data_batch, mode=mode)

        device = self.betas.device
        batch_size = shape[0]
        if self.current_perturbation_guidance.current_guidance is not None and not apply_guidance:
            print('DIFFUSER: Note, not using guidance during sampling, only evaluating guidance loss at very end...')

        # sample from base distribution
        x = torch.randn(shape, device=device) # (B, N, M, T(+T_hist), transition_dim)
        x = TensorUtils.join_dimensions(x, begin_axis=0, end_axis=2) # (B*N, M, T(+T_hist), transition_dim)
        x_clean_model_out = None

        # (B, M, C) -> (B*N, M, C)
        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=num_samp, dim=0)
        if return_diffusion: diffusion = [x] #(1, B*N, M, T, transition_dim)
        progress = Progress(self.n_timesteps) if verbose else Silent()

        steps = [i for i in reversed(range(0, self.n_timesteps, self.stride))]
        attn_weights = []
        for i in steps:
            # (B*N)
            timesteps = torch.full((batch_size*num_samp,), i, device=device, dtype=torch.long)
            x, guide_losses, x_clean_model_out, info = self.p_sample(x, timesteps, data_batch, aux_info=aux_info, num_samp=num_samp, class_free_guide_w=class_free_guide_w,
            apply_guidance=apply_guidance, guide_clean=guide_clean, eval_final_guide_loss=(i == steps[-1]), x_clean_model_out=x_clean_model_out, data_batch_for_guidance=data_batch_for_guidance)

            # apply hard constraints (overwrite waypoints at certain timesteps)
            if self.current_constraints is not None: 
                # apply constraints expects traj in shape (B, N, T, D) and metric space
                x = self.descale_traj(x.reshape((shape[0], shape[1], shape[2], -1)))
                x = apply_constraints(x, data_batch['scene_index'], self.current_constraints)
                x = self.scale_traj(x.reshape((shape[0]*shape[1], shape[2], -1)))


            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
            if i == 0:
                attn_weights = info['attn_weights']

        progress.close()

        if any(guide_losses):
            print('===== GUIDANCE LOSSES ======')
            for k,v in guide_losses.items():
                if torch.isnan(v).any() or len(v) == 0:
                    v_mean = np.nan
                else:
                    v_mean = np.nanmean(v.cpu())
                print('%s: %.012f' % (k, v_mean))
        # (B*N, M, T(+T_hist), output_dim) -> (B, N, M, T(+T_hist), output_dim)
        x = TensorUtils.reshape_dimensions(x, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp))

        out_dict = {'pred_traj' : x, 'attn_weights': attn_weights}
        if return_guidance_losses:
            out_dict['guide_losses'] = guide_losses
        if return_diffusion:
            diffusion = [TensorUtils.reshape_dimensions(cur_diff, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp))
                         for cur_diff in diffusion]
            out_dict['diffusion'] = torch.stack(diffusion, dim=3)

        return out_dict


    @torch.no_grad()
    def conditional_sample(self, data_batch, horizon=None, num_samp=1, class_free_guide_w=0.0, **kwargs):
        batch_size, num_agents = data_batch['history_positions'].size()[:2]
        horizon = horizon or self.horizon
        shape = (batch_size, num_samp, num_agents, horizon, self.transition_dim)

        return self.p_sample_loop(shape, data_batch, num_samp, class_free_guide_w=class_free_guide_w, **kwargs)

    def q_sample(self, x_start, t, noise):        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
    
    def p_losses(self, x_start_init, t, data_batch, aux_info={}):
        noise_init = torch.randn_like(x_start_init)
        x_start = x_start_init
        noise = noise_init
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        t_inp = t

        if self.diffuser_input_mode == 'state_and_action':
            x_action_noisy = x_noisy[..., [4, 5]]
            x_noisy = self.convert_action_to_state_and_action(x_action_noisy, aux_info['curr_states'])
        
        x_noisy_in = x_noisy
        if self.map_embed_method in ['cnn', 'cnn_and_input']:
            if self.use_map_feat_grid and self.map_encoder is not None:
                # get features from map and append to the trajectory
                map_feat_traj = self.query_map_feats(x_noisy.detach().clone(),
                                                    aux_info['map_grid_feat'],
                                                    aux_info['raster_from_agent'])
                x_noisy_in = torch.cat([x_noisy_in, map_feat_traj], dim=-1)

        if self.neigh_hist_embed_method in ['interaction_edge', 'interaction_edge_and_input']:
            if self.use_noisy_fut_edge:
                x_edge = x_noisy.detach().clone()
            else:
                x_edge = x_start_init.detach().clone()
            aux_info = self.get_neighbor_future_relative_states(x_edge, aux_info, data_batch)

            if self.neigh_hist_embed_method == 'interaction_edge_and_input':
                # (B*N, M, M, T_fut, K_neigh) -> (B*N, M, T_fut, K_d)
                neighbor_fut_feat = self.query_neigh_feats(aux_info['neighbor_fut_feat'])
                # (B*N, M, T_fut, D) + (B*N, M, T_fut, K_d) -> (B*N, M, T_fut, D+K_d)
                x_noisy_in = torch.cat([x_noisy_in, neighbor_fut_feat], dim=-1)
        noise, _ = self.model(x_noisy_in, aux_info, t_inp)
        if self.agent_hist_embed_method == 'concat':
            noise_hist = noise[..., :-self.horizon, :]
            noise_fut = noise[..., -self.horizon:, :]
        else:
            noise_fut = noise

        if self.diffuser_input_mode == 'state_and_action':
            # Note: if predict_eps, we convert noise into x_start for loss estimation since we need to apply forward dynamics
            x_recon_action = self.predict_start_from_noise(x_action_noisy, t=t, noise=noise_fut)
            x_recon = self.convert_action_to_state_and_action(x_recon_action, aux_info['curr_states'])
        else:
            x_recon = self.predict_start_from_noise(x_noisy_in, t=t, noise=noise_fut)
        


        if self.diffuser_input_mode in ['action', 'state_and_action', 'state_and_action_no_dyn'] and self.action_loss_only:
            x_recon_selected, x_start_selected = x_recon[..., -2:], x_start[..., -2:]
        else:
            x_recon_selected, x_start_selected = x_recon, x_start
        
        if self.use_target_availabilities:
            x_recon_selected = x_recon_selected * data_batch['target_availabilities'][..., :self.horizon].unsqueeze(-1)
            x_start_selected = x_start_selected * data_batch['target_availabilities'][..., :self.horizon].unsqueeze(-1)
        # currently other losses can only used if agent_centric
        if self.coordinate == 'agent_centric':
            # merge B and M to be compatible with guidance loss developed for agent-centric models
            data_batch_for_guidance = extract_data_batch_for_guidance(data_batch, mode='training')
            loss, info = self.loss_fn(x_recon_selected, x_start_selected, data_batch_for_guidance)
        else:
            loss, info = self.loss_fn(x_recon_selected, x_start_selected, {})

        if self.agent_hist_embed_method == 'concat':
            # TBD: currently only support predicting x0
            assert not self.predict_epsilon
            history_reconstruction_loss = self.estimate_history_reconstruction_loss(noise_hist, data_batch)
            info['history_reconstruction_loss'] = history_reconstruction_loss

        return loss, info

    def loss(self, x, data_batch, aux_info={}):
        batch_size = len(x)

        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            
        return self.p_losses(x, t, data_batch, aux_info=aux_info)

    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return torch.tensor(list(reversed(out))).flatten().cpu().numpy()

    def estimate_history_reconstruction_loss(self, noise_hist, data_batch):
        '''
        noise_hist: [B, M, T_hist, 2]
        TBD: get history diffuser norm info to apply
        '''
        x_recon_action_hist = noise_hist

        # Get initial history state
        hist_start_state = torch.zeros(x_recon_action_hist.shape[0], x_recon_action_hist.shape[1], 4).to(x_recon_action_hist.device)  # [x, y, vel, yaw]
        hist_start_state[..., :2] = data_batch["history_positions"][..., 0, :]
        hist_start_state[..., 2] = data_batch["history_speeds"][..., 0]
        hist_start_state[..., 3] = data_batch["history_yaws"][..., 0, 0]

        # Get predicted state and action
        x_recon_hist = self.convert_action_to_state_and_action(x_recon_action_hist[...,1:,:], hist_start_state)

        # Get GT state and action
        traj_state_history = torch.cat(
            (data_batch["history_positions"][..., 1:, :], data_batch["history_yaws"][..., 1:, :]), dim=-1)
        vel_init = hist_start_state[..., 2]

        x_start_hist = convert_state_to_state_and_action(traj_state_history, vel_init, self.dt, data_type='torch')
        x_start_hist = self.scale_traj(x_start_hist)

        # Select relevant fields
        if self.diffuser_input_mode in ['action', 'state_and_action', 'state_and_action_no_dyn'] and self.action_loss_only:
            x_recon_hist_selected, x_start_hist_selected = x_recon_hist[..., -2:], x_start_hist[..., -2:]
        else:
            x_recon_hist_selected, x_start_hist_selected = x_recon_hist, x_start_hist

        # Apply availabilities
        if self.use_target_availabilities:
            x_recon_hist_selected = x_recon_hist_selected * data_batch['history_availabilities'][..., 1:].unsqueeze(-1)
            x_start_hist_selected = x_start_hist_selected * data_batch['history_availabilities'][..., 1:].unsqueeze(-1)
            # invalidate those that are not available at the first timestep in history
            x_recon_hist_selected[data_batch['history_availabilities'][..., 0]==False] = 0
            x_start_hist_selected[data_batch['history_availabilities'][..., 0]==False] = 0
        
        # Estimate loss
        history_loss = torch.mean(torch.nn.functional.mse_loss(x_recon_hist_selected, x_start_hist_selected, reduction='none'))

        return history_loss


class CCFlowMatching(CCDiffuserModel):
    """Flow-matching transport with the unchanged CCDiff SceneTransformer.

    The transported variable is the two-dimensional normalized unicycle action.
    At every flow time the six-dimensional state/action input is reconstructed by
    the same dynamics used by :class:`CCDiffuserModel`.
    """

    def __init__(self, *args, n_flow_steps=20, **kwargs):
        super().__init__(*args, **kwargs)
        if self.diffuser_input_mode != 'state_and_action':
            raise ValueError(
                "CCFlowMatching requires diffuser_input_mode='state_and_action'"
            )
        if self.predict_epsilon:
            raise ValueError("CCFlowMatching requires predict_epsilon=False")
        if int(n_flow_steps) <= 0:
            raise ValueError("n_flow_steps must be positive")
        self.n_flow_steps = int(n_flow_steps)
        # Default-off inference-only switch. Existing training, validation and
        # native sampling retain the historical detached conditioning graph.
        # Conditional-FM may enable this only for source-gradient evaluation;
        # tensor values must remain identical to the native path.
        self.conditional_flow_differentiable_conditioning = False
        # Inference-only source-control proposal.  ``None`` preserves the exact
        # native M2 path.  The dictionary contains validated dataclass configs;
        # it is deliberately absent from the checkpoint/state_dict contract.
        self.conditional_flow_source_control = None
        self._conditional_flow_last_audit = None
        # Optional inference-only typed collision projection.  Training,
        # checkpoint tensors and the learned FM transport remain unchanged.
        self.typed_collision_projection = None
        self._typed_projection_calls = 0
        self._typed_projection_min_distance_m = float("inf")
        self._typed_projection_latched = False
        self._typed_projection_previous_joint_world_positions = None
        self._typed_projection_previous_track_ids = None

    def set_conditional_flow_differentiable_conditioning(self, enabled):
        """Expose noisy-edge Jacobians without changing frozen model values."""
        enabled = bool(enabled)
        if enabled and self.training:
            raise RuntimeError(
                "differentiable Conditional-FM conditioning is inference-only"
            )
        if enabled and self.typed_collision_projection is not None:
            raise RuntimeError(
                "Conditional-FM cannot coexist with endpoint projection"
            )
        if enabled and self.current_constraints is not None:
            raise RuntimeError(
                "Conditional-FM cannot coexist with output constraints"
            )
        self.conditional_flow_differentiable_conditioning = enabled

    def set_conditional_flow_source_control(
        self, source_config, objective_config
    ):
        """Install a complete-joint-source Conditional-FM proposal arm.

        This is intentionally inference-only and mutually exclusive with all
        endpoint guidance/projection/constraint paths.  The selected K mode is
        optimized as one complete Gaussian source; the remaining K-1 modes are
        retained bit-for-bit from the native source bank.
        """
        if self.training:
            raise RuntimeError("Conditional-FM source control is inference-only")
        source_config.validate()
        if self.typed_collision_projection is not None:
            raise RuntimeError("Conditional-FM forbids endpoint projection")
        if self.current_constraints is not None:
            raise RuntimeError("Conditional-FM forbids output constraints")
        if self.current_perturbation_guidance.current_guidance is not None:
            raise RuntimeError("Conditional-FM forbids legacy gradient guidance")
        self.conditional_flow_source_control = {
            "source_config": source_config,
            "objective_config": objective_config,
        }
        self._conditional_flow_last_audit = None

    def clear_conditional_flow_source_control(self):
        self.conditional_flow_source_control = None
        self._conditional_flow_last_audit = None
        self.set_conditional_flow_differentiable_conditioning(False)

    def set_typed_collision_projection(self, config):
        """Install a typed action-space projection for one target pair."""
        required = {"target_idx", "reference_idx", "collision_type"}
        missing = required - set(config)
        if missing:
            raise ValueError("typed projection missing {}".format(sorted(missing)))
        collision_type = str(config["collision_type"])
        if collision_type not in {"rear_end", "side", "cut_in", "head_on"}:
            raise ValueError("unsupported collision type {}".format(collision_type))
        parsed = dict(config)
        parsed["target_idx"] = int(parsed["target_idx"])
        parsed["reference_idx"] = int(parsed["reference_idx"])
        parsed.setdefault("activation_distance_m", 25.0)
        parsed.setdefault("projection_step_size", 0.35)
        parsed.setdefault("damping", 1e-3)
        parsed.setdefault("max_normalized_delta", 0.75)
        parsed.setdefault("min_collision_step", 5)
        parsed.setdefault("max_collision_step", 15)
        parsed.setdefault("collateral_margin_m", 0.5)
        parsed.setdefault("latch_after_distance_rebound_m", 1.0)
        parsed.setdefault("temporal_only", False)
        parsed["temporal_only"] = bool(parsed["temporal_only"])
        parsed.setdefault("native_joint_selection", False)
        parsed["native_joint_selection"] = bool(
            parsed["native_joint_selection"]
        )
        parsed.setdefault("final_step_only", False)
        parsed["final_step_only"] = bool(parsed["final_step_only"])
        parsed.setdefault("phase_shift_steps", 0)
        parsed["phase_shift_steps"] = int(parsed["phase_shift_steps"])
        parsed.setdefault("max_phase_path_displacement_m", 6.0)
        parsed.setdefault("max_phase_heading_change_deg", 10.0)
        parsed.setdefault("joint_continuity_enabled", False)
        parsed["joint_continuity_enabled"] = bool(
            parsed["joint_continuity_enabled"]
        )
        parsed.setdefault("joint_continuity_shift_steps", 5)
        parsed["joint_continuity_shift_steps"] = int(
            parsed["joint_continuity_shift_steps"]
        )
        parsed.setdefault("joint_continuity_risk_slack", 1.0)
        parsed.setdefault("joint_continuity_max_mean_error_m", 2.0)
        if parsed["joint_continuity_shift_steps"] < 0:
            raise ValueError("joint continuity shift must be non-negative")
        if parsed["joint_continuity_risk_slack"] < 0:
            raise ValueError("joint continuity risk slack must be non-negative")
        if parsed["joint_continuity_max_mean_error_m"] <= 0:
            raise ValueError("joint continuity error gate must be positive")
        if parsed["phase_shift_steps"] and not parsed["temporal_only"]:
            raise ValueError("phase control requires temporal_only=True")
        if parsed["phase_shift_steps"] and not parsed["native_joint_selection"]:
            raise ValueError("phase control requires native_joint_selection=True")
        self.typed_collision_projection = parsed
        self._typed_projection_calls = 0
        self._typed_projection_min_distance_m = float("inf")
        self._typed_projection_latched = False
        self._typed_projection_previous_joint_world_positions = None
        self._typed_projection_previous_track_ids = None
        print(
            "TYPED_COLLISION_PROJECTION_ATTESTATION target={} reference={} "
            "type={} activation_distance_m={} step_size={} temporal_only={} "
            "native_joint_selection={} final_step_only={} phase_shift_steps={} "
            "joint_continuity_enabled={} "
            "training_changed=False transport=flow_matching".format(
                parsed["target_idx"], parsed["reference_idx"],
                parsed["collision_type"], parsed["activation_distance_m"],
                parsed["projection_step_size"], parsed["temporal_only"],
                parsed["native_joint_selection"], parsed["final_step_only"],
                parsed["phase_shift_steps"],
                parsed["joint_continuity_enabled"],
            )
        )

    def clear_typed_collision_projection(self):
        self.typed_collision_projection = None
        self._typed_projection_calls = 0
        self._typed_projection_min_distance_m = float("inf")
        self._typed_projection_latched = False
        self._typed_projection_previous_joint_world_positions = None
        self._typed_projection_previous_track_ids = None

    def _typed_projection_latch_state(self, current_distance_m, step_index):
        """One-shot event latch evaluated once per closed-loop replan."""
        if step_index != 0:
            return bool(self._typed_projection_latched)
        current = float(current_distance_m)
        previous_min = float(self._typed_projection_min_distance_m)
        activation = float(
            self.typed_collision_projection["activation_distance_m"]
        )
        rebound = float(
            self.typed_collision_projection["latch_after_distance_rebound_m"]
        )
        if (
            not self._typed_projection_latched
            and previous_min <= activation
            and current > previous_min + rebound
        ):
            self._typed_projection_latched = True
            print(
                "TYPED_COLLISION_PROJECTION_LATCHED current_distance_m={:.3f} "
                "previous_min_distance_m={:.3f} reason=passed_closest_approach".format(
                    current, previous_min
                )
            )
        self._typed_projection_min_distance_m = min(previous_min, current)
        return bool(self._typed_projection_latched)

    @staticmethod
    def _temporal_only_gradient(gradient):
        """Restrict a projection gradient to longitudinal FM action only.

        The second action channel is yaw-rate.  Returning a cloned tensor keeps
        the unmodified FM endpoint and its autograd buffer isolated from the
        inference-only timing intervention.
        """
        temporal_gradient = gradient.clone()
        temporal_gradient[..., 1] = 0.0
        return temporal_gradient

    @staticmethod
    def _restore_native_yaw_rate(projected_target, native_target):
        """Preserve the FM sample's steering path exactly in temporal mode."""
        restored = projected_target.clone()
        restored[..., 1] = native_target[..., 1]
        return restored

    @staticmethod
    def _select_native_joint_candidate(residual, num_samp, eligible=None):
        """Rank complete native FM joint modes by typed residual.

        The current closed-loop evaluator is scene-batch one.  Requiring that
        shape here prevents accidental per-agent or cross-scene mode mixing.
        """
        if residual.ndim != 2 or residual.shape[0] != int(num_samp):
            raise ValueError(
                "native joint selection requires one scene with K candidates"
            )
        score = torch.linalg.vector_norm(residual, dim=-1)
        if not torch.isfinite(score).all():
            raise ValueError("native joint candidate scores are non-finite")
        if eligible is not None:
            if eligible.shape != score.shape:
                raise ValueError("native joint eligibility shape disagrees")
            if not bool(torch.any(eligible)):
                raise ValueError("no native joint candidate passes phase gates")
            score = torch.where(
                eligible, score,
                torch.full_like(score, torch.finfo(score.dtype).max),
            )
        return int(torch.argmin(score).detach().cpu()), score.detach()

    @staticmethod
    def _select_continuous_native_joint_candidate(
        residual, joint_world_positions, previous_joint_world_positions,
        shift_steps, risk_slack, max_mean_error_m, eligible=None,
    ):
        """Choose a risky complete joint mode without changing joint intent.

        Native K indices have no identity across closed-loop replans.  The
        selector therefore matches trajectories in world space: candidates
        within ``risk_slack`` of the best typed-risk score compete on mean
        overlap error to the previously committed complete joint trajectory.
        If even the closest candidate violates the continuity gate, callers
        must retain the ordinary FM endpoint rather than force a mode jump.
        """
        if joint_world_positions.ndim != 4:
            raise ValueError("joint continuity expects [K,M,T,2] positions")
        if previous_joint_world_positions.ndim != 3:
            raise ValueError("previous joint plan must have shape [M,T,2]")
        if joint_world_positions.shape[1] != previous_joint_world_positions.shape[0]:
            raise ValueError("joint continuity agent rows changed")
        selected_risk, risk_scores = CCFlowMatching._select_native_joint_candidate(
            residual, joint_world_positions.shape[0], eligible
        )
        shift_steps = int(shift_steps)
        overlap = min(
            joint_world_positions.shape[2],
            previous_joint_world_positions.shape[1] - shift_steps,
        )
        if overlap <= 0:
            return selected_risk, risk_scores, None, True
        reference = previous_joint_world_positions[
            :, shift_steps:shift_steps + overlap
        ]
        continuity = torch.linalg.vector_norm(
            joint_world_positions[:, :, :overlap] - reference[None], dim=-1
        ).mean(dim=(1, 2))
        if not torch.isfinite(continuity).all():
            raise ValueError("joint continuity scores are non-finite")
        best_risk = risk_scores[selected_risk]
        risk_eligible = risk_scores <= best_risk + float(risk_slack)
        if eligible is not None:
            risk_eligible = risk_eligible & eligible
        masked_continuity = torch.where(
            risk_eligible, continuity,
            torch.full_like(continuity, torch.finfo(continuity.dtype).max),
        )
        selected = int(torch.argmin(masked_continuity).detach().cpu())
        accepted = bool(
            continuity[selected] <= float(max_mean_error_m)
        )
        return selected, risk_scores, continuity.detach(), accepted

    @staticmethod
    def _candidate_closest_approach_steps(distance, min_step):
        """Return each native joint mode's own closest-approach time.

        Averaging distance over K before choosing a time creates a trajectory
        which is not one of the FM samples.  Candidate-wise CPA preserves the
        semantics of K complete joint modes and lets selection rank every mode
        at the event time that it actually proposes.
        """
        if distance.ndim != 2 or distance.shape[1] == 0:
            raise ValueError("candidate CPA requires a non-empty [K,T] distance")
        if not torch.isfinite(distance).all():
            raise ValueError("candidate CPA distances are non-finite")
        return torch.argmin(distance, dim=1) + int(min_step)

    @staticmethod
    def _candidate_phase_event_steps(
        target_position, reference_position, min_step, max_step,
        phase_shift_steps,
    ):
        """Find each joint mode's best asynchronous path encounter.

        ``phase_shift_steps > 0`` means that at event time ``t`` the target
        should occupy the point it would natively reach at ``t + shift``.  In
        other words it arrives early without changing its steering path.  A
        negative shift represents delayed arrival.  Selection observes only
        the native FM joint mode and the frozen planner reference.
        """
        if target_position.ndim != 3 or target_position.shape[-1] != 2:
            raise ValueError("phase event search expects target [K,T,2]")
        if reference_position.ndim != 2 or reference_position.shape[-1] != 2:
            raise ValueError("phase event search expects reference [T,2]")
        horizon = min(target_position.shape[1], reference_position.shape[0])
        event_steps = torch.arange(
            int(min_step), int(max_step) + 1,
            device=target_position.device, dtype=torch.long,
        )
        source_steps = event_steps + int(phase_shift_steps)
        valid = (source_steps >= 0) & (source_steps < horizon)
        event_steps = event_steps[valid]
        source_steps = source_steps[valid]
        if event_steps.numel() == 0:
            raise ValueError("phase shift leaves no valid event steps")
        distance = torch.linalg.vector_norm(
            target_position[:, source_steps]
            - reference_position[None, event_steps], dim=-1,
        )
        if not torch.isfinite(distance).all():
            raise ValueError("phase event distances are non-finite")
        best = torch.argmin(distance, dim=1)
        return event_steps[best], source_steps[best], distance

    @staticmethod
    def _promote_projected_joint_candidate(
        endpoint, projected_target, target_idx, selected_idx
    ):
        """Put one complete joint mode in sample-0, changing only its target."""
        selected_idx = int(selected_idx)
        target_idx = int(target_idx)
        if endpoint.ndim != 4 or projected_target.ndim != 3:
            raise ValueError("unexpected endpoint/projected-target rank")
        if endpoint.shape[0] != projected_target.shape[0]:
            raise ValueError("candidate counts disagree")
        promoted = endpoint.clone()
        selected_joint = endpoint[selected_idx].clone()
        selected_joint[target_idx] = projected_target[selected_idx]
        promoted[0] = selected_joint
        return promoted

    @staticmethod
    def _local_pose_to_world(position, yaw, world_from_agent):
        rotation = world_from_agent[:2, :2]
        translation = world_from_agent[:2, 2]
        world_position = torch.matmul(position, rotation.transpose(0, 1)) + translation
        yaw_offset = torch.atan2(world_from_agent[1, 0], world_from_agent[0, 0])
        return world_position, yaw + yaw_offset

    @staticmethod
    def _typed_collision_residual(
        target_position, target_yaw, target_speed, target_extent,
        reference_position, reference_yaw, reference_speed, reference_extent,
        collision_type,
    ):
        """Differentiable contact/heading/speed residuals for one event type."""
        target_forward = torch.stack(
            (torch.cos(target_yaw), torch.sin(target_yaw)), dim=-1
        )
        reference_forward = torch.stack(
            (torch.cos(reference_yaw), torch.sin(reference_yaw)), dim=-1
        )
        reference_right = torch.stack(
            (torch.sin(reference_yaw), -torch.cos(reference_yaw)), dim=-1
        )
        if collision_type == "head_on":
            target_contact = target_position + 0.5 * target_extent[0] * target_forward
            reference_contact = (
                reference_position + 0.5 * reference_extent[0] * reference_forward
            )
            desired_cos = -1.0
            speed_residual = torch.relu(8.0 - target_speed - reference_speed)
        elif collision_type == "rear_end":
            relative_initial = target_position - reference_position
            target_in_front = (
                torch.sum(relative_initial * reference_forward, dim=-1) > 0
            ).to(target_position.dtype).detach()
            target_front = target_position + 0.5 * target_extent[0] * target_forward
            target_rear = target_position - 0.5 * target_extent[0] * target_forward
            reference_front = (
                reference_position + 0.5 * reference_extent[0] * reference_forward
            )
            reference_rear = (
                reference_position - 0.5 * reference_extent[0] * reference_forward
            )
            target_contact = (
                target_in_front[..., None] * target_rear
                + (1.0 - target_in_front[..., None]) * target_front
            )
            reference_contact = (
                target_in_front[..., None] * reference_front
                + (1.0 - target_in_front[..., None]) * reference_rear
            )
            desired_cos = 1.0
            closing = torch.where(
                target_in_front > 0.5,
                reference_speed - target_speed,
                target_speed - reference_speed,
            )
            speed_residual = torch.relu(2.0 - closing)
        elif collision_type == "side":
            lateral = torch.sum(
                (target_position - reference_position) * reference_right, dim=-1
            )
            side_sign = torch.where(
                lateral >= 0, torch.ones_like(lateral), -torch.ones_like(lateral)
            ).detach()
            target_contact = target_position
            reference_contact = (
                reference_position
                + 0.5 * reference_extent[1] * side_sign[..., None] * reference_right
            )
            desired_cos = 0.0
            speed_residual = torch.zeros_like(target_speed)
        else:  # cut_in
            target_contact = target_position
            reference_contact = (
                reference_position + 0.5 * reference_extent[0] * reference_forward
            )
            desired_cos = float(np.cos(np.deg2rad(30.0)))
            speed_residual = torch.relu(
                4.0 - torch.abs(target_speed - reference_speed)
            )
        contact = (target_contact - reference_contact) / 2.0
        heading_cos = torch.sum(target_forward * reference_forward, dim=-1)
        heading = 2.0 * (heading_cos - desired_cos)
        return torch.cat(
            (contact, heading[..., None], 0.25 * speed_residual[..., None]), dim=-1
        )

    def _project_typed_collision_endpoint(
        self, endpoint, data_batch, aux_info, num_samp, step_index
    ):
        """One damped Gauss--Newton step on only the target agent actions."""
        config = self.typed_collision_projection
        if config is None:
            return endpoint
        target_idx = config["target_idx"]
        reference_idx = config["reference_idx"]
        if endpoint.ndim != 4:
            raise ValueError("typed projection expects [B*K,M,T,2]")
        if max(target_idx, reference_idx) >= endpoint.shape[1]:
            raise IndexError("typed projection target/reference exceeds scene agents")

        planner_positions = data_batch.get("planner_reference_positions")
        planner_yaws = data_batch.get("planner_reference_yaws")
        planner_mask = data_batch.get("planner_reference_mask")
        world_from_agent = data_batch.get("world_from_agent")
        extent = data_batch.get("extent")
        if any(value is None for value in (
            planner_positions, planner_yaws, planner_mask, world_from_agent, extent
        )):
            raise ValueError("typed projection requires frozen planner reference and geometry")
        if not bool(planner_mask[reference_idx]):
            raise ValueError("typed projection reference planner is unavailable")

        target_origin = world_from_agent[target_idx, :2, 2]
        reference_origin = world_from_agent[reference_idx, :2, 2]
        current_distance = torch.linalg.vector_norm(target_origin - reference_origin)
        if self._typed_projection_latch_state(
            float(current_distance.detach().cpu()), step_index
        ):
            return endpoint
        if float(current_distance.detach().cpu()) > float(config["activation_distance_m"]):
            return endpoint
        if bool(config["final_step_only"]) and step_index != self.n_flow_steps - 1:
            # The step-0 path above must still update the closest-approach latch
            # once per closed-loop replan; only the expensive projection waits
            # until the final FM endpoint.
            return endpoint

        horizon = min(
            endpoint.shape[2], planner_positions.shape[-2], planner_yaws.shape[-2]
        )
        min_step = min(int(config["min_collision_step"]), horizon - 1)
        max_step = min(int(config["max_collision_step"]), horizon - 1)
        if max_step < min_step:
            return endpoint

        curr_states = aux_info["curr_states"]
        target_actions = endpoint[:, target_idx].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            target_traj = self.convert_action_to_state_and_action(
                target_actions, curr_states[:, target_idx],
                scaled_input=True, descaled_output=True,
            )
            target_world_pos, target_world_yaw = self._local_pose_to_world(
                target_traj[..., :2], target_traj[..., 3],
                world_from_agent[target_idx],
            )
            reference_local_pos = planner_positions[reference_idx, :horizon]
            reference_local_yaw = planner_yaws[reference_idx, :horizon]
            if reference_local_yaw.ndim > 1:
                reference_local_yaw = reference_local_yaw[..., 0]
            reference_world_pos, reference_world_yaw = self._local_pose_to_world(
                reference_local_pos, reference_local_yaw,
                world_from_agent[reference_idx],
            )
            nominal_distance = torch.linalg.vector_norm(
                target_world_pos[:, min_step:max_step + 1]
                - reference_world_pos[None, min_step:max_step + 1], dim=-1,
            )
            phase_shift_steps = int(config["phase_shift_steps"])
            candidate_phase_source_steps = None
            if phase_shift_steps:
                if not bool(config["native_joint_selection"]):
                    raise ValueError(
                        "phase control requires complete native joint selection"
                    )
                (
                    candidate_collision_steps,
                    candidate_phase_source_steps,
                    _,
                ) = self._candidate_phase_event_steps(
                    target_world_pos.detach(), reference_world_pos,
                    min_step, max_step, phase_shift_steps,
                )
            elif bool(config["native_joint_selection"]):
                candidate_collision_steps = self._candidate_closest_approach_steps(
                    nominal_distance, min_step
                )
            else:
                mean_step = min_step + int(
                    torch.argmin(nominal_distance.mean(dim=0)).detach().cpu()
                )
                candidate_collision_steps = torch.full(
                    (target_world_pos.shape[0],), mean_step,
                    device=target_world_pos.device, dtype=torch.long,
                )
            candidate_indices = torch.arange(
                target_world_pos.shape[0], device=target_world_pos.device
            )
            target_event_pos = target_world_pos[
                candidate_indices, candidate_collision_steps
            ]
            target_event_yaw = target_world_yaw[
                candidate_indices, candidate_collision_steps
            ]
            target_speed = target_traj[
                candidate_indices, candidate_collision_steps, 2
            ]
            if candidate_phase_source_steps is not None:
                # The phase goal is a point on this exact FM mode's native
                # target path.  It is detached so optimization changes only
                # arrival timing, never the chosen spatial goal or joint mode.
                phase_goal_pos = target_world_pos[
                    candidate_indices, candidate_phase_source_steps
                ].detach()
                phase_goal_yaw = target_world_yaw[
                    candidate_indices, candidate_phase_source_steps
                ].detach()
                phase_goal_speed = target_traj[
                    candidate_indices, candidate_phase_source_steps, 2
                ].detach()
                phase_path_displacement = torch.linalg.vector_norm(
                    phase_goal_pos - target_event_pos.detach(), dim=-1
                )
                phase_heading_change = torch.abs(torch.atan2(
                    torch.sin(phase_goal_yaw - target_event_yaw.detach()),
                    torch.cos(phase_goal_yaw - target_event_yaw.detach()),
                ))
                phase_candidate_eligible = (
                    (phase_path_displacement <= float(
                        config["max_phase_path_displacement_m"]
                    ))
                    & (phase_heading_change <= float(np.deg2rad(
                        config["max_phase_heading_change_deg"]
                    )))
                )
            else:
                phase_goal_pos = target_event_pos
                phase_goal_yaw = target_event_yaw
                phase_goal_speed = target_speed
                phase_path_displacement = None
                phase_heading_change = None
                phase_candidate_eligible = None
            reference_event_pos = reference_world_pos[candidate_collision_steps]
            reference_event_yaw = reference_world_yaw[candidate_collision_steps]
            previous_steps = torch.clamp(candidate_collision_steps - 1, min=0)
            reference_speed = torch.linalg.vector_norm(
                reference_event_pos - reference_world_pos[previous_steps], dim=-1
            ) / float(self.dt)
            reference_speed = torch.where(
                candidate_collision_steps == 0,
                torch.zeros_like(reference_speed), reference_speed,
            )
            residual = self._typed_collision_residual(
                phase_goal_pos, phase_goal_yaw, phase_goal_speed,
                extent[target_idx, :2],
                reference_event_pos, reference_event_yaw,
                reference_speed, extent[reference_idx, :2],
                config["collision_type"],
            )
            native_selection_residual = residual
            if candidate_phase_source_steps is not None:
                # Selection must agree with the actual asynchronous phase
                # encounter, not with a synchronous typed-contact surrogate.
                native_selection_residual = torch.cat((
                    (phase_goal_pos - reference_event_pos) / 2.0,
                    residual[..., 2:],
                ), dim=-1)
                # Explicit signed phase control: reach the mode's own
                # t+Delta point at event time t.  Unlike contact-distance GN,
                # this cannot accidentally choose the opposite timing update.
                residual = (target_event_pos - phase_goal_pos) / 2.0
            elif bool(config["temporal_only"]):
                # Geometry/type is used to rank the native joint modes, but an
                # acceleration-only projection can satisfy only contact x/y.
                residual = residual[..., :2]
            # SAFE-SIM-style collateral veto folded into the same projection:
            # other agents are detached (their FM marginals are untouched),
            # while a positive residual pushes only the target away from them.
            collateral_residuals = []
            target_radius = 0.5 * torch.linalg.vector_norm(
                extent[target_idx, :2]
            )
            for other_idx in range(endpoint.shape[1]):
                if other_idx in {target_idx, reference_idx}:
                    continue
                other_traj = self.convert_action_to_state_and_action(
                    endpoint[:, other_idx].detach(), curr_states[:, other_idx],
                    scaled_input=True, descaled_output=True,
                )
                other_world_pos, _ = self._local_pose_to_world(
                    other_traj[..., :2], other_traj[..., 3],
                    world_from_agent[other_idx],
                )
                other_radius = 0.5 * torch.linalg.vector_norm(
                    extent[other_idx, :2]
                )
                collateral_eval_pos = (
                    phase_goal_pos
                    if candidate_phase_source_steps is not None
                    else target_event_pos
                )
                surface_gap = (
                    torch.linalg.vector_norm(
                        collateral_eval_pos
                        - other_world_pos[
                            candidate_indices, candidate_collision_steps
                        ], dim=-1,
                    )
                    - target_radius - other_radius
                )
                collateral_residuals.append(
                    0.5 * torch.relu(
                        float(config["collateral_margin_m"]) - surface_gap
                    )[..., None]
                )
            if collateral_residuals:
                residual = torch.cat((residual, *collateral_residuals), dim=-1)
                native_selection_residual = torch.cat(
                    (native_selection_residual, *collateral_residuals), dim=-1
                )
            selected_native_idx = None
            selected_native_score = None
            selected_continuity_error_m = None
            selected_phase_source_step = None
            selected_phase_path_displacement = None
            selected_phase_heading_change_deg = None
            if bool(config["native_joint_selection"]):
                if (
                    phase_candidate_eligible is not None
                    and not bool(torch.any(phase_candidate_eligible))
                ):
                    print(
                        "TYPED_COLLISION_PROJECTION_REJECT "
                        "reason=no_phase_candidate_passes_path_gates"
                    )
                    return endpoint
                joint_world_positions = None
                if bool(config["joint_continuity_enabled"]):
                    native_joint_world = []
                    for joint_agent_idx in range(endpoint.shape[1]):
                        # The reference ego row is replaced by the downstream
                        # planner wrapper and is never executed from this FM
                        # sample.  Excluding it prevents an irrelevant model
                        # ego prediction from determining background intent
                        # continuity.
                        if joint_agent_idx == reference_idx:
                            continue
                        if joint_agent_idx == target_idx:
                            agent_world_pos = target_world_pos.detach()
                        else:
                            agent_traj = self.convert_action_to_state_and_action(
                                endpoint[:, joint_agent_idx].detach(),
                                curr_states[:, joint_agent_idx],
                                scaled_input=True, descaled_output=True,
                            )
                            agent_world_pos, _ = self._local_pose_to_world(
                                agent_traj[..., :2], agent_traj[..., 3],
                                world_from_agent[joint_agent_idx],
                            )
                        native_joint_world.append(agent_world_pos)
                    joint_world_positions = torch.stack(
                        native_joint_world, dim=1
                    ).detach()
                if (
                    joint_world_positions is not None
                    and self._typed_projection_previous_joint_world_positions
                    is not None
                    and self._typed_projection_previous_joint_world_positions.shape[0]
                    == joint_world_positions.shape[1]
                ):
                    (
                        selected_native_idx, native_scores,
                        continuity_scores, continuity_accepted,
                    ) = self._select_continuous_native_joint_candidate(
                        native_selection_residual.detach(),
                        joint_world_positions,
                        self._typed_projection_previous_joint_world_positions,
                        config["joint_continuity_shift_steps"],
                        config["joint_continuity_risk_slack"],
                        config["joint_continuity_max_mean_error_m"],
                        phase_candidate_eligible,
                    )
                    selected_continuity_error_m = float(
                        continuity_scores[selected_native_idx].detach().cpu()
                    )
                    if not continuity_accepted:
                        # The unmodified sample-0 is the actual fallback plan;
                        # remember it so the following replan does not compare
                        # against an intent which was never executed.
                        self._typed_projection_previous_joint_world_positions = (
                            joint_world_positions[0].detach().clone()
                        )
                        print(
                            "TYPED_COLLISION_PROJECTION_REJECT "
                            "reason=joint_continuity_gate "
                            "selected_continuity_error_m={:.6f}".format(
                                selected_continuity_error_m
                            )
                        )
                        return endpoint
                else:
                    selected_native_idx, native_scores = (
                        self._select_native_joint_candidate(
                            native_selection_residual.detach(), num_samp,
                            phase_candidate_eligible,
                        )
                    )
                    if bool(config["joint_continuity_enabled"]):
                        # The first active replan establishes the ordinary FM
                        # sample-0 intent.  Do not attack before a real executed
                        # baseline exists; this prevents an unconstrained first
                        # mode jump and makes the pre-conflict prefix native.
                        self._typed_projection_previous_joint_world_positions = (
                            joint_world_positions[0].detach().clone()
                        )
                        print(
                            "TYPED_COLLISION_PROJECTION_BASELINE "
                            "sample=0 action=retain_native_endpoint"
                        )
                        return endpoint
                if joint_world_positions is not None:
                    self._typed_projection_previous_joint_world_positions = (
                        joint_world_positions[selected_native_idx].detach().clone()
                    )
                selected_native_score = float(
                    native_scores[selected_native_idx].detach().cpu()
                )
                collision_step = int(
                    candidate_collision_steps[selected_native_idx].detach().cpu()
                )
                if candidate_phase_source_steps is not None:
                    selected_phase_source_step = int(
                        candidate_phase_source_steps[
                            selected_native_idx
                        ].detach().cpu()
                    )
                    selected_phase_path_displacement = float(
                        phase_path_displacement[
                            selected_native_idx
                        ].detach().cpu()
                    )
                    selected_phase_heading_change_deg = float(np.degrees(
                        phase_heading_change[
                            selected_native_idx
                        ].detach().cpu()
                    ))
                else:
                    selected_phase_path_displacement = None
                    selected_phase_heading_change_deg = None
                # Differentiate only the selected scene-level joint mode while
                # retaining the K-sized batch required by the batched GN solve.
                # Every non-selected mode therefore receives an exact zero
                # residual and update.
                selected_mask = torch.zeros(
                    residual.shape[0], 1,
                    device=residual.device, dtype=residual.dtype,
                )
                selected_mask[selected_native_idx] = 1.0
                residual = residual * selected_mask
            else:
                collision_step = int(candidate_collision_steps[0].detach().cpu())
            optimize_steps = min(collision_step + 1, target_actions.shape[1])
            jacobian_rows = []
            for residual_index in range(residual.shape[1]):
                gradient = torch.autograd.grad(
                    residual[:, residual_index].sum(), target_actions,
                    retain_graph=residual_index < residual.shape[1] - 1,
                    create_graph=False,
                )[0]
                if bool(config["temporal_only"]):
                    # Human-error timing intervention: keep the target's FM
                    # steering/yaw-rate exactly unchanged and alter only its
                    # longitudinal acceleration / arrival time.
                    gradient = self._temporal_only_gradient(gradient)
                jacobian_rows.append(
                    gradient[:, :optimize_steps].reshape(endpoint.shape[0], -1)
                )
            jacobian = torch.stack(jacobian_rows, dim=1).detach()
        residual_detached = residual.detach()
        jjt = torch.bmm(jacobian, jacobian.transpose(1, 2))
        identity = torch.eye(
            residual.shape[1], device=endpoint.device, dtype=endpoint.dtype
        )[None]
        system = jjt + float(config["damping"]) * identity
        solution = torch.linalg.solve(system, residual_detached[..., None])
        delta = torch.bmm(jacobian.transpose(1, 2), solution).squeeze(-1)
        delta = delta.reshape(endpoint.shape[0], optimize_steps, 2)
        tau = float(step_index + 1) / float(max(self.n_flow_steps, 1))
        delta = tau * float(config["projection_step_size"]) * delta
        delta = torch.clamp(
            delta, -float(config["max_normalized_delta"]),
            float(config["max_normalized_delta"]),
        )
        projected_target = target_actions.detach().clone()
        projected_target[:, :optimize_steps] -= delta
        if bool(config["temporal_only"]):
            projected_target = self._restore_native_yaw_rate(
                projected_target, target_actions.detach()
            )
        projected_target[..., 0].clamp_(-10.0, 10.0)
        projected_target[..., 1].clamp_(-15.0, 15.0)
        if not torch.isfinite(projected_target).all():
            print("TYPED_COLLISION_PROJECTION_REJECT reason=nonfinite")
            return endpoint
        if bool(config["native_joint_selection"]):
            projected = self._promote_projected_joint_candidate(
                endpoint, projected_target, target_idx, selected_native_idx
            )
        else:
            projected = endpoint.clone()
            projected[:, target_idx] = projected_target
        self._typed_projection_calls += 1
        if step_index == self.n_flow_steps - 1:
            print(
                "TYPED_COLLISION_PROJECTION_STEP active=True target={} reference={} "
                "type={} collision_step={} current_distance_m={:.3f} "
                "residual_mean={:.6f} delta_abs_max={:.6f} temporal_only={} "
                "native_joint_selection={} selected_native_sample={} "
                "selected_native_score={} phase_shift_steps={} "
                "selected_continuity_error_m={} "
                "phase_source_step={} phase_path_displacement_m={} "
                "phase_heading_change_deg={}".format(
                    target_idx, reference_idx, config["collision_type"],
                    collision_step, float(current_distance.detach().cpu()),
                    float(torch.linalg.vector_norm(
                        residual_detached, dim=-1
                    ).mean().cpu()), float(delta.abs().max().cpu()),
                    bool(config["temporal_only"]),
                    bool(config["native_joint_selection"]),
                    selected_native_idx,
                    selected_native_score,
                    phase_shift_steps,
                    selected_continuity_error_m,
                    selected_phase_source_step,
                    selected_phase_path_displacement,
                    selected_phase_heading_change_deg,
                )
            )
        return projected.detach()

    @staticmethod
    def _expand_flow_time(flow_time, tensor):
        return flow_time.reshape(
            flow_time.shape[0], *((1,) * (tensor.ndim - 1))
        )

    def _interpolate_actions(self, action_start, action_end, flow_time):
        """Linear probability path: noise at 0, data at 1."""
        flow_time_expanded = self._expand_flow_time(flow_time, action_end)
        return (
            (1.0 - flow_time_expanded) * action_start
            + flow_time_expanded * action_end
        )

    def _predict_flow_velocity(
        self,
        x_flow,
        flow_time,
        aux_info,
        data_batch,
        class_free_guide_w=0.0,
        x_clean_model_out=None,
    ):
        """Predict velocity while preserving CCDiff's sampling condition path."""
        t_inp = flow_time * float(self.n_timesteps - 1)
        x_model_in = x_flow

        if self.map_embed_method in ['cnn', 'cnn_local_patch']:
            if self.use_map_feat_grid and self.map_encoder is not None:
                map_feat_traj = self.query_map_feats(
                    x_flow.detach().clone(),
                    aux_info['map_grid_feat'],
                    aux_info['raster_from_agent'],
                )
                x_model_in = torch.cat([x_model_in, map_feat_traj], dim=-1)

        if self.neigh_hist_embed_method in [
            'interaction_edge',
            'interaction_edge_and_input',
        ]:
            if self.use_noisy_fut_edge:
                x_edge = (
                    x_flow
                    if self.conditional_flow_differentiable_conditioning
                    else x_flow.detach().clone()
                )
            elif x_clean_model_out is None:
                x_edge = None
            else:
                x_edge = self.convert_action_to_state_and_action(
                    x_clean_model_out.detach(), aux_info['curr_states']
                )
            aux_info = self.get_neighbor_future_relative_states(
                x_edge, aux_info, data_batch
            )

            if self.neigh_hist_embed_method == 'interaction_edge_and_input':
                neighbor_fut_feat = self.query_neigh_feats(
                    aux_info['neighbor_fut_feat']
                )
                x_model_in = torch.cat(
                    [x_model_in, neighbor_fut_feat], dim=-1
                )

        aux_info['attn_name'] = ""
        velocity, info = self.model(
            x_model_in,
            aux_info,
            t_inp,
            motion_dist=self.motion_dist,
            use_cond=False,
        )
        if self.agent_hist_embed_method == 'concat':
            velocity = velocity[..., -self.horizon:, :]

        if class_free_guide_w != 0.0:
            velocity_non_cond, _ = self.model(
                x_model_in,
                aux_info,
                t_inp,
                motion_dist=self.motion_dist,
                use_cond=True,
            )
            if self.agent_hist_embed_method == 'concat':
                velocity_non_cond = velocity_non_cond[..., -self.horizon:, :]
            velocity = (
                (1.0 + class_free_guide_w) * velocity
                - class_free_guide_w * velocity_non_cond
            )

        return velocity, info

    def p_losses(self, x_start_init, flow_time, data_batch, aux_info={}):
        # CCDiff's state_and_action mode transports only normalized actions.
        action_end = x_start_init[..., [4, 5]]
        action_start = torch.randn_like(action_end)
        action_flow = self._interpolate_actions(
            action_start, action_end, flow_time
        )
        t_inp = flow_time * float(self.n_timesteps - 1)

        x_flow = self.convert_action_to_state_and_action(
            action_flow, aux_info['curr_states']
        )
        x_flow_in = x_flow
        if self.map_embed_method in ['cnn', 'cnn_and_input']:
            if self.use_map_feat_grid and self.map_encoder is not None:
                map_feat_traj = self.query_map_feats(
                    x_flow.detach().clone(),
                    aux_info['map_grid_feat'],
                    aux_info['raster_from_agent'],
                )
                x_flow_in = torch.cat([x_flow_in, map_feat_traj], dim=-1)

        if self.neigh_hist_embed_method in [
            'interaction_edge',
            'interaction_edge_and_input',
        ]:
            if self.use_noisy_fut_edge:
                x_edge = x_flow.detach().clone()
            else:
                x_edge = x_start_init.detach().clone()
            aux_info = self.get_neighbor_future_relative_states(
                x_edge, aux_info, data_batch
            )

            if self.neigh_hist_embed_method == 'interaction_edge_and_input':
                neighbor_fut_feat = self.query_neigh_feats(
                    aux_info['neighbor_fut_feat']
                )
                x_flow_in = torch.cat(
                    [x_flow_in, neighbor_fut_feat], dim=-1
                )

        velocity, _ = self.model(x_flow_in, aux_info, t_inp)
        if self.agent_hist_embed_method == 'concat':
            velocity_hist = velocity[..., :-self.horizon, :]
            velocity_fut = velocity[..., -self.horizon:, :]
        else:
            velocity_fut = velocity

        flow_time_expanded = self._expand_flow_time(flow_time, action_flow)
        action_recon = (
            action_flow + (1.0 - flow_time_expanded) * velocity_fut
        )
        x_recon = self.convert_action_to_state_and_action(
            action_recon, aux_info['curr_states']
        )

        x_start = x_start_init
        if (
            self.diffuser_input_mode
            in ['action', 'state_and_action', 'state_and_action_no_dyn']
            and self.action_loss_only
        ):
            x_recon_selected = x_recon[..., -2:]
            x_start_selected = x_start[..., -2:]
        else:
            x_recon_selected = x_recon
            x_start_selected = x_start

        if self.use_target_availabilities:
            target_availabilities = data_batch['target_availabilities'][
                ..., :self.horizon
            ].unsqueeze(-1)
            x_recon_selected = x_recon_selected * target_availabilities
            x_start_selected = x_start_selected * target_availabilities

        if self.coordinate == 'agent_centric':
            data_batch_for_guidance = extract_data_batch_for_guidance(
                data_batch, mode='training'
            )
            loss, info = self.loss_fn(
                x_recon_selected,
                x_start_selected,
                data_batch_for_guidance,
            )
        else:
            loss, info = self.loss_fn(
                x_recon_selected, x_start_selected, {}
            )

        if self.agent_hist_embed_method == 'concat':
            # This auxiliary term has zero weight in the controlled M1/M2 config.
            # Keep the original CCDiff bookkeeping unchanged.
            history_reconstruction_loss = self.estimate_history_reconstruction_loss(
                velocity_hist, data_batch
            )
            info['history_reconstruction_loss'] = history_reconstruction_loss

        return loss, info

    def loss(self, x, data_batch, aux_info={}):
        batch_size = len(x)
        flow_time = torch.rand(
            (batch_size,), device=x.device, dtype=x.dtype
        )
        return self.p_losses(
            x, flow_time, data_batch, aux_info=aux_info
        )

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        data_batch,
        num_samp,
        aux_info={},
        verbose=True,
        return_diffusion=False,
        return_guidance_losses=False,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=False,
        mode='testing',
    ):
        """Integrate the learned ODE from noise (0) to data (1)."""
        data_batch_for_guidance = {}
        if apply_guidance:
            data_batch_for_guidance = extract_data_batch_for_guidance(
                data_batch, mode=mode
            )

        device = self.betas.device
        batch_size = shape[0]
        if (
            self.current_perturbation_guidance.current_guidance is not None
            and not apply_guidance
        ):
            print(
                'FLOW MATCHING: Note, not using guidance during sampling, '
                'only evaluating guidance loss at the end...'
            )

        action_shape = shape[:-1] + (self.action_dim,)
        correlated_prior = self.marginal_preserving_correlated_flow_prior
        pcn_beta = self.pcn_joint_flow_prior_beta
        persistent_prior = self.persistent_copula_flow_prior
        if sum(
            item is not None
            for item in (correlated_prior, pcn_beta, persistent_prior)
        ) > 1:
            raise RuntimeError(
                "correlated, pCN, and persistent-copula FM priors are mutually exclusive"
            )
        if persistent_prior is not None:
            action_flow = persistent_prior["sampler"].sample(
                action_shape,
                global_replan_step=self._persistent_copula_replan_call,
                device=device,
            )
            self._persistent_copula_replan_call += 1
            self._persistent_copula_last_shape = tuple(
                int(value) for value in action_shape
            )
        elif pcn_beta is not None:
            action_flow = self.sample_pcn_joint_prior(
                action_shape, pcn_beta, device=device
            )
        elif correlated_prior is None:
            action_flow = torch.randn(action_shape, device=device)
        else:
            action_flow = self.sample_marginal_preserving_correlated_prior(
                action_shape,
                correlated_prior["agent_pairs"],
                correlated_prior["rho"],
                lag_steps=correlated_prior.get("lag_steps", 0),
                action_dims=correlated_prior.get("action_dims"),
                device=device,
            )
        native_action_source_bank = action_flow.detach().clone()

        source_control = self.conditional_flow_source_control
        velocity_residual_runtime = None
        if (
            source_control is not None
            and source_control["source_config"].target_velocity_residual
        ):
            if apply_guidance:
                raise RuntimeError(
                    "target velocity residual requires apply_guidance=False"
                )
            source_config = source_control["source_config"]
            objective_config = source_control["objective_config"]
            if not source_config.enabled:
                raise ValueError(
                    "target velocity residual cannot be combined with source_disabled"
                )
            required_online = (
                "planner_reference_positions",
                "world_from_agent",
                "curr_speed",
            )
            missing_online = [
                key for key in required_online if key not in data_batch
            ]
            if missing_online:
                raise KeyError(
                    "velocity residual online CPA missing fields {}".format(
                        missing_online
                    )
                )
            online_cpa_audit = estimate_online_planner_target_cpa(
                data_batch["planner_reference_positions"][
                    0, int(objective_config.executed_planner_actor_idx)
                ],
                data_batch["world_from_agent"][0],
                data_batch["curr_speed"][0].reshape(-1),
                planner_actor_idx=int(
                    objective_config.executed_planner_actor_idx
                ),
                target_actor_idx=int(objective_config.target_actor_idx),
                dt=float(self.dt),
                max_horizon=int(action_flow.shape[-2]),
            )
            arrival_aligned = bool(
                source_config.target_velocity_residual_arrival_aligned
            )
            if arrival_aligned:
                online_cpa_active = bool(
                    online_cpa_audit["arrival_alignment_valid"]
                    and online_cpa_audit["arrival_lateral_distance_m"]
                    <= source_config.target_velocity_residual_arrival_max_lateral_distance_m
                    and online_cpa_audit["arrival_control_sign"] != 0.0
                )
            else:
                online_cpa_active = bool(
                    online_cpa_audit["cpa_center_distance_m"]
                    <= source_config.target_mean_shift_max_cpa_distance_m
                )
            online_cpa_audit["active"] = online_cpa_active
            online_cpa_audit["max_center_distance_m"] = float(
                source_config.target_mean_shift_max_cpa_distance_m
            )
            proposed_coefficients = tuple(
                float(value)
                for value in source_config.target_mean_shift_coefficients
            )
            if arrival_aligned and any(value < 0.0 for value in proposed_coefficients[:2]):
                raise ValueError(
                    "arrival-aligned acceleration coefficients are unsigned magnitudes"
                )
            arrival_sign = float(online_cpa_audit["arrival_control_sign"])
            signed_coefficients = (
                proposed_coefficients[0] * arrival_sign,
                proposed_coefficients[1] * arrival_sign,
                proposed_coefficients[2],
                proposed_coefficients[3],
            ) if arrival_aligned else proposed_coefficients
            applied_coefficients = signed_coefficients if online_cpa_active else (0.0, 0.0, 0.0, 0.0)
            pulse_center_fraction = float(
                online_cpa_audit[
                    "arrival_pulse_center_fraction" if arrival_aligned else "pulse_center_fraction"
                ]
            )
            online_cpa_audit["alignment_mode"] = (
                "target_forward_arrival_time" if arrival_aligned else "simultaneous_cpa"
            )
            online_cpa_audit["arrival_max_lateral_distance_m"] = float(
                source_config.target_velocity_residual_arrival_max_lateral_distance_m
            )
            online_cpa_audit["signed_applied_coefficients"] = applied_coefficients
            velocity_residual_runtime = {
                "config": TargetVelocityResidualConfig(
                    target_actor_idx=int(objective_config.target_actor_idx),
                    selected_sample_idx=int(objective_config.selected_sample_idx),
                    coefficients=applied_coefficients,
                    pulse_center_fraction=pulse_center_fraction,
                    pulse_width_fraction=float(
                        source_config.target_mean_shift_pulse_width_fraction
                    ),
                    max_abs_coefficient=float(
                        source_config.target_mean_shift_max_abs_coefficient
                    ),
                    max_global_residual_rms=float(
                        source_config.max_source_delta_rms
                    ),
                    max_target_residual_rms=float(
                        source_config.max_per_actor_source_delta_rms
                    ),
                    max_path_energy=float(
                        source_config.max_source_energy_increase
                    ),
                ),
                "online_cpa": online_cpa_audit,
                "proposed_coefficients": proposed_coefficients,
                "step_audits": [],
            }
            # The source bank stays exactly native.  The residual is injected
            # only after each frozen FM velocity evaluation below.
            source_control = None
        if (
            source_control is not None
            and source_control["source_config"].target_mean_shift_coefficients
            is not None
        ):
            if apply_guidance:
                raise RuntimeError(
                    "target source mean shift requires apply_guidance=False"
                )
            source_config = source_control["source_config"]
            objective_config = source_control["objective_config"]
            if not source_config.enabled:
                raise ValueError(
                    "target source mean shift cannot be combined with source_disabled"
                )
            pulse_center_fraction = 0.5
            online_cpa_audit = None
            applied_coefficients = tuple(
                float(value)
                for value in source_config.target_mean_shift_coefficients
            )
            if source_config.target_mean_shift_basis_kind == "gaussian_pulse":
                if source_config.target_mean_shift_online_cpa:
                    required_online = (
                        "planner_reference_positions",
                        "world_from_agent",
                        "curr_speed",
                    )
                    missing_online = [
                        key for key in required_online if key not in data_batch
                    ]
                    if missing_online:
                        raise KeyError(
                            "online CPA missing fields {}".format(missing_online)
                        )
                    online_cpa_audit = estimate_online_planner_target_cpa(
                        data_batch["planner_reference_positions"][
                            0, int(objective_config.executed_planner_actor_idx)
                        ],
                        data_batch["world_from_agent"][0],
                        data_batch["curr_speed"][0].reshape(-1),
                        planner_actor_idx=int(
                            objective_config.executed_planner_actor_idx
                        ),
                        target_actor_idx=int(objective_config.target_actor_idx),
                        dt=float(self.dt),
                        max_horizon=int(action_flow.shape[-2]),
                    )
                    pulse_center_fraction = online_cpa_audit[
                        "pulse_center_fraction"
                    ]
                    online_cpa_active = bool(
                        online_cpa_audit["cpa_center_distance_m"]
                        <= source_config.target_mean_shift_max_cpa_distance_m
                    )
                    online_cpa_audit["active"] = online_cpa_active
                    online_cpa_audit["max_center_distance_m"] = float(
                        source_config.target_mean_shift_max_cpa_distance_m
                    )
                    if not online_cpa_active:
                        applied_coefficients = (0.0, 0.0, 0.0, 0.0)
                elif objective_config.global_conflict_step is None:
                    raise ValueError(
                        "gaussian-pulse source action requires global_conflict_step"
                    )
                else:
                    relative_conflict_step = int(
                        objective_config.global_conflict_step
                        - objective_config.current_global_step
                    )
                    if not 0 <= relative_conflict_step < action_flow.shape[-2]:
                        raise ValueError(
                            "global conflict step is outside the current FM horizon"
                        )
                    pulse_center_fraction = relative_conflict_step / float(
                        max(action_flow.shape[-2] - 1, 1)
                    )
            action_flow, mean_shift_audit = apply_target_source_mean_shift(
                action_flow,
                TargetSourceMeanShiftConfig(
                    target_actor_idx=int(objective_config.target_actor_idx),
                    selected_sample_idx=int(objective_config.selected_sample_idx),
                    coefficients=applied_coefficients,
                    basis_kind=str(
                        source_config.target_mean_shift_basis_kind
                    ),
                    pulse_center_fraction=float(pulse_center_fraction),
                    pulse_width_fraction=float(
                        source_config.target_mean_shift_pulse_width_fraction
                    ),
                    max_abs_coefficient=float(
                        source_config.target_mean_shift_max_abs_coefficient
                    ),
                    max_global_delta_rms=float(
                        source_config.max_source_delta_rms
                    ),
                    max_target_delta_rms=float(
                        source_config.max_per_actor_source_delta_rms
                    ),
                    max_mean_kl_per_coordinate=float(
                        source_config.target_mean_shift_max_kl_per_coordinate
                    ),
                    max_source_energy_increase=float(
                        source_config.max_source_energy_increase
                    ),
                ),
            )
            self._conditional_flow_last_audit = {
                **mean_shift_audit,
                "native_k": int(num_samp),
                "exact_frozen_flow_steps": int(self.n_flow_steps),
                "source_bank_other_modes_identical": bool(
                    mean_shift_audit["other_modes_bit_exact"]
                ),
                "final_native_reintegration": True,
                "external_hard_acceptance_required": True,
                "proposed_coefficients": tuple(
                    source_config.target_mean_shift_coefficients
                ),
                "online_cpa": online_cpa_audit,
            }
            print(
                "TARGET_SOURCE_MEAN_SHIFT status={} coefficients={} basis={} "
                "pulse_center_fraction={} online_cpa_step={} "
                "online_cpa_distance_m={} online_cpa_active={} "
                "global_delta_rms={} target_delta_rms={} mean_kl={} "
                "energy_increase={}".format(
                    mean_shift_audit["status"],
                    applied_coefficients,
                    source_config.target_mean_shift_basis_kind,
                    pulse_center_fraction,
                    None if online_cpa_audit is None else online_cpa_audit["cpa_step"],
                    None if online_cpa_audit is None else online_cpa_audit["cpa_center_distance_m"],
                    None if online_cpa_audit is None else online_cpa_audit["active"],
                    mean_shift_audit["global_delta_rms"],
                    mean_shift_audit["target_delta_rms"],
                    mean_shift_audit["mean_kl_per_coordinate"],
                    mean_shift_audit["source_energy_increase"],
                )
            )
            # This arm is a four-dimensional proposal action, not the old
            # source-gradient optimizer. The unchanged sampler below consumes
            # the accepted source bank directly.
            source_control = None
        if source_control is not None:
            if apply_guidance:
                raise RuntimeError(
                    "Conditional-FM source control requires apply_guidance=False"
                )
            if batch_size != 1:
                raise RuntimeError(
                    "initial Conditional-FM protocol requires one scene"
                )
            required = (
                "planner_reference_positions",
                "planner_reference_mask",
                "planner_safety_reference_positions",
                "planner_safety_reference_mask",
                "world_from_agent",
                "extent",
                "history_availabilities",
            )
            missing = [key for key in required if key not in data_batch]
            if missing:
                raise KeyError(
                    "Conditional-FM missing planner/geometry fields {}".format(
                        missing
                    )
                )
            objective_config = source_control["objective_config"]
            selected_index = int(objective_config.selected_sample_idx)
            if not 0 <= selected_index < num_samp:
                raise IndexError("Conditional-FM selected K index is invalid")

            attacked_mask = data_batch["planner_reference_mask"][0].bool()
            safety_mask = data_batch["planner_safety_reference_mask"][0].bool()
            if int(attacked_mask.sum().item()) != 1:
                raise ValueError("attacked planner mask must select exactly one ego")
            if not torch.equal(attacked_mask, safety_mask):
                raise ValueError("attacked and safety planners must share ego row")
            planner_idx = int(torch.nonzero(attacked_mask)[0, 0].item())
            if planner_idx != int(objective_config.executed_planner_actor_idx):
                raise ValueError("configured planner row disagrees with injected mask")

            # Optimize exactly one complete joint K mode.  K20 is still sampled
            # and returned; K-1 sources are never evaluated or modified here.
            selected_source = action_flow[:, selected_index].clone()
            selected_aux = TensorUtils.repeat_by_expand_at(
                aux_info, repeats=1, dim=0
            )
            valid_actor_mask = data_batch["history_availabilities"][
                0, :, -1
            ].bool()
            source_with_k = selected_source[:, None]
            optimization_mask = selected_joint_background_mask(
                source_with_k,
                selected_sample_idx=0,
                ego_actor_idx=planner_idx,
                valid_actor_mask=valid_actor_mask,
                target_actor_idx=int(objective_config.target_actor_idx),
                optimization_scope=source_control[
                    "source_config"
                ].optimization_scope,
                optimization_start_fraction=source_control[
                    "source_config"
                ].optimization_start_fraction,
            ).reshape_as(selected_source)
            local_objective_config = PlannerSeparationObjectiveConfig(
                **{
                    **objective_config.__dict__,
                    "selected_sample_idx": 0,
                    "executed_planner_actor_idx": planner_idx,
                }
            )
            native_selected = selected_source.detach().clone()

            self.set_conditional_flow_differentiable_conditioning(True)
            gradient_contract = verify_noisy_edge_gradient_contract(self)
            try:
                objective = lambda source: planner_separation_source_objective(
                    self,
                    source,
                    source_shape=(
                        batch_size, 1, shape[2], shape[3], self.action_dim
                    ),
                    data_batch=data_batch,
                    repeated_aux_info=selected_aux,
                    target_planner_positions_local=data_batch[
                        "planner_reference_positions"
                    ][0, planner_idx],
                    safety_planner_positions_local=data_batch[
                        "planner_safety_reference_positions"
                    ][0, planner_idx],
                    world_from_agent=data_batch["world_from_agent"][0],
                    extents=data_batch["extent"][0],
                    config=local_objective_config,
                    native_source=native_selected,
                    valid_actor_mask=valid_actor_mask,
                )
                optimized_source, source_audit = optimize_complete_joint_source(
                    native_selected,
                    objective,
                    source_control["source_config"],
                    optimization_mask=optimization_mask,
                )
            finally:
                # The accepted source is re-integrated by the exact native
                # sampler below.  No graph or altered-conditioning path leaks
                # into the executed output.
                self.set_conditional_flow_differentiable_conditioning(False)
            action_flow = torch.cat(
                (
                    action_flow[:, :selected_index],
                    optimized_source[:, None],
                    action_flow[:, selected_index + 1:],
                ),
                dim=1,
            )
            self._conditional_flow_last_audit = {
                **source_audit,
                "gradient_contract": gradient_contract,
                "native_k": int(num_samp),
                "optimized_complete_joint_modes": 1,
                "selected_native_sample_idx": selected_index,
                "target_actor_idx": int(local_objective_config.target_actor_idx),
                "executed_planner_actor_idx": planner_idx,
                "source_bank_other_modes_identical": bool(
                    torch.equal(
                        torch.cat(
                            (
                                action_flow[:, :selected_index],
                                action_flow[:, selected_index + 1:],
                            ),
                            dim=1,
                        ),
                        torch.cat(
                            (
                                native_action_source_bank[:, :selected_index],
                                native_action_source_bank[:, selected_index + 1:],
                            ),
                            dim=1,
                        ),
                    )
                ),
                "final_native_reintegration": True,
                "external_hard_acceptance_required": True,
                "optimization_scope": source_control[
                    "source_config"
                ].optimization_scope,
                "optimization_start_fraction": source_control[
                    "source_config"
                ].optimization_start_fraction,
            }
            trace_row = (
                source_audit.get("trace", [{}])[0]
                if source_audit.get("trace") else {}
            )
            first_trial = (
                trace_row.get("trials", [{}])[0]
                if trace_row.get("trials") else {}
            )
            print(
                "CONDITIONAL_FM_SOURCE_TRACE initial_objective={} "
                "raw_active_gradient_rms={} normalized_gradient_rms={} "
                "first_trial_trust_pass={} first_trial_source_delta_rms={} "
                "first_trial_energy_delta={} first_trial_objective={} "
                "first_trial_improvement={}".format(
                    source_audit.get("initial_objective"),
                    trace_row.get("raw_active_gradient_rms"),
                    trace_row.get("gradient_rms"),
                    first_trial.get("trust_pass"),
                    first_trial.get("source_delta_rms"),
                    first_trial.get("source_energy_delta"),
                    first_trial.get("objective"),
                    first_trial.get("improvement"),
                )
            )
            print(
                "CONDITIONAL_FM_SOURCE_ATTESTATION status={} accepted_steps={} "
                "source_delta_rms={:.8f} native_k={} optimized_modes=1 "
                "target_actor={} planner_actor={} endpoint_projection=False "
                "external_hard_acceptance_required=True".format(
                    source_audit["status"],
                    source_audit["accepted_steps"],
                    source_audit["source_delta_rms"],
                    num_samp,
                    local_objective_config.target_actor_idx,
                    planner_idx,
                )
            )
        action_flow = TensorUtils.join_dimensions(
            action_flow, begin_axis=0, end_axis=2
        )

        aux_info = TensorUtils.repeat_by_expand_at(
            aux_info, repeats=num_samp, dim=0
        )
        x_flow = self.convert_action_to_state_and_action(
            action_flow, aux_info['curr_states']
        )
        if return_diffusion:
            diffusion = [x_flow]

        progress = Progress(self.n_flow_steps) if verbose else Silent()
        step_size = 1.0 / float(self.n_flow_steps)
        x_clean_model_out = None
        guide_losses = dict()
        attn_weights = []

        for step_index in range(self.n_flow_steps):
            flow_value = step_index * step_size
            flow_time = torch.full(
                (batch_size * num_samp,),
                flow_value,
                device=device,
                dtype=x_flow.dtype,
            )
            velocity, info = self._predict_flow_velocity(
                x_flow,
                flow_time,
                aux_info,
                data_batch,
                class_free_guide_w=class_free_guide_w,
                x_clean_model_out=x_clean_model_out,
            )

            if velocity_residual_runtime is not None:
                velocity_bank = velocity.reshape(
                    batch_size,
                    num_samp,
                    *velocity.shape[1:],
                )
                velocity_bank, residual_audit = apply_target_velocity_residual(
                    velocity_bank,
                    velocity_residual_runtime["config"],
                    flow_fraction=float(flow_value),
                    euler_step_size=float(step_size),
                )
                velocity = velocity_bank.reshape_as(velocity)
                velocity_residual_runtime["step_audits"].append(
                    residual_audit
                )

            remaining = 1.0 - flow_value
            endpoint = action_flow + remaining * velocity

            if self.disable_control_on_stationary and self.stationary_mask is not None:
                endpoint_stationary = endpoint[self.stationary_mask]
                endpoint_stationary = self.descale_traj(
                    endpoint_stationary, [4, 5]
                )
                endpoint_stationary[...] = 0
                endpoint_stationary = self.scale_traj(
                    endpoint_stationary, [4, 5]
                )
                endpoint[self.stationary_mask] = endpoint_stationary

            is_final_step = step_index == self.n_flow_steps - 1
            use_step_guidance = apply_guidance
            opt_params = None
            if use_step_guidance and self.guidance_optimization_params is not None:
                if is_final_step:
                    use_step_guidance = self.apply_guidance_output
                    if use_step_guidance:
                        opt_params = self.final_step_opt_params
                else:
                    use_step_guidance = self.apply_guidance_intermediate
                    if use_step_guidance:
                        if self.guidance_optimization_params['grad_steps'] <= 0:
                            raise ValueError(
                                'guidance grad_steps must be positive'
                            )
                        opt_params = deepcopy(
                            self.guidance_optimization_params
                        )
                        if opt_params['perturb_th'] is None:
                            opt_params['perturb_th'] = remaining
                        if opt_params['lr'] is None:
                            opt_params['lr'] = remaining

            endpoint_guided = None
            if (
                use_step_guidance
                and opt_params is not None
                and self.current_perturbation_guidance.current_guidance
                is not None
            ):
                if self.flow_velocity_guidance_config is not None:
                    native_guidance_velocity = (
                        endpoint.detach() - action_flow.detach()
                    ) / remaining
                    velocity_guided, guide_losses, velocity_audit = (
                        self.current_perturbation_guidance.perturb_flow_velocity(
                            native_guidance_velocity,
                            action_flow,
                            remaining,
                            data_batch_for_guidance,
                            opt_params,
                            num_samp=num_samp,
                            trust_max_rms=self.flow_velocity_guidance_config[
                                "trust_max_rms"
                            ],
                            trust_max_linf=self.flow_velocity_guidance_config[
                                "trust_max_linf"
                            ],
                        )
                    )
                    endpoint_guided = (
                        action_flow + remaining * velocity_guided
                    )
                    print(
                        "FM_VELOCITY_GUIDANCE_ATTESTATION "
                        + json.dumps({
                            **velocity_audit,
                            "flow_step": int(step_index),
                            "remaining": float(remaining),
                        }, sort_keys=True),
                        flush=True,
                    )
                else:
                    endpoint_initial = (
                        endpoint.detach().clone().requires_grad_(True)
                    )
                    endpoint_guided, guide_losses = (
                        self.current_perturbation_guidance.perturb(
                            endpoint_initial,
                            data_batch_for_guidance,
                            opt_params,
                            num_samp=num_samp,
                            return_grad_of=endpoint_initial,
                        )
                    )

            if endpoint_guided is None:
                endpoint_guided = endpoint
                if (
                    self.current_perturbation_guidance.current_guidance
                    is not None
                    and is_final_step
                ):
                    _, guide_losses = (
                        self.current_perturbation_guidance.compute_guidance_loss(
                            endpoint_guided,
                            data_batch_for_guidance,
                            num_samp=num_samp,
                        )
                    )

            if self.typed_collision_projection is not None:
                endpoint_guided = self._project_typed_collision_endpoint(
                    endpoint_guided, data_batch_for_guidance, aux_info,
                    num_samp, step_index,
                )

            # Explicit five-class contract; only selected actor actions are mutable.
            if getattr(self, "five_class_guide", None) is not None:
                if self.current_perturbation_guidance.current_guidance is not None or self.typed_collision_projection is not None:
                    raise RuntimeError("Five-class guide cannot mix with legacy guidance")
                endpoint_guided = self.five_class_guide.project(
                    self, endpoint_guided, aux_info, num_samp, step_index)

            # Guidance operates on the predicted clean endpoint. Convert it back
            # to a flow velocity before the deterministic Euler update.
            guided_velocity = (
                endpoint_guided.detach() - action_flow
            ) / remaining
            native_euler_actions = action_flow + step_size * guided_velocity
            action_flow = native_euler_actions
            x_clean_model_out = endpoint_guided.detach()
            x_flow = self.convert_action_to_state_and_action(
                action_flow, aux_info['curr_states']
            )

            if self.current_constraints is not None:
                x_flow = self.descale_traj(
                    x_flow.reshape((shape[0], shape[1], shape[2], -1))
                )
                x_flow = apply_constraints(
                    x_flow,
                    data_batch['scene_index'],
                    self.current_constraints,
                )
                x_flow = self.scale_traj(
                    x_flow.reshape((shape[0] * shape[1], shape[2], -1))
                )
                action_flow = x_flow[..., [4, 5]]

            progress.update({'lambda': flow_value})
            if return_diffusion:
                diffusion.append(x_flow)
            if is_final_step:
                attn_weights = info['attn_weights']

        progress.close()

        if velocity_residual_runtime is not None:
            step_audits = velocity_residual_runtime["step_audits"]
            accepted_steps = sum(
                audit["status"] == "accepted_velocity_residual"
                for audit in step_audits
            )
            rejected_steps = sum(
                audit["status"] == "native_fallback_trust_reject"
                for audit in step_audits
            )
            residual_audit = {
                "contract": "target_actor_fm_velocity_residual_rollout_v1",
                "status": (
                    "accepted_velocity_residual"
                    if accepted_steps > 0 and rejected_steps == 0
                    else "native_or_rejected_velocity_residual"
                ),
                "native_k": int(num_samp),
                "exact_frozen_flow_steps": int(self.n_flow_steps),
                "accepted_flow_steps": int(accepted_steps),
                "rejected_flow_steps": int(rejected_steps),
                "source_bank_native_bit_exact": True,
                "other_modes_bit_exact_every_step": bool(
                    all(audit["other_modes_bit_exact"] for audit in step_audits)
                ),
                "other_actors_bit_exact_every_step": bool(
                    all(audit["other_actors_bit_exact"] for audit in step_audits)
                ),
                "endpoint_projection": False,
                "trajectory_modified_after_sampling": False,
                "agent_row_mixing": False,
                "exact_kl_claim": False,
                "path_energy_regularizer": True,
                "maximum_global_residual_rms": float(
                    max(audit["global_residual_rms"] for audit in step_audits)
                ),
                "maximum_target_residual_rms": float(
                    max(audit["target_residual_rms"] for audit in step_audits)
                ),
                "maximum_one_step_delta_rms": float(
                    max(audit["one_step_delta_rms"] for audit in step_audits)
                ),
                "maximum_path_energy": float(
                    max(audit["path_energy"] for audit in step_audits)
                ),
                "proposed_coefficients": velocity_residual_runtime[
                    "proposed_coefficients"
                ],
                "online_cpa": velocity_residual_runtime["online_cpa"],
                "step_audits": step_audits,
            }
            self._conditional_flow_last_audit = residual_audit
            print(
                "TARGET_VELOCITY_RESIDUAL status={} coefficients={} "
                "online_cpa_step={} online_cpa_distance_m={} "
                "online_cpa_active={} alignment_mode={} arrival_step={} "
                "arrival_lateral_m={} arrival_required_speed_mps={} "
                "arrival_current_speed_mps={} arrival_speed_error_mps={} "
                "arrival_control_sign={} signed_coefficients={} "
                "accepted_flow_steps={} "
                "rejected_flow_steps={} max_global_rms={} "
                "max_target_rms={} max_step_delta_rms={} "
                "max_path_energy={} endpoint_projection=False "
                "row_mixing=False".format(
                    residual_audit["status"],
                    residual_audit["proposed_coefficients"],
                    residual_audit["online_cpa"]["cpa_step"],
                    residual_audit["online_cpa"]["cpa_center_distance_m"],
                    residual_audit["online_cpa"]["active"],
                    residual_audit["online_cpa"]["alignment_mode"],
                    residual_audit["online_cpa"]["arrival_step"],
                    residual_audit["online_cpa"]["arrival_lateral_distance_m"],
                    residual_audit["online_cpa"]["arrival_required_target_speed_mps"],
                    residual_audit["online_cpa"]["arrival_current_target_speed_mps"],
                    residual_audit["online_cpa"]["arrival_speed_error_mps"],
                    residual_audit["online_cpa"]["arrival_control_sign"],
                    residual_audit["online_cpa"]["signed_applied_coefficients"],
                    residual_audit["accepted_flow_steps"],
                    residual_audit["rejected_flow_steps"],
                    residual_audit["maximum_global_residual_rms"],
                    residual_audit["maximum_target_residual_rms"],
                    residual_audit["maximum_one_step_delta_rms"],
                    residual_audit["maximum_path_energy"],
                )
            )

        if any(guide_losses):
            print('===== GUIDANCE LOSSES ======')
            for key, value in guide_losses.items():
                if torch.isnan(value).any() or len(value) == 0:
                    value_mean = np.nan
                else:
                    value_mean = np.nanmean(value.cpu())
                print('%s: %.012f' % (key, value_mean))

        x_flow = TensorUtils.reshape_dimensions(
            x_flow,
            begin_axis=0,
            end_axis=1,
            target_dims=(batch_size, num_samp),
        )
        out_dict = {
            'pred_traj': x_flow,
            'attn_weights': attn_weights,
        }
        if return_guidance_losses:
            out_dict['guide_losses'] = guide_losses
        if return_diffusion:
            diffusion = [
                TensorUtils.reshape_dimensions(
                    cur_flow,
                    begin_axis=0,
                    end_axis=1,
                    target_dims=(batch_size, num_samp),
                )
                for cur_flow in diffusion
            ]
            out_dict['diffusion'] = torch.stack(diffusion, dim=3)

        return out_dict
