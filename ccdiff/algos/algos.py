#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

from collections import OrderedDict
from dataclasses import is_dataclass, replace
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import tbsim.utils.torch_utils as TorchUtils

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.utils.batch_utils import batch_utils
from tbsim.policies.common import Plan, Action

from ccdiff.models.diffuser_helpers import EMA
from ccdiff.models.ccdiff import CCDiffuserModel
from tbsim.utils.guidance_loss import choose_action_from_guidance, choose_action_from_gt
from tbsim.utils.trajdata_utils import convert_scene_data_to_agent_coordinates,  add_scene_dim_to_agent_data, get_stationary_mask


def _conditional_flow_runtime_objective_config(objective_config, global_step):
    """Bind a frozen objective template to the current simulator step."""
    if is_dataclass(objective_config):
        return replace(objective_config, current_global_step=int(global_step))
    return objective_config


def apply_semantic_reaction_hold(
    positions,
    yaws,
    receiver_idx,
    hold_steps,
    speed,
    yaw_rate,
    dt,
    accel_bias_mps2=0.0,
    elapsed_steps=0,
):
    """Hold a driver's pre-response command for a finite reaction interval.

    The intervention has no collision target and never projects coordinates onto
    a desired contact geometry.  It integrates a constant speed/yaw-rate command
    in the receiver's current agent frame, then translates the untouched M2 tail
    so the first post-delay waypoint remains continuous.
    """
    if positions.ndim != 4 or yaws.ndim != 4:
        raise ValueError("expected positions [B,M,T,2] and yaws [B,M,T,1]")
    if positions.shape[:3] != yaws.shape[:3]:
        raise ValueError("position/yaw shapes disagree")
    if positions.shape[-1] != 2 or yaws.shape[-1] != 1:
        raise ValueError("unexpected action dimensions")
    if not 0 <= int(receiver_idx) < positions.shape[1]:
        raise IndexError("reaction-delay receiver exceeds modeled agent count")
    if int(hold_steps) < 0 or int(elapsed_steps) < 0 or float(dt) <= 0:
        raise ValueError("hold_steps must be non-negative and dt positive")

    hold_steps = min(int(hold_steps), positions.shape[2])
    if hold_steps == 0:
        return positions, yaws

    out_positions = positions.clone()
    out_yaws = yaws.clone()
    batch_size = positions.shape[0]
    speed = torch.as_tensor(speed, dtype=positions.dtype, device=positions.device).reshape(-1)
    yaw_rate = torch.as_tensor(
        yaw_rate, dtype=positions.dtype, device=positions.device
    ).reshape(-1)
    if speed.numel() == 1:
        speed = speed.expand(batch_size)
    if yaw_rate.numel() == 1:
        yaw_rate = yaw_rate.expand(batch_size)
    if speed.numel() != batch_size or yaw_rate.numel() != batch_size:
        raise ValueError("speed/yaw-rate batch size mismatch")

    accel_bias_mps2 = float(accel_bias_mps2)
    if not np.isfinite(accel_bias_mps2):
        raise ValueError("accel_bias_mps2 must be finite")
    times = (
        torch.arange(1, hold_steps + 1, device=positions.device, dtype=positions.dtype)
        * float(dt)
    )
    theta = yaw_rate[:, None] * times[None]
    if accel_bias_mps2 == 0.0:
        omega = yaw_rate[:, None]
        near_zero = omega.abs() < 1.0e-5
        safe_omega = torch.where(near_zero, torch.ones_like(omega), omega)
        x_turn = speed[:, None] * torch.sin(theta) / safe_omega
        y_turn = speed[:, None] * (1.0 - torch.cos(theta)) / safe_omega
        x_straight = speed[:, None] * times[None]
        x = torch.where(near_zero, x_straight, x_turn)
        y = torch.where(near_zero, torch.zeros_like(y_turn), y_turn)
        held_positions = torch.stack((x, y), dim=-1)
    else:
        # Midpoint integration of a constant longitudinal acceleration bias.
        # The yaw-rate command still comes from the untouched M2 sample, so this
        # changes a human-interpretable driver command rather than targeting any
        # contact coordinate.
        # `positions` is expressed in the receiver's current local frame at each
        # receding-horizon call.  Yaw therefore restarts at zero, but the speed
        # schedule must retain elapsed time since the intervention began.
        midpoint_times = (
            times - 0.5 * float(dt) + int(elapsed_steps) * float(dt)
        )
        held_speed = torch.clamp(
            speed[:, None] + accel_bias_mps2 * midpoint_times[None],
            min=0.0,
            max=20.0,
        )
        midpoint_theta = yaw_rate[:, None] * midpoint_times[None]
        dx = held_speed * float(dt) * torch.cos(midpoint_theta)
        dy = held_speed * float(dt) * torch.sin(midpoint_theta)
        held_positions = torch.stack(
            (torch.cumsum(dx, dim=1), torch.cumsum(dy, dim=1)), dim=-1
        )
    held_yaws = theta.unsqueeze(-1)

    original_boundary_position = positions[:, receiver_idx, hold_steps - 1].clone()
    original_boundary_yaw = yaws[:, receiver_idx, hold_steps - 1].clone()
    out_positions[:, receiver_idx, :hold_steps] = held_positions
    out_yaws[:, receiver_idx, :hold_steps] = held_yaws
    if hold_steps < positions.shape[2]:
        position_offset = held_positions[:, -1] - original_boundary_position
        yaw_offset = held_yaws[:, -1] - original_boundary_yaw
        out_positions[:, receiver_idx, hold_steps:] = (
            positions[:, receiver_idx, hold_steps:] + position_offset[:, None]
        )
        out_yaws[:, receiver_idx, hold_steps:] = (
            yaws[:, receiver_idx, hold_steps:] + yaw_offset[:, None]
        )
    return out_positions, out_yaws


class CCDiffTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, registered_name, do_log=True, guidance_config=None, constraint_config=None, policy_model_cls=CCDiffuserModel, policy_model_kwargs=None):
        """
        The traffic model for the CCDiff
        """
        super(CCDiffTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        # assigned at run-time according to the given data batch
        self.data_centric = None
        # ['agent_centric', 'scene_centric']
        self.coordinate = algo_config.coordinate

        self.scene_agent_max_neighbor_dist = algo_config.scene_agent_max_neighbor_dist
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary
        self.moving_speed_th = algo_config.moving_speed_th
        self.stationary_mask = None
        
        if algo_config.diffuser_input_mode == 'state_and_action':
            observation_dim = 4 
            action_dim = 2
            output_dim = 2
        else:
            raise
        
        print('registered_name', registered_name)
        diffuser_norm_info = ([-17.5, 0, 0, 0, 0, 0],[22.5, 10, 40, 3.14, 500, 31.4])
        agent_hist_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        neighbor_hist_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        neighbor_fut_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        if 'nusc' in registered_name:
            diffuser_norm_info = algo_config.nusc_norm_info['diffuser']
            agent_hist_norm_info = algo_config.nusc_norm_info['agent_hist']
            if 'neighbor_hist' in algo_config.nusc_norm_info:
                neighbor_hist_norm_info = algo_config.nusc_norm_info['neighbor_hist']
            if 'neighbor_fut' in algo_config.nusc_norm_info:
                neighbor_fut_norm_info = algo_config.nusc_norm_info['neighbor_fut']
        else:
            raise


        self.cond_drop_map_p = algo_config.conditioning_drop_map_p
        self.cond_drop_neighbor_p = algo_config.conditioning_drop_neighbor_p
        self.cond_drop_graph_p = algo_config.conditioning_drop_graph_p
        min_cond_drop_p = min([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        max_cond_drop_p = max([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        assert min_cond_drop_p >= 0.0 and max_cond_drop_p <= 1.0
        self.use_cond = self.cond_drop_map_p < 1.0 and self.cond_drop_neighbor_p < 1.0 # no need for conditioning arch if always dropping
        self.cond_fill_val = algo_config.conditioning_drop_fill

        self.use_rasterized_map = algo_config.rasterized_map
        self.use_rasterized_hist = algo_config.rasterized_history

        if self.use_cond:
            print(self.cond_drop_graph_p)
            if self.cond_drop_graph_p > 0:
                print('DIFFUSER: Dropping Causal Graph mask conditioning with p = %f during training...' % (self.cond_drop_map_p))
                

        policy_model_kwargs = {} if policy_model_kwargs is None else dict(policy_model_kwargs)
        self.nets["policy"] = policy_model_cls(
            rasterized_map=algo_config.rasterized_map,
            use_map_feat_global=algo_config.use_map_feat_global,
            use_map_feat_grid=algo_config.use_map_feat_grid,
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            map_grid_feature_dim=algo_config.map_grid_feature_dim,

            rasterized_hist=algo_config.rasterized_history,
            hist_num_frames=algo_config.history_num_frames+1, # the current step is concat to the history
            hist_feature_dim=algo_config.history_feature_dim,

            diffuser_model_arch=algo_config.diffuser_model_arch,
            horizon=algo_config.horizon,

            observation_dim=observation_dim, 
            action_dim=action_dim,

            output_dim=output_dim,

            n_timesteps=algo_config.n_diffusion_steps,
            
            loss_type=algo_config.loss_type, 
            clip_denoised=algo_config.clip_denoised,

            predict_epsilon=algo_config.predict_epsilon,
            action_weight=algo_config.action_weight, 
            loss_discount=algo_config.loss_discount, 
            loss_weights=algo_config.loss_weights,
            loss_decay_rates=algo_config.loss_decay_rates,

            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,

            action_loss_only = algo_config.action_loss_only,
            
            diffuser_input_mode=algo_config.diffuser_input_mode,
            use_reconstructed_state=algo_config.use_reconstructed_state,

            use_conditioning=self.use_cond,
            cond_fill_value=self.cond_fill_val,

            diffuser_norm_info=diffuser_norm_info,
            agent_hist_norm_info=agent_hist_norm_info,
            neighbor_hist_norm_info=neighbor_hist_norm_info,
            neighbor_fut_norm_info=neighbor_fut_norm_info,

            agent_hist_embed_method=algo_config.agent_hist_embed_method,
            neigh_hist_embed_method=algo_config.neigh_hist_embed_method,
            map_embed_method=algo_config.map_embed_method,
            interaction_edge_speed_repr=algo_config.interaction_edge_speed_repr,
            normalize_rel_states=algo_config.normalize_rel_states,
            mask_social_interaction=algo_config.mask_social_interaction,
            mask_edge=algo_config.mask_edge,
            neighbor_inds=algo_config.neighbor_inds,
            edge_attr_separation=algo_config.edge_attr_separation,
            social_attn_radius=algo_config.social_attn_radius,
            use_last_hist_step=algo_config.use_last_hist_step,
            use_noisy_fut_edge=algo_config.use_noisy_fut_edge,
            semantic_visibility_dropout_p=algo_config.semantic_visibility_dropout_p,
            semantic_visibility_dropout_radius=algo_config.semantic_visibility_dropout_radius,
            use_const_speed_edge=algo_config.use_const_speed_edge,
            all_interactive_social=algo_config.all_interactive_social,
            mask_time=algo_config.mask_time,
            layer_num_per_edge_decoder=algo_config.layer_num_per_edge_decoder,
            attn_combination=algo_config.attn_combination,
            single_cond_feat=algo_config.single_cond_feat,

            disable_control_on_stationary=self.disable_control_on_stationary,

            coordinate=self.coordinate,

            # controllable_agent=algo_config.controllable_agent,
            **policy_model_kwargs,
        )

        # set up initial guidance and constraints
        if guidance_config is not None:
            self.set_guidance(guidance_config)
        if constraint_config is not None:
            self.set_constraints(constraint_config)

        # set up EMA
        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            self.ema_policy = copy.deepcopy(self.nets["policy"])
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()

        self.cur_train_step = 0

    @property
    def checkpoint_monitor_keys(self):
        if self.use_ema:
            return {"valLoss": "val/ema_losses_diffusion_loss"}
        else:
            return {"valLoss": "val/losses_diffusion_loss"}
    
    def forward(self, obs_dict, plan=None, step_index=0, num_samp=1, class_free_guide_w=0.0, guide_as_filter_only=False, guide_clean=False, global_t=0):
        if self.disable_control_on_stationary and global_t == 0:
            self.stationary_mask = get_stationary_mask(obs_dict, self.disable_control_on_stationary, self.moving_speed_th)
            target = getattr(self, "five_class_stationary_target", None)
            if target is not None:
                from ccdiff.evaluation.five_class_guidance import release_selected_stationary
                self.stationary_mask = release_selected_stationary(self.stationary_mask, target)
            B, M = self.stationary_mask.shape
            # (B, M) -> (B, N, M) -> (B*N, M)
            stationary_mask_expand =  self.stationary_mask.unsqueeze(1).expand(B, num_samp, M).reshape(B*num_samp, M)
        else:
            stationary_mask_expand = None
        cur_policy = self.nets["policy"]

        # this function is only called at validation time, so use ema
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)

        return cur_policy(obs_dict, plan, num_samp, return_diffusion=True,
                                   return_guidance_losses=True, class_free_guide_w=class_free_guide_w,
                                   apply_guidance=(not guide_as_filter_only),
                                   guide_clean=guide_clean, global_t=global_t, stationary_mask=stationary_mask_expand)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        sample_preds = predictions["positions"]
        B, N, M, T, _ = sample_preds.shape
        # (B, N, M, T, 2) -> (B, M, N, T, 2) -> (B*M, N, T, 2)
        sample_preds = TensorUtils.to_numpy(sample_preds.permute(0, 2, 1, 3, 4).reshape(B*M, N, T, -1))
        # (B, M, T, 2) -> (B*M, T, 2)
        gt = TensorUtils.to_numpy(data_batch["target_positions"].reshape(B*M, T, -1))
        # (B, M, T) -> (B*M, T)
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"].reshape(B*M, T))
        
        # compute ADE & FDE based on trajectory samples
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.nets["policy"].state_dict())

    def on_train_start(self):
        # cur_train_step is a plain Python counter and is therefore not part of
        # the checkpoint state_dict. Align it with Lightning's restored global
        # step so a resumed run does not incorrectly re-enter the 4000-step EMA
        # warm-up and overwrite the restored EMA weights.
        self.cur_train_step = int(self.global_step)

    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.nets["policy"])

    def training_step_end(self, batch_parts):
        self.cur_train_step += 1

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number (relative to the CURRENT epoch) - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        if self.use_ema and self.cur_train_step % self.ema_update_every == 0:
            self.step_ema(self.cur_train_step)
        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, max_neighbor_dist=self.scene_agent_max_neighbor_dist, keep_order_of_neighbors=True)
        elif self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            batch = add_scene_dim_to_agent_data(batch)
        elif self.data_centric == 'scene' and self.coordinate == 'scene_centric':
            pass
        else:
            raise NotImplementedError
       
        # drop out conditioning if desired
        if self.use_cond:
            if self.use_rasterized_map:
                num_sem_layers = batch['maps'].size(1) # NOTE: this assumes a trajdata-based loader. Will not work with lyft-specific loader.
                if self.cond_drop_map_p > 0:
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_map_p
                    # only fill the last num_sem_layers as these correspond to semantic map
                    batch["image"][drop_mask, -num_sem_layers:] = self.cond_fill_val

            if self.use_rasterized_hist:
                # drop layers of map corresponding to histories
                num_sem_layers = batch['maps'].size(1) if batch['maps'] is not None else None
                if self.cond_drop_neighbor_p > 0:
                    # sample different mask so sometimes both get dropped, sometimes only one
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_neighbor_p
                    if num_sem_layers is None:
                        batch["image"][drop_mask] = self.cond_fill_val
                    else:
                        # only fill the layers before semantic map corresponding to trajectories (neighbors and ego)
                        batch["image"][drop_mask, :-num_sem_layers] = self.cond_fill_val
        
        # diffuser only take the data to estimate loss
        losses = self.nets["policy"].compute_losses(batch)

        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        # metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)

        return {
            "loss": total_loss,
            "all_losses": losses,
        }
    
    def validation_step(self, batch, batch_idx):
        cur_policy = self.nets["policy"]
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)

        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, max_neighbor_dist=self.scene_agent_max_neighbor_dist, keep_order_of_neighbors=True)
        elif self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            batch = add_scene_dim_to_agent_data(batch)
        elif self.data_centric == 'scene' and self.coordinate == 'scene_centric':
            pass
        else:
            raise NotImplementedError
        
        losses = TensorUtils.detach(cur_policy.compute_losses(batch))
        
        pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False, mode='training')
        metrics = self._compute_metrics(pout, batch)
        return_dict =  {"losses": losses, "metrics": metrics}

        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
            ema_losses = TensorUtils.detach(cur_policy.compute_losses(batch))
            pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False, mode='training')
            ema_metrics = self._compute_metrics(pout, batch)
            return_dict["ema_losses"] = ema_losses
            return_dict["ema_metrics"] = ema_metrics

        return return_dict


    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            # Under torchrun/DDP each rank sees a different validation shard.
            # Synchronize the per-rank means so checkpoint monitoring and the
            # TensorBoard curves describe the full four-GPU validation sample.
            self.log("val/losses_" + k, m, sync_dist=True)
        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m, sync_dist=True)
        
        if self.use_ema:
            for k in outputs[0]["ema_losses"]:
                m = torch.stack([o["ema_losses"][k] for o in outputs]).mean()
                self.log("val/ema_losses_" + k, m, sync_dist=True)
            for k in outputs[0]["ema_metrics"]:
                m = np.stack([o["ema_metrics"][k] for o in outputs]).mean()
                self.log("val/ema_metrics_" + k, m, sync_dist=True)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.nets["policy"].parameters(),
            lr=optim_params["learning_rate"]["initial"],
        )

    def get_plan(self, obs_dict, **kwargs):
        plan = kwargs.get("plan", None)
        preds = self(obs_dict, plan)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}
    
    def get_action(self, obs_dict,
                    num_action_samples=1,
                    class_free_guide_w=0.0, 
                    guide_as_filter_only=False,
                    guide_with_gt=False,
                    guide_clean=False,
                    **kwargs):
        plan = kwargs.get("plan", None)

        cur_policy = self.nets["policy"]

        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy

        schedule = getattr(self, "_interaction_edge_cut_schedule", None)
        if schedule is not None:
            global_step = int(kwargs["step_index"])
            active = global_step >= schedule["start_step"] and (
                schedule["end_step"] is None
                or global_step < schedule["end_step"]
            )
            policies = [self.nets["policy"]]
            if self.use_ema:
                policies.append(self.ema_policy)
            for scheduled_policy in policies:
                if active:
                    scheduled_policy.set_interaction_edge_cuts(
                        self._interaction_edge_cut_pairs,
                        bidirectional=self._interaction_edge_cut_bidirectional,
                    )
                else:
                    scheduled_policy.clear_interaction_edge_cut()

        full_visibility_schedule = getattr(
            self, "_interaction_full_visibility_cut_schedule", None
        )
        if full_visibility_schedule is not None:
            global_step = int(kwargs["step_index"])
            active = global_step >= full_visibility_schedule["start_step"] and (
                full_visibility_schedule["end_step"] is None
                or global_step < full_visibility_schedule["end_step"]
            )
            policies = [self.nets["policy"]]
            if self.use_ema:
                policies.append(self.ema_policy)
            for scheduled_policy in policies:
                if active:
                    scheduled_policy.set_interaction_full_visibility_cuts(
                        self._interaction_full_visibility_cut_pairs,
                        bidirectional=(
                            self._interaction_full_visibility_cut_bidirectional
                        ),
                    )
                else:
                    scheduled_policy.clear_interaction_full_visibility_cut()

        delay_schedule = getattr(self, "_interaction_edge_delay_schedule", None)
        if delay_schedule is not None:
            global_step = int(kwargs["step_index"])
            active = global_step >= delay_schedule["start_step"] and (
                delay_schedule["end_step"] is None
                or global_step < delay_schedule["end_step"]
            )
            policies = [self.nets["policy"]]
            if self.use_ema:
                policies.append(self.ema_policy)
            for scheduled_policy in policies:
                if active:
                    scheduled_policy.set_interaction_edge_delays(
                        self._interaction_edge_delays,
                        bidirectional=self._interaction_edge_delay_bidirectional,
                    )
                else:
                    scheduled_policy.clear_interaction_edge_delay()

        intent_schedule = getattr(
            self, "_interaction_future_intent_hold_schedule", None
        )
        if intent_schedule is not None:
            global_step = int(kwargs["step_index"])
            active = global_step >= intent_schedule["start_step"] and (
                intent_schedule["end_step"] is None
                or global_step < intent_schedule["end_step"]
            )
            policies = [self.nets["policy"]]
            if self.use_ema:
                policies.append(self.ema_policy)
            for scheduled_policy in policies:
                if active:
                    scheduled_policy.set_interaction_future_intent_holds(
                        self._interaction_future_intent_hold_pairs,
                        bidirectional=(
                            self._interaction_future_intent_hold_bidirectional
                        ),
                    )
                else:
                    scheduled_policy.clear_interaction_future_intent_hold()

        yield_schedule = getattr(
            self, "_interaction_yield_belief_schedule", None
        )
        if yield_schedule is not None:
            global_step = int(kwargs["step_index"])
            active = global_step >= yield_schedule["start_step"] and (
                yield_schedule["end_step"] is None
                or global_step < yield_schedule["end_step"]
            )
            policies = [self.nets["policy"]]
            if self.use_ema:
                policies.append(self.ema_policy)
            for scheduled_policy in policies:
                if active:
                    scheduled_policy.set_interaction_yield_beliefs(
                        self._interaction_yield_beliefs,
                        bidirectional=self._interaction_yield_belief_bidirectional,
                    )
                else:
                    scheduled_policy.clear_interaction_yield_belief()

        assertive_schedule = getattr(
            self, "_semantic_assertive_receiver_schedule", None
        )
        if assertive_schedule is not None:
            global_step = int(kwargs["step_index"])
            active = global_step >= assertive_schedule["start_step"] and (
                assertive_schedule["end_step"] is None
                or global_step < assertive_schedule["end_step"]
            )
            policies = [self.nets["policy"]]
            if self.use_ema:
                policies.append(self.ema_policy)
            for scheduled_policy in policies:
                if active:
                    scheduled_policy.set_semantic_assertive_receivers(
                        self._semantic_assertive_receiver_indices,
                        scale=self._semantic_assertive_token_scale,
                    )
                else:
                    scheduled_policy.clear_semantic_assertive_receivers()

        correlated_prior_schedule = getattr(
            self, "_marginal_preserving_correlated_flow_prior_schedule", None
        )
        if correlated_prior_schedule is not None:
            global_step = int(kwargs["step_index"])
            active = global_step >= correlated_prior_schedule["start_step"] and (
                correlated_prior_schedule["end_step"] is None
                or global_step < correlated_prior_schedule["end_step"]
            )
            policies = [self.nets["policy"]]
            if self.use_ema:
                policies.append(self.ema_policy)
            for scheduled_policy in policies:
                if active:
                    scheduled_policy.set_marginal_preserving_correlated_flow_prior(
                        self._marginal_preserving_correlated_flow_prior_pairs,
                        self._marginal_preserving_correlated_flow_prior_rho,
                        self._marginal_preserving_correlated_flow_prior_lag_steps,
                        self._marginal_preserving_correlated_flow_prior_action_dims,
                    )
                else:
                    scheduled_policy.clear_marginal_preserving_correlated_flow_prior()

        pcn_prior_schedule = getattr(
            self, "_pcn_joint_flow_prior_schedule", None
        )
        if pcn_prior_schedule is not None:
            global_step = int(kwargs["step_index"])
            active = global_step >= pcn_prior_schedule["start_step"] and (
                pcn_prior_schedule["end_step"] is None
                or global_step < pcn_prior_schedule["end_step"]
            )
            policies = [self.nets["policy"]]
            if self.use_ema:
                policies.append(self.ema_policy)
            for scheduled_policy in policies:
                if active:
                    scheduled_policy.set_pcn_joint_flow_prior(
                        self._pcn_joint_flow_prior_beta
                    )
                else:
                    scheduled_policy.clear_pcn_joint_flow_prior()

        conditional_flow_schedule = getattr(
            self, "_conditional_flow_source_control_schedule", None
        )
        if conditional_flow_schedule is not None:
            global_step = int(kwargs["step_index"])
            active = global_step >= conditional_flow_schedule["start_step"] and (
                conditional_flow_schedule["end_step"] is None
                or global_step < conditional_flow_schedule["end_step"]
            )
            policies = [self.nets["policy"]]
            if self.use_ema:
                policies.append(self.ema_policy)
            for scheduled_policy in policies:
                if active:
                    objective_config = _conditional_flow_runtime_objective_config(
                        conditional_flow_schedule["objective_config"], global_step
                    )
                    scheduled_policy.set_conditional_flow_source_control(
                        conditional_flow_schedule["source_config"],
                        objective_config,
                    )
                else:
                    scheduled_policy.clear_conditional_flow_source_control()
            if getattr(self, "_conditional_flow_source_control_active", None) != active:
                print(
                    "CONDITIONAL_FM_GLOBAL_SCHEDULE step={} active={} start_step={} end_step={}".format(
                        global_step,
                        active,
                        conditional_flow_schedule["start_step"],
                        conditional_flow_schedule["end_step"],
                    )
                )
                self._conditional_flow_source_control_active = active
        
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)

        # already called in policy_composer, but just for good measure...
        cur_policy.eval()

        # update with current "global" timestep
        cur_policy.update_guidance(global_t=kwargs['step_index'])
        preds = self(obs_dict, plan, num_samp=num_action_samples,
                    class_free_guide_w=class_free_guide_w, guide_as_filter_only=guide_as_filter_only,
                    guide_clean=guide_clean, global_t=kwargs['step_index']) 

        # [B, N, M, T, 2]
        B, N, M, _, _ = preds["positions"].shape

        # arbitrarily use the first sample as the action by default
        act_idx = torch.zeros((M), dtype=torch.long, device=preds["positions"].device)
        if guide_with_gt and "target_positions" in obs_dict:
            act_idx = choose_action_from_gt(preds, obs_dict)
        elif cur_policy.current_perturbation_guidance.current_guidance is not None:
            # choose sample closest to desired guidance
            guide_losses = preds.pop("guide_losses", None)
            act_idx = choose_action_from_guidance(preds, obs_dict, cur_policy.current_perturbation_guidance.current_guidance.guide_configs, guide_losses)          

        def map_act_idx(x):
            # Assume B == 1 during generation. TBD: need to change this to support general batchsize
            if len(x.shape) == 4:
                # [N, T, M1, M2] -> [M1, N, T, M2]
                x = x.permute(2,0,1,3)
            elif len(x.shape) == 5:
                # [B, N, M, T, 2] -> [N, M, T, 2] -> [M, N, T, 2]
                x = x[0].permute(1,0,2,3)
            elif len(x.shape) == 6: # for "diffusion_steps"
                x = x[0].permute(1,0,2,3,4)
            else:
                raise NotImplementedError
            # [M, N, T, 2] -> [M, T, 2]
            x = x[torch.arange(M), act_idx]
            # [M, T, 2] -> [B, M, T, 2]
            x = x.unsqueeze(0)
            return x
        
        preds_positions = preds["positions"]
        preds_yaws = preds["yaws"]
        preds_trajectories = preds["trajectories"]

        action_preds_positions = map_act_idx(preds_positions)
        action_preds_yaws = map_act_idx(preds_yaws)

        reaction_cfg = getattr(self, "_semantic_reaction_delay", None)
        reaction_audit = None
        if reaction_cfg is not None:
            global_step = int(kwargs["step_index"])
            delay_end = reaction_cfg["start_step"] + reaction_cfg["delay_steps"]
            if reaction_cfg["end_step"] is not None:
                delay_end = min(delay_end, reaction_cfg["end_step"])
            active = reaction_cfg["start_step"] <= global_step < delay_end
            receiver_idx = reaction_cfg["receiver_idx"]
            if receiver_idx >= M:
                raise IndexError(
                    "reaction-delay receiver {} exceeds modeled agent count {}".format(
                        receiver_idx, M
                    )
                )
            if active:
                state = getattr(self, "_semantic_reaction_delay_state", None)
                if state is None:
                    dt = float(getattr(cur_policy, "dt", 0.1))
                    initial_speed = obs_dict["curr_speed"][:, receiver_idx].detach().clone()
                    initial_yaw_rate = (
                        action_preds_yaws[:, receiver_idx, 0, 0].detach().clone() / dt
                    )
                    state = {
                        "speed": initial_speed,
                        "yaw_rate": initial_yaw_rate,
                        "captured_step": global_step,
                    }
                    self._semantic_reaction_delay_state = state
                remaining_steps = delay_end - global_step
                action_preds_positions, action_preds_yaws = apply_semantic_reaction_hold(
                    action_preds_positions,
                    action_preds_yaws,
                    receiver_idx=receiver_idx,
                    hold_steps=remaining_steps,
                    speed=state["speed"],
                    yaw_rate=state["yaw_rate"],
                    dt=float(getattr(cur_policy, "dt", 0.1)),
                    accel_bias_mps2=reaction_cfg["accel_bias_mps2"],
                    elapsed_steps=global_step - state["captured_step"],
                )
                reaction_audit = {
                    "active": True,
                    "receiver_idx": receiver_idx,
                    "remaining_steps": remaining_steps,
                    "captured_step": state["captured_step"],
                    "accel_bias_mps2": reaction_cfg["accel_bias_mps2"],
                }
            else:
                self._semantic_reaction_delay_state = None
                reaction_audit = {
                    "active": False,
                    "receiver_idx": receiver_idx,
                    "remaining_steps": 0,
                }
        
        attn_weights = map_act_idx(preds["attn_weights"]).squeeze(0)

        if self.disable_control_on_stationary and self.stationary_mask is not None:
            
            stationary_mask_expand = self.stationary_mask.unsqueeze(1).expand(B, N, M)
            
            preds_positions[stationary_mask_expand] = 0
            preds_yaws[stationary_mask_expand] = 0
            preds_trajectories[stationary_mask_expand] = 0

            action_preds_positions[self.stationary_mask] = 0
            action_preds_yaws[self.stationary_mask] = 0

        info = dict(
            action_samples=Action(
                positions=preds_positions, # (B, N, M, T, 2)
                yaws=preds_yaws
            ).to_dict(),
            trajectories=preds_trajectories,
            act_idx=act_idx,
            dyn=self.nets["policy"].dyn,
            attn_weights=attn_weights,
            semantic_reaction_delay=reaction_audit,
            conditional_flow_source_audit=getattr(
                cur_policy, "_conditional_flow_last_audit", None
            ),
        )
        action = Action(
            positions=action_preds_positions, # (B, M, T, 2)
            yaws=action_preds_yaws
        )
        return action, info

    def set_guidance(self, guidance_config, example_batch=None):
        '''
        Resets the test-time guidance functions to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)
        cur_policy.set_guidance(guidance_config, example_batch)
    
    def clear_guidance(self):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.clear_guidance()

    def set_flow_velocity_guidance(self, config=None):
        """Forward FM velocity-field guidance to policy and EMA copies.

        The rollout wrappers receive ``CCDiffTrafficModel`` as ``model``;
        the actual Euler sampler lives one level deeper in the policy module.
        Keep both copies synchronized so evaluation cannot accidentally set
        state only on the inactive non-EMA policy.
        """
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            if not hasattr(policy, "set_flow_velocity_guidance"):
                raise RuntimeError(
                    "flow-velocity guidance requires a compatible policy"
                )
            policy.set_flow_velocity_guidance(config)

    def set_typed_collision_projection(self, config):
        """Forward one inference-only typed projector to policy and EMA copy."""
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_typed_collision_projection(config)

    def clear_typed_collision_projection(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_typed_collision_projection()

    def set_conditional_flow_source_control(
        self, source_config, objective_config
    ):
        """Forward an inference-only source-control config to policy and EMA."""
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.eval()
            policy.set_conditional_flow_source_control(
                source_config, objective_config
            )

    def clear_conditional_flow_source_control(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_conditional_flow_source_control()

    def set_conditional_flow_source_control_schedule(
        self,
        source_config,
        objective_config,
        start_step=0,
        end_step=None,
    ):
        """Activate source control at simulation steps, not source-horizon indices."""
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        if start_step < 0:
            raise ValueError("conditional-flow start_step must be non-negative")
        if end_step is not None and end_step <= start_step:
            raise ValueError("conditional-flow end_step must exceed start_step")
        self._conditional_flow_source_control_schedule = {
            "source_config": source_config,
            "objective_config": objective_config,
            "start_step": start_step,
            "end_step": end_step,
        }
        self._conditional_flow_source_control_active = None
        # Identity before activation is explicit, including the EMA copy.
        self.clear_conditional_flow_source_control()

    def set_interaction_edge_cut(self, receiver_idx, sender_idx, bidirectional=False):
        self.set_interaction_edge_cuts(
            [(receiver_idx, sender_idx)], bidirectional=bidirectional
        )

    def set_interaction_edge_cuts(self, edge_pairs, bidirectional=False):
        edge_pairs = [(int(receiver), int(sender)) for receiver, sender in edge_pairs]
        self._interaction_edge_cut_pairs = edge_pairs
        self._interaction_edge_cut_bidirectional = bool(bidirectional)
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_interaction_edge_cuts(
                edge_pairs, bidirectional=bidirectional
            )
        print(
            "EDGE_CUT_ATTESTATION pairs={} bidirectional={} "
            "gradient_guidance=False transport=flow_matching".format(
                edge_pairs, bool(bidirectional)
            )
        )

    def set_interaction_edge_cut_schedule(self, start_step=0, end_step=None):
        """Activate the configured edge cut only inside a rollout-step window."""
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("invalid interaction edge-cut schedule")
        if not hasattr(self, "_interaction_edge_cut_pairs"):
            raise RuntimeError("set interaction edge cuts before their schedule")
        self._interaction_edge_cut_schedule = {
            "start_step": start_step,
            "end_step": end_step,
        }
        print(
            "EDGE_CUT_SCHEDULE_ATTESTATION start_step={} end_step={}".format(
                start_step, end_step
            )
        )

    def clear_interaction_edge_cut(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_interaction_edge_cut()
        self._interaction_edge_cut_schedule = None

    def set_interaction_full_visibility_cuts(
        self, edge_pairs, bidirectional=False
    ):
        edge_pairs = [
            (int(receiver), int(sender)) for receiver, sender in edge_pairs
        ]
        self._interaction_full_visibility_cut_pairs = edge_pairs
        self._interaction_full_visibility_cut_bidirectional = bool(bidirectional)
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_interaction_full_visibility_cuts(
                edge_pairs, bidirectional=bidirectional
            )
        print(
            "FULL_VISIBILITY_CUT_ATTESTATION pairs={} bidirectional={} "
            "paths=history_encoder,edge_decoder,later_social_decoder "
            "gradient_guidance=False projection=False trajectory_modified=False "
            "transport=flow_matching".format(
                edge_pairs, bool(bidirectional)
            )
        )

    def set_interaction_full_visibility_cut_schedule(
        self, start_step=0, end_step=None
    ):
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("invalid full-visibility cut schedule")
        if not hasattr(self, "_interaction_full_visibility_cut_pairs"):
            raise RuntimeError(
                "set full-visibility cuts before their schedule"
            )
        self._interaction_full_visibility_cut_schedule = {
            "start_step": start_step,
            "end_step": end_step,
        }
        print(
            "FULL_VISIBILITY_CUT_SCHEDULE_ATTESTATION start_step={} "
            "end_step={}".format(start_step, end_step)
        )

    def clear_interaction_full_visibility_cut(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_interaction_full_visibility_cut()
        self._interaction_full_visibility_cut_schedule = None

    def set_interaction_edge_delays(self, edge_delays, bidirectional=False):
        edge_delays = [
            (int(receiver), int(sender), int(delay_steps))
            for receiver, sender, delay_steps in edge_delays
        ]
        self._interaction_edge_delays = edge_delays
        self._interaction_edge_delay_bidirectional = bool(bidirectional)
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_interaction_edge_delays(
                edge_delays, bidirectional=bidirectional
            )
        print(
            "EDGE_DELAY_ATTESTATION edges={} bidirectional={} "
            "gradient_guidance=False projection=False transport=flow_matching".format(
                edge_delays, bool(bidirectional)
            )
        )

    def set_interaction_edge_delay_schedule(self, start_step=0, end_step=None):
        """Activate configured perception delays only inside a rollout window."""
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("invalid interaction edge-delay schedule")
        if not hasattr(self, "_interaction_edge_delays"):
            raise RuntimeError("set interaction edge delays before their schedule")
        self._interaction_edge_delay_schedule = {
            "start_step": start_step,
            "end_step": end_step,
        }
        print(
            "EDGE_DELAY_SCHEDULE_ATTESTATION start_step={} end_step={}".format(
                start_step, end_step
            )
        )

    def clear_interaction_edge_delay(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_interaction_edge_delay()
        self._interaction_edge_delay_schedule = None

    def set_interaction_future_intent_holds(
        self, edge_pairs, bidirectional=False
    ):
        pairs = [(int(receiver), int(sender)) for receiver, sender in edge_pairs]
        self._interaction_future_intent_hold_pairs = pairs
        self._interaction_future_intent_hold_bidirectional = bool(bidirectional)
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_interaction_future_intent_holds(
                pairs, bidirectional=bidirectional
            )
        print(
            "FUTURE_INTENT_HOLD_ATTESTATION pairs={} bidirectional={} "
            "history=untouched future_belief=last_observed_relative_state "
            "gradient_guidance=False projection=False trajectory_modified=False "
            "transport=flow_matching".format(pairs, bool(bidirectional))
        )

    def set_interaction_future_intent_hold_schedule(
        self, start_step=0, end_step=None
    ):
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("invalid future-intent hold schedule")
        if not hasattr(self, "_interaction_future_intent_hold_pairs"):
            raise RuntimeError("set future-intent hold edges before schedule")
        self._interaction_future_intent_hold_schedule = {
            "start_step": start_step, "end_step": end_step,
        }
        print(
            "FUTURE_INTENT_HOLD_SCHEDULE_ATTESTATION start_step={} end_step={}".format(
                start_step, end_step
            )
        )

    def clear_interaction_future_intent_hold(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_interaction_future_intent_hold()
        self._interaction_future_intent_hold_schedule = None

    def set_interaction_yield_beliefs(
        self, edge_beliefs, bidirectional=False
    ):
        beliefs = [
            (int(receiver), int(sender), float(deceleration_mps2))
            for receiver, sender, deceleration_mps2 in edge_beliefs
        ]
        self._interaction_yield_beliefs = beliefs
        self._interaction_yield_belief_bidirectional = bool(bidirectional)
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_interaction_yield_beliefs(
                beliefs, bidirectional=bidirectional
            )
        print(
            "YIELD_BELIEF_ATTESTATION beliefs={} bidirectional={} "
            "history=untouched future_belief=constant_deceleration "
            "gradient_guidance=False projection=False trajectory_modified=False "
            "transport=flow_matching".format(beliefs, bool(bidirectional))
        )

    def set_interaction_yield_belief_schedule(
        self, start_step=0, end_step=None
    ):
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("invalid yield-belief schedule")
        if not hasattr(self, "_interaction_yield_beliefs"):
            raise RuntimeError("set yield beliefs before their schedule")
        self._interaction_yield_belief_schedule = {
            "start_step": start_step, "end_step": end_step,
        }
        print(
            "YIELD_BELIEF_SCHEDULE_ATTESTATION start_step={} end_step={}".format(
                start_step, end_step
            )
        )

    def clear_interaction_yield_belief(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_interaction_yield_belief()
        self._interaction_yield_belief_schedule = None

    def set_semantic_assertive_receivers(self, receiver_indices, scale=1.0):
        receiver_indices = sorted({int(index) for index in receiver_indices})
        if not receiver_indices or receiver_indices[0] < 0:
            raise ValueError("assertive receivers must be non-negative indices")
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("assertive token scale must be finite and positive")
        self._semantic_assertive_receiver_indices = receiver_indices
        self._semantic_assertive_token_scale = scale
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_semantic_assertive_receivers(
                receiver_indices, scale=scale
            )
        print(
            "SEMANTIC_ASSERTIVE_TOKEN_ATTESTATION receivers={} scale={} "
            "gradient_guidance=False projection=False trajectory_modified=False "
            "transport=flow_matching".format(receiver_indices, scale)
        )

    def set_semantic_assertive_receiver_schedule(
        self, start_step=0, end_step=None
    ):
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("invalid assertive-token schedule")
        if not hasattr(self, "_semantic_assertive_receiver_indices"):
            raise RuntimeError("set assertive receivers before their schedule")
        self._semantic_assertive_receiver_schedule = {
            "start_step": start_step,
            "end_step": end_step,
        }
        print(
            "SEMANTIC_ASSERTIVE_TOKEN_SCHEDULE_ATTESTATION "
            "start_step={} end_step={}".format(start_step, end_step)
        )

    def clear_semantic_assertive_receivers(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_semantic_assertive_receivers()
        self._semantic_assertive_receiver_schedule = None

    def set_marginal_preserving_correlated_flow_prior(
        self, agent_pairs, rho, lag_steps=0, action_dims=None
    ):
        pairs = [(int(left), int(right)) for left, right in agent_pairs]
        rho = float(rho)
        lag_steps = int(lag_steps)
        parsed_dims = (
            None if action_dims is None else [int(dim) for dim in action_dims]
        )
        self._marginal_preserving_correlated_flow_prior_pairs = pairs
        self._marginal_preserving_correlated_flow_prior_rho = rho
        self._marginal_preserving_correlated_flow_prior_lag_steps = lag_steps
        self._marginal_preserving_correlated_flow_prior_action_dims = parsed_dims
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_marginal_preserving_correlated_flow_prior(
                pairs, rho, lag_steps, parsed_dims
            )
        anchors = [right for _, right in pairs]
        copula_name = (
            "shared_anchor_lagged_gaussian"
            if len(set(anchors)) < len(anchors)
            else "pairwise_lagged_gaussian"
        )
        print(
            "CORRELATED_FLOW_PRIOR_ATTESTATION pairs={} rho={} lag_steps={} action_dims={} "
            "marginals=standard_gaussian copula={} "
            "gradient_guidance=False projection=False trajectory_modified=False "
            "transport=flow_matching".format(
                pairs, rho, lag_steps, parsed_dims, copula_name
            )
        )

    def set_marginal_preserving_correlated_flow_prior_schedule(
        self, start_step=0, end_step=None
    ):
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("invalid correlated flow-prior schedule")
        if not hasattr(self, "_marginal_preserving_correlated_flow_prior_pairs"):
            raise RuntimeError("set correlated flow-prior pairs before schedule")
        self._marginal_preserving_correlated_flow_prior_schedule = {
            "start_step": start_step,
            "end_step": end_step,
        }
        print(
            "CORRELATED_FLOW_PRIOR_SCHEDULE_ATTESTATION start_step={} "
            "end_step={}".format(start_step, end_step)
        )

    def clear_marginal_preserving_correlated_flow_prior(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_marginal_preserving_correlated_flow_prior()
        self._marginal_preserving_correlated_flow_prior_schedule = None

    def set_pcn_joint_flow_prior(self, beta):
        beta = float(beta)
        self._pcn_joint_flow_prior_beta = beta
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_pcn_joint_flow_prior(beta)
        print(
            "PCN_JOINT_FLOW_PRIOR_ATTESTATION beta={} "
            "each_complete_proposal=original_standard_gaussian "
            "sample0=independent_policy_baseline gradients=False "
            "projection=False trajectory_modified=False transport=flow_matching".format(
                beta
            )
        )

    def set_pcn_joint_flow_prior_schedule(self, start_step=0, end_step=None):
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("invalid pCN flow-prior schedule")
        if not hasattr(self, "_pcn_joint_flow_prior_beta"):
            raise RuntimeError("set pCN flow-prior beta before schedule")
        self._pcn_joint_flow_prior_schedule = {
            "start_step": start_step,
            "end_step": end_step,
        }
        print(
            "PCN_JOINT_FLOW_PRIOR_SCHEDULE_ATTESTATION start_step={} "
            "end_step={}".format(start_step, end_step)
        )

    def clear_pcn_joint_flow_prior(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_pcn_joint_flow_prior()
        self._pcn_joint_flow_prior_schedule = None

    def set_persistent_copula_flow_prior(
        self, persistence, execute_steps, actor_pairs=(), copula_rho=0.0,
        action_dims=None, proposal_start_index=1, temporal_rho=0.0,
        temporal_action_dims=None,
    ):
        pairs = [
            (int(receiver), int(anchor))
            for receiver, anchor in actor_pairs
        ]
        dims = (
            None if action_dims is None else [int(dim) for dim in action_dims]
        )
        temporal_dims = (
            None
            if temporal_action_dims is None
            else [int(dim) for dim in temporal_action_dims]
        )
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.set_persistent_copula_flow_prior(
                persistence=float(persistence),
                execute_steps=int(execute_steps),
                actor_pairs=pairs,
                copula_rho=float(copula_rho),
                action_dims=dims,
                proposal_start_index=int(proposal_start_index),
                temporal_rho=float(temporal_rho),
                temporal_action_dims=temporal_dims,
            )
        print(
            "PERSISTENT_COPULA_FLOW_PRIOR_ATTESTATION persistence={} "
            "execute_steps={} pairs={} rho={} action_dims={} "
            "temporal_rho={} temporal_action_dims={} "
            "sample0=independent_policy_baseline proposal_marginal_process="
            "stationary_standard_gaussian gradients=False projection=False "
            "trajectory_modified=False transport=flow_matching".format(
                float(persistence), int(execute_steps), pairs,
                float(copula_rho), dims, float(temporal_rho), temporal_dims,
            )
        )

    def clear_persistent_copula_flow_prior(self):
        policies = [self.nets["policy"]]
        if self.use_ema:
            policies.append(self.ema_policy)
        for policy in policies:
            policy.clear_persistent_copula_flow_prior()

    def set_semantic_reaction_delay(
        self, receiver_idx, delay_steps, start_step=0, end_step=None,
        accel_bias_mps2=0.0,
    ):
        """Configure a finite driver-response delay without geometric guidance."""
        receiver_idx = int(receiver_idx)
        delay_steps = int(delay_steps)
        start_step = int(start_step)
        end_step = None if end_step is None else int(end_step)
        accel_bias_mps2 = float(accel_bias_mps2)
        if receiver_idx < 0 or delay_steps <= 0 or start_step < 0:
            raise ValueError("invalid semantic reaction-delay configuration")
        if end_step is not None and end_step <= start_step:
            raise ValueError("reaction-delay end must follow its start")
        if not np.isfinite(accel_bias_mps2):
            raise ValueError("reaction-delay acceleration bias must be finite")
        self._semantic_reaction_delay = {
            "receiver_idx": receiver_idx,
            "delay_steps": delay_steps,
            "start_step": start_step,
            "end_step": end_step,
            "accel_bias_mps2": accel_bias_mps2,
        }
        self._semantic_reaction_delay_state = None
        print(
            "REACTION_DELAY_ATTESTATION receiver={} delay_steps={} start_step={} "
            "end_step={} accel_bias_mps2={} gradient_guidance=False collision_target=False "
            "projection=False transport=flow_matching".format(
                receiver_idx, delay_steps, start_step, end_step, accel_bias_mps2
            )
        )

    def clear_semantic_reaction_delay(self):
        self._semantic_reaction_delay = None
        self._semantic_reaction_delay_state = None


    def set_constraints(self, constraint_config):
        '''
        Resets the test-time hard constraints to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_constraints(constraint_config)

    def set_guidance_optimization_params(self, guidance_optimization_params):
        '''
        Resets the test-time guidance_optimization_params.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance_optimization_params(guidance_optimization_params)
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)
    
    def set_diffusion_specific_params(self, diffusion_specific_params):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_diffusion_specific_params(diffusion_specific_params)
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)
