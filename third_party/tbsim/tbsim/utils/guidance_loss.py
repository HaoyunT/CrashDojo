#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import time

import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F

import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.metrics import batch_detect_off_road

from tbsim.utils.geometry_utils import (
    transform_agents_to_world,
)
from tbsim.utils.trajdata_utils import get_current_lane_projection, get_left_lane_projection, get_right_lane_projection, select_agent_ind, transform_coord_agents_to_world, transform_coord_world_to_agent_i

from torch.autograd import Variable
import tbsim.utils.tensor_utils as TensorUtils

### utils for choosing from samples ####

def choose_action_from_guidance(preds, obs_dict, guide_configs, guide_losses):
    '''
    preds: dict of predictions from model, preds["positions"] (M, N, T, 2) or (B, N, M, T, 2)
    '''
    if len(preds["positions"].shape) == 4:
        # for agent-centric model the batch dimension is always 1
        B = 1
        M, N, *_ = preds["positions"].shape
    else:
        B, N, M, *_ = preds["positions"].shape
    BM = B*M
    # arbitrarily use the first sample as the action if no guidance given
    act_idx = torch.zeros((BM), dtype=torch.long, device=preds["positions"].device)
    # choose sample closest to desired guidance
    accum_guide_loss = torch.stack([v for k,v in guide_losses.items()], dim=2)
    # each scene separately since may contain different guidance
    scount = 0
    for sidx in range(len(guide_configs)):
        scene_guide_cfg = guide_configs[sidx]
        ends = scount + len(scene_guide_cfg)
        # (BM, N, num_of_guidance)
        scene_guide_loss = accum_guide_loss[..., scount:ends]
        scount = ends
        # scene_mask = ~torch.isnan(torch.sum(scene_guide_loss, dim=[1,2]))
        # scene_guide_loss = scene_guide_loss[scene_mask].cpu()
        # (BM, N, num_of_guidance) -> (BM, N)
        scene_guide_loss = torch.nansum(scene_guide_loss, dim=-1)
        is_scene_level = np.array([guide_cfg.name in ['agent_collision', 'social_group', 'gptcollision', 'gptkeepdistance', 'safesim_target_pair'] for guide_cfg in scene_guide_cfg])
        if np.sum(is_scene_level) > 0: 
            # choose which sample minimizes at the scene level (where each sample is a "scene")
            # (1)
            scene_act_idx = torch.argmin(torch.sum(scene_guide_loss, dim=0))
            # (BM,N) -> (B,M,N) -> (B,N) -> (B)
            scene_act_idx = torch.argmin(scene_guide_loss.reshape(B, M, N).sum(dim=1), dim=1)
            scene_act_idx = scene_act_idx.unsqueeze(-1).expand(B,M).view(BM)
        else:
            # each agent can choose the sample that minimizes guidance loss independently
            # (BM)
            scene_act_idx = torch.argmin(scene_guide_loss, dim=-1)

        # act_idx[scene_mask] = scene_act_idx.to(act_idx.device)
        act_idx = scene_act_idx.to(act_idx.device)

    return act_idx

def choose_action_from_gt(preds, obs_dict):
    '''
    preds: dict of predictions from model, preds["positions"] (M, N, T, 2) or (B, N, M, T, 2)
    '''
    if len(preds["positions"].shape) == 4:
        # for agent-centric model the batch dimension is always 1
        B = 1
        M, N, T, _ = preds["positions"].shape
    else:
        B, N, M, T, _ = preds["positions"].shape
    BM = B*M

    # arbitrarily use the first sample as the action if no gt given
    act_idx = torch.zeros((BM), dtype=torch.long, device=preds["positions"].device)
    if "target_positions" in obs_dict:
        print("DIFFUSER: WARNING using sample closest to GT from diffusion model!")
        # use the sample closest to GT
        # pred and gt may not be the same if gt is missing data at the end
        endT = min(T, obs_dict["target_positions"].size(1))
        pred_pos = preds["positions"][:,:, :, :endT]
        gt_pos = obs_dict["target_positions"][: :, :,:endT].unsqueeze(1)

        gt_valid = obs_dict["target_availabilities"][...,:endT].unsqueeze(1).expand((1, N, BM, endT)) # expand((BM, N, endT))
        err = torch.norm(pred_pos - gt_pos, dim=-1)
        err[~gt_valid] = torch.nan # so doesn't affect
        ade = torch.nanmean(err, dim=-1) # BM x N
        # print('ADE:', err.shape, ade.shape)
        res_valid = torch.sum(torch.isnan(ade), dim=-2) == 0
        if torch.sum(res_valid) > 0:
            min_ade_idx = torch.argmin(ade, dim=-1)
            print('res : ', res_valid.shape, act_idx, min_ade_idx.shape)
            act_idx[res_valid[0]] = min_ade_idx[0, res_valid[0]]
    else:
        print('Could not choose sample based on GT, as no GT in data')

    return act_idx


############## GUIDANCE config ########################

class GuidanceConfig(object):
    def __init__(self, name, weight, params, agents, func=None, partition=None):
        '''
        - name : name of the guidance function (i.e. the type of guidance), must be in GUIDANCE_FUNC_MAP
        - weight : alpha weight, how much affects denoising
        - params : guidance loss specific parameters
        - agents : agent indices within the scene to apply this guidance to. Applies to ALL if is None.
        - func : the function to call to evaluate this guidance loss value.
        '''
        assert name in GUIDANCE_FUNC_MAP, 'Guidance name must be one of: ' + ', '.join(map(str, GUIDANCE_FUNC_MAP.keys()))
        self.name = name
        self.weight = weight
        self.params = params
        self.agents = agents
        self.func = func
        if partition not in (None, "attack", "feasibility"):
            raise ValueError(
                "guidance partition must be attack, feasibility, or None"
            )
        self.partition = partition

    @staticmethod
    def from_dict(config_dict):
        required = {'name', 'weight', 'params', 'agents'}
        allowed = required | {'partition'}
        assert required.issubset(config_dict.keys()) and set(config_dict.keys()).issubset(allowed), \
                'Guidance config must include [name, weight, params, agents] and optional partition'
        return GuidanceConfig(**config_dict)

    def __repr__(self):
        return '<\n%s\n>' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

def verify_guidance_config_list(guidance_config_list):
    '''
    Returns true if there list contains some valid guidance that needs to be applied.
    Does not check to make sure each guidance dict is structured properly, only that
    the list structure is valid.
    '''
    assert len(guidance_config_list) > 0
    valid_guidance = False
    for guide in guidance_config_list:
        valid_guidance = valid_guidance or len(guide) > 0
    return valid_guidance

def verify_constraint_config(constraint_config_list):
    '''
    Given a hard constraint config dict, verifies it's structured as expected.
    Should contain fields 'agents', 'loc', and 'times'
    '''
    for constraint_config in constraint_config_list:
        if constraint_config is not None and len(constraint_config.keys()) > 0:
            assert constraint_config.keys() == {'agents', 'locs', 'times'}, \
                        'Constraint config must include only [agents, locs, times].'
            num_constraints = len(constraint_config['agents'])
            assert num_constraints == len(constraint_config['locs']), \
                        'all config fields should be same length'
            assert num_constraints == len(constraint_config['times']), \
                        'all config fields should be same length'
            if num_constraints > 0:
                assert len(constraint_config['locs'][0]) == 2, \
                    'Constraint locations must be 2d (x,y) waypoints'


############## GUIDANCE functions ########################

def apply_constraints(x, batch_scene_idx, cfg):
    '''
    Applies hard constraints to positions (x,y) specified by the given configuration.
    - x : trajectory to update with constraints. (B, N, T, D) where N is num samples and B is num agents
    - batch_scene_idx : (B,) boolean saying which scene each agent belongs to
    - cfg : list of dicts, which agents and times to apply constraints in each scene
    '''
    all_scene_inds = torch.unique_consecutive(batch_scene_idx).cpu().numpy()
    assert len(cfg) == len(all_scene_inds), "Must give the same num of configs as there are scenes in each batch"
    for i, si in enumerate(all_scene_inds):
        cur_cfg = cfg[i]
        if cur_cfg is not None and len(cur_cfg.keys()) > 0:
            cur_scene_inds = torch.nonzero(batch_scene_idx == si, as_tuple=True)[0]
            loc_tgt = torch.tensor(cur_cfg['locs']).to(x.device)
            x[cur_scene_inds[cur_cfg['agents']], :, cur_cfg['times'], :2] = loc_tgt.unsqueeze(1)
    return x

class GuidanceLoss(nn.Module):
    '''
    Abstract guidance function. This is a loss (not a reward), i.e. guidance will seek to
    MINIMIZE the implemented function.
    '''
    def __init__(self):
        super().__init__()
        self.global_t = 0

    def init_for_batch(self, example_batch):
        '''
        Initializes this loss to be used repeatedly only for the given scenes/agents in the example_batch.
        e.g. this function could use the extents of agents or num agents in each scene to cache information
              that is used while evaluating the loss
        '''
        pass

    def update(self, global_t=None):
        '''
        Update any persistant state needed by guidance loss functions.
        - global_t : the current global timestep of rollout
        '''
        if global_t is not None:
            self.global_t = global_t


    def forward(self, x, data_batch, agt_mask=None):
        '''
        Computes and returns loss value.

        Inputs:
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        - agt_mask : size B boolean list specifying which agents to apply guidance to. Applies to ALL agents if is None.

        Output:
        - loss : (B, N) loss for each sample of each batch index. Final loss will be mean of this.
        '''
        raise NotImplementedError('Must implement guidance function evaluation')

class TargetSpeedLoss(GuidanceLoss):
    '''
    Agent should follow specific target speed.
    '''
    def __init__(self, dt, target_speed, fut_valid):
        super().__init__()
        self.target_speed = target_speed
        self.fut_valid = fut_valid
        self.dt = dt

    def forward(self, x, data_batch, agt_mask=None):
        T = x.shape[2]
        cur_tgt_speed = self.target_speed[..., self.global_t:self.global_t+T]
        fut_valid = self.fut_valid[..., self.global_t:self.global_t+T]
        
        cur_tgt_speed = torch.tensor(cur_tgt_speed, dtype=torch.float32).to(x.device)
        fut_valid = torch.tensor(fut_valid).to(x.device)

        if agt_mask is not None:
            x = x[agt_mask]
            cur_tgt_speed = cur_tgt_speed[agt_mask]
            fut_valid = fut_valid[agt_mask]

        cur_speed = x[..., 2]
        
        valid_T = cur_tgt_speed.shape[-1]
        if valid_T > 0:
            speed_dev = torch.abs(cur_speed[..., :valid_T] - cur_tgt_speed[:, None, :])
            speed_dev = torch.nan_to_num(speed_dev, nan=0)
            loss = torch.mean(speed_dev, dim=-1)
        else:
            # dummy loss
            loss = torch.mean(x, dim=[-1, -2]) * 0.
        # print('loss.shape', loss.shape)
        # print('x.shape', x.shape)
        return loss

class AgentCollisionLoss(GuidanceLoss):
    '''
    Agents should not collide with each other.
    NOTE: this assumes full control over the scene. 
    '''
    def __init__(self, num_disks=5, buffer_dist=0.2, decay_rate=0.9, guide_moving_speed_th=5e-1, excluded_agents=None):
        '''
        - num_disks : the number of disks to use to approximate the agent for collision detection.
                        more disks improves accuracy
        - buffer_dist : additional space to leave between agents
        - decay_rate : how much to decay the loss as time goes on
        - excluded_agents : the collisions among these agents will not be penalized
        '''
        super().__init__()
        self.num_disks = num_disks
        self.buffer_dist = buffer_dist
        self.decay_rate = decay_rate
        self.guide_moving_speed_th = guide_moving_speed_th

        self.centroids = None
        self.penalty_dists = None
        self.scene_mask = None
        self.excluded_agents = excluded_agents

    def init_for_batch(self, example_batch):
        '''
        Caches disks and masking ahead of time.
        '''
        # return 
        # pre-compute disks to approximate each agent
        data_extent = example_batch["extent"]
        self.centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
        B = self.centroids.size(0)
        # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
        self.penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        
        # pre-compute masking for vectorized pairwise distance computation
        self.scene_mask = self.init_mask(example_batch['scene_index'], self.centroids.device)

    def init_disks(self, num_disks, extents):
        NA = extents.size(0)
        agt_rad = extents[:, 1] / 2. # assumes lenght > width
        cent_min = -(extents[:, 0] / 2.) + agt_rad
        cent_max = (extents[:, 0] / 2.) - agt_rad
        # sample disk centroids along x axis
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_disks) \
                                for vidx in range(NA)], dim=0).to(extents.device)
        centroids = torch.stack([cent_x, torch.zeros_like(cent_x)], dim=2)      
        return centroids, agt_rad

    # TODO why are results when using num_scenes_per_batch > 1 different than = 1?
    def init_mask(self, batch_scene_index, device):
        _, data_scene_index = torch.unique_consecutive(batch_scene_index, return_inverse=True)
        scene_block_list = []
        scene_inds = torch.unique_consecutive(data_scene_index)
        for scene_idx in scene_inds:
            cur_scene_mask = data_scene_index == scene_idx
            num_agt_in_scene = torch.sum(cur_scene_mask)
            cur_scene_block = ~torch.eye(num_agt_in_scene, dtype=torch.bool)
            scene_block_list.append(cur_scene_block)
        scene_mask = torch.block_diag(*scene_block_list).to(device)
        return scene_mask

    def forward(self, x, data_batch, agt_mask=None):
        data_extent = data_batch["extent"]
        data_world_from_agent = data_batch["world_from_agent"]
        curr_speed = data_batch['curr_speed']
        scene_index = data_batch['scene_index']

        # consider collision gradients only for those moving vehicles
        moving = torch.abs(curr_speed) > self.guide_moving_speed_th
        stationary = ~moving
        stationary = stationary.view(-1, 1, 1, 1).expand_as(x)
        x[stationary] = x[stationary].detach()

        pos_pred = x[..., :2]
        yaw_pred = x[..., 3:4]

        pos_pred_global, yaw_pred_global = transform_agents_to_world(pos_pred, yaw_pred, data_world_from_agent)
        if agt_mask is not None:
            # only want gradient to backprop to agents being guided
            pos_pred_detach = pos_pred_global.detach().clone()
            yaw_pred_detach = yaw_pred_global.detach().clone()

            pos_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(pos_pred_global),
                                          pos_pred_global,
                                          pos_pred_detach)
            yaw_pred_global = torch.where(agt_mask[:,None,None,None].expand_as(yaw_pred_global),
                                          yaw_pred_global,
                                          yaw_pred_detach)

        # create disks and transform to world frame (centroids)
        B, N, T, _ = pos_pred_global.size()
        if self.centroids is None or self.penalty_dists is None:
            centroids, agt_rad = self.init_disks(self.num_disks, data_extent) # B x num_disks x 2
            # minimum distance that two vehicle circle centers can be apart without collision (+ buffer)
            penalty_dists = agt_rad.view(B, 1).expand(B, B) + agt_rad.view(1, B).expand(B, B) + self.buffer_dist
        else:
            centroids, penalty_dists = self.centroids, self.penalty_dists
        centroids = centroids[:,None,None].expand(B, N, T, self.num_disks, 2)
        # to world
        s = torch.sin(yaw_pred_global).unsqueeze(-1)
        c = torch.cos(yaw_pred_global).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
        centroids = torch.matmul(centroids, rotM) + pos_pred_global.unsqueeze(-2)

        # NOTE: debug viz sanity check
        # import matplotlib
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # for ni in range(centroids.size(0)):
        #     plt.plot(centroids[ni,0,:,2,0].detach().cpu().numpy(),
        #              centroids[ni,0,:,2,1].detach().cpu().numpy(),
        #              '-')
        # plt.gca().set_xlim([15, -15])
        # plt.gca().set_ylim([-15, 15])
        # plt.show()
        # plt.close(fig)

        # NOTE: assume each sample is a different scene for the sake of computing collisions
        if self.scene_mask is None:
            scene_mask = self.init_mask(scene_index, centroids.device)
        else:
            scene_mask = self.scene_mask

        # TODO technically we do not need all BxB comparisons
        #       only need the lower triangle of this matrix (no self collisions and only one way distance)
        #       but this may be slower to assemble than masking

        # TODO B could contain multiple scenes, could just pad each scene to the max_agents and compare MaxA x MaxA to avoid unneeded comparisons across scenes

        centroids = centroids.transpose(0,2) # T x NS x B x D x 2
        centroids = centroids.reshape((T*N, B, self.num_disks, 2))
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = centroids.view(T*N, B, 1, self.num_disks, 2).expand(T*N, B, B, self.num_disks, 2).reshape(T*N*B*B, self.num_disks, 2)
        cur_cent2 = centroids.view(T*N, 1, B, self.num_disks, 2).expand(T*N, B, B, self.num_disks, 2).reshape(T*N*B*B, self.num_disks, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*N*B*B, self.num_disks*self.num_disks)
        # get minimum distance over all circle pairs between each pair of agents
        pair_dists = torch.min(pair_dists, 1)[0].view(T*N, B, B)

        penalty_dists = penalty_dists.view(1, B, B)
        is_colliding_mask = torch.logical_and(pair_dists <= penalty_dists,
                                              scene_mask.view(1, B, B))
        
        # self.excluded_agents = torch.tensor([i for i in range(5, N)], dtype=torch.long) # TODO: debug
        if self.excluded_agents is not None:
            # for all row and column pairs that are both in the excluded agents list, set to 0
            excluded_agents_mask = torch.ones((1, B, B), device=is_colliding_mask.device)
            excluded_agents_tensor = torch.tensor(self.excluded_agents, device=is_colliding_mask.device)
            i_indices, j_indices = torch.meshgrid(excluded_agents_tensor, excluded_agents_tensor, indexing='ij')
            excluded_agents_mask[0, i_indices, j_indices] = 0    

            is_colliding_mask = torch.logical_and(is_colliding_mask, excluded_agents_mask)
        
        # # consider collision only for those involving at least one vehicle moving
        # moving = torch.abs(data_batch['curr_speed']) > self.guide_moving_speed_th
        # moving1 = moving.view(1, B, 1).expand(1, B, B)
        # moving2 = moving.view(1, 1, B).expand(1, B, B)
        # moving_mask = torch.logical_or(moving1, moving2) 
        # is_colliding_mask = torch.logical_and(is_colliding_mask,
        #                                       moving_mask)

        # penalty is inverse normalized distance apart for those already colliding
        cur_penalties = 1.0 - (pair_dists / penalty_dists)
        # only compute loss where it's valid and colliding
        cur_penalties = torch.where(is_colliding_mask,
                                    cur_penalties,
                                    torch.zeros_like(cur_penalties))
                                        
        # summing over timesteps and all other agents to get B x N
        cur_penalties = cur_penalties.reshape((T, N, B, B))
        # cur_penalties = cur_penalties.sum(0).sum(-1).transpose(0, 1)
        # penalize early steps more than later steps
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=cur_penalties.device)
        exp_weights /= exp_weights.sum()
        cur_penalties = cur_penalties * exp_weights[:, None, None, None]
        cur_penalties = cur_penalties.sum(0).mean(-1).transpose(0, 1)

        # consider loss only for those agents that are moving (note: since the loss involves interaction those stationary vehicles will still be indirectly penalized from the loss of other moving vehicles)
        cur_penalties = torch.where(moving.unsqueeze(-1).expand(B, N), cur_penalties, torch.zeros_like(cur_penalties))

        # print(cur_penalties)
        if agt_mask is not None:
            return cur_penalties[agt_mask]
        else:
            return cur_penalties


class SafeSimTargetPairLoss(GuidanceLoss):
    """Target one adversary at a frozen ego plan and penalize collateral hits.

    This is a scene-centric port of the two public SAFE-SIM semantics used for
    conflict generation: ``causecollision`` for one ego/adversary pair and
    ``collision(adv_mode=True)`` for every other pair.  It is deliberately a
    strong geometric baseline, not a claim that the result stays on the FM
    manifold.
    """

    def __init__(
        self,
        target_ind,
        ref_ind,
        loss_timesteps=20,
        sigma=1.0,
        heading_weight=1.0,
        buff_dist=1.5,
        cause_weight=1.0,
        collateral_weight=1.0,
        desired_clearance=0.0,
        severity_mode="collision",
        softmin_temperature=0.5,
        collateral_margin=1.0,
        collateral_barrier_weight=5.0,
        collision_type=None,
        conflict_step=None,
        conflict_point_world=None,
        arrival_weight=0.0,
        typed_heading_weight=0.0,
        marginal_weight=0.0,
        dt=0.1,
        max_speed_mps=25.0,
        max_abs_acc_mps2=4.0,
        max_abs_jerk_mps3=8.0,
        max_abs_yaw_rate_rps=1.0,
    ):
        super().__init__()
        if severity_mode not in {"near_miss", "emergency", "collision"}:
            raise ValueError("unknown severity_mode {}".format(severity_mode))
        self.target_ind = int(target_ind)
        self.ref_ind = int(ref_ind)
        self.loss_timesteps = int(loss_timesteps)
        self.sigma = float(sigma)
        self.heading_weight = float(heading_weight)
        self.buff_dist = float(buff_dist)
        self.cause_weight = float(cause_weight)
        self.collateral_weight = float(collateral_weight)
        self.desired_clearance = float(desired_clearance)
        self.severity_mode = severity_mode
        self.softmin_temperature = float(softmin_temperature)
        self.collateral_margin = float(collateral_margin)
        self.collateral_barrier_weight = float(collateral_barrier_weight)
        if collision_type not in {None, "rear_end", "side", "cut_in", "head_on"}:
            raise ValueError("unknown collision_type {}".format(collision_type))
        self.collision_type = collision_type
        self.conflict_step = (
            None if conflict_step is None else int(conflict_step)
        )
        self.conflict_point_world = conflict_point_world
        self.arrival_weight = float(arrival_weight)
        self.typed_heading_weight = float(typed_heading_weight)
        self.marginal_weight = float(marginal_weight)
        self.dt = float(dt)
        self.max_speed_mps = float(max_speed_mps)
        self.max_abs_acc_mps2 = float(max_abs_acc_mps2)
        self.max_abs_jerk_mps3 = float(max_abs_jerk_mps3)
        self.max_abs_yaw_rate_rps = float(max_abs_yaw_rate_rps)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self._printed_reference_attestation = False

    @staticmethod
    def _scene_agents(scene_index, selected, target_ind, ref_ind):
        if selected.numel() == 0:
            raise ValueError("SAFE-SIM guidance received an empty agent mask")
        scene_value = scene_index[selected[0]]
        agents = torch.nonzero(scene_index == scene_value, as_tuple=True)[0]
        if target_ind >= agents.numel() or ref_ind >= agents.numel():
            raise IndexError("SAFE-SIM target/reference index exceeds scene size")
        return agents[target_ind], agents[ref_ind], agents

    def _target_cost(self, target_pos, ref_pos, target_extent, ref_extent):
        clearance = (
            torch.linalg.norm(target_pos - ref_pos, dim=-1)
            - 0.5 * (target_extent[1] + ref_extent[1])
        )
        tau = self.softmin_temperature
        if self.conflict_step is None:
            soft_min = -tau * torch.logsumexp(-clearance / tau, dim=-1)
        else:
            step = min(max(self.conflict_step, 0), clearance.shape[-1] - 1)
            soft_min = clearance[..., step]
        if self.severity_mode == "collision":
            return F.softplus((soft_min - self.desired_clearance) / tau) * tau
        return (soft_min - self.desired_clearance) ** 2

    def _arrival_cost(self, target_pos):
        if self.conflict_step is None or self.conflict_point_world is None:
            return torch.zeros(
                target_pos.shape[0], device=target_pos.device,
                dtype=target_pos.dtype,
            )
        step = min(max(self.conflict_step, 0), target_pos.shape[1] - 1)
        point = torch.as_tensor(
            self.conflict_point_world,
            device=target_pos.device,
            dtype=target_pos.dtype,
        )
        if point.shape != (2,):
            raise ValueError("conflict_point_world must contain exactly x,y")
        distance = torch.linalg.norm(target_pos - point[None, None], dim=-1)
        event_distance = distance[:, step]
        if distance.shape[1] <= 1:
            return event_distance
        before = distance[:, :step]
        after = distance[:, step + 1:]
        alternatives = torch.cat((before, after), dim=-1)
        tau = self.softmin_temperature
        alternative_softmin = -tau * torch.logsumexp(
            -alternatives / tau, dim=-1
        )
        phase_penalty = F.softplus(
            (event_distance - alternative_softmin) / tau
        ) * tau
        return event_distance + phase_penalty

    def _typed_heading_cost(self, target_yaw, ref_yaw):
        if self.conflict_step is None or self.collision_type is None:
            return torch.zeros(
                target_yaw.shape[0], device=target_yaw.device,
                dtype=target_yaw.dtype,
            )
        step = min(max(self.conflict_step, 0), target_yaw.shape[1] - 1)
        relative = target_yaw[:, step] - ref_yaw[:, step]
        if self.collision_type == "side":
            return torch.cos(relative) ** 2
        if self.collision_type == "head_on":
            return 1.0 + torch.cos(relative)
        if self.collision_type == "rear_end":
            return 1.0 - torch.cos(relative)
        wrapped = torch.remainder(relative + torch.pi, 2.0 * torch.pi) - torch.pi
        return (torch.abs(wrapped) - torch.pi / 6.0) ** 2

    def _marginal_cost(self, target_state):
        speed = target_state[..., 2]
        acceleration = target_state[..., 4]
        yaw_rate = target_state[..., 5]
        jerk = (acceleration[:, 1:] - acceleration[:, :-1]) / self.dt
        speed_cost = (
            F.relu(-speed) ** 2
            + F.relu(speed - self.max_speed_mps) ** 2
        ).mean(dim=-1) / max(self.max_speed_mps ** 2, 1e-6)
        acc_cost = F.relu(
            torch.abs(acceleration) - self.max_abs_acc_mps2
        ).pow(2).mean(dim=-1) / max(self.max_abs_acc_mps2 ** 2, 1e-6)
        yaw_cost = F.relu(
            torch.abs(yaw_rate) - self.max_abs_yaw_rate_rps
        ).pow(2).mean(dim=-1) / max(self.max_abs_yaw_rate_rps ** 2, 1e-6)
        if jerk.shape[-1] == 0:
            jerk_cost = torch.zeros_like(speed_cost)
        else:
            jerk_cost = F.relu(
                torch.abs(jerk) - self.max_abs_jerk_mps3
            ).pow(2).mean(dim=-1) / max(self.max_abs_jerk_mps3 ** 2, 1e-6)
        return speed_cost + acc_cost + jerk_cost + yaw_cost

    def _collateral_cost(self, pos, yaw, extent, target, reference, agents):
        target_pos = pos[target]
        target_yaw = yaw[target, ..., 0]
        target_extent = extent[target]
        total = torch.zeros(
            target_pos.shape[:2], device=target_pos.device, dtype=target_pos.dtype
        )
        long_vec = torch.stack(
            (torch.cos(target_yaw), torch.sin(target_yaw)), dim=-1
        )
        lat_vec = torch.stack(
            (-torch.sin(target_yaw), torch.cos(target_yaw)), dim=-1
        )
        for other in agents:
            if int(other) in {int(target), int(reference)}:
                continue
            delta = pos[other] - target_pos
            long_clear = (
                torch.abs(torch.sum(delta * long_vec, dim=-1))
                - 0.5 * (target_extent[0] + extent[other][0])
            )
            lat_clear = (
                torch.abs(torch.sum(delta * lat_vec, dim=-1))
                - 0.5 * (target_extent[1] + extent[other][1])
            )
            gaussian = torch.exp(
                -0.5 * (
                    (long_clear / (self.sigma * self.heading_weight)) ** 2
                    + (lat_clear / self.sigma) ** 2
                    - self.buff_dist ** 2
                )
            )
            surface = (
                torch.linalg.norm(delta, dim=-1)
                - 0.5 * (target_extent[1] + extent[other][1])
            )
            barrier = F.softplus(
                (self.collateral_margin - surface) / self.sigma
            )
            total = total + gaussian + self.collateral_barrier_weight * barrier
        return total.mean(dim=-1)

    def forward(self, x, data_batch, agt_mask=None):
        if agt_mask is None:
            raise ValueError("SAFE-SIM guidance requires one controlled adversary")
        selected = torch.nonzero(agt_mask, as_tuple=True)[0]
        target, reference, agents = self._scene_agents(
            data_batch["scene_index"], selected, self.target_ind, self.ref_ind
        )
        if not bool(agt_mask[target]):
            raise ValueError("configured target must be the controlled agent")

        pos, yaw = transform_agents_to_world(
            x[..., :2], x[..., 3:4], data_batch["world_from_agent"]
        )
        pos = torch.where(agt_mask[:, None, None, None], pos, pos.detach())
        yaw = torch.where(agt_mask[:, None, None, None], yaw, yaw.detach())
        horizon = min(pos.shape[2], self.loss_timesteps)
        pos, yaw = pos[:, :, :horizon], yaw[:, :, :horizon]
        ref_pos = pos[reference]

        planner_pos = data_batch.get("planner_reference_positions")
        planner_mask = data_batch.get("planner_reference_mask")
        if planner_pos is not None and planner_mask is not None and bool(planner_mask[reference]):
            local = planner_pos[reference, :horizon]
            homogeneous = torch.cat((local, torch.ones_like(local[..., :1])), -1)
            transform = data_batch["world_from_agent"][reference]
            world = torch.matmul(homogeneous, transform.transpose(-1, -2))[..., :2]
            ref_pos = world.unsqueeze(0).expand(x.shape[1], -1, -1).detach()
            if not self._printed_reference_attestation:
                print(
                    "SAFESIM_GUIDANCE_ATTESTATION planner_reference_used=True "
                    "target_global={} ref_global={}".format(int(target), int(reference))
                )
                self._printed_reference_attestation = True

        target_cost = self._target_cost(
            pos[target], ref_pos, data_batch["extent"][target],
            data_batch["extent"][reference]
        )
        arrival = self._arrival_cost(pos[target])
        typed_heading = self._typed_heading_cost(
            yaw[target, ..., 0], yaw[reference, ..., 0]
        )
        marginal = self._marginal_cost(x[target, :, :horizon])
        collateral = self._collateral_cost(
            pos, yaw, data_batch["extent"], target, reference, agents
        )
        output = torch.zeros(
            (selected.numel(), x.shape[1]), device=x.device, dtype=x.dtype
        )
        target_row = torch.nonzero(selected == target, as_tuple=True)[0]
        output[target_row] = (
            self.cause_weight * target_cost
            + self.collateral_weight * collateral
            + self.arrival_weight * arrival
            + self.typed_heading_weight * typed_heading
            + self.marginal_weight * marginal
        )
        return output


# TODO target waypoint guidance
#       - Really the target positions should be global not local, will have to do some extra work to transform into
#           the local frame.
class TargetPosAtTimeLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at a specific time step (within the current planning horizon).
    '''
    def __init__(self, target_pos, target_time):
        '''
        - target_pos : (B,2) batch of positions to hit, B must equal the number of agents after applying mask in forward.
        - target_time: (B,) batch of times at which to hit the given positions
        '''
        super().__init__()
        self.set_target(target_pos, target_time)

    def set_target(self, target_pos, target_time):
        if isinstance(target_pos, torch.Tensor):
            self.target_pos = target_pos
        else:
            self.target_pos = torch.tensor(target_pos)
        if isinstance(target_time, torch.Tensor):
            self.target_time = target_time
        else:
            self.target_time = torch.tensor(target_time)

    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        if agt_mask is not None:
            x = x[agt_mask]
        assert x.size(0) == self.target_pos.size(0)
        assert x.size(0) == self.target_time.size(0)
        
        x_pos = x[torch.arange(x.size(0)), :, self.target_time, :2]
        tgt_pos = self.target_pos.to(x_pos.device)[:,None] # (B,1,2)
        # MSE
        # loss = torch.sum((x_pos - tgt_pos)**2, dim=-1)
        loss = torch.norm(x_pos - tgt_pos, dim=-1)
        # # Normalization Change: clip to 1
        # loss = torch.clip(loss, max=1)
        return loss

class TargetPosLoss(GuidanceLoss):
    '''
    Hit a specific waypoint at some time step (within the current planning horizon).
    '''
    def __init__(self, target_pos, min_target_time=0.0):
        '''
        - target_pos : (B,2) batch of positions to hit, B must equal the number of agents after applying mask in forward.
        - min_target_time : float, only tries to hit the target after the initial min_target_time*horizon_num_steps of the trajectory
                            e.g. if = 0.5 then only the last half of the trajectory will attempt to go through target
        '''
        super().__init__()
        self.min_target_time = min_target_time
        self.set_target(target_pos)

    def set_target(self, target_pos):
        if isinstance(target_pos, torch.Tensor):
            self.target_pos = target_pos
        else:
            self.target_pos = torch.tensor(target_pos)


    def forward(self, x, data_batch, agt_mask=None):
        '''
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        '''
        if agt_mask is not None:
            x = x[agt_mask]
        assert x.size(0) == self.target_pos.size(0)
        
        min_t = int(self.min_target_time*x.size(2))
        x_pos = x[:,:,min_t:,:2]
        tgt_pos = self.target_pos.to(x_pos.device)[:,None,None] # (B,1,1,2)
        dist = torch.norm(x_pos - tgt_pos, dim=-1)
        # give higher loss weight to the closest valid timesteps
        loss_weighting = F.softmin(dist, dim=-1)
        loss = loss_weighting * torch.sum((x_pos - tgt_pos)**2, dim=-1) # (B, N, T)
        # loss = loss_weighting * torch.norm(x_pos - tgt_pos, dim=-1)
        loss = torch.mean(loss, dim=-1) # (B, N)
        # # Normalization Change: clip to 1
        # loss = torch.clip(loss, max=1)
        return loss

# TODO: this currently depends on the map that's also passed into the network.
#       if the network map viewport is small and the future horizon is long enough,
#       it may go outside the range of the map and then this is really inaccurate.
class MapCollisionLoss(GuidanceLoss):
    '''
    Agents should not go offroad.
    '''
    def __init__(self, num_points_lw=(10, 10), decay_rate=0.9, guide_moving_speed_th=5e-1):
        '''
        - num_points_lw : how many points will be sampled within each agent bounding box
                            to detect map collisions. e.g. (15, 10) will sample a 15 x 10 grid
                            of points where 15 is along the length and 10 along the width.
        '''
        super().__init__()
        self.num_points_lw = num_points_lw
        self.decay_rate = decay_rate
        self.guide_moving_speed_th = guide_moving_speed_th
        lwise = torch.linspace(-0.5, 0.5, self.num_points_lw[0])
        wwise = torch.linspace(-0.5, 0.5, self.num_points_lw[1])
        self.local_coords = torch.cartesian_prod(lwise, wwise)
        # TODO could cache initial (local) point samplings if given extents at instantiation

    def gen_agt_coords(self, pos, yaw, lw, raster_from_agent):
        '''
        - pos : B x 2
        - yaw : B x 1
        - lw : B x 2
        '''
        B = pos.size(0)
        cur_loc_coords = self.local_coords.to(pos.device).unsqueeze(0).expand((B, -1, -1))
        # scale by the extents
        cur_loc_coords = cur_loc_coords * lw.unsqueeze(-2)

        # transform initial coords to given pos, yaw
        s = torch.sin(yaw).unsqueeze(-1)
        c = torch.cos(yaw).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
        agt_coords_agent_frame = cur_loc_coords @ rotM + pos.unsqueeze(-2)
        
        # then transform to raster frame
        agt_coords_raster_frame = GeoUtils.transform_points_tensor(agt_coords_agent_frame, raster_from_agent)

        # # NOTE: debug viz sanity check
        # import matplotlib
        # import matplotlib.pyplot as plt
        # agt_coords = agt_coords.reshape((8, 10, 52, 25, 2))
        # fig = plt.figure()
        # for t in range(agt_coords.size(2)):
        #     plt.scatter(agt_coords[3,0,t,:,0].cpu().detach().numpy(),
        #                 agt_coords[3,0,t,:,1].cpu().detach().numpy())
        # # plt.gca().set_xlim([-1, 7])
        # # plt.gca().set_ylim([-4, 4])
        # plt.axis('equal')
        # plt.show()
        # plt.close(fig)

        return agt_coords_agent_frame, agt_coords_raster_frame

    def forward(self, x, data_batch, agt_mask=None):   
        drivable_map = data_batch["drivable_map"]
        data_extent = data_batch["extent"]
        data_raster_from_agent = data_batch["raster_from_agent"]
        moving = torch.abs(data_batch['curr_speed']) > self.guide_moving_speed_th

        if agt_mask is not None:
            x = x[agt_mask]
            drivable_map = drivable_map[agt_mask]
            data_extent = data_extent[agt_mask]
            data_raster_from_agent = data_raster_from_agent[agt_mask]
            moving = moving[agt_mask]

        _, H, W = drivable_map.size()

        B, N, T, _ = x.size()
        traj = x.reshape((-1, 6)) # B*N*T x 6
        pos_pred = traj[:,:2]
        yaw_pred = traj[:, 3:4] 
        lw = data_extent[:,None,None].expand((B, N, T, 3)).reshape((-1, 3))[:,:2]
        diag_len = torch.sqrt(torch.sum(lw*lw, dim=-1))
        data_raster_from_agent = data_raster_from_agent[:,None,None].expand((B, N, T, 3, 3)).reshape((-1, 3, 3))

        # sample points within each agent to check if drivable
        agt_samp_pts, agt_samp_pix = self.gen_agt_coords(pos_pred, yaw_pred, lw, data_raster_from_agent)
        # agt_samp_pts = agt_samp_pts.reshape((B, N, T, -1, 2))
        agt_samp_pix = agt_samp_pix.reshape((B, N, T, -1, 2)).long().detach() # only used to query drivable map, not to compute loss
        # NOTE: this projects pixels outside the map onto the edge
        agt_samp_l = torch.clamp(agt_samp_pix[..., 0:1], 0, W-1)
        agt_samp_w = torch.clamp(agt_samp_pix[..., 1:2], 0, H-1)
        agt_samp_pix = torch.cat([agt_samp_l, agt_samp_w], dim=-1)

        # query these points in the drivable area to determine collision
        _, P, _ = agt_samp_pts.size()
        map_coll_mask = torch.isclose(batch_detect_off_road(agt_samp_pix, drivable_map), torch.ones((1)).to(agt_samp_pix.device))
        map_coll_mask = map_coll_mask.reshape((-1, P))

        # only apply loss to timesteps that are partially overlapping
        per_step_coll = torch.sum(map_coll_mask, dim=-1)
        overlap_mask = ~torch.logical_or(per_step_coll == 0, per_step_coll == P)

        overlap_coll_mask = map_coll_mask[overlap_mask]
        overlap_agt_samp = agt_samp_pts[overlap_mask]
        overlap_diag_len = diag_len[overlap_mask]

        #
        # The idea here: for each point that is offroad, we want to compute
        #   the minimum distance to a point that is on the road to give a nice
        #   gradient to push it back.
        #

        # compute dist mat between all pairs of points at each step
        # NOTE: the detach here is a very subtle but IMPORTANT point
        #       since these sample points are a function of the pos/yaw, if we compute
        #       the distance between them the gradients will always be 0, no matter how
        #       we change the pos and yaw the distance will never change. But if we detach
        #       one and compute distance to these arbitrary points we've selected, then
        #       we get a useful gradient.
        #           Moreover, it's also importan the columns are the ones detached here!
        #       these correspond to the points that ARE colliding. So if we try to max
        #       distance b/w these and the points inside the agent, it will push the agent
        #       out of the offroad area. If we do it the other way it will pull the agent
        #       into the offroad (if we max the dist) or just be a small pull in the correct dir
        #       (if we min the dist).
        pt_samp_dist = torch.cdist(overlap_agt_samp, overlap_agt_samp.clone().detach())
        # get min dist just for points still on the road
        # so we mask out points off the road (this also removes diagonal for off-road points which excludes self distances)
        pt_samp_dist = torch.where(overlap_coll_mask.unsqueeze(-1).expand(-1, -1, P),
                                   torch.ones_like(pt_samp_dist)*np.inf,
                                   pt_samp_dist)
        pt_samp_min_dist_all = torch.amin(pt_samp_dist, dim=1) # previously masked rows, so compute min over cols
        # compute actual loss
        pt_samp_loss_all = 1.0 - (pt_samp_min_dist_all / overlap_diag_len.unsqueeze(1))
        # only want a loss for off-road points
        pt_samp_loss_offroad = torch.where(overlap_coll_mask,
                                               pt_samp_loss_all,
                                               torch.zeros_like(pt_samp_loss_all))

        overlap_coll_loss = torch.sum(pt_samp_loss_offroad, dim=-1)
        # expand back to all steps, other non-overlap steps will be zero
        all_coll_loss = torch.zeros((agt_samp_pts.size(0))).to(overlap_coll_loss.device)
        all_coll_loss[overlap_mask] = overlap_coll_loss
        
        # summing over timesteps
        # all_coll_loss = all_coll_loss.reshape((B, N, T)).sum(-1)

        # consider offroad only for those moving vehicles
        all_coll_loss = all_coll_loss.reshape((B, N, T))
        moving_mask = moving.view((B,1,1)).expand(B, N, T)
        all_coll_loss = torch.where(moving_mask,
                                    all_coll_loss,
                                    torch.zeros_like(all_coll_loss))

        # penalize early steps more than later steps
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=all_coll_loss.device)
        exp_weights /= exp_weights.sum()
        all_coll_loss = all_coll_loss * exp_weights[None, None, :]
        all_coll_loss = all_coll_loss.sum(-1)

        return all_coll_loss

#
# Global waypoint target losses
#   (i.e. at some future planning horizon)
#
def compute_progress_loss(pos_pred, tgt_pos, urgency,
                          tgt_time=None,
                          pref_speed=1.42,
                          dt=0.1,
                          min_progress_dist=0.5):
    '''
    Evaluate progress towards a goal that we want to hit.
    - pos_pred : (B x N x T x 2)
    - tgt_pos : (B x 2)
    - urgency : (B) in (0.0, 1.0]
    - tgt_time : [optional] (B) local target time, i.e. starting from the current t0 how many steps in the
                    future will we need to hit the target. If given, loss is computed to cover the distance
                    necessary to hit the goal at the given time
    - pref_speed: speed used to determine how much distance should be covered in a time interval
    - dt : step interval of the trajectories
    - min_progress_dist : float (in meters). if not using tgt_time, the minimum amount of progress that should be made in
                            each step no matter what the urgency is
    '''
    # TODO: use velocity or heading to avoid degenerate case of trying to whip around immediately
    #       and getting stuck from unicycle dynamics?

    # distance from final trajectory timestep to the goal position
    final_dist = torch.norm(pos_pred[:,:,-1] - tgt_pos[:,None], dim=-1)

    if tgt_time is not None:
        #
        # have a target time: distance covered is based on arrival time
        #
        # distance of straight path from current pos to goal at the average speed

        goal_dist = tgt_time * dt * pref_speed

        # factor in urgency (shortens goal_dist since we can't expect to always go on a straight path)
        goal_dist = goal_dist * (1.0 - urgency)
        # only apply loss if above the goal distance
        progress_loss = F.relu(final_dist - goal_dist[:,None])
    else:
        #
        # don't have a target time: distance covered based on making progress
        #       towards goal with the specified urgency
        #
        # following straight line path from current pos to goal
        max_horizon_dist = pos_pred.size(2) * dt * pref_speed
        # at max urgency, want to cover distance of this straight line path
        # at min urgency, just make minimum progress
        goal_dist = torch.maximum(urgency * max_horizon_dist, torch.tensor([min_progress_dist]).to(urgency.device))

        init_dist = torch.norm(pos_pred[:,:,0] - tgt_pos[:,None], dim=-1)
        progress_dist = init_dist - final_dist
        # only apply loss if less progress than goal
        progress_loss = F.relu(goal_dist[:,None] - progress_dist)

    return progress_loss

class ConditionalMarginalSupportLoss(GuidanceLoss):
    """Differentiable B4 barrier on matched-support marginal kinematics.

    The loss recomputes kinematics from positions/yaws instead of trusting
    proposal-side cached speed or acceleration channels.  It evaluates the
    preregistered logged-matched cadence (0.5 s by default), while executed
    0.1 s microdynamics remain a separate reporting contract.
    """

    FEATURE_NAMES = ("speed", "acceleration", "jerk", "yaw_rate")

    def __init__(
        self,
        empirical_upper,
        absolute_upper,
        base_dt=0.1,
        matched_stride=5,
    ):
        super().__init__()
        if float(base_dt) <= 0.0:
            raise ValueError("support base_dt must be positive")
        if int(matched_stride) <= 0:
            raise ValueError("support matched_stride must be positive")
        if set(empirical_upper.keys()) != set(self.FEATURE_NAMES):
            raise ValueError("empirical support requires four fixed features")
        if set(absolute_upper.keys()) != set(self.FEATURE_NAMES):
            raise ValueError("absolute support requires four fixed features")
        self.empirical_upper = {
            name: list(values) if isinstance(values, (list, tuple)) else [values]
            for name, values in empirical_upper.items()
        }
        self.absolute_upper = {
            name: float(absolute_upper[name]) for name in self.FEATURE_NAMES
        }
        if any(value <= 0.0 for value in self.absolute_upper.values()):
            raise ValueError("absolute support thresholds must be positive")
        self.base_dt = float(base_dt)
        self.matched_stride = int(matched_stride)

    @staticmethod
    def _wrap(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _threshold(self, name, full_batch, mask, value):
        threshold = torch.as_tensor(
            self.empirical_upper[name], dtype=value.dtype, device=value.device
        )
        if threshold.numel() == 1:
            threshold = threshold.expand(full_batch)
        elif threshold.numel() != full_batch:
            raise ValueError(
                "{} empirical threshold count {} does not match actor count {}"
                .format(name, threshold.numel(), full_batch)
            )
        if not bool(torch.isfinite(threshold).all()) or bool((threshold <= 0).any()):
            raise ValueError("{} empirical thresholds must be finite positive".format(name))
        if mask is not None:
            threshold = threshold[mask]
        absolute = torch.as_tensor(
            self.absolute_upper[name], dtype=value.dtype, device=value.device
        )
        return torch.minimum(threshold, absolute)[:, None]

    def forward(self, x, data_batch, agt_mask=None):
        del data_batch
        if x.ndim != 4 or x.shape[-1] < 4:
            raise ValueError("support loss expects [actors,samples,time,>=4]")
        full_batch = x.shape[0]
        if agt_mask is not None:
            if agt_mask.shape != (full_batch,):
                raise ValueError("support actor mask shape mismatch")
            x = x[agt_mask]
        sampled = x[:, :, ::self.matched_stride]
        if sampled.shape[2] < 4:
            raise ValueError("support loss needs four matched-cadence states")
        dt = self.base_dt * self.matched_stride
        velocity = torch.diff(sampled[..., :2], dim=2) / dt
        acceleration = torch.diff(velocity, dim=2) / dt
        jerk = torch.diff(acceleration, dim=2) / dt
        yaw_rate = self._wrap(torch.diff(sampled[..., 3], dim=2)) / dt
        features = {
            "speed": torch.linalg.vector_norm(velocity, dim=-1).max(dim=2).values,
            "acceleration": torch.linalg.vector_norm(
                acceleration, dim=-1
            ).max(dim=2).values,
            "jerk": torch.linalg.vector_norm(jerk, dim=-1).max(dim=2).values,
            "yaw_rate": yaw_rate.abs().max(dim=2).values,
        }
        output = x.new_zeros((x.shape[0], x.shape[1]))
        for name, value in features.items():
            limit = self._threshold(
                name, full_batch, agt_mask, value
            )
            excess = F.relu(value - limit) / torch.clamp(limit, min=1.0e-6)
            output = output + excess.square()
        return output


############## GUIDANCE utilities ########################

GUIDANCE_FUNC_MAP = {
    'target_speed' : TargetSpeedLoss,
    'agent_collision' : AgentCollisionLoss,
    'safesim_target_pair' : SafeSimTargetPairLoss,
    'conditional_marginal_support' : ConditionalMarginalSupportLoss,
    'map_collision' : MapCollisionLoss,
    'target_pos_at_time' : TargetPosAtTimeLoss,
    'target_pos' : TargetPosLoss,
}

class DiffuserGuidance(object):
    '''
    Handles initializing guidance functions and computing gradients at test-time.
    '''
    def __init__(self, guidance_config_list, example_batch=None):
        '''
        - example_obs [optional] - if this guidance will only be used on a single batch repeatedly,
                                    i.e. the same set of scenes/agents, an example data batch can
                                    be passed in a used to init some guidance making test-time more efficient.
        '''
        self.num_scenes = len(guidance_config_list)
        assert self.num_scenes > 0, "Guidance config list must include list of guidance for each scene"
        self.guide_configs = [[]]*self.num_scenes
        for si in range(self.num_scenes):
            if len(guidance_config_list[si]) > 0:
                self.guide_configs[si] = [GuidanceConfig.from_dict(cur_cfg) for cur_cfg in guidance_config_list[si]]
                # initialize each guidance function
                for guide_cfg in self.guide_configs[si]:
                    guide_cfg.func = GUIDANCE_FUNC_MAP[guide_cfg.name](**guide_cfg.params)
                    if example_batch is not None:
                        guide_cfg.func.init_for_batch(example_batch)
    
    def init_for_batch(self, example_batch):
        '''
        Initializes this loss to be used repeatedly only for the given scenes/agents in the example_batch.
        e.g. this function could use the extents of agents or num agents in each scene to cache information
              that is used while evaluating the loss
        '''
        pass
    
    def update(self, **kwargs):
        for si in range(self.num_scenes):
            cur_guide = self.guide_configs[si]
            if len(cur_guide) > 0:
                for guide_cfg in cur_guide:
                    guide_cfg.func.update(**kwargs)

    def compute_guidance_loss(self, x_loss, data_batch, return_loss_tot_traj=False):
        '''
        Evaluates all guidance losses and total and individual values.
        - x_loss: (B, N, T, 6) the trajectory to use to compute losses and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations

        - loss_tot_traj: bool, if True, returns the total loss over the each trajectory (B, N)
        '''
        bsize, num_samp, _, _ = x_loss.size()
        guide_losses = dict()
        loss_tot = 0.0
        _, local_scene_index = torch.unique_consecutive(data_batch['scene_index'], return_inverse=True)
        for si in range(self.num_scenes):
            cur_guide = self.guide_configs[si]
            if len(cur_guide) > 0:
                # mask out non-current current scene
                for gidx, guide_cfg in enumerate(cur_guide):
                    agt_mask = local_scene_index == si
                    if guide_cfg.agents is not None:
                        # mask out non-requested agents within the scene
                        cur_scene_inds = torch.nonzero(agt_mask, as_tuple=True)[0]
                        agt_mask_inds = cur_scene_inds[guide_cfg.agents]
                        agt_mask = torch.zeros_like(agt_mask)
                        agt_mask[agt_mask_inds] = True
                    # compute loss
                    cur_loss = guide_cfg.func(x_loss, data_batch,
                                            agt_mask=agt_mask)
                    indiv_loss = torch.ones((bsize, num_samp)).to(cur_loss.device) * np.nan # return indiv loss for whole batch, not just masked ones
                    indiv_loss[agt_mask] = cur_loss.detach().clone()
                    guide_losses[guide_cfg.name + '_scene_%03d_%02d' % (si, gidx)] = indiv_loss
                    loss_tot = loss_tot + torch.mean(cur_loss) * guide_cfg.weight
                    # print('GUIDANCE INFO: ', gidx, guide_cfg, torch.mean(cur_loss).item(), guide_cfg.weight)
        return loss_tot, guide_losses

    def compute_b4_energy_partition(self, x_loss, data_batch):
        """Evaluate explicit B4 attack and feasibility energies.

        Every configured loss must declare exactly one partition.  This is a
        fail-closed contract: legacy unpartitioned guidance cannot silently be
        treated as B4.  Both partitions must be non-empty in every call.
        """
        bsize, num_samp, _, _ = x_loss.size()
        attack = x_loss.new_zeros(())
        feasibility = x_loss.new_zeros(())
        partition_losses = {}
        partition_counts = {"attack": 0, "feasibility": 0}
        _, local_scene_index = torch.unique_consecutive(
            data_batch['scene_index'], return_inverse=True
        )
        for si in range(self.num_scenes):
            cur_guide = self.guide_configs[si]
            for gidx, guide_cfg in enumerate(cur_guide):
                if guide_cfg.partition not in partition_counts:
                    raise ValueError(
                        "B4 requires explicit attack/feasibility partition "
                        "for {} scene {} config {}".format(
                            guide_cfg.name, si, gidx
                        )
                    )
                agt_mask = local_scene_index == si
                if guide_cfg.agents is not None:
                    cur_scene_inds = torch.nonzero(
                        agt_mask, as_tuple=True
                    )[0]
                    agt_mask_inds = cur_scene_inds[guide_cfg.agents]
                    agt_mask = torch.zeros_like(agt_mask)
                    agt_mask[agt_mask_inds] = True
                cur_loss = guide_cfg.func(
                    x_loss, data_batch, agt_mask=agt_mask
                )
                mean_loss = torch.mean(cur_loss) * guide_cfg.weight
                if guide_cfg.partition == "attack":
                    attack = attack + mean_loss
                else:
                    feasibility = feasibility + mean_loss
                partition_counts[guide_cfg.partition] += 1
                key = '{}_scene_{:03d}_{:02d}'.format(
                    guide_cfg.name, si, gidx
                )
                indiv_loss = torch.ones(
                    (bsize, num_samp), device=cur_loss.device
                ) * np.nan
                indiv_loss[agt_mask] = cur_loss.detach().clone()
                partition_losses[key] = {
                    "partition": guide_cfg.partition,
                    "weight": float(guide_cfg.weight),
                    "value": indiv_loss,
                }
        if partition_counts["attack"] == 0:
            raise ValueError("B4 attack partition is empty")
        if partition_counts["feasibility"] == 0:
            raise ValueError("B4 feasibility partition is empty")
        partition_losses["partition_counts"] = partition_counts
        return attack, feasibility, partition_losses



############## ITERATIVE PERTURBATION ########################
class PerturbationGuidance(object):
    """
    Guide trajectory to satisfy rules by directly perturbing it
    """
    def __init__(self, transform, transform_params, scale_traj=lambda x,y:x, descale_traj=lambda x,y:x, controllable_agent=-1) -> None:
        
        self.transform = transform
        self.transform_params = transform_params
        
        self.scale_traj = scale_traj
        self.descale_traj = descale_traj

        self.current_guidance = None
        self.controllable_agent = controllable_agent # -1
        # print('controllable agents: ', self.controllable_agent)

    def update(self, **kwargs):
        self.current_guidance.update(**kwargs)

    def set_guidance(self, guidance_config_list, example_batch=None):
        self.current_guidance = DiffuserGuidance(guidance_config_list, example_batch)
    
    def clear_guidance(self):
        self.current_guidance = None

    def perturb_actions_dict(self, actions_dict, data_batch, opt_params, num_samp=1):
        """Given the observation object, add Gaussian noise to positions and yaws

        Args:
            data_batch(Dict[torch.tensor]): observation dict

        Returns:
            data_batch(Dict[torch.tensor]): perturbed observation
        """
        x_initial = torch.cat((actions_dict["target_positions"], actions_dict["target_yaws"]), dim=-1)

        x_guidance, _ = self.perturb(x_initial, data_batch, opt_params, num_samp)
        # print('x_guidance.shape', x_guidance.shape)
        # x_guidance: [B*N, T, 3]
        actions_dict["target_positions"] = x_guidance[..., :2].type(torch.float32)
        actions_dict["target_yaws"] = x_guidance[..., 2:3].type(torch.float32)
        
        return actions_dict

    def perturb(self, x_initial, data_batch, opt_params, num_samp=1, decoder=None, return_grad_of=None):
        '''
        perturb the gradient and estimate the guidance loss w.r.t. the input trajectory
        Input:
            x_initial: [batch_size*num_samp, (num_agents), time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            opt_params: optimization parameters.
            num_samp: number of samples in x_initial.
            decoder: decode the perturbed variable to get the trajectory.
            return_grad_of: apply the gradient to which variable.
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'

        perturb_th = opt_params['perturb_th']
        
        x_guidance = x_initial

        # x_guidance may not have gradient enabled when BITS is used
        if not x_guidance.requires_grad:
            x_guidance.requires_grad_()

        if len(x_guidance.shape) == 4:
            with torch.enable_grad():
                BN, M, T, _ = x_guidance.shape
                B = int(BN // num_samp)
                x_guidance_reshaped = x_guidance.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)
        else:
            x_guidance_reshaped = x_guidance
        # print("X guide: ", x_guidance.shape)
        # guide_dim = min(15, x_guidance.shape[0])
        # guide_dim = x_guidance.shape[-3]
        if self.controllable_agent > 0: 
            # print("controllable_agent: ", self.controllable_agent)
            guide_dim = min(self.controllable_agent, x_guidance.shape[-3]) # for CCDiff    
        else: 
            guide_dim = x_guidance.shape[-3]
        # print("guide dim: ", guide_dim, x_guidance.shape)
        if opt_params['optimizer'] == 'adam': # TODO: different for diffusion and CCDiff
            # opt = torch.optim.Adam([x_guidance], lr=opt_params['lr'])
            if len(x_guidance.shape) == 3: 
                opt = torch.optim.Adam([x_guidance[:guide_dim]], lr=opt_params['lr'])
            elif len(x_guidance.shape) == 4: 
                opt = torch.optim.Adam([x_guidance[:, :guide_dim]], lr=opt_params['lr'])
        elif opt_params['optimizer'] == 'sgd':
            # opt = torch.optim.SGD([x_guidance], lr=opt_params['lr'])
            if len(x_guidance.shape) == 3: 
                opt = torch.optim.SGD([x_guidance[:guide_dim]], lr=opt_params['lr'])
            elif len(x_guidance.shape) == 4:
                opt = torch.optim.SGD([x_guidance[:, :guide_dim]], lr=opt_params['lr'])
        else: 
            raise NotImplementedError('Optimizer not implemented')
        # if opt_params['optimizer'] == 'adam': # TODO: different for diffusion and CCDiff
        #     opt = torch.optim.Adam([x_guidance], lr=opt_params['lr'])
        # elif opt_params['optimizer'] == 'sgd':
        #     opt = torch.optim.SGD([x_guidance], lr=opt_params['lr'])
        per_losses = dict()
        for _ in range(opt_params['grad_steps']):
            with torch.enable_grad():
                # for CVAE, we need to decode the latent
                if decoder is not None:
                    x_guidance_decoded = decoder(x_guidance_reshaped)
                else:
                    x_guidance_decoded = x_guidance_reshaped
                bsize = int(x_guidance_decoded.size(0) / num_samp)

                x_all = self.transform(x_guidance_decoded, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)

                x_loss = x_all.reshape((bsize, num_samp, -1, 6))
                tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)
            
            tot_loss.backward()
            opt.step()
            opt.zero_grad()
            if perturb_th is not None:
                with torch.no_grad():
                    x_delta = x_guidance - x_initial
                    x_delta_clipped = torch.clip(x_delta, -1*perturb_th, perturb_th)
                    x_guidance.data = x_initial + x_delta_clipped
        # print('x_guidance.data - x_initial', x_guidance.data - x_initial)
        # print("per_losses: ", per_losses)
        return x_guidance, per_losses
    
    def perturb_video_diffusion(self, x_initial, data_batch, opt_params, num_samp=1, return_grad_of=None):
        '''
        video_diffusion only
        perturb the gradient and estimate the guidance loss w.r.t. the input trajectory
        Input:
            x_initial: [batch_size*num_samp, (num_agents), time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            opt_params: optimization parameters.
            num_samp: number of samples in x_initial.
            return_grad_of: apply the gradient to which variable.
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'

        perturb_th = opt_params['perturb_th']
        
        x_guidance = x_initial

        if len(x_guidance.shape) == 4:
            with torch.enable_grad():
                BN, M, T, _ = x_guidance.shape
                B = int(BN // num_samp)
                x_guidance = x_guidance.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)

        per_losses = dict()
        for _ in range(opt_params['grad_steps']):
            with torch.enable_grad():
                bsize = int(x_guidance.size(0) / num_samp)

                x_all = self.transform(x_guidance, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)
                
                x_loss = x_all.reshape((bsize, num_samp, -1, 6))
                tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)
            
            tot_loss.backward()
            x_delta = opt_params['lr'] * return_grad_of.grad
            if x_initial.shape[-1] == 2:
                # only need the grad w.r.t noisy action
                x_delta = x_delta[..., [4,5]]

            x_guidance = x_initial + x_delta
            if perturb_th is not None:
                with torch.no_grad():
                    x_delta_clipped = torch.clip(x_delta, -1*perturb_th, perturb_th)
                    x_guidance.data = x_initial + x_delta_clipped

        return x_guidance, per_losses
    
    @torch.no_grad()
    def compute_guidance_loss(self, x_initial, data_batch, num_samp=1):
        '''
        -x_initial: [B*N, T, 2/3]
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'
        if len(x_initial.shape) == 4:
            BN, M, T, _ = x_initial.shape
            B = int(BN // num_samp)
            x_initial = x_initial.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)
        bsize = int(x_initial.size(0) / num_samp)
        num_t = x_initial.size(1)

        x_initial_copy = x_initial.clone().detach()
        x_guidance = Variable(x_initial_copy, requires_grad=True)

        x_all = self.transform(x_guidance, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)
        
        x_loss = x_all.reshape((bsize, num_samp, num_t, 6))

        tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)

        return tot_loss, per_losses
