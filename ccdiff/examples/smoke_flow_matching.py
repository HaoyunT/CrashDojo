#!/usr/bin/env python3
"""Three-part, one-GPU smoke test for the CCDiff flow-matching transport."""

import argparse
import json
import math
import os

import torch

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.batch_utils import batch_utils, set_global_batch_type
from tbsim.utils.trajdata_utils import (
    add_scene_dim_to_agent_data,
    convert_scene_data_to_agent_coordinates,
    set_global_trajdata_batch_env,
    set_global_trajdata_batch_raster_cfg,
)

from ccdiff.algos.factory import algo_factory


def prepare_batch(batch, algo, device):
    batch = TensorUtils.to_device(batch, device)
    data_centric = 'scene' if 'num_agents' in batch else 'agent'
    batch = batch_utils().parse_batch(batch)
    if data_centric == 'scene' and algo.coordinate == 'agent_centric':
        batch = convert_scene_data_to_agent_coordinates(
            batch,
            max_neighbor_dist=algo.scene_agent_max_neighbor_dist,
            keep_order_of_neighbors=True,
        )
    elif data_centric == 'agent' and algo.coordinate == 'agent_centric':
        batch = add_scene_dim_to_agent_data(batch)
    elif not (
        data_centric == 'scene' and algo.coordinate == 'scene_centric'
    ):
        raise NotImplementedError(
            '{} data with {} coordinates'.format(
                data_centric, algo.coordinate
            )
        )
    return batch


def scalar_dict(values):
    result = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            result[key] = float(value.detach().mean().cpu())
        elif isinstance(value, (float, int)):
            result[key] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='/path/to/workspace/nuscenes')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_collision_attempts', type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            'Smoke requires exactly one visible CUDA GPU; found {}'.format(
                torch.cuda.device_count()
            )
        )
    device = torch.device('cuda:0')
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    cfg = get_registered_experiment_config(
        'trajdata_nusc_ccflow_matching'
    )
    cfg.devices.num_gpus = 1
    cfg.train.training.batch_size = 1
    cfg.train.training.num_data_workers = 0
    cfg.train.validation.num_data_workers = 0
    cfg.train.logging.log_tb = False
    cfg.train.logging.log_wandb = False
    for dataset_key in cfg.train.trajdata_data_dirs:
        if dataset_key.startswith('nusc'):
            cfg.train.trajdata_data_dirs[dataset_key] = args.dataset_path
    cfg.train.dataset_path = args.dataset_path

    set_global_batch_type('trajdata')
    set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])
    set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer)

    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )
    datamodule.setup()
    raw_batch = next(iter(datamodule.train_dataloader()))
    algo = algo_factory(cfg, datamodule.modality_shapes).to(device)
    algo.train()
    batch = prepare_batch(raw_batch, algo, device)
    policy = algo.nets['policy']

    target_traj = policy.get_state_and_action_from_data_batch(batch)
    if policy.use_reconstructed_state:
        aux_for_target = policy.get_aux_info(batch)
        target_traj = policy.convert_action_to_state_and_action(
            target_traj[..., [4, 5]],
            aux_for_target['curr_states'],
            scaled_input=False,
        )
    x_start = policy.scale_traj(target_traj)
    action_end = x_start[..., [4, 5]]
    action_start = torch.randn_like(action_end)
    lambda_zero = torch.zeros(
        (len(x_start),), device=device, dtype=x_start.dtype
    )
    lambda_one = torch.ones_like(lambda_zero)
    at_zero = policy._interpolate_actions(
        action_start, action_end, lambda_zero
    )
    at_one = policy._interpolate_actions(
        action_start, action_end, lambda_one
    )
    endpoint_report = {
        'lambda_0_mean': float(at_zero.mean().detach().cpu()),
        'lambda_0_std': float(at_zero.std().detach().cpu()),
        'lambda_0_max_abs_error_vs_noise': float(
            (at_zero - action_start).abs().max().detach().cpu()
        ),
        'lambda_1_mean': float(at_one.mean().detach().cpu()),
        'lambda_1_std': float(at_one.std().detach().cpu()),
        'lambda_1_max_abs_error_vs_clean_action': float(
            (at_one - action_end).abs().max().detach().cpu()
        ),
        'time_embedding_zero': 0.0,
        'time_embedding_one': float(policy.n_timesteps - 1),
    }
    if endpoint_report['lambda_0_max_abs_error_vs_noise'] != 0.0:
        raise AssertionError('lambda=0 is not the Gaussian endpoint')
    if endpoint_report['lambda_1_max_abs_error_vs_clean_action'] != 0.0:
        raise AssertionError('lambda=1 is not the clean-action endpoint')

    collision_gradient_report = None
    loss_report = None
    for attempt in range(args.max_collision_attempts):
        captured = {}

        def capture_velocity(_module, _inputs, output):
            captured['velocity'] = output[0]
            captured['velocity'].retain_grad()

        hook = policy.model.register_forward_hook(capture_velocity)
        flow_time = torch.rand(
            (len(x_start),), device=device, dtype=x_start.dtype
        )
        aux_info = policy.get_aux_info(batch)
        loss, info = policy.p_losses(
            x_start, flow_time, batch, aux_info=aux_info
        )
        hook.remove()

        if not torch.isfinite(loss):
            raise AssertionError('non-finite flow reconstruction loss')
        collision_loss = info.get('collision_loss')
        if collision_loss is None:
            raise AssertionError('loss_fn did not expose collision_loss')
        collision_grad = torch.autograd.grad(
            collision_loss,
            captured['velocity'],
            retain_graph=True,
            allow_unused=True,
        )[0]
        collision_grad_norm = (
            0.0
            if collision_grad is None
            else float(collision_grad.norm().detach().cpu())
        )
        if math.isfinite(collision_grad_norm) and collision_grad_norm > 0.0:
            algo.zero_grad(set_to_none=True)
            loss.backward()
            parameter_grad_sq = 0.0
            for parameter in policy.parameters():
                if parameter.grad is not None:
                    parameter_grad_sq += float(
                        parameter.grad.detach().float().pow(2).sum().cpu()
                    )
            collision_gradient_report = {
                'attempt': attempt + 1,
                'flow_time': [
                    float(value)
                    for value in flow_time.detach().cpu().tolist()
                ],
                'collision_loss': float(collision_loss.detach().cpu()),
                'collision_grad_norm_wrt_velocity': collision_grad_norm,
                'training_loss_grad_norm': math.sqrt(parameter_grad_sq),
            }
            loss_report = {
                'flow_reconstruction_loss': float(loss.detach().cpu()),
                'components': scalar_dict(info),
            }
            break

    if collision_gradient_report is None:
        raise AssertionError(
            'collision_loss had zero gradient to v_pred in {} attempts'.format(
                args.max_collision_attempts
            )
        )

    policy.eval()
    # The training-coordinate conversion does not retain extras.full_fut_traj,
    # while the configured policy uses the current-speed stationary criterion.
    stationary_mask = batch['curr_speed'] <= algo.moving_speed_th
    with torch.no_grad():
        sampled = policy(
            batch,
            num_samp=1,
            return_diffusion=True,
            return_guidance_losses=False,
            apply_guidance=False,
            stationary_mask=stationary_mask,
            mode='training',
        )['predictions']
    trajectories = sampled['trajectories']
    if not torch.isfinite(trajectories).all():
        raise AssertionError('FM Euler sample contains NaN or Inf')
    acceleration = trajectories[..., 4]
    yaw_rate = trajectories[..., 5]
    speed = trajectories[..., 2]
    dynamics_report = {
        'n_flow_steps': int(policy.n_flow_steps),
        'diffusion_snapshots': int(
            sampled['diffusion_steps'].shape[3]
        ),
        'acceleration_min': float(acceleration.min().cpu()),
        'acceleration_max': float(acceleration.max().cpu()),
        'yaw_rate_abs_max': float(yaw_rate.abs().max().cpu()),
        'speed_min': float(speed.min().cpu()),
        'speed_max': float(speed.max().cpu()),
        'configured_acceleration_bounds': [-10.0, 8.0],
        'configured_max_speed': 40.0,
    }
    tolerance = 1e-5
    if dynamics_report['acceleration_min'] < -10.0 - tolerance:
        raise AssertionError('sampled acceleration is below the configured bound')
    if dynamics_report['acceleration_max'] > 8.0 + tolerance:
        raise AssertionError('sampled acceleration is above the configured bound')
    if dynamics_report['speed_max'] > 40.0 + tolerance:
        raise AssertionError('sampled speed exceeds max_speed')
    if dynamics_report['speed_min'] < -10.0 - tolerance:
        raise AssertionError('sampled speed is below the Unicycle lower bound')
    if dynamics_report['diffusion_snapshots'] != policy.n_flow_steps + 1:
        raise AssertionError('Euler trace does not contain K+1 states')

    report = {
        'status': 'pass',
        'device': torch.cuda.get_device_name(0),
        'config_name': 'trajdata_nusc_ccflow_matching',
        'endpoint_direction': endpoint_report,
        'gradient': collision_gradient_report,
        'loss': loss_report,
        'sampling_dynamics': dynamics_report,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as file_obj:
        json.dump(report, file_obj, indent=2, sort_keys=True)
    print('FLOW_SMOKE_REPORT {}'.format(json.dumps(report, sort_keys=True)))


if __name__ == '__main__':
    main()
