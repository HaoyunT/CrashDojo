"""Process-local availability export without mutating raw simulator states."""

import torch


def masked_observation(observation):
    import numpy as np

    result = dict(observation)
    available = result['availability']
    for key in ('centroid', 'yaw', 'curr_agent_state'):
        if key in result:
            value = result[key]
            result[key] = value.clone() if torch.is_tensor(value) else np.array(value, copy=True)
            result[key][~available] = float('nan')
    return result


def install():
    import tbsim.utils.trajdata_utils as utils
    import tbsim.envs.env_trajdata as environment
    from tbsim.utils.rollout_logger import RolloutLogger

    original = utils.parse_trajdata_batch
    if getattr(original, '_availability_export_v9', False):
        raise RuntimeError('Availability adapter already installed')

    @torch.no_grad()
    def parse(batch, overwrite_nan=True):
        if 'num_agents' in batch:
            raise ValueError('The frozen protocol requires agent-centric batches')
        state = batch['curr_agent_state']
        present = (torch.isfinite(state.position).all(-1)
                   & torch.isfinite(state.heading).all(-1)).clone()
        # The native parser zero-fills this tensor in place even in metric mode.
        # A private copy prevents numerical padding from changing simulator data.
        private = dict(batch)
        private['curr_agent_state'] = state.clone()
        parsed = original(private, overwrite_nan=overwrite_nan)
        parsed['current_pose_available'] = present.clone()
        parsed['availability'] = present.clone()
        absent = ~present
        if bool(absent.any()):
            if bool(parsed['history_availabilities'][absent].any()):
                raise ValueError('Absent current frame has a valid own coordinate history')
            if bool(parsed['target_availabilities'][absent].any()):
                raise ValueError('Unsupported background reappearance')
            if overwrite_nan:
                for key in ('world_from_agent', 'agent_from_world'):
                    parsed[key] = parsed[key].clone()
                    parsed[key][absent] = torch.eye(3, device=state.device, dtype=parsed[key].dtype)
                parsed['raster_from_world'] = parsed['raster_from_world'].clone()
                parsed['raster_from_world'][absent] = parsed['raster_from_agent'][absent]
                parsed['agents_from_world_tf'] = parsed['agents_from_world_tf'].clone()
                parsed['agents_from_world_tf'][absent] = parsed['agent_from_world'][absent]
            else:
                for key in ('centroid', 'yaw', 'curr_agent_state'):
                    parsed[key] = parsed[key].clone()
                    parsed[key][absent] = float('nan')
        return parsed

    class AvailabilityLogger(RolloutLogger):
        def __init__(self, obs_keys=None, info_keys=None, save_action_samples=False):
            keys = list(obs_keys or [])
            if 'availability' not in keys:
                keys.append('availability')
            super().__init__(obs_keys=keys, info_keys=list(info_keys or []),
                             save_action_samples=save_action_samples)

        def log_step(self, observation, action):
            observation = {key: masked_observation(value) if value is not None else None
                           for key, value in observation.items()}
            return super().log_step(observation, action)

    for name in ('EnvUnifiedSimulation', 'EnvSplitEgoSimulation'):
        cls = getattr(environment, name, None)
        if cls is None:
            continue
        original_metrics = cls._add_per_step_metrics

        def metrics(self, observation, *args, _original=original_metrics, **kwargs):
            return _original(self, masked_observation(observation), *args, **kwargs)

        cls._add_per_step_metrics = metrics

    parse._availability_export_v9 = True
    utils.parse_trajdata_batch = parse
    environment.parse_trajdata_batch = parse
    environment.RolloutLogger = AvailabilityLogger
