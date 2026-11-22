"""Keep replay absence out of occupancy while retaining finite masked model rows."""
import contextvars
import functools
import inspect
import textwrap

import torch


METRIC_SCOPE = contextvars.ContextVar('native_common_metric_scope', default=False)


def metric_view(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        token = METRIC_SCOPE.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            METRIC_SCOPE.reset(token)
    return wrapped


def install(legacy):
    import tbsim.utils.trajdata_utils as utils
    import tbsim.envs.env_trajdata as environment
    from tbsim.utils.rollout_logger import RolloutLogger

    original = utils.parse_trajdata_batch
    if getattr(original, '_native_common_v5', False):
        raise RuntimeError('Common observation adapter installed twice')

    @torch.no_grad()
    def parse(batch, *args, **kwargs):
        if 'num_agents' in batch:
            return original(batch, *args, **kwargs)
        current = batch['curr_agent_state']
        present = (torch.isfinite(current.position).all(-1)
                   & torch.isfinite(current.heading).all(-1)).clone()
        # Capture raw replay before the legacy parser zero-fills or lane-filters it.
        future = batch['agent_fut'].as_format('x,y,v_lon,h').as_tensor().clone()
        positions, yaws = future[..., :2], future[..., 3:4]
        available = torch.isfinite(future).all(-1)
        private = dict(batch)
        private['curr_agent_state'] = current.clone()
        parsed = original(private, *args, **kwargs)
        parsed.update(current_pose_available=present,
                      availability=present.clone(),
                      logged_target_positions=positions,
                      logged_target_yaws=yaws,
                      logged_target_availabilities=available)
        absent = ~present
        if not bool(absent.any()):
            return parsed
        if bool(parsed['history_availabilities'][absent].any()):
            raise ValueError('Absent current frames must have masked own histories')
        if bool(available[absent].any()):
            raise ValueError('Absent rows cannot have an expressed logged future')
        metric = METRIC_SCOPE.get() or kwargs.get('overwrite_nan') is False
        if metric:
            for key in ('centroid', 'yaw', 'curr_agent_state'):
                parsed[key] = parsed[key].clone()
                parsed[key][absent] = float('nan')
        else:
            for key in ('world_from_agent', 'agent_from_world', 'agents_from_world_tf'):
                if key in parsed:
                    value = parsed[key].clone()
                    value[absent] = torch.eye(3, device=value.device, dtype=value.dtype)
                    parsed[key] = value
            if parsed.get('raster_from_world') is not None:
                value = parsed['raster_from_world'].clone()
                value[absent] = parsed['raster_from_agent'][absent]
                parsed['raster_from_world'] = value
        return parsed

    parse._native_common_v5 = True
    utils.parse_trajdata_batch = parse
    environment.parse_trajdata_batch = parse

    class AvailabilityLogger(RolloutLogger):
        def __init__(self, obs_keys=None, info_keys=None, **kwargs):
            keys = list(obs_keys or [])
            if 'availability' not in keys:
                keys.append('availability')
            if 'info_keys' in inspect.signature(RolloutLogger.__init__).parameters:
                kwargs['info_keys'] = list(info_keys or [])
            super().__init__(obs_keys=keys, **kwargs)

    environment.RolloutLogger = AvailabilityLogger
    if legacy:
        cls = environment.EnvSplitUnifiedSimulation
        source = textwrap.dedent(inspect.getsource(cls._step))
        old = 'next_state = np.zeros(3, dtype=obs["agent_fut"].dtype)'
        if source.count(old) != 1:
            raise ValueError('Unsupported legacy simulator implementation')
        source = source.replace(old, 'next_state = np.full(3, np.nan, dtype=obs["agent_fut"].dtype)')
        namespace = {}
        exec(compile(source, '<native_common_v5_step>', 'exec'), vars(environment), namespace)
        cls._step = metric_view(namespace['_step'])
        observation_source = textwrap.dedent(inspect.getsource(cls.get_observation_skimp))
        selection = 'keys_to_take = ['
        if observation_source.count(selection) != 1:
            raise ValueError('Unsupported legacy logging observation selection')
        observation_source = observation_source.replace(selection, selection + "'availability', ")
        namespace = {}
        exec(compile(observation_source, '<native_common_v5_log_observation>', 'exec'),
             vars(environment), namespace)
        cls.get_observation_skimp = metric_view(namespace['get_observation_skimp'])
    else:
        for cls in (environment.EnvUnifiedSimulation, environment.EnvSplitUnifiedSimulation):
            # Split logging uses the parser's default zero filling. Keep actual
            # absence in step/log views while model observations stay masked.
            cls._step = metric_view(cls._step)
            cls.get_observation_skimp = metric_view(cls.get_observation_skimp)

    return dict(legacy_nan_step_patch=legacy,
                step_and_log_absence_preserved=True,
                raw_future_availability_before_lane_filter=True,
                absent_occupancy_encoding='NaN', absent_model_rows_history_masked=True,
                explicit_frame_availability_exported=True, raw_current_state_padding_isolated=True,
                frozen_repository_files_modified=False)
