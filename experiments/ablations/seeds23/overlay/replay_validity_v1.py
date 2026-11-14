"""Missing logged background poses are absent, never finite local-zero actions.

Fixed row/track identity is retained. Only observation tensors for an absent row
get finite coordinate placeholders, with all own-history availability false;
the simulator cache, metric observations and executed actions retain NaNs.
No extrapolation or stationary continuation is introduced. The forecast uses
current/history availability only, never future replay masks.
"""
import torch


def apply_replay_validity(positions, yaws, obs, replay_indices):
    """Mutate only unavailable replay slots; validate all controlled outputs."""
    horizon = positions.shape[1]
    mask = obs['target_availabilities'][:, :horizon].bool()
    replay = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)
    replay[replay_indices] = True
    # Dataset endpoint padding is supported. An internal gap/reappearance must
    # receive an explicit lifecycle design, not silently lose a real track.
    if horizon > 1 and bool(((~mask[:,:-1]) & mask[:,1:] & replay[:,None]).any()):
        raise ValueError('Unsupported reappearance inside a logged replay horizon')
    invalid = replay[:,None] & ~mask
    valid = replay[:,None] & mask
    if not torch.equal(positions[valid], obs['target_positions'][:,:horizon][valid]):
        raise AssertionError('Available background replay positions changed')
    if not torch.equal(yaws[valid], obs['target_yaws'][:,:horizon][valid]):
        raise AssertionError('Available background replay yaws changed')
    positions[invalid] = float('nan'); yaws[invalid] = float('nan')
    finite = torch.isfinite(positions).all(-1) & torch.isfinite(yaws).all(-1)
    if not bool(finite[~invalid].all()):
        raise FloatingPointError('Controlled or available replay action is nonfinite')
    assert bool(torch.isnan(positions[invalid]).all()) and bool(torch.isnan(yaws[invalid]).all())
    missing_now = replay & ~obs['history_availabilities'][:,-1].bool()
    if bool((mask & missing_now[:,None]).any()):
        raise ValueError('Absent background unexpectedly acquired a logged future')
    return dict(policy='logged_pose_availability_v1', valid_replay_positions_exact=True,
                valid_replay_yaws_exact=True, missing_replay_action_slots=int(invalid.sum()),
                missing_replay_next5_slots=int(invalid[:,:5].sum()),
                currently_absent_replay_indices=missing_now.nonzero(as_tuple=False)[:,0].cpu().tolist(),
                absent_pose_encoding='NaN', stationary_tail_or_extrapolation=False)


def install_observation_validity():
    """Process-local adapter; never modify frozen trajdata/tbsim source files."""
    import tbsim.utils.trajdata_utils as utils
    import tbsim.envs.env_trajdata as environment
    original = utils.parse_trajdata_batch
    assert not getattr(original, '_replay_validity_v1', False)

    @torch.no_grad()
    def parse(batch, overwrite_nan=True):
        if 'num_agents' in batch:
            raise ValueError('Replay validity v1 requires the frozen agent-centric protocol')
        state = batch['curr_agent_state']
        present = torch.isfinite(state.position).all(-1) & torch.isfinite(state.heading).all(-1)
        parsed = original(batch, overwrite_nan=overwrite_nan)
        parsed['current_pose_available'] = present.clone()
        absent = ~present
        if not bool(absent.any()):
            return parsed
        # A missing current coordinate frame cannot express a valid own history.
        assert not bool(parsed['history_availabilities'][absent].any())
        assert not bool(parsed['target_availabilities'][absent].any())
        if overwrite_nan:
            # Finite identities are only numerical padding for masked model
            # rows. Neither the replay action nor occupancy uses these origins.
            for key in ['world_from_agent', 'agent_from_world']:
                value = parsed[key].clone()
                value[absent] = torch.eye(3, device=value.device, dtype=value.dtype)
                parsed[key] = value
            parsed['raster_from_world'] = parsed['raster_from_world'].clone()
            parsed['raster_from_world'][absent] = parsed['raster_from_agent'][absent]
            parsed['agents_from_world_tf'] = parsed['agents_from_world_tf'].clone()
            parsed['agents_from_world_tf'][absent] = parsed['agent_from_world'][absent]
        else:
            # Native collision/off-road code already treats NaN poses as absent.
            # The old parser zero-filled curr_agent_state even in metric mode.
            for key in ['centroid', 'yaw', 'curr_agent_state']:
                parsed[key] = parsed[key].clone()
                parsed[key][absent] = float('nan')
        return parsed

    parse._replay_validity_v1 = True
    utils.parse_trajdata_batch = parse
    environment.parse_trajdata_batch = parse
