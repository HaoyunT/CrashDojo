"""Common transport for native selected actors, supplied ego, and logged traffic."""
import torch


def compose(observation, generated, ego_action, selected_indices, ego_index):
    reference = observation.get('logged_target_positions', observation['target_positions'])
    reference_yaw = observation.get('logged_target_yaws', observation['target_yaws'])
    count = reference.shape[0]
    if reference.ndim != 3 or reference.shape[-1] != 2:
        raise ValueError('Expected agent-centric target positions')
    if reference_yaw.shape != reference.shape[:-1] + (1,):
        raise ValueError('Reference yaw layout differs')
    if not 0 <= ego_index < count:
        raise ValueError('Invalid protected ego index')
    selected = [int(i) for i in selected_indices]
    if len(set(selected)) != len(selected) or any(i < 0 or i >= count for i in selected):
        raise ValueError('Native selected identities are not unique valid rows')
    present = observation.get('current_pose_available')
    if present is None:
        raise ValueError('Common transport requires observed current pose availability')
    if not bool(present[ego_index]):
        raise ValueError('The supplied ego has no current coordinate frame')
    adversaries = [i for i in selected if i != ego_index and bool(present[i])]
    if ego_action.positions.shape[0] != 1:
        raise ValueError('Exactly one supplied ego trajectory is required')
    horizon = min(reference.shape[1], generated.positions.shape[1], ego_action.positions.shape[1])
    if horizon < 5:
        raise ValueError('Supplied trajectories cannot cover one execution block')
    positions = reference[:, :horizon].clone()
    yaws = reference_yaw[:, :horizon].clone()
    positions[adversaries] = generated.positions[adversaries, :horizon]
    yaws[adversaries] = generated.yaws[adversaries, :horizon]
    positions[ego_index] = ego_action.positions[0, :horizon]
    yaws[ego_index] = ego_action.yaws[0, :horizon]

    replay = torch.ones(count, dtype=torch.bool, device=positions.device)
    replay[adversaries + [ego_index]] = False
    available = observation.get('logged_target_availabilities',
                                observation['target_availabilities'])[:, :horizon].bool()
    if horizon > 1 and bool((replay[:, None] & ~available[:, :-1] & available[:, 1:]).any()):
        raise ValueError('Logged background reappearance needs an explicit lifecycle policy')
    if bool((replay[:, None] & ~present[:, None] & available).any()):
        raise ValueError('An absent background cannot regain a logged coordinate frame')
    valid_replay = replay[:, None] & available
    if not torch.equal(positions[valid_replay], reference[:, :horizon][valid_replay]):
        raise AssertionError('Available replay positions changed')
    if not torch.equal(yaws[valid_replay], reference_yaw[:, :horizon][valid_replay]):
        raise AssertionError('Available replay yaws changed')
    absent_replay = replay[:, None] & ~available
    positions[absent_replay] = float('nan')
    yaws[absent_replay] = float('nan')
    finite = torch.isfinite(positions).all(-1) & torch.isfinite(yaws).all(-1)
    if not bool(finite[~absent_replay].all()):
        raise FloatingPointError('Controlled or available replay output is nonfinite')
    return positions, yaws, dict(
        execution_protocol='supplied_ego_native_selected_logged_background_v5',
        supplied_ego_executed_exactly=True,
        protected_ego_index=ego_index,
        selected_non_ego_indices=adversaries,
        selected_absent_indices=[i for i in selected if not bool(present[i])],
        replay_indices=replay.nonzero(as_tuple=False)[:, 0].cpu().tolist(),
        background_exact_replay=True,
        missing_replay_slots=int(absent_replay.sum()),
        missing_replay_next5_slots=int(absent_replay[:, :5].sum()),
        native_selected_ego_overridden_by_external_planner=ego_index in selected,
        horizon=horizon,
    )
