"""Screen five causal speed profiles against the actual current ego forecast."""
import copy
import json

import torch


def speed_profile(parameters, target, limits):
    result = copy.deepcopy(parameters)
    initial = float(parameters['initial_speed_mps'])
    target = max(0., min(float(target), limits['max_speed_mps']))
    change = abs(target - initial)
    acceleration = min(2.5, .85 * limits['max_acceleration_mps2'])
    ramp = .6
    if change < .01:
        result.update(target_speed_mps=target, speed_transition_s=2.)
        result.pop('speed_profile', None)
    else:
        acceleration = min(acceleration, change / ramp)
        result.update(target_speed_mps=target, speed_profile='bounded_jerk',
                      profile_acceleration_mps2=acceleration, profile_jerk_ramp_s=ramp,
                      speed_transition_s=change / acceleration + ramp)
    return result


def screen_request(request, context, path):
    from ccdiff.evaluation.five_class_guidance import (
        reference_route_progress, route_frame_at_s, sat_separation)
    from execution_completion import offset_reference

    result = copy.deepcopy(request)
    family = request['family']
    assert family == 'C2', 'First timing experiment is deliberately limited to complete lane changes'
    horizon = len(context['ego_pos'])
    times = context['route'].new_tensor(list(range(1, horizon + 1))) * context['dt']
    assert context['elapsed'] == 0.
    proposals = []
    original = request['parameters']['target_speed_mps']
    # The original proposal remains available; four proposals change longitudinal motion.
    for index, target in enumerate((original, 6., 8., 10., 12.)):
        candidate = copy.deepcopy(request)
        if index:
            candidate['parameters'] = speed_profile(request['parameters'], target, request['engineering_limits'])
        parameters = candidate['parameters']
        station = reference_route_progress(times, parameters, context['initial_route_s'])
        center, tangent = route_frame_at_s(station, context['route'])
        normal = torch.stack((-tangent[:, 1], tangent[:, 0]), -1)
        offset = offset_reference(times, context, candidate)
        position = center + offset[:, None] * normal
        initial = context['transform'][:2, 2]
        delta = torch.diff(torch.cat((initial[None], position)), dim=0)
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        separation = sat_separation(position[None], yaw[None], context['ego_pos'],
                                    context['ego_yaw'], context['extent'], context['ego_extent'])[0]
        window = ((times >= candidate['interaction_window_s'][0]) &
                  (times <= candidate['interaction_window_s'][1]))
        if bool(window.any()):
            cost = torch.relu(separation[window] + .05).square() + .05 * times[window]
            selected_time = int(cost.argmin())
            score = float(cost[selected_time])
            gap = float(separation[window][selected_time])
            encounter = float(times[window][selected_time])
        else:
            score, gap, encounter = None, None, None
        proposals.append(dict(index=index, target_speed_mps=float(parameters['target_speed_mps']),
            parameters=parameters, score=score, predicted_separation_m=gap,
            predicted_encounter_s=encounter, actual_ego_forecast_frames=horizon))
    usable = [value for value in proposals if value['score'] is not None]
    chosen = min(usable, key=lambda value: (value['score'], value['index'])) if usable else proposals[0]
    result['parameters'] = chosen['parameters']
    report = dict(version='interaction_timing_v11', family=family,
        actor_and_start_unchanged=True, recorded_future_used=False,
        forecast_source='actual_current_ego_planner_detached', ego_horizon_extrapolated=False,
        extra_planner_queries=0, extra_closed_loop_trials=0, analytic_proposals=proposals,
        selected_index=chosen['index'], selected_target_speed_mps=chosen['target_speed_mps'],
        status='causal_initial_speed_profile_selected' if usable else 'no_interaction_window_in_actual_forecast',
        validity='reference screening only; execution and contacts independently audited',
        background_forecast_screened=False)
    result['interaction_timing_audit_v11'] = report
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return result
