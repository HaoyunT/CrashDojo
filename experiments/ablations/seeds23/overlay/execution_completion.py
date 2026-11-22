"""Causal geometric completion targets and native-control warm starts."""
import copy
import hashlib
import json
import math

import torch


def transition(v0, v1, limits):
    change = abs(v1 - v0)
    acceleration = limits['max_acceleration_mps2'] if v1 >= v0 else limits['max_braking_mps2']
    return max(1.5, 1.875 * change / (.85 * acceleration), math.sqrt(5.774 * change / 7.))


def phased_speed(times, parameters):
    from ccdiff.evaluation.five_class_guidance import smooth
    result = torch.full_like(times, parameters['initial_speed_mps'])
    for phase in parameters['speed_phases_v10']:
        result = result + (phase['v1'] - phase['v0']) * smooth((times - phase['start']) / phase['duration'])
    return result


def phased_progress(times, parameters, initial):
    result = initial + parameters['initial_speed_mps'] * times
    for phase in parameters['speed_phases_v10']:
        u = ((times - phase['start']) / phase['duration']).clamp(0, 1)
        integral = phase['duration'] * (2.5 * u**4 - 3 * u**5 + u**6)
        integral = integral + torch.relu(times - phase['start'] - phase['duration'])
        result = result + (phase['v1'] - phase['v0']) * integral
    return result


def boundary_return(times, context, parameters):
    from ccdiff.evaluation.five_class_guidance import reference_route_progress, route_frame_at_s, route_coordinates
    station = reference_route_progress(times, parameters, context['initial_route_s'])
    center, tangent = route_frame_at_s(station, context['route'])
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), -1)
    bs, bd = route_coordinates(center, context['boundary'])
    _, bt = route_frame_at_s(bs, context['boundary'])
    bn = torch.stack((-bt[:, 1], bt[:, 0]), -1)
    sign = context['boundary_source_sign']
    derivative = sign * (normal * bn).sum(-1)
    if bool((derivative.abs() < .5).any()):
        raise ValueError('Boundary return requires locally parallel map geometry')
    support = (tangent * bn).sum(-1).abs() * context['extent'][0] / 2
    support = support + (normal * bn).sum(-1).abs() * context['extent'][1] / 2
    return (support + .12 - sign * bd) / derivative


def offset_reference(times, context, request):
    from ccdiff.evaluation.five_class_guidance import smooth, boundary_intrusion_offset
    p, family = request['parameters'], request['family']
    d0, start = p.get('initial_offset_m', 0.), p.get('t0', 0.)
    result = torch.full_like(times, d0)
    if family == 'C2':
        result = result + p['delta_d_m'] * smooth((times - start) / p['T_lc_s'])
    elif family == 'C3B':
        amplitude = boundary_intrusion_offset(times, context, p) - d0
        returned = boundary_return(times, context, p) - d0
        result = result + amplitude * smooth((times - start) / p['T_in_s'])
        result = result + (returned - amplitude) * smooth((times - start - p['T_in_s'] - p['T_hold_s']) / p['T_out_s'])
    elif family == 'C5':
        result = result * (1 - smooth((times - start) / p['centerline_transition_s']))
    return result


def prepare_request(request, context, path):
    from shapely import wkt
    from available_geometry import box
    from ccdiff.evaluation.five_class_guidance import route_frame_at_s

    effective = copy.deepcopy(request)
    p, family = effective['parameters'], effective['family']
    report = dict(version='execution_completion_v10', family=family, actor_and_start_unchanged=True,
                  recorded_future_used=False, extra_planner_queries=0, extra_closed_loop_trials=0,
                  native_geometry_only=True, screening_not_closed_loop_guarantee=True)
    if family == 'C2':
        # The quintic lateral peak speed must fit a moderate heading angle.
        average_speed = max(.75, (p['initial_speed_mps'] + p['target_speed_mps']) / 2)
        required = 1.875 * abs(p['delta_d_m']) / (.65 * average_speed)
        required = max(required, (60 * abs(p['delta_d_m']) / 7.) ** (1 / 3))
        duration = max(p['T_lc_s'], required)
        report.update(status='causal_lateral_duration_selected', previous_duration_s=p['T_lc_s'], selected_duration_s=duration)
        if p['t0'] + duration + .5 <= 9.9:
            p['T_lc_s'] = duration
        else:
            report['status'] = 'duration_exceeds_episode_action_budget'
        effective['interaction_window_s'] = [p['t0'] + .5 * p['T_lc_s'], min(9.4, p['t0'] + p['T_lc_s'] + 2.)]
    elif family == 'C3B':
        report.update(status='body_relative_return_target', source_side_clearance_m=.12,
                      legacy_constant_return_offset_m=p['return_offset_m'])
    elif family == 'C5':
        lane = wkt.loads(effective['native_task_geometry_v10']['receiving_lane_wkt']).buffer(.10)
        first = max(context['initial_route_s'], p['arrival_route_s_m'] + float(context['extent'][0]) / 2 + .1)
        stations = torch.arange(first, first + 35., .2, dtype=context['route'].dtype, device=context['route'].device)
        center, tangent = route_frame_at_s(stations, context['route'])
        heading = torch.atan2(tangent[:, 1], tangent[:, 0])
        coverage = [box(x, y, context['extent'].cpu().numpy()).intersection(lane).area /
                    box(x, y, context['extent'].cpu().numpy()).area
                    for x, y in zip(center.cpu().numpy(), heading.cpu().numpy())]
        valid = [i for i, value in enumerate(coverage) if value >= .995]
        report['fully_inside_post_event_stations'] = len(valid)
        proposals = []
        if valid:
            # Stay inside one verified contiguous receiving-lane interval.
            start_index = valid[0]
            end_index = start_index
            while end_index + 1 in valid:
                end_index += 1
            # A stopping region is an interval, not a fixed midpoint. Use its
            # early interior so braking can finish before the terminal hold.
            goal_index = min(end_index, start_index + max(1, (end_index - start_index) // 4))
            goal = float(stations[goal_index])
            distance = goal - context['initial_route_s']
            initial = p['initial_speed_mps']
            for peak in (6., 7., 8., 9., 10.):
                up = transition(initial, peak, effective['engineering_limits'])
                down = transition(peak, 0., effective['engineering_limits'])
                cruise = (distance - initial * .2 - (initial + peak) * up / 2 - peak * down / 2) / peak
                end = .2 + up + max(0., cruise) + down
                feasible = cruise >= 0 and end <= 9.3
                proposals.append(dict(peak_mps=peak, up_s=up, down_s=down,
                                      cruise_s=cruise, stopped_at_s=end, feasible=feasible))
            accepted = [value for value in proposals if value['feasible']]
            if accepted:
                chosen = min(accepted, key=lambda value: value['stopped_at_s'])
                p['speed_phases_v10'] = [dict(start=.2, duration=chosen['up_s'], v0=initial, v1=chosen['peak_mps']),
                    dict(start=.2 + chosen['up_s'] + chosen['cruise_s'], duration=chosen['down_s'], v0=chosen['peak_mps'], v1=0.)]
                p['target_speed_mps'] = 0.
                p['speed_transition_s'] = chosen['stopped_at_s'] - .2
                lo, hi = 0., chosen['stopped_at_s']
                for _ in range(40):
                    middle = (lo + hi) / 2
                    if float(phased_progress(stations.new_tensor([middle]), p, context['initial_route_s'])[0]) < p['arrival_route_s_m']:
                        lo = middle
                    else:
                        hi = middle
                p['arrival_s'] = (lo + hi) / 2
                effective['interaction_window_s'] = [max(.1, p['arrival_s'] - 1.), min(9.8, chosen['stopped_at_s'] + .5)]
                report.update(status='merge_then_brake_inside_receiving_lane', goal_route_s_m=goal,
                              arrival_s=p['arrival_s'], chosen=chosen)
            else:
                report['status'] = 'no_acceleration_braking_profile_within_episode'
        else:
            report['status'] = 'no_verified_post_event_stopping_region_on_route'
        report['profile_proposals'] = proposals
    effective['execution_completion_audit_v10'] = report
    effective['_effective_sha256'] = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    path.with_name('effective_request.json').write_text(json.dumps(effective, indent=2, allow_nan=False))
    return effective


def completion_loss(trajectory, context, request):
    from ccdiff.evaluation.five_class_guidance import world_pose, route_coordinates, band
    T = trajectory.shape[1]
    times = context['elapsed'] + torch.arange(1, T + 1, dtype=trajectory.dtype, device=trajectory.device) * context['dt']
    position, yaw = world_pose(trajectory, context['transform'])
    _, offset = route_coordinates(position, context['route'])
    reference = offset_reference(times, context, request)
    error = ((offset - reference) / .2).square()
    # The executed prefix and worst error matter; a good unexecuted tail is insufficient.
    cost = 5 * error[:, :5].mean(-1) + error.mean(-1) + error.amax(-1)
    p, family = request['parameters'], request['family']
    if family == 'C2':
        settled = times >= p['t0'] + p['T_lc_s']
        if bool(settled.any()):
            residual = torch.relu((offset[:, settled] - reference[settled]).abs() - .15).square() / .04
            cost = cost + 10 * residual.amax(-1) + 5 * residual.mean(-1)
    if family == 'C3B':
        u = torch.stack((yaw.cos(), yaw.sin()), -1) * context['extent'][0] / 2
        v = torch.stack((-yaw.sin(), yaw.cos()), -1) * context['extent'][1] / 2
        corners = torch.stack((position + u + v, position + u - v, position - u - v, position - u + v), -2)
        _, side = route_coordinates(corners, context['boundary'])
        depth = (-context['boundary_source_sign'] * side).amax(-1)
        cost = cost + 5 * torch.relu(depth - p['z_max_m'] + .10).square().amax(-1) / .04
        returned = times >= p['t0'] + p['T_in_s'] + p['T_hold_s'] + p['T_out_s']
        if bool(returned.any()):
            cost = cost + 10 * torch.relu(depth[:, returned] + .06).square().amax(-1) / .04
    heading = torch.cat((trajectory.new_zeros((len(trajectory), 1)), trajectory[:, :-1, 3]), -1)
    velocity = torch.diff(torch.cat((trajectory.new_zeros((len(trajectory), 1, 2)), trajectory[..., :2]), 1), dim=1) / context['dt']
    signed = (velocity * torch.stack((heading.cos(), heading.sin()), -1)).sum(-1)
    cost = cost + 100 * torch.relu(-signed).square().amax(-1)
    return cost


def tracking_seed(model, base, state, context, request):
    from ccdiff.evaluation.five_class_guidance import route_coordinates, route_frame_at_s, speed_reference
    if request['family'] not in ('C2', 'C3B', 'C5'):
        return base
    physical = model.descale_traj(base, [4, 5]).detach().clone()
    initial = state.reshape(-1, 4)[0]
    xy, speed, heading = initial[:2].clone(), initial[2].clone(), initial[3].clone()
    dt, tf = context['dt'], context['transform']
    times = context['elapsed'] + torch.arange(1, base.shape[1] + 2, dtype=base.dtype, device=base.device) * dt
    offsets = offset_reference(times, context, request)
    offset_speed = torch.diff(offsets) / dt
    reference_speed = speed_reference(times, request['parameters'])
    world_angle = torch.atan2(tf[1, 0], tf[0, 0])
    previous_acceleration = torch.zeros_like(speed)
    for i in range(base.shape[1]):
        world = xy @ tf[:2, :2].T + tf[:2, 2]
        station, lateral = route_coordinates(world, context['route'])
        _, tangent = route_frame_at_s(station + speed.abs().clamp_min(.5) * .4, context['route'])
        desired_angle = torch.atan2(tangent[1], tangent[0]) - world_angle
        desired_angle = desired_angle + torch.atan2(offset_speed[i] + 1.5 * (offsets[i] - lateral), speed.abs().clamp_min(.75))
        difference = torch.atan2(torch.sin(desired_angle - heading), torch.cos(desired_angle - heading))
        bound = torch.minimum(model.dyn.max_steer * speed.abs(), model.dyn.max_yawvel / speed.abs().clamp_min(.1)).clamp_min(.1)
        omega = (difference / .45).clamp(-bound, bound)
        acceleration = ((reference_speed[i] - speed) / .35).clamp(-.85 * request['engineering_limits']['max_braking_mps2'], .85 * request['engineering_limits']['max_acceleration_mps2'])
        acceleration = torch.minimum(torch.maximum(acceleration, previous_acceleration - .7), previous_acceleration + .7)
        next_speed = (speed + acceleration * dt).clamp(min=0., max=request['engineering_limits']['max_speed_mps'])
        xy = xy + (speed + next_speed) / 2 * torch.stack((heading.cos(), heading.sin())) * dt
        heading = heading + omega * dt
        speed, previous_acceleration = next_speed, acceleration
        physical[0, i, 0], physical[0, i, 1] = acceleration, omega
    return model.scale_traj(physical, [4, 5]).detach()
