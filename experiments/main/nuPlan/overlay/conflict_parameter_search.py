"""Choose one causal action template before generating the fixed five FM samples."""

import copy
import hashlib
import json
import math

import torch


def transition_duration(v0, target, limits, old):
    delta = abs(target - v0)
    acceleration = .9 * limits['max_acceleration_mps2' if target >= v0 else 'max_braking_mps2']
    return max(float(old), 1.875 * delta / acceleration,
               math.sqrt(5.774 * delta / 8.))


def parameter_options(request):
    original = copy.deepcopy(request['parameters'])
    family, limits = request['family'], request['engineering_limits']
    options = [original]
    if family == 'C1':
        v0 = original['initial_speed_mps']
        role = request['geometry_evidence'].get('longitudinal_role')
        if role not in ('adversary_ahead', 'adversary_behind'):
            return options, 'unknown_longitudinal_role_no_parameter_expansion'
        ahead = role == 'adversary_ahead'
        if ahead:
            if v0 < .5:
                return options, 'static_lead_has_no_active_braking_room'
            proposals = [(0., .2), (0., .8), (.35 * v0, .2), (.15 * v0, 1.2)]
        else:
            proposals = [(v0 + 2., .2), (v0 + 4., .2), (v0 + 6., .6), (v0 + 8., .2)]
        for target, onset in proposals:
            target = min(limits['max_speed_mps'], max(0., target))
            if abs(target - v0) < .5:
                continue
            p = copy.deepcopy(original)
            p.update(target_speed_mps=target, speed_t0=onset)
            p['speed_transition_s'] = transition_duration(v0, target, limits, 2.)
            if onset + p['speed_transition_s'] <= 8.5:
                options.append(p)
    elif family == 'C2':
        displacement = abs(original['delta_d_m'])
        minimum = max(math.sqrt(5.774 * displacement / (.9 * limits['max_lateral_mps2'])),
                      (60 * displacement / 8.) ** (1 / 3))
        for onset, scale in [(0., 1.), (.4, .85), (.8, 1.), (.2, 1.15)]:
            p = copy.deepcopy(original)
            p.update(t0=onset, T_lc_s=max(minimum, original['T_lc_s'] * scale))
            options.append(p)
    elif family in ('C3A', 'C3B'):
        for onset, scale in [(0., 1.), (.4, .9), (.8, 1.), (.2, 1.1)]:
            p = copy.deepcopy(original)
            p.update(t0=onset, T_in_s=original['T_in_s'] * scale,
                     T_out_s=original['T_out_s'] * scale)
            options.append(p)
    elif family in ('C4', 'C5'):
        for scale in (.8, 1., 1.2, 1.4):
            p = copy.deepcopy(original)
            target = min(limits['max_speed_mps'], max(.5, original['target_speed_mps'] * scale))
            p.update(target_speed_mps=target, speed_t0=.2)
            p['speed_transition_s'] = transition_duration(p['initial_speed_mps'], target, limits, 2.)
            options.append(p)
    return options[:5], 'parameter_proposals_available'


def choose_request(request, context, audit_path):
    from ccdiff.evaluation.five_class_guidance import (
        boundary_intrusion_offset, reference_route_progress, route_frame_at_s,
        sat_separation, smooth, speed_reference)
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from available_geometry import box

    options, status = parameter_options(request)
    effective = copy.deepcopy(request)
    records = []
    if status == 'static_lead_has_no_active_braking_room':
        # Preserve the scene and stationary obstacle, but do not pressure a
        # nonnegative-speed actor backwards to rescue a degenerate request.
        effective['weights']['pressure'] = 0.
    times = torch.arange(1, context['ego_pos'].shape[0] + 1,
                         dtype=context['route'].dtype, device=context['route'].device) * context['dt']
    corridor = unary_union([Polygon(p) for p in request['corridor_world_polygons']])
    best = None
    for index, parameters in enumerate(options):
        candidate = copy.deepcopy(request)
        candidate['parameters'] = parameters
        family = request['family']
        if family in ('C4', 'C5'):
            target_s = parameters['arrival_route_s_m']
            end = float(reference_route_progress(times[-1:], parameters, context['initial_route_s'])[0])
            if end < target_s:
                records.append(dict(index=index, feasible=False, reason='event_not_reachable_in_planner_horizon', parameters=parameters))
                continue
            lo, hi = 0., float(times[-1])
            for _ in range(40):
                middle = (lo + hi) / 2
                value = float(reference_route_progress(times.new_tensor([middle]), parameters, context['initial_route_s'])[0])
                if value < target_s:
                    lo = middle
                else:
                    hi = middle
            parameters['arrival_s'] = (lo + hi) / 2
            candidate['interaction_window_s'] = [max(.1, parameters['arrival_s'] - 1.5),
                                                 min(float(times[-1]), parameters['arrival_s'] + 1.5)]
        station = reference_route_progress(times, parameters, context['initial_route_s'])
        center, tangent = route_frame_at_s(station, context['route'])
        offset = torch.full_like(times, parameters.get('initial_offset_m', 0.))
        start = parameters.get('t0', 0.)
        if family in ('C4', 'C5') and candidate.get('center_receiving_route'):
            offset = offset * (1 - smooth((times - start) / parameters['centerline_transition_s']))
        if family == 'C2':
            offset = offset + parameters['delta_d_m'] * smooth((times - start) / parameters['T_lc_s'])
        elif family in ('C3A', 'C3B'):
            amplitude = parameters['A_m']
            if candidate.get('boundary_relative_template'):
                amplitude = boundary_intrusion_offset(times, context, parameters) - parameters['initial_offset_m']
            returned = 0. if family == 'C3A' else parameters['return_offset_m'] - parameters['initial_offset_m']
            offset = offset + amplitude * smooth((times - start) / parameters['T_in_s'])
            offset = offset + (returned - amplitude) * smooth(
                (times - start - parameters['T_in_s'] - parameters['T_hold_s']) / parameters['T_out_s'])
        normal = torch.stack((-tangent[:, 1], tangent[:, 0]), -1)
        position = center + offset[:, None] * normal
        initial = context['transform'][:2, 2]
        velocity = torch.diff(torch.cat((initial[None], position)), dim=0) / context['dt']
        heading = torch.atan2(velocity[:, 1], velocity[:, 0])
        stationary = velocity.norm(dim=-1) < .01
        heading[stationary] = torch.atan2(tangent[stationary, 1], tangent[stationary, 0])
        separation = sat_separation(position[None], heading[None], context['ego_pos'],
                                    context['ego_yaw'], context['extent'], context['ego_extent'])[0]
        window = (times >= candidate['interaction_window_s'][0]) & (times <= candidate['interaction_window_s'][1])
        if candidate.get('contact_phase_window'):
            if family == 'C2':
                window &= times >= start + .5 * parameters['T_lc_s']
            elif family in ('C3A', 'C3B'):
                window &= ((times >= start + .65 * parameters['T_in_s'])
                           & (times <= start + parameters['T_in_s'] + parameters['T_hold_s'] + .25 * parameters['T_out_s']))
        coverage = min(box(p, y, context['extent'].cpu().numpy()).intersection(corridor).area /
                       box(p, y, context['extent'].cpu().numpy()).area
                       for p, y in zip(position.cpu().numpy(), heading.cpu().numpy()))
        feasible = bool(window.any()) and coverage >= .98
        speed = speed_reference(times, parameters)
        delta_speed = torch.diff(torch.cat((speed[:1], speed))) / context['dt']
        feasible &= bool((speed >= 0).all() and (speed <= candidate['engineering_limits']['max_speed_mps']).all())
        feasible &= bool(delta_speed.max() <= .9 * candidate['engineering_limits']['max_acceleration_mps2'] + .01)
        feasible &= bool(delta_speed.min() >= -.9 * candidate['engineering_limits']['max_braking_mps2'] - .01)
        cost = None
        if feasible:
            cost = float((torch.relu(separation[window] + .05).square() / 4 + times[window]).min())
            if best is None or cost < best[0]:
                best = cost, index, candidate
        records.append(dict(index=index, feasible=bool(feasible), score=cost,
                            sampled_body_coverage_min=coverage, parameters=parameters,
                            interaction_window_s=candidate['interaction_window_s']))
    if best is not None and status != 'static_lead_has_no_active_braking_room':
        effective = best[2]
    report = dict(version='causal_parameter_search_v9', status=status,
                  selected_index=best[1] if best is not None else None,
                  options=records, screening_only_not_closed_loop_reachability_proof=True,
                  scene_retained=True, actor_and_start_unchanged=True,
                  additional_planner_queries=0, additional_closed_loop_trials=0,
                  fm_candidate_budget=5, parameter_screening_count=len(options),
                  information_source='current_history_native_map_actual_ego_planner_prediction',
                  recorded_future_used=False)
    effective['parameter_search_audit'] = report
    audit_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    body = json.dumps(effective, sort_keys=True, allow_nan=False).encode()
    effective['_effective_sha256'] = hashlib.sha256(body).hexdigest()
    audit_path.with_name('effective_request.json').write_text(json.dumps(effective, indent=2, allow_nan=False))
    return effective
