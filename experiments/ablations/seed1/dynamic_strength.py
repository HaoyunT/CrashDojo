"""V26 pressure scheduling with an auditable hierarchical motion prior."""
import math

import numpy as np

BASE_PRESSURE = {'C1': 20., 'C2': 2., 'C3A': 2., 'C3B': 2., 'C4': 2., 'C5': 2.}


def prior_factor(table, dataset, request, context):
    evidence = request.get('geometry_evidence', {})
    params = request.get('parameters', {})
    if (request['family'] != 'C1' or evidence.get('longitudinal_role') != 'adversary_ahead'
            or params.get('target_speed_mps', 0.) >= params.get('initial_speed_mps', 0.) - .05):
        return 1., 'unsupported_native_mechanism_role', None
    if not context.get('complete_neighbor_history') or not context.get('eligible'):
        return 1., 'incomplete_causal_context', None
    semantic = context.get('semantic_label', 'UNKNOWN')
    road = context.get('road', {}).get('road_type', 'UNKNOWN')
    traffic = context.get('traffic_state', 'UNKNOWN')
    supported = [c for c in table['cells'] if c['supported'] and c['dataset'] == dataset]
    for level in ['joint', 'semantic_road', 'semantic', 'road', 'family_role']:
        matches = [c for c in supported if c['level'] == level
                   and (c['semantic_label'] == 'ALL' or c['semantic_label'] == semantic)
                   and (c['road_type'] == 'ALL' or c['road_type'] == road)
                   and (c['traffic_state'] == 'ALL' or c['traffic_state'] == traffic)]
        if matches:
            cell = matches[0]
            return cell['relative_observed_intensity_factor'], 'supported_' + level, cell
    return 1., 'unsupported_domain_context', None


def pressure(base, scale, beta, prior, relative_position, relative_velocity, radius):
    values = np.r_[base, scale, beta, prior, relative_position, relative_velocity, radius]
    if not np.isfinite(values).all() or base < 0 or scale <= 0 or beta < 0 or prior <= 0:
        raise ValueError('Nonfinite or invalid pressure scheduling inputs')
    relative_position, relative_velocity = np.asarray(relative_position), np.asarray(relative_velocity)
    vv = float(relative_velocity @ relative_velocity)
    t = float(np.clip(-float(relative_position @ relative_velocity) / max(vv, 1e-8), 0., 4.))
    clearance = float(np.linalg.norm(relative_position + t * relative_velocity) - radius)
    risk = math.exp(-max(0., clearance) / 8.) * (.5 + .5 * math.exp(-t / 4.))
    factor = math.exp(beta * (1. - 2. * risk))
    value = float(np.clip(base * scale * prior * factor, 0., base * 16.))
    return dict(pressure=value, risk=risk, closest_approach_s=t,
                predicted_disc_clearance_m=clearance, temporal_factor=factor,
                base_pressure=base, calibration_scale=scale, beta=beta, prior_factor=prior)


def observed_pair(obs, target, ego, dt):
    def array(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    transforms = array(obs['world_from_agent'])
    history = array(obs['history_positions'])
    available = array(obs['history_availabilities']).astype(bool)
    extent = array(obs['extent'])
    velocities = []
    for actor in [target, ego]:
        good = np.flatnonzero(available[actor])
        if len(good) < 3 or good[-1] != history.shape[1] - 1:
            raise ValueError('Current and historical positions required for causal pressure')
        first, last = good[max(0, len(good) - 6)], good[-1]
        if not available[actor, first:last+1].all():
            raise ValueError('Causal pressure cannot bridge missing history')
        local = (history[actor, last] - history[actor, first]) / ((last - first) * dt)
        velocities.append(transforms[actor, :2, :2] @ local)
    return (transforms[target, :2, 2] - transforms[ego, :2, 2],
            velocities[0] - velocities[1],
            .5 * float(np.linalg.norm(extent[target, :2]) + np.linalg.norm(extent[ego, :2])))


def install(module):
    original = module.FiveClassGuide

    class DynamicFiveClassGuide(original):
        def prepare(self, obs, ego_action, target, ego, elapsed, dt):
            super().prepare(obs, ego_action, target, ego, elapsed, dt)
            config = self.request.get('dynamic_strength_v26')
            if not config or config['mode'] == 'fixed_reference':
                return
            relative, velocity, radius = observed_pair(obs, target, ego, dt)
            update = pressure(BASE_PRESSURE[self.request['family']], config['scale'],
                              config['beta'], config['prior_factor'], relative, velocity, radius)
            self.request['weights']['pressure'] = update['pressure']
            import json
            update.update(elapsed_s=float(elapsed), recorded_future_used=False,
                          inputs='current world poses and last 0.5s observed positions',
                          prior_reason=config['prior_reason'],
                          prior_support_level=config.get('prior_support_level'),
                          prior_events=config.get('prior_events', 0),
                          prior_scenes=config.get('prior_scenes', 0),
                          prior_logs=config.get('prior_logs', 0),
                          prior_fallback_reason=config.get('prior_fallback_reason'),
                          calibration_state=config['calibration_state'],
                          context_conditioning_v26=self.request['context_conditioning_v26'])
            with self.audit_path.with_name('dynamic_strength_steps.jsonl').open('a') as stream:
                stream.write(json.dumps(update, allow_nan=False) + '\n')

    module.FiveClassGuide = DynamicFiveClassGuide
