"""Executable manuscript settings and immutable actor/mechanism requests.

This profile defines new implementation runs. It cannot reconstruct historical
campaign inputs, source revisions, candidate evidence, or experiment outcomes.
"""
import copy
import hashlib
import json
import math
from pathlib import Path


PROFILE_PATH = Path(__file__).resolve().parents[2] / 'configs/paper_runtime.json'
REWRITE_FLAGS = ('causal_parameter_search_v9', 'interaction_timing_v11',
                 'execution_completion_v10')
FROZEN_FIELDS = (
    'scene_id', 'adversary_vehicle_id', 'start_step_10hz', 'family', 'parameters',
    'interaction_window_s', 'engineering_limits', 'route_world_xy',
    'corridor_world_polygons', 'route_offset_bounds_m', 'geometry_status',
    'geometry_evidence', 'selection_mode', 'route_progress_guidance',
    'center_receiving_route', 'boundary_relative_template',
    'phase_specific_intrusion', 'context_prior_v26', 'context_conditioning_v26',
    'dynamic_strength_v26',
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_profile():
    return json.loads(PROFILE_PATH.read_text())


class FrozenRequest:
    """Allow live pressure updates, but never change the frozen challenge."""
    def __init__(self, request):
        self.snapshot = {key: copy.deepcopy(request.get(key)) for key in FROZEN_FIELDS}
        self.sha256 = digest(self.snapshot)
        self.check(request)

    def check(self, request):
        enabled = [key for key in REWRITE_FLAGS if request.get(key)]
        if enabled:
            raise ValueError('The paper freezes request parameters before rollout; '
                             'disable runtime rewrites: ' + ', '.join(enabled))
        changed = [key for key in FROZEN_FIELDS
                   if canonical(request.get(key)) != canonical(self.snapshot[key])]
        if changed:
            raise ValueError('Frozen request changed during rollout: ' + ', '.join(changed))


def _require_numbers(mapping, keys, label):
    for key in keys:
        value = mapping.get(key)
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value < 0):
            raise ValueError('Supply the original request setting ' + label + '.' + key)


def prepare_paper_request(request, dataset, arm='A0', steps=None, smoke=False):
    """Apply declared constants; preserve geometry and unspecified request values.

    ``arm`` uses internal labels. Missing per-request penalty weights/margins are
    errors, not guessed replacements for unavailable frozen inputs.
    """
    profile = read_profile()
    dataset = dataset.lower()
    if dataset not in profile['domains'] or arm not in profile['paper_arms'].values():
        raise ValueError('Unknown paper dataset or internal ablation arm')
    result = copy.deepcopy(request)
    before = copy.deepcopy(result)
    ablation = 'ablation_v26' in result
    if ((ablation and result['ablation_v26'].get('arm') != arm)
            or (not ablation and arm != 'A0')):
        raise ValueError('Internal arm must match the explicit ablation request')
    expected_steps = (profile['ablation_steps'] if ablation else
                      profile['domains'][dataset]['main_steps'])
    if steps is not None and not smoke and steps != expected_steps:
        raise ValueError('Paper rollout horizon must be %d frames' % expected_steps)
    if result.get('family') == 'NONE':
        raise ValueError('The reported paper cohorts require an explicit target request')
    if result.get('family') not in ('C1', 'C2', 'C3A', 'C3B', 'C4', 'C5'):
        raise ValueError('Unknown paper request family')
    original_identity = {key: copy.deepcopy(result.get(key)) for key in FROZEN_FIELDS
                         if key != 'engineering_limits'}
    deviation = copy.deepcopy(result.get('deviation_guidance') or {})
    _require_numbers(deviation, ('ade_weight', 'fde_weight'), 'deviation_guidance')
    deviation.update(profile['deviation'])
    background = copy.deepcopy(result.get('non_target_avoidance') or {})
    _require_numbers(background, ('margin_m', 'prefix_margin_m', 'uncertainty_cap_m'),
                     'non_target_avoidance')
    if not 0 < background['uncertainty_cap_m'] <= 2:
        raise ValueError('A bounded positive background uncertainty cap is required')
    background.update(enabled=True, weight=profile['weights']['background'],
                      forecast='observed_cv_with_velocity_spread_v1')
    generic = arm == 'A4'
    unguided = arm == 'U'
    qualified = None
    if not generic and not unguided:
        qualified = copy.deepcopy(result.get('qualified_contact_v2') or {})
        _require_numbers(qualified, ('prefix_safety_priority',), 'qualified_contact_v2')
        if qualified['prefix_safety_priority'] <= 0:
            raise ValueError('Prefix safety priority must be positive')
        qualified.update(profile['contact'])
    result.update(copy.deepcopy(profile['request_controls']))
    result['engineering_limits'] = copy.deepcopy(profile['engineering_limits'])
    result['deviation_guidance'] = deviation
    result['non_target_avoidance'] = background
    result['qualified_contact_v2'] = qualified
    result['conditional_distribution_guidance'] = {'enabled': False}
    result['ablation_generic_collision'] = generic
    result['weights'] = dict(result.get('weights', {}),
        action=0. if generic or unguided else profile['weights'][
            'longitudinal_action' if result['family'] == 'C1' else 'other_action'],
        prior=profile['weights']['prior'], constraint=profile['weights']['constraint'])
    if generic or unguided:
        result['contact_phase_window'] = False
        result['multimodal_obstacle_seed'] = False
    if unguided:
        result['weights']['pressure'] = 0.
    dynamic = result.get('dynamic_strength_v26', {})
    domain = profile['domains'][dataset]
    for key, expected in [('scale', domain['pressure_scale']), ('beta', domain['pressure_beta'])]:
        if dynamic.get(key) != expected:
            raise ValueError('Frozen domain calibration disagrees with paper: ' + key)
    # Geometry, clock, mechanism and context are never synthesized by the profile.
    for key, value in original_identity.items():
        if canonical(result.get(key)) != canonical(value):
            raise AssertionError('Profile changed request identity: ' + key)
    result['paper_runtime_contract'] = dict(revision=profile['revision'],
        profile_sha256=digest(profile), dataset=dataset, internal_arm=arm,
        expected_steps=expected_steps, historical_results_verified=False)
    frozen = FrozenRequest(result)
    changed = {key: {'before': before.get(key), 'after': result.get(key)}
               for key in sorted(set(before) | set(result))
               if canonical(before.get(key)) != canonical(result.get(key))}
    return result, dict(profile_sha256=digest(profile), input_request_sha256=digest(before),
        effective_request_sha256=digest(result), frozen_identity_sha256=frozen.sha256,
        changes=changed, historical_results_verified=False)


def validate_generation(request, model, endpoint, samples, dt):
    if not request.get('paper_runtime_contract'):
        return
    profile = read_profile()
    if request['paper_runtime_contract']['profile_sha256'] != digest(profile):
        raise ValueError('Paper runtime profile changed after request preparation')
    expected = profile['generation']
    if (model.n_flow_steps != expected['flow_steps'] or samples != expected['candidates']
            or endpoint.shape[-2] != expected['future_frames']
            or not math.isclose(dt, expected['dt'], rel_tol=0., abs_tol=1e-10)):
        raise ValueError('Generation budget disagrees with the frozen paper profile')


class PaperExecutionMixin:
    """Keep the final solver's physical candidates and their native references."""
    def initialize_contract(self):
        self._request_contract = FrozenRequest(self.request)
        self._candidate_native = self._candidate_trajectory = None
        self._candidate_elapsed = None

    def begin_replan(self):
        self._request_contract.check(self.request)
        self._candidate_native = self._candidate_trajectory = None
        self._candidate_elapsed = None

    def retain_candidates(self, native, trajectory):
        self._candidate_native = native.detach().clone()
        self._candidate_trajectory = trajectory.detach().clone()
        self._candidate_elapsed = self.context['elapsed']

    def score_candidates(self, positions, yaws, loss_function):
        import torch
        self._request_contract.check(self.request)
        native, trajectory = self._candidate_native, self._candidate_trajectory
        if (native is None or trajectory is None
                or self._candidate_elapsed != self.context['elapsed']):
            raise RuntimeError('No final-flow candidate reference for this replan')
        count, horizon = positions.shape[:2]
        if (trajectory.shape[0] != count or trajectory.shape[1] < horizon
                or yaws.shape != positions.shape[:2] or horizon < 1):
            raise ValueError('Candidate shape/order does not match final-flow output')
        candidate = trajectory[:, :horizon]
        observed = positions.to(device=candidate.device, dtype=candidate.dtype)
        angles = yaws.to(device=candidate.device, dtype=candidate.dtype)
        # Allow only decoder float32/float64 roundoff, not a transformed or
        # reordered candidate set. Preserve physical speed/actions from decoding.
        angle_error = torch.atan2(torch.sin(candidate[..., 3] - angles),
                                 torch.cos(candidate[..., 3] - angles))
        if (not torch.allclose(candidate[..., :2], observed, atol=1e-4, rtol=1e-5)
                or not bool((angle_error.abs() <= 1e-4).all())):
            raise ValueError('Candidate values/order changed after endpoint optimization')
        return loss_function(candidate, native[:, :horizon], self.context, self.request)
