"""Regression checks for immutable requests and complete execution scoring."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ccdiff.evaluation.paper_contract import (
    FrozenRequest, prepare_paper_request, validate_generation)

ROOT = Path(__file__).resolve().parents[1]
SUITES = [ROOT/'experiments/main'/s for s in ('nuScenes', 'nuPlan')]
SUITES += [ROOT/'experiments/ablations'/s for s in ('seed1', 'seeds23')]


def load_guide(suite):
    with patch.object(sys, 'path', [str(suite/'overlay'), str(suite)] + sys.path):
        spec = importlib.util.spec_from_file_location('paper_guide_' + suite.name,
                  suite/'overlay/five_class_guidance.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def request_fixture():
    # Synthetic unit-test geometry and numerical settings, not experiment data.
    return dict(scene_id='unit-scene', adversary_vehicle_id='target', start_step_10hz=30,
        family='C1', parameters=dict(initial_speed_mps=2., target_speed_mps=3.,
            initial_offset_m=0., speed_t0=0., t0=0., speed_transition_s=1.),
        mode='contact_search', interaction_window_s=[0., 2.],
        route_world_xy=[[0., 0.], [100., 0.]], route_offset_bounds_m=[-4., 4.],
        corridor_world_polygons=[[[0., -5.], [100., -5.], [100., 5.], [0., 5.], [0., -5.]]],
        geometry_evidence={}, geometry_status='native_map_verified', capability_status='test',
        engineering_limits=dict(max_speed_mps=25., max_acceleration_mps2=3.,
            max_braking_mps2=6., max_lateral_mps2=4., max_jerk_mps3=10.),
        weights=dict(action=1., pressure=0., prior=.1, constraint=0.),
        trajectory_solver='lbfgs', gradient_steps=3, last_flow_steps=1,
        normalized_action_trust=2., dynamic_strength_v26=dict(scale=.25, beta=1., prior_factor=1.),
        deviation_guidance=dict(ade_weight=1., fde_weight=1.),
        non_target_avoidance=dict(enabled=False, margin_m=.1, prefix_margin_m=.1, uncertainty_cap_m=.5),
        qualified_contact_v2=dict(prefix_safety_priority=1.),
        tracking_scales=dict(lateral_m=.35, speed_mps=.5))


def observation():
    tf = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    tf[1, 0, 2] = 8.
    return dict(world_from_agent=tf,
        extent=torch.tensor([[4., 2., 1.], [4., 2., 1.]], dtype=torch.float64))


def ego_action():
    return SimpleNamespace(positions=torch.zeros(1, 10, 2, dtype=torch.float64),
                           yaws=torch.zeros(1, 10, 1, dtype=torch.float64))


def decoder():
    return SimpleNamespace(n_flow_steps=20, dt=.1,
        dyn=SimpleNamespace(acce_bound=(-10., 8.), vbound=(0., 40.), max_steer=.5, max_yawvel=6.28),
        descale_traj=lambda x, idx:x, scale_traj=lambda x, idx:x,
        add_coeffs=torch.zeros(6), div_coeffs=torch.ones(6))


class PaperProfileTests(unittest.TestCase):
    def test_profile_replaces_legacy_controls_but_preserves_request_identity(self):
        source = request_fixture()
        source.update(causal_parameter_search_v9=True, interaction_timing_v11=True,
                      execution_completion_v10=True, select_guided_candidate=False)
        before = copy.deepcopy(source)
        result, audit = prepare_paper_request(source, 'nuScenes', steps=100)
        self.assertEqual(source, before)
        for key in ('parameters', 'interaction_window_s', 'adversary_vehicle_id',
                    'route_world_xy', 'geometry_status', 'geometry_evidence'):
            self.assertEqual(result[key], before[key])
        self.assertEqual(result['gradient_steps'], 30)
        self.assertEqual(result['normalized_action_trust'], 2.)
        self.assertTrue(result['select_guided_candidate'])
        self.assertFalse(result['execution_completion_v10'])
        self.assertEqual(result['qualified_contact_v2']['delay_cost_per_s'], 1.)
        self.assertEqual(result['weights']['prior'], .1)
        self.assertFalse(audit['historical_results_verified'])
        FrozenRequest(result).check(result)

    def test_paper_ablation_overrides_keep_common_controls(self):
        for arm in ('A0', 'A1', 'A2', 'A3', 'A4'):
            source = request_fixture(); source['ablation_v26'] = {'arm': arm}
            result, _ = prepare_paper_request(source, 'nuscenes', arm, steps=70)
            self.assertTrue(result['exact_body_prefix_gate'])
            self.assertTrue(result['executed_jerk_guidance'])
            self.assertEqual(result['non_target_avoidance']['weight'], 100.)
            if arm in ('A3', 'A4'):
                self.assertEqual(result['weights']['action'], 0.)
                self.assertIsNone(result['qualified_contact_v2'])
                self.assertFalse(result['contact_phase_window'])
            if arm == 'A4':
                self.assertEqual(result['weights']['pressure'], 0.)

    def test_unspecified_inputs_are_required_not_fabricated(self):
        source = request_fixture(); del source['deviation_guidance']['ade_weight']
        with self.assertRaisesRegex(ValueError, 'original request setting'):
            prepare_paper_request(source, 'nuscenes')

    def test_wrong_horizon_and_calibration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '100 frames'):
            prepare_paper_request(request_fixture(), 'nuscenes', steps=70)
        with self.assertRaisesRegex(ValueError, 'calibration'):
            prepare_paper_request(request_fixture(), 'nuplan', steps=70)
        source = request_fixture(); source['dynamic_strength_v26']['scale'] = 4.
        result, _ = prepare_paper_request(source, 'nuplan', steps=70)
        self.assertEqual(result['paper_runtime_contract']['expected_steps'], 70)

    def test_ablation_label_cannot_silently_select_a_different_condition(self):
        with self.assertRaisesRegex(ValueError, 'Ablation arm'):
            prepare_paper_request(request_fixture(), 'nuscenes', arm='A3')
        source = request_fixture(); source['ablation_v26'] = {'arm': 'A0'}
        with self.assertRaisesRegex(ValueError, 'Ablation arm'):
            prepare_paper_request(source, 'nuscenes', arm='A4')

    def test_cli_writes_loadable_request_and_refuses_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'input.json'; output = Path(temp)/'request.json'
            source.write_text(json.dumps(request_fixture()))
            command = [sys.executable, str(ROOT/'scripts/prepare_paper_request.py'),
                       '--request', str(source), '--dataset', 'nuscenes',
                       '--steps', '100', '--output', str(output)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            prepared = json.loads(output.read_text())
            self.assertEqual(prepared['parameters'], request_fixture()['parameters'])
            self.assertEqual(prepared['family'], 'C1')
            audit = json.loads(output.with_suffix('.audit.json').read_text())
            self.assertFalse(audit['historical_results_verified'])
            original_bytes = output.read_bytes()
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(output.read_bytes(), original_bytes)

    def test_generation_budget_is_checked_at_runtime(self):
        result, _ = prepare_paper_request(request_fixture(), 'nuscenes')
        model = decoder(); endpoint = torch.zeros(5, 2, 52, 2)
        validate_generation(result, model, endpoint, 5, .1)
        for samples, horizon, dt in [(4, 52, .1), (5, 50, .1), (5, 52, .2)]:
            with self.assertRaisesRegex(ValueError, 'Generation budget'):
                validate_generation(result, model, torch.zeros(samples, 2, horizon, 2), samples, dt)


class FrozenExecutionTests(unittest.TestCase):
    def test_nested_parameter_or_clock_change_is_rejected_but_pressure_can_change(self):
        request = request_fixture(); guard = FrozenRequest(request)
        request['weights']['pressure'] = 13.; guard.check(request)
        for field, value in [('parameters', dict(request['parameters'], target_speed_mps=7.)),
                             ('interaction_window_s', [0., 3.]),
                             ('adversary_vehicle_id', 'other')]:
            altered = copy.deepcopy(request); altered[field] = value
            with self.assertRaisesRegex(ValueError, 'Frozen request changed'):
                guard.check(altered)

    def test_full_guide_rejects_online_request_rewriters(self):
        for suite in SUITES:
            module = load_guide(suite)
            for flag in ('causal_parameter_search_v9', 'interaction_timing_v11', 'execution_completion_v10'):
                request = request_fixture(); request[flag] = True
                with self.assertRaisesRegex(ValueError, 'disable runtime rewrites'):
                    module.FiveClassGuide(request, 'unused.jsonl')

    def test_final_candidate_scores_keep_native_prior_and_reset_between_replans(self):
        torch.optim.LBFGS([torch.zeros(1, requires_grad=True)])
        for suite in SUITES:
            with self.subTest(suite=suite.name), tempfile.TemporaryDirectory() as temp:
                module = load_guide(suite); request = request_fixture()
                # Keep this numerical regression focused on tracking + prior.
                request.pop('deviation_guidance'); request['qualified_contact_v2'] = None
                guide = module.FiveClassGuide(request, Path(temp)/'guide.jsonl')
                guide.prepare(observation(), ego_action(), 0, 1, 0., .1)
                model = decoder()
                endpoint = torch.zeros(2, 2, 10, 2, dtype=torch.float64)
                endpoint[1, 0, :, 0] = -.1
                state = torch.tensor([[[0., 0., 2., 0.], [0., 0., 0., 0.]]], dtype=torch.float64).repeat(2, 1, 1)
                output = guide.project(model, endpoint, {'curr_states': state}, 2, 19)
                self.assertTrue(torch.equal(output[:, 1], endpoint[:, 1]))
                candidate = guide._candidate_trajectory
                costs, terms, _ = guide.score_candidates(candidate[..., :2], candidate[..., 3], module.loss_terms)
                self.assertGreater(float(terms['prior'].sum()), 0.)
                expected, _, _ = module.loss_terms(candidate, guide._candidate_native, guide.context, request)
                torch.testing.assert_close(costs, expected)
                old, _, _ = module.loss_terms(candidate, candidate, guide.context, request)
                torch.testing.assert_close(costs - old, .1 * terms['prior'])
                with self.assertRaisesRegex(ValueError, 'values/order changed'):
                    guide.score_candidates(candidate[..., :2] + .01, candidate[..., 3], module.loss_terms)
                before = copy.deepcopy(request['parameters'])
                guide.prepare(observation(), ego_action(), 0, 1, .5, .1)
                self.assertEqual(request['parameters'], before)
                self.assertEqual(guide.context['elapsed'], .5)
                with self.assertRaisesRegex(RuntimeError, 'No final-flow'):
                    guide.score_candidates(candidate[..., :2], candidate[..., 3], module.loss_terms)


if __name__ == '__main__':
    unittest.main()
