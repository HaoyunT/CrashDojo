import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parent
name = 'ccdiff.evaluation.five_class_guidance'
if name not in sys.modules:
    spec = importlib.util.spec_from_file_location(name, HERE / 'five_class_guidance.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

from execution_completion import boundary_return, phased_progress, phased_speed, prepare_request, transition


class ExecutionCompletionTest(unittest.TestCase):
    def test_phased_speed_and_integrated_progress_agree(self):
        p = dict(initial_speed_mps=2., speed_phases_v10=[
            dict(start=.2, duration=3., v0=2., v1=7.),
            dict(start=4., duration=2.8, v0=7., v1=0.)])
        t = torch.linspace(.01, 9.9, 1000, dtype=torch.float64)
        epsilon = 1e-5
        derivative = (phased_progress(t + epsilon, p, 0.) - phased_progress(t - epsilon, p, 0.)) / (2 * epsilon)
        torch.testing.assert_close(derivative, phased_speed(t, p), atol=1e-7, rtol=1e-7)
        self.assertAlmostEqual(float(phased_progress(torch.tensor([10.], dtype=torch.float64), p, 0.)),
                               .4 + 4.5 * 3. + .8 * 7. + 3.5 * 2.8)

    def test_transition_respects_quintic_acceleration_and_jerk(self):
        limits = dict(max_acceleration_mps2=3., max_braking_mps2=6.)
        for a, b in [(2., 9.), (9., 0.)]:
            duration = transition(a, b, limits)
            self.assertLessEqual(1.875 * abs(b - a) / duration,
                                 .85 * limits['max_acceleration_mps2' if b > a else 'max_braking_mps2'] + 1e-8)
            self.assertLessEqual(5.774 * abs(b - a) / duration**2, 7. + 1e-8)

    def test_body_relative_return_leaves_clearance(self):
        context = dict(initial_route_s=0., route=torch.tensor([[0., 0.], [100., 0.]], dtype=torch.float64),
                       boundary=torch.tensor([[0., 1.], [100., 1.]], dtype=torch.float64),
                       boundary_source_sign=-1., extent=torch.tensor([4., 2.], dtype=torch.float64))
        parameters = dict(initial_speed_mps=2., speed_transition_s=2.)
        returned = boundary_return(torch.tensor([1., 2., 3.], dtype=torch.float64), context, parameters)
        torch.testing.assert_close(returned, torch.full_like(returned, -.12))

    def test_merge_profile_stops_in_verified_native_lane(self):
        request = dict(family='C5', parameters=dict(initial_speed_mps=2., target_speed_mps=8.,
            speed_transition_s=3., arrival_route_s_m=25., arrival_s=4., t0=.4, centerline_transition_s=3.),
            engineering_limits=dict(max_acceleration_mps2=3., max_braking_mps2=6.),
            native_task_geometry_v10=dict(receiving_lane_wkt='POLYGON ((25 -2, 45 -2, 45 2, 25 2, 25 -2))'))
        context = dict(initial_route_s=0., route=torch.tensor([[0., 0.], [100., 0.]], dtype=torch.float64),
                       extent=torch.tensor([4., 2.], dtype=torch.float64))
        with tempfile.TemporaryDirectory() as temp:
            effective = prepare_request(request, context, Path(temp) / 'audit.json')
        self.assertEqual(effective['execution_completion_audit_v10']['status'], 'merge_then_brake_inside_receiving_lane')
        goal = effective['execution_completion_audit_v10']['goal_route_s_m']
        final = phased_progress(torch.tensor([10.], dtype=torch.float64), effective['parameters'], 0.)
        self.assertAlmostEqual(float(final), goal, places=6)
        self.assertEqual(float(phased_speed(torch.tensor([10.]), effective['parameters'])), 0.)


if __name__ == '__main__':
    unittest.main()
