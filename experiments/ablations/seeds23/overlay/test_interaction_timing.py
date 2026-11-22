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
from ccdiff.evaluation.five_class_guidance import reference_route_progress, speed_reference
from interaction_timing import screen_request, speed_profile


class InteractionTimingTest(unittest.TestCase):
    def test_bounded_profile_speed_integral_and_acceleration(self):
        p = speed_profile(dict(initial_speed_mps=1., target_speed_mps=3.), 10.,
                          dict(max_speed_mps=25., max_acceleration_mps2=3.))
        p['speed_t0'] = .2
        times = torch.linspace(.001, 8., 2000, dtype=torch.float64)
        epsilon = 1e-5
        derivative = (reference_route_progress(times + epsilon, p, 0.) -
                      reference_route_progress(times - epsilon, p, 0.)) / (2 * epsilon)
        torch.testing.assert_close(derivative, speed_reference(times, p), atol=1e-6, rtol=1e-6)
        acceleration = torch.diff(speed_reference(times, p)) / float(times[1] - times[0])
        self.assertLessEqual(float(acceleration.max()), 2.5 + 1e-6)

    def test_rear_lane_change_screens_actual_forecast_without_extrapolation(self):
        times = torch.arange(1, 51, dtype=torch.float64) * .1
        context = dict(route=torch.tensor([[0., 0.], [200., 0.]], dtype=torch.float64),
            transform=torch.tensor([[1., 0., 0.], [0., 1., 3.5], [0., 0., 1.]], dtype=torch.float64),
            dt=.1, elapsed=0., initial_route_s=0.,
            ego_pos=torch.stack((10. + 3. * times, torch.zeros_like(times)), -1),
            ego_yaw=torch.zeros_like(times), extent=torch.tensor([4., 2.], dtype=torch.float64),
            ego_extent=torch.tensor([4., 2.], dtype=torch.float64))
        request = dict(family='C2', parameters=dict(initial_speed_mps=1., target_speed_mps=3.,
            initial_offset_m=3.5, delta_d_m=-3.5, t0=.4, T_lc_s=3.5, speed_t0=.2, speed_transition_s=2.),
            interaction_window_s=[2.15, 6.], engineering_limits=dict(max_speed_mps=25., max_acceleration_mps2=3.))
        with tempfile.TemporaryDirectory() as folder:
            result = screen_request(request, context, Path(folder) / 'audit.json')
        audit = result['interaction_timing_audit_v11']
        self.assertGreater(audit['selected_target_speed_mps'], 3.)
        self.assertEqual(len(audit['analytic_proposals']), 5)
        self.assertTrue(all(row['actual_ego_forecast_frames'] == 50 for row in audit['analytic_proposals']))
        self.assertFalse(audit['ego_horizon_extrapolated'])
        self.assertEqual(audit['extra_planner_queries'], 0)
        selected = audit['analytic_proposals'][audit['selected_index']]
        self.assertLess(selected['score'], audit['analytic_proposals'][0]['score'])


if __name__ == '__main__':
    unittest.main()
