import unittest

import torch

from availability_export import masked_observation
from conflict_parameter_search import parameter_options, transition_duration


LIMITS = dict(max_speed_mps=25., max_acceleration_mps2=3., max_braking_mps2=6.,
              max_lateral_mps2=4., max_jerk_mps3=10.)


class ParameterSearchTest(unittest.TestCase):
    def request(self, speed, role):
        return dict(family='C1', engineering_limits=LIMITS,
                    geometry_evidence=dict(longitudinal_role=role),
                    parameters=dict(initial_speed_mps=speed, target_speed_mps=0.,
                                    speed_transition_s=2., speed_t0=.2))

    def test_static_lead_does_not_get_acceleration_or_reverse_proposals(self):
        options, status = parameter_options(self.request(0., 'adversary_ahead'))
        self.assertEqual(len(options), 1)
        self.assertEqual(status, 'static_lead_has_no_active_braking_room')

    def test_unknown_role_is_neutral(self):
        self.assertEqual(len(parameter_options(self.request(5., None))[0]), 1)

    def test_rear_proposals_accelerate_with_frozen_budget(self):
        options, _ = parameter_options(self.request(5., 'adversary_behind'))
        self.assertEqual(len(options), 5)
        self.assertTrue(all(p['target_speed_mps'] > 5 for p in options[1:]))

    def test_quintic_profile_respects_continuous_acceleration_and_jerk(self):
        for initial, target in [(0., 8.), (15., 0.), (4., 6.)]:
            duration = transition_duration(initial, target, LIMITS, 2.)
            acceleration = 3. if target >= initial else 6.
            self.assertLessEqual(1.875 * abs(target-initial) / duration, .9 * acceleration + 1e-9)
            self.assertLessEqual(5.774 * abs(target-initial) / duration**2, 8. + 1e-9)

    def test_mask_does_not_change_valid_origin_or_input(self):
        source = dict(centroid=torch.zeros(2, 2), yaw=torch.zeros(2),
                      availability=torch.tensor([True, False]))
        result = masked_observation(source)
        self.assertTrue(torch.isfinite(result['centroid'][0]).all())
        self.assertTrue(torch.isnan(result['centroid'][1]).all())
        self.assertTrue(torch.isfinite(source['centroid']).all())


if __name__ == '__main__':
    unittest.main()
