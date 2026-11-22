import unittest

import numpy as np
from shapely.geometry import box

from available_geometry import evaluate_arrays


class LifecycleGeometryTest(unittest.TestCase):
    def sample(self):
        xy = np.array([[[0., 0.], [0., 0.]], [[4., 0.], [4., 0.]],
                       [[20., 0.], [np.nan, np.nan]]])
        yaw = np.zeros((3, 2))
        yaw[2, 1] = np.nan
        return xy, yaw, np.ones((3, 2, 2)), np.array([[1, 1], [1, 1], [1, 0]], bool)

    def run_case(self, values):
        return evaluate_arrays(*values, 0, 1, box(-30, -30, 30, 30))

    def test_absent_actor_does_not_create_origin_collision(self):
        result = self.run_case(self.sample())
        self.assertTrue(result['geometry_evaluable'])
        self.assertFalse(result['scene_any_contact'])

    def test_valid_world_origin_actor_is_not_removed(self):
        values = list(self.sample())
        values[0][1] = 0
        self.assertTrue(self.run_case(values)['target_contact'])

    def test_between_frame_contact_is_detected(self):
        values = list(self.sample())
        values[0][1, :, 0] = [-2, 2]
        result = self.run_case(values)
        self.assertTrue(result['target_contact'])
        self.assertGreater(result['first_target_contact_s'], 0)
        self.assertLess(result['first_target_contact_s'], .1)

    def test_missing_controlled_actor_is_unknown(self):
        values = list(self.sample())
        values[3][0, 1] = False
        self.assertIsNone(self.run_case(values)['target_contact'])

    def test_valid_teleport_still_rejected(self):
        values = list(self.sample())
        values[0][1, 1, 0] = 1000
        self.assertEqual(self.run_case(values)['geometry_rejection_reason'],
                         'unreasonable_between_frame_motion')

    def test_reappearance_not_silently_dropped(self):
        values = list(self.sample())
        values[3][2] = [False, True]
        values[0][2, 1] = [20, 0]
        values[1][2, 1] = 0
        self.assertEqual(self.run_case(values)['geometry_rejection_reason'], 'unsupported_reappearance')


if __name__ == '__main__':
    unittest.main()
