import unittest

import torch

from target_source_mean_shift import (
    TargetSourceMeanShiftConfig,
    apply_target_source_mean_shift,
    estimate_online_planner_target_cpa,
)


class TargetSourceMeanShiftTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.source = torch.randn(1, 20, 12, 16, 2)

    def test_zero_action_is_bit_exact(self):
        result, audit = apply_target_source_mean_shift(
            self.source, TargetSourceMeanShiftConfig(target_actor_idx=9)
        )
        self.assertTrue(torch.equal(result, self.source))
        self.assertEqual(audit["status"], "zero_action_native_identity")

    def test_action_changes_only_one_target_and_mode(self):
        result, audit = apply_target_source_mean_shift(
            self.source,
            TargetSourceMeanShiftConfig(
                target_actor_idx=9,
                coefficients=(0.08, -0.04, 0.03, 0.02),
            ),
        )
        self.assertTrue(audit["accepted"])
        self.assertFalse(torch.equal(result[:, 0, 9], self.source[:, 0, 9]))
        self.assertTrue(audit["other_modes_bit_exact"])
        self.assertTrue(audit["other_actors_bit_exact"])
        self.assertFalse(audit["endpoint_projection"])

    def test_trust_reject_falls_back_bit_exact(self):
        result, audit = apply_target_source_mean_shift(
            self.source,
            TargetSourceMeanShiftConfig(
                target_actor_idx=9,
                coefficients=(0.2, 0.2, 0.2, 0.2),
                max_target_delta_rms=0.01,
            ),
        )
        self.assertFalse(audit["accepted"])
        self.assertTrue(torch.equal(result, self.source))

    def test_gaussian_pulse_is_local_and_target_only(self):
        result, audit = apply_target_source_mean_shift(
            self.source,
            TargetSourceMeanShiftConfig(
                target_actor_idx=9,
                coefficients=(-0.12, 0.04, 0.0, 0.0),
                basis_kind="gaussian_pulse",
                pulse_center_fraction=0.70,
                pulse_width_fraction=0.10,
            ),
        )
        self.assertTrue(audit["accepted"])
        delta = result[0, 0, 9] - self.source[0, 0, 9]
        self.assertGreater(float(delta[11:13, 0].abs().max()), 0.05)
        self.assertLess(float(delta[:3, 0].abs().max()), 1e-4)
        self.assertTrue(audit["other_modes_bit_exact"])
        self.assertTrue(audit["other_actors_bit_exact"])

    def test_invalid_target_fails_closed(self):
        with self.assertRaises(IndexError):
            apply_target_source_mean_shift(
                self.source, TargetSourceMeanShiftConfig(target_actor_idx=99)
            )

    def test_online_cpa_uses_planner_and_current_target_state(self):
        planner = torch.stack((torch.arange(1, 11, dtype=torch.float32), torch.zeros(10)), dim=-1)
        transforms = torch.eye(3).repeat(2, 1, 1)
        transforms[1, 0, 2] = 6.0
        speeds = torch.tensor([0.0, 0.0])
        audit = estimate_online_planner_target_cpa(
            planner,
            transforms,
            speeds,
            planner_actor_idx=0,
            target_actor_idx=1,
            dt=1.0,
            max_horizon=10,
        )
        self.assertEqual(audit["cpa_step"], 5)
        self.assertEqual(audit["cpa_center_distance_m"], 0.0)
        self.assertFalse(audit["uses_logged_target_future"])

    def test_online_cpa_respects_target_velocity(self):
        planner = torch.stack((torch.arange(1, 7, dtype=torch.float32), torch.zeros(6)), dim=-1)
        transforms = torch.eye(3).repeat(2, 1, 1)
        transforms[1, 0, 2] = 2.0
        speeds = torch.tensor([0.0, 0.5])
        audit = estimate_online_planner_target_cpa(
            planner,
            transforms,
            speeds,
            planner_actor_idx=0,
            target_actor_idx=1,
            dt=1.0,
            max_horizon=6,
        )
        self.assertEqual(audit["cpa_step"], 3)

    def test_arrival_alignment_sign_is_geometry_covariant(self):
        planner = torch.stack((torch.arange(1, 7, dtype=torch.float32), torch.zeros(6)), dim=-1)
        transforms = torch.eye(3).repeat(2, 1, 1)
        transforms[1, 0, 2] = 2.0
        slow = estimate_online_planner_target_cpa(
            planner, transforms, torch.tensor([0.0, 0.0]),
            planner_actor_idx=0, target_actor_idx=1, dt=1.0, max_horizon=6,
        )
        fast = estimate_online_planner_target_cpa(
            planner, transforms, torch.tensor([0.0, 4.0]),
            planner_actor_idx=0, target_actor_idx=1, dt=1.0, max_horizon=6,
        )
        self.assertTrue(slow["arrival_alignment_valid"])
        self.assertEqual(slow["arrival_lateral_distance_m"], 0.0)
        self.assertLessEqual(slow["arrival_required_target_speed_mps"], 20.0)
        self.assertEqual(slow["arrival_control_sign"], 1.0)
        self.assertEqual(fast["arrival_control_sign"], -1.0)
        self.assertFalse(slow["uses_logged_target_future"])


if __name__ == "__main__":
    unittest.main()
