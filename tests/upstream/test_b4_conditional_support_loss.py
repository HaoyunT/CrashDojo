import unittest

import torch

from tbsim.utils.guidance_loss import ConditionalMarginalSupportLoss


def _thresholds(actor_count=2):
    empirical = {
        "speed": [2.0] * actor_count,
        "acceleration": [1.0] * actor_count,
        "jerk": [2.0] * actor_count,
        "yaw_rate": [0.5] * actor_count,
    }
    absolute = {
        "speed": 20.0,
        "acceleration": 4.0,
        "jerk": 8.37,
        "yaw_rate": 0.95,
    }
    return empirical, absolute


class ConditionalMarginalSupportLossTest(unittest.TestCase):
    def test_constant_velocity_inside_support_is_zero(self):
        empirical, absolute = _thresholds()
        loss_fn = ConditionalMarginalSupportLoss(empirical, absolute)
        time = torch.arange(26, dtype=torch.float32) * 0.1
        x = torch.zeros((2, 3, 26, 6), dtype=torch.float32)
        x[..., 0] = time
        value = loss_fn(x, {}, agt_mask=torch.tensor([True, True]))
        self.assertEqual(tuple(value.shape), (2, 3))
        self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_joint_kink_has_positive_finite_gradient(self):
        empirical, absolute = _thresholds()
        loss_fn = ConditionalMarginalSupportLoss(empirical, absolute)
        time = torch.arange(26, dtype=torch.float32) * 0.1
        base = torch.zeros((2, 2, 26, 6), dtype=torch.float32)
        base[..., 0] = time
        base[:, :, 15:, 0] += 5.0
        x = base.requires_grad_(True)
        value = loss_fn(x, {}, agt_mask=torch.tensor([True, True])).mean()
        value.backward()
        self.assertGreater(float(value), 0.0)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().max()), 0.0)

    def test_actor_specific_thresholds_follow_mask(self):
        empirical, absolute = _thresholds()
        empirical["speed"] = [0.5, 2.0]
        loss_fn = ConditionalMarginalSupportLoss(empirical, absolute)
        time = torch.arange(26, dtype=torch.float32) * 0.1
        x = torch.zeros((2, 1, 26, 6), dtype=torch.float32)
        x[..., 0] = time
        first = loss_fn(x, {}, agt_mask=torch.tensor([True, False]))
        second = loss_fn(x, {}, agt_mask=torch.tensor([False, True]))
        self.assertGreater(float(first), 0.0)
        self.assertEqual(float(second), 0.0)

    def test_threshold_count_mismatch_fails_closed(self):
        empirical, absolute = _thresholds(actor_count=3)
        loss_fn = ConditionalMarginalSupportLoss(empirical, absolute)
        x = torch.zeros((2, 1, 26, 6), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "does not match actor count"):
            loss_fn(x, {})


if __name__ == "__main__":
    unittest.main()
