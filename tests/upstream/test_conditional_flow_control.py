import math
import unittest

import torch

from conditional_flow_control import (
    ConditionalFlowSourceConfig,
    gaussian_energy,
    optimize_complete_joint_source,
    per_actor_source_delta_rms,
)


class ConditionalFlowControlTest(unittest.TestCase):
    def test_disabled_is_exact_identity(self):
        source = torch.randn(2, 3, 4, dtype=torch.float64)
        out, audit = optimize_complete_joint_source(
            source, lambda z: z.square().mean(), ConditionalFlowSourceConfig()
        )
        self.assertTrue(torch.equal(out, source))
        self.assertTrue(audit["fallback_native"])
        self.assertEqual(audit["accepted_steps"], 0)

    def test_zero_gradient_is_exact_fallback(self):
        source = torch.randn(2, 3, 4, dtype=torch.float64)
        config = ConditionalFlowSourceConfig(enabled=True)
        out, audit = optimize_complete_joint_source(
            source, lambda z: (z * 0).sum(), config
        )
        self.assertTrue(torch.equal(out, source))
        self.assertEqual(audit["status"], "native_fallback_no_valid_improvement")

    def test_quadratic_objective_decreases_inside_trust_region(self):
        source = torch.zeros(2, 3, dtype=torch.float64)
        target = torch.full_like(source, 0.05)
        config = ConditionalFlowSourceConfig(
            enabled=True,
            num_steps=6,
            step_size=0.5,
            max_source_delta_rms=0.10,
            max_source_energy_increase=0.02,
        )
        out, audit = optimize_complete_joint_source(
            source, lambda z: (z - target).square().mean(), config
        )
        self.assertFalse(audit["fallback_native"])
        self.assertLess(audit["final_objective"], audit["initial_objective"])
        self.assertLessEqual(audit["source_delta_rms"], 0.10 + 1e-12)
        self.assertLessEqual(audit["source_energy_delta"], 0.02 + 1e-12)
        self.assertTrue(torch.isfinite(out).all())

    def test_mask_freezes_third_actor(self):
        source = torch.zeros(3, 2, dtype=torch.float64)
        target = torch.ones_like(source) * 0.05
        mask = torch.ones_like(source)
        mask[2] = 0
        config = ConditionalFlowSourceConfig(
            enabled=True, num_steps=4, step_size=0.5,
            max_source_delta_rms=0.10, max_source_energy_increase=0.02,
        )
        out, _ = optimize_complete_joint_source(
            source, lambda z: (z - target).square().mean(), config, mask
        )
        self.assertTrue(torch.equal(out[2], source[2]))
        self.assertFalse(torch.equal(out[:2], source[:2]))

    def test_tight_trust_gate_causes_exact_fallback(self):
        source = torch.zeros(2, 3, dtype=torch.float64)
        config = ConditionalFlowSourceConfig(
            enabled=True,
            num_steps=2,
            step_size=1.0,
            backtracking_steps=3,
            max_source_delta_rms=0.0,
            max_source_energy_increase=0.0,
        )
        out, audit = optimize_complete_joint_source(
            source, lambda z: (z - 1).square().mean(), config
        )
        self.assertTrue(torch.equal(out, source))
        self.assertTrue(audit["fallback_native"])

    def test_attested_energy_reconstructs(self):
        source = torch.zeros(2, 2, dtype=torch.float64)
        config = ConditionalFlowSourceConfig(
            enabled=True, num_steps=2, step_size=0.2,
            max_source_delta_rms=0.2, max_source_energy_increase=0.1,
        )
        out, audit = optimize_complete_joint_source(
            source, lambda z: (z - 0.05).square().mean(), config
        )
        reconstructed = float((gaussian_energy(out) - gaussian_energy(source)).item())
        self.assertTrue(math.isclose(
            reconstructed, audit["source_energy_delta"], rel_tol=0, abs_tol=1e-12
        ))
        self.assertFalse(audit["endpoint_projection"])
        self.assertFalse(audit["world_coordinate_variable"])
        self.assertFalse(audit["per_agent_row_mixing"])

    def test_per_actor_gate_rejects_concentrated_change(self):
        source = torch.zeros(1, 3, 4, 2, dtype=torch.float64)
        target = source.clone()
        target[:, 1] = 1.0
        config = ConditionalFlowSourceConfig(
            enabled=True,
            num_steps=1,
            step_size=1.0,
            backtracking_steps=1,
            max_source_delta_rms=1.0,
            max_per_actor_source_delta_rms=0.01,
            max_source_energy_increase=1.0,
        )
        out, audit = optimize_complete_joint_source(
            source, lambda z: (z - target).square().mean(), config
        )
        self.assertTrue(torch.equal(out, source))
        self.assertTrue(audit["fallback_native"])
        trial = audit["trace"][0]["trials"][0]
        self.assertLess(trial["source_delta_rms"], 1.0)
        self.assertGreater(trial["max_per_actor_source_delta_rms"], 0.01)

    def test_per_actor_rms_shape_matches_joint_source_rows(self):
        source = torch.zeros(2, 3, 4, 2, dtype=torch.float64)
        changed = source.clone()
        changed[1, 2] = 0.07
        delta = per_actor_source_delta_rms(changed, source)
        self.assertEqual(tuple(delta.shape), (2, 3))
        self.assertAlmostEqual(float(delta[1, 2]), 0.07, places=12)

    def test_normalized_gradient_step_has_interpretable_active_rms(self):
        source = torch.zeros(1, 3, 4, 2, dtype=torch.float64)
        mask = torch.zeros_like(source)
        mask[:, 1:] = 1
        target = torch.ones_like(source)
        config = ConditionalFlowSourceConfig(
            enabled=True,
            num_steps=1,
            step_size=0.02,
            max_source_delta_rms=0.03,
            max_per_actor_source_delta_rms=0.03,
            max_source_energy_increase=0.01,
            normalize_masked_gradient_rms=True,
        )
        out, audit = optimize_complete_joint_source(
            source, lambda z: (z - target).square().mean(), config, mask
        )
        self.assertFalse(audit["fallback_native"])
        self.assertAlmostEqual(float((out[mask.bool()]).square().mean().sqrt()), 0.02)
        self.assertTrue(audit["trace"][0]["gradient_normalized"])


if __name__ == "__main__":
    unittest.main()
