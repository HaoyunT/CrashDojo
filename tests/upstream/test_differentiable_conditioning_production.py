"""Value/Jacobian contract for production Conditional-FM conditioning.

This test imports the installed M2 module.  It proves that the default-off
native inference values are unchanged while the opt-in inference-only branch
provides a finite full relation-conditioning Jacobian.
"""

import types
import unittest

import torch

from ccdiff.models.ccdiff import CCDiffuserModel, CCFlowMatching


class DifferentiableConditioningProductionTest(unittest.TestCase):
    def _inputs(self):
        generator = torch.Generator().manual_seed(7)
        pos = torch.randn(
            1, 3, 3, 5, 2, generator=generator, dtype=torch.float64
        )
        pos = pos + torch.arange(3, dtype=torch.float64)[
            None, None, :, None, None
        ]
        pos.requires_grad_(True)
        yaw = torch.randn(
            1, 3, 3, 5, 1, generator=generator, dtype=torch.float64
        )
        speed = torch.rand(
            1, 3, 3, 5, generator=generator, dtype=torch.float64
        ) + 2.0
        extent = torch.tensor(
            [[[[4.0, 2.0, 1.5]] * 3] * 3], dtype=torch.float64
        ).reshape(1, 3, 3, 3)
        avail = torch.ones(1, 3, 3, 5, dtype=torch.bool)
        norm_info = [[0.0] * 5, [1.0] * 5]
        return pos, yaw, speed, extent, avail, norm_info

    def test_values_match_and_full_jacobian_is_finite(self):
        pos, yaw, speed, extent, avail, norm_info = self._inputs()
        owner = types.SimpleNamespace(
            conditional_flow_differentiable_conditioning=False
        )
        native, _ = CCDiffuserModel.prepare_scene_agent_hist(
            owner, pos, yaw, speed, extent, avail, norm_info,
            scale=True, speed_repr="rel_vel_new_new",
        )
        native_grad = torch.autograd.grad(
            native[:, 0, 1, :, 9].square().mean(),
            pos,
            retain_graph=True,
        )[0]

        owner.conditional_flow_differentiable_conditioning = True
        differentiable, _ = CCDiffuserModel.prepare_scene_agent_hist(
            owner, pos, yaw, speed, extent, avail, norm_info,
            scale=True, speed_repr="rel_vel_new_new",
        )
        differentiable_grad = torch.autograd.grad(
            differentiable.square().mean(), pos
        )[0]

        self.assertTrue(torch.equal(native, differentiable))
        self.assertFalse(torch.isfinite(native_grad).all())
        self.assertTrue(torch.isfinite(differentiable_grad).all())
        self.assertGreater(float(differentiable_grad.abs().max()), 0.0)

    def test_switch_is_default_off_and_inference_only(self):
        instance = types.SimpleNamespace(
            training=True,
            typed_collision_projection=None,
            current_constraints=None,
            conditional_flow_differentiable_conditioning=False,
        )
        with self.assertRaises(RuntimeError):
            CCFlowMatching.set_conditional_flow_differentiable_conditioning(
                instance, True
            )
        self.assertFalse(instance.conditional_flow_differentiable_conditioning)


if __name__ == "__main__":
    unittest.main()
