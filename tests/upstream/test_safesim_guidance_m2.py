import unittest

import torch

from tbsim.utils.guidance_loss import SafeSimTargetPairLoss


class SafeSimTargetPairLossM2Test(unittest.TestCase):
    @staticmethod
    def _batch():
        return {
            "scene_index": torch.zeros(3, dtype=torch.long),
            "world_from_agent": torch.eye(3).repeat(3, 1, 1),
            "extent": torch.tensor(
                [[4.5, 2.0, 1.5], [4.5, 2.0, 1.5], [4.5, 2.0, 1.5]]
            ),
        }

    def test_target_is_attracted_to_reference(self):
        trajectory = torch.zeros((3, 1, 6, 6), requires_grad=True)
        with torch.no_grad():
            trajectory[1, :, :, 0] = 12.0
            trajectory[2, :, :, 0] = 40.0
        loss_fn = SafeSimTargetPairLoss(
            target_ind=1,
            ref_ind=0,
            cause_weight=1.0,
            collateral_weight=0.0,
            loss_timesteps=6,
        )
        mask = torch.tensor([False, True, False])
        loss = loss_fn(trajectory, self._batch(), agt_mask=mask).mean()
        loss.backward()
        self.assertGreater(float(trajectory.grad[1, :, :, 0].mean()), 0.0)
        self.assertEqual(float(trajectory.grad[0].abs().sum()), 0.0)

    def test_target_pair_is_excluded_from_collateral(self):
        mask = torch.tensor([False, True, False])
        loss_fn = SafeSimTargetPairLoss(
            target_ind=1,
            ref_ind=0,
            cause_weight=0.0,
            collateral_weight=1.0,
            loss_timesteps=6,
        )
        close = torch.zeros((3, 1, 6, 6), requires_grad=True)
        far = torch.zeros((3, 1, 6, 6), requires_grad=True)
        with torch.no_grad():
            close[1, :, :, 0] = 10.0
            close[2, :, :, 0] = 10.5
            far[1, :, :, 0] = 10.0
            far[2, :, :, 0] = 40.0
        close_loss = loss_fn(close, self._batch(), agt_mask=mask).mean()
        far_loss = loss_fn(far, self._batch(), agt_mask=mask).mean()
        self.assertGreater(float(close_loss), float(far_loss))

    def test_frozen_planner_reference_is_used(self):
        trajectory = torch.zeros((3, 1, 6, 6), requires_grad=True)
        with torch.no_grad():
            trajectory[1, :, :, 0] = 12.0
        batch = self._batch()
        batch["planner_reference_positions"] = torch.zeros((3, 6, 2))
        batch["planner_reference_positions"][0, :, 0] = 20.0
        batch["planner_reference_mask"] = torch.tensor([True, False, False])
        loss_fn = SafeSimTargetPairLoss(
            target_ind=1,
            ref_ind=0,
            cause_weight=1.0,
            collateral_weight=0.0,
            loss_timesteps=6,
        )
        mask = torch.tensor([False, True, False])
        loss = loss_fn(trajectory, batch, agt_mask=mask).mean()
        loss.backward()
        # The frozen planner is to the right of the target. Gradient descent
        # must move the target right, hence a negative x gradient.
        self.assertLess(float(trajectory.grad[1, :, :, 0].mean()), 0.0)

    def test_arrival_term_prefers_the_selected_conflict_step(self):
        loss_fn = SafeSimTargetPairLoss(
            target_ind=1,
            ref_ind=0,
            cause_weight=0.0,
            collateral_weight=0.0,
            conflict_step=3,
            conflict_point_world=(5.0, 0.0),
            arrival_weight=1.0,
        )
        correct = torch.zeros((3, 1, 6, 6))
        wrong = torch.zeros((3, 1, 6, 6))
        correct[1, 0, 3, 0] = 5.0
        wrong[1, 0, 1, 0] = 5.0
        mask = torch.tensor([False, True, False])
        correct_loss = loss_fn(correct, self._batch(), agt_mask=mask).mean()
        wrong_loss = loss_fn(wrong, self._batch(), agt_mask=mask).mean()
        self.assertLess(float(correct_loss), float(wrong_loss))

    def test_typed_heading_side_prefers_perpendicular_arrival(self):
        loss_fn = SafeSimTargetPairLoss(
            target_ind=1,
            ref_ind=0,
            cause_weight=0.0,
            collateral_weight=0.0,
            collision_type="side",
            conflict_step=2,
            typed_heading_weight=1.0,
        )
        reference = torch.zeros((1, 6))
        perpendicular = torch.full((1, 6), torch.pi / 2.0)
        parallel = torch.zeros((1, 6))
        self.assertLess(
            float(loss_fn._typed_heading_cost(perpendicular, reference)),
            float(loss_fn._typed_heading_cost(parallel, reference)),
        )

    def test_marginal_barrier_penalizes_acceleration_tail(self):
        loss_fn = SafeSimTargetPairLoss(
            target_ind=1,
            ref_ind=0,
            cause_weight=0.0,
            collateral_weight=0.0,
            marginal_weight=1.0,
            max_abs_acc_mps2=4.0,
        )
        natural = torch.zeros((1, 6, 6))
        tail = natural.clone()
        tail[..., 4] = 12.0
        self.assertLess(
            float(loss_fn._marginal_cost(natural)),
            float(loss_fn._marginal_cost(tail)),
        )


if __name__ == "__main__":
    unittest.main()
