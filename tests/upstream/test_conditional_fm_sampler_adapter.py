import unittest

import torch

from conditional_fm_sampler_adapter import (
    PlannerSeparationObjectiveConfig,
    integrate_frozen_probability_flow,
    local_to_world,
    objective_time_slice,
    per_actor_source_delta_rms,
    persistent_soft_min,
    planner_separation_source_objective,
    selected_joint_background_mask,
    smooth_max_violation,
    verify_noisy_edge_gradient_contract,
)


class DummyFlow:
    action_dim = 2
    n_flow_steps = 4
    disable_control_on_stationary = False
    stationary_mask = None
    typed_collision_projection = None
    current_constraints = None

    def convert_action_to_state_and_action(
        self, actions, curr_states, scaled_input=True, descaled_output=False
    ):
        del curr_states, scaled_input, descaled_output
        speed = torch.cumsum(actions[..., :1], dim=-2)
        yaw = torch.cumsum(actions[..., 1:2], dim=-2)
        position = torch.cat((torch.cumsum(speed, dim=-2), yaw), dim=-1)
        return torch.cat((position, speed, yaw, actions), dim=-1)

    def _predict_flow_velocity(
        self, x_flow, flow_time, aux_info, data_batch,
        class_free_guide_w=0.0, x_clean_model_out=None,
    ):
        del flow_time, aux_info, data_batch, class_free_guide_w, x_clean_model_out
        return -0.2 * x_flow[..., 4:6], {"attn_weights": x_flow[..., :1]}

    def descale_traj(self, value, chosen_inds=None):
        del chosen_inds
        return value


class ConditionalFMSamplerAdapterTest(unittest.TestCase):
    def test_global_conflict_anchor_tracks_same_episode_frame(self):
        early = PlannerSeparationObjectiveConfig(
            target_actor_idx=1, global_conflict_step=47,
            current_global_step=20, conflict_anchor_radius_steps=2,
        )
        late = PlannerSeparationObjectiveConfig(
            target_actor_idx=1, global_conflict_step=47,
            current_global_step=35, conflict_anchor_radius_steps=2,
        )
        self.assertEqual(objective_time_slice(52, early), slice(25, 30))
        self.assertEqual(objective_time_slice(52, late), slice(10, 15))

    def test_global_conflict_anchor_fails_after_window(self):
        cfg = PlannerSeparationObjectiveConfig(
            target_actor_idx=1, global_conflict_step=47,
            current_global_step=55, conflict_anchor_radius_steps=2,
        )
        with self.assertRaises(ValueError):
            objective_time_slice(52, cfg)

    def test_integration_keeps_source_gradient(self):
        model = DummyFlow()
        source = torch.ones(2, 3, 5, 2, dtype=torch.float64, requires_grad=True)
        aux = {"curr_states": torch.zeros(2, 3, 4, dtype=torch.float64)}
        trajectory, actions, _ = integrate_frozen_probability_flow(
            model, source, {}, aux
        )
        loss = trajectory[..., :2].square().mean() + actions.square().mean()
        gradient = torch.autograd.grad(loss, source)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().max()), 0.0)

    def test_selected_mask_changes_all_valid_background_in_one_joint_mode(self):
        source = torch.zeros(1, 4, 3, 5, 2)
        valid = torch.tensor([True, True, True])
        mask = selected_joint_background_mask(source, 2, 0, valid).reshape_as(source)
        self.assertEqual(int(mask.sum()), 20)
        self.assertTrue(torch.all(mask[0, 2, 1] == 1))
        self.assertTrue(torch.all(mask[0, 2, 2] == 1))
        self.assertEqual(int(mask[0, 2, 0].sum()), 0)
        self.assertEqual(int(mask[0, :2].sum()), 0)
        self.assertEqual(int(mask[0, 3].sum()), 0)

    def test_per_actor_source_trust_catches_concentrated_change(self):
        native = torch.zeros(1, 2, 3, 5, 2)
        changed = native.clone()
        changed[0, 0, 2] = 0.2
        rms = per_actor_source_delta_rms(changed, native)
        self.assertAlmostEqual(float(rms[0, 0, 2]), 0.2, places=6)
        self.assertEqual(int(torch.count_nonzero(rms)), 1)

    def test_target_only_late_mask_changes_one_actor_and_suffix(self):
        source = torch.zeros(1, 4, 3, 10, 2)
        valid = torch.tensor([True, True, True])
        mask = selected_joint_background_mask(
            source, 2, 0, valid, target_actor_idx=1,
            optimization_scope="target_only", optimization_start_fraction=0.6,
        ).reshape_as(source)
        self.assertEqual(int(mask.sum()), 8)
        self.assertEqual(int(mask[0, 2, 1, :6].sum()), 0)
        self.assertTrue(torch.all(mask[0, 2, 1, 6:] == 1))
        self.assertEqual(int(mask[0, 2, 2].sum()), 0)

    def test_noisy_edge_contract_fails_closed(self):
        model = DummyFlow()
        model.use_noisy_fut_edge = True
        with self.assertRaises(RuntimeError):
            verify_noisy_edge_gradient_contract(model)

    def test_noisy_edge_contract_accepts_explicit_enabled_path(self):
        model = DummyFlow()
        model.use_noisy_fut_edge = True
        model.conditional_flow_differentiable_conditioning = True
        contract = verify_noisy_edge_gradient_contract(model)
        self.assertTrue(contract["full_probability_flow_jacobian"])

    def test_smooth_violation_is_near_zero_when_safe(self):
        safe = smooth_max_violation(torch.full((20,), 10.0), 1.0, 0.25)
        unsafe = smooth_max_violation(torch.zeros(20), 1.0, 0.25)
        self.assertLess(float(safe), 1e-6)
        self.assertGreater(float(unsafe), 1.0)

    def test_world_transform(self):
        positions = torch.tensor([[[[1.0, 2.0]]]])
        transform = torch.eye(3).unsqueeze(0)
        transform[0, 0, 2] = 3.0
        transform[0, 1, 2] = -1.0
        world = local_to_world(positions, transform)
        self.assertTrue(torch.equal(world, torch.tensor([[[[4.0, 1.0]]]])))

    def test_persistent_soft_min_prefers_three_frame_event(self):
        safe = torch.tensor([3.0, 3.0, 3.0, 3.0])
        event = torch.tensor([3.0, 0.2, 0.2, 0.2])
        self.assertLess(
            persistent_soft_min(event, 3, 0.1),
            persistent_soft_min(safe, 3, 0.1),
        )

    def test_planner_objective_backpropagates_to_target_source_only(self):
        model = DummyFlow()
        batch, samples, agents, horizon, action_dim = 1, 2, 3, 5, 2
        source = torch.zeros(
            batch * samples, agents, horizon, action_dim,
            dtype=torch.float64, requires_grad=True,
        )
        aux = {
            "curr_states": torch.zeros(
                batch * samples, agents, 4, dtype=torch.float64
            )
        }
        world_from_agent = torch.eye(3, dtype=torch.float64).repeat(agents, 1, 1)
        world_from_agent[1, 0, 2] = 8.0
        world_from_agent[2, 1, 2] = 20.0
        extents = torch.tensor(
            [[4.0, 2.0], [4.0, 2.0], [4.0, 2.0]], dtype=torch.float64
        )
        attacked_plan = torch.stack(
            (torch.linspace(0, 6, horizon), torch.zeros(horizon)), dim=-1
        ).to(torch.float64)
        safety_plan = attacked_plan.clone()
        safety_plan[:, 1] = -10.0
        cfg = PlannerSeparationObjectiveConfig(
            target_actor_idx=1, selected_sample_idx=0,
            persistence_steps=3,
        )
        objective = planner_separation_source_objective(
            model,
            source,
            (batch, samples, agents, horizon, action_dim),
            {},
            aux,
            attacked_plan,
            safety_plan,
            world_from_agent,
            extents,
            cfg,
        )
        gradient = torch.autograd.grad(objective, source)[0]
        self.assertTrue(torch.isfinite(objective))
        self.assertGreater(float(gradient[0, 1].abs().sum()), 0.0)

    def test_planner_objective_excludes_padded_collateral_actor(self):
        model = DummyFlow()
        source = torch.zeros(1, 3, 5, 2, dtype=torch.float64)
        aux = {"curr_states": torch.zeros(1, 3, 4, dtype=torch.float64)}
        world = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1)
        world[1, 0, 2] = 8.0
        # Actor 2 is deliberately placed on the target, but declared padding.
        world[2, 0, 2] = 8.0
        extents = torch.tensor(
            [[4.0, 2.0], [4.0, 2.0], [4.0, 2.0]], dtype=torch.float64
        )
        plan = torch.zeros(5, 2, dtype=torch.float64)
        cfg = PlannerSeparationObjectiveConfig(target_actor_idx=1)
        without_padding = planner_separation_source_objective(
            model, source, (1, 1, 3, 5, 2), {}, aux, plan, plan - 20,
            world, extents, cfg, valid_actor_mask=torch.tensor([1, 1, 0]),
        )
        with_collateral = planner_separation_source_objective(
            model, source, (1, 1, 3, 5, 2), {}, aux, plan, plan - 20,
            world, extents, cfg, valid_actor_mask=torch.tensor([1, 1, 1]),
        )
        self.assertGreater(float(with_collateral), float(without_padding))


if __name__ == "__main__":
    unittest.main()
