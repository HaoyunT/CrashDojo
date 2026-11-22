import types
import unittest

import numpy as np

from tbsim.policies.common import Action
from tbsim.policies.wrappers import PlannerOverrideRolloutWrapper


class _Planner:
    def __init__(self, marker):
        self.marker = float(marker)

    def eval(self):
        return self

    def get_action(self, obs, **kwargs):
        count = obs["target_positions"].shape[0]
        horizon = obs["target_positions"].shape[1]
        return Action(
            positions=np.full((count, horizon, 2), self.marker, dtype=np.float32),
            yaws=np.full((count, horizon, 1), self.marker, dtype=np.float32),
        ), {"marker": self.marker}


class _Agents:
    device = "cpu"

    def __init__(self):
        self.last_obs = None

    def eval(self):
        return self

    def get_action(self, obs, **kwargs):
        self.last_obs = obs
        return Action(
            positions=np.zeros_like(obs["target_positions"]),
            yaws=np.zeros_like(obs["target_yaws"]),
        ), {}


class ConditionalFMPlannerAttachmentTest(unittest.TestCase):
    def test_attacked_plan_is_executed_and_both_plans_reach_sampler(self):
        agents = _Agents()
        wrapper = PlannerOverrideRolloutWrapper(
            ego_policy=_Planner(3.0),
            agents_policy=agents,
            strategy_depth=0,
            marginal_tail_config={"mode": "conditional_fm_source"},
            marginal_tail_target_ego_policy=_Planner(7.0),
        )
        obs = {
            "agents": {
                "target_positions": np.zeros((3, 5, 2), dtype=np.float32),
                "target_yaws": np.zeros((3, 5, 1), dtype=np.float32),
                "agent_name": np.asarray(["ego", "a", "b"]),
            }
        }
        action = wrapper.get_action(obs, step_index=0)

        np.testing.assert_array_equal(
            action.agents.positions[0], np.full((5, 2), 3.0)
        )
        np.testing.assert_array_equal(
            agents.last_obs["planner_reference_positions"][0],
            np.full((5, 2), 3.0),
        )
        np.testing.assert_array_equal(
            agents.last_obs["planner_safety_reference_positions"][0],
            np.full((5, 2), 7.0),
        )
        self.assertTrue(agents.last_obs["planner_reference_mask"][0])
        self.assertTrue(agents.last_obs["planner_safety_reference_mask"][0])

    def test_planner_aligned_native_mode_attaches_plans_before_sampling(self):
        agents = _Agents()
        wrapper = PlannerOverrideRolloutWrapper(
            ego_policy=_Planner(3.0),
            agents_policy=agents,
            strategy_depth=0,
            marginal_tail_config={
                "mode": "planner_aligned_native_joint",
                "start_step": 10,
                "end_step": 20,
                "num_samples": 20,
            },
            marginal_tail_target_ego_policy=_Planner(7.0),
        )
        obs = {
            "agents": {
                "target_positions": np.zeros((3, 5, 2), dtype=np.float32),
                "target_yaws": np.zeros((3, 5, 1), dtype=np.float32),
                "agent_name": np.asarray(["ego", "a", "b"]),
            }
        }
        with self.assertRaisesRegex(
            RuntimeError, "requires samples/transforms/extents"
        ):
            wrapper.get_action(obs, step_index=10)
        np.testing.assert_array_equal(
            agents.last_obs["planner_reference_positions"][0],
            np.full((5, 2), 3.0),
        )
        np.testing.assert_array_equal(
            agents.last_obs["planner_safety_reference_positions"][0],
            np.full((5, 2), 7.0),
        )

    def test_planner_aligned_prefix_is_unconditioned(self):
        agents = _Agents()
        wrapper = PlannerOverrideRolloutWrapper(
            ego_policy=_Planner(3.0),
            agents_policy=agents,
            strategy_depth=0,
            marginal_tail_config={
                "mode": "planner_aligned_native_joint",
                "start_step": 10,
                "end_step": 20,
                "num_samples": 20,
            },
            marginal_tail_target_ego_policy=_Planner(7.0),
            marginal_tail_attack_ego_policy=_Planner(3.0),
        )
        obs = {
            "agents": {
                "target_positions": np.zeros((3, 5, 2), dtype=np.float32),
                "target_yaws": np.zeros((3, 5, 1), dtype=np.float32),
                "agent_name": np.asarray(["ego", "a", "b"]),
            }
        }
        action = wrapper.get_action(obs, step_index=9)

        np.testing.assert_array_equal(
            action.agents.positions[0], np.full((5, 2), 3.0)
        )
        self.assertNotIn("planner_reference_positions", agents.last_obs)
        self.assertNotIn("planner_safety_reference_positions", agents.last_obs)

    def test_executed_planner_is_decoupled_from_selector_attack_planner(self):
        agents = _Agents()
        wrapper = PlannerOverrideRolloutWrapper(
            ego_policy=_Planner(7.0),
            agents_policy=agents,
            strategy_depth=0,
            marginal_tail_config={
                "mode": "conditional_fm_source",
            },
            marginal_tail_target_ego_policy=_Planner(7.0),
            marginal_tail_attack_ego_policy=_Planner(3.0),
        )
        obs = {
            "agents": {
                "target_positions": np.zeros((3, 5, 2), dtype=np.float32),
                "target_yaws": np.zeros((3, 5, 1), dtype=np.float32),
                "agent_name": np.asarray(["ego", "a", "b"]),
            }
        }
        action = wrapper.get_action(obs, step_index=10)

        np.testing.assert_array_equal(
            action.agents.positions[0], np.full((5, 2), 7.0)
        )
        np.testing.assert_array_equal(
            agents.last_obs["planner_reference_positions"][0],
            np.full((5, 2), 3.0),
        )
        np.testing.assert_array_equal(
            agents.last_obs["planner_safety_reference_positions"][0],
            np.full((5, 2), 7.0),
        )

    def test_planner_aligned_selector_roles_are_not_reversed(self):
        values = [np.asarray([marker], dtype=np.float32) for marker in range(4)]
        reference_pos, reference_yaw, target_pos, target_yaw = (
            PlannerOverrideRolloutWrapper._joint_selector_planner_roles(
                "planner_aligned_native_joint", *values
            )
        )
        np.testing.assert_array_equal(reference_pos, values[2])
        np.testing.assert_array_equal(reference_yaw, values[3])
        np.testing.assert_array_equal(target_pos, values[0])
        np.testing.assert_array_equal(target_yaw, values[1])

        legacy = PlannerOverrideRolloutWrapper._joint_selector_planner_roles(
            "constrained_native_joint", *values
        )
        for actual, expected in zip(legacy, values):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
