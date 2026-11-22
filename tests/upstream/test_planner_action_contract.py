import unittest

import torch

from ccdiff.planners.safesim import (
    IDMPlannerPolicy,
    STRIVEPlannerPolicy,
    _ccdiff_state_8d_to_safesim_9d,
)
from tbsim.policies.common import Action
from tbsim.policies.wrappers import PlannerOverrideRolloutWrapper


class _MockScenePolicy:
    def __init__(self):
        self.device = torch.device("cpu")

    def eval(self):
        return self

    def get_action(self, obs, step_index=None):
        batch_size, horizon = obs["target_positions"].shape[:2]
        positions = torch.zeros((batch_size, horizon, 2))
        positions[:, :, 0] = 50.0
        return Action(
            positions=positions,
            yaws=torch.zeros((batch_size, horizon, 1)),
        ), {"source": torch.ones(batch_size)}


class _MockEgoPlanner:
    supports_candidate_future = False

    def __init__(self):
        self.device = torch.device("cpu")

    def eval(self):
        return self

    def get_action(self, obs, **kwargs):
        horizon = obs["target_positions"].shape[1]
        x = torch.arange(1, horizon + 1, dtype=torch.float32)
        return Action(
            positions=torch.stack((x, torch.zeros_like(x)), dim=-1)[None],
            yaws=torch.zeros((1, horizon, 1)),
        ), {"contract_test": torch.ones(1)}


class PlannerActionContractTest(unittest.TestCase):
    def test_8d_to_9d_preserves_velocity_and_acceleration(self):
        state = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.6, 0.8])
        converted = _ccdiff_state_8d_to_safesim_9d(state)
        expected = torch.tensor(
            [1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0, 0.6, 0.8]
        )
        self.assertTrue(torch.equal(converted, expected))

    def test_native_idm_straight_road_preserves_ten_mps(self):
        planner = IDMPlannerPolicy(torch.device("cpu"), horizon=10)
        lane = torch.zeros((1, 1, 150, 4))
        lane[0, 0, :, 0] = torch.arange(150, dtype=torch.float32)
        action, info = planner.get_action(
            {
                "curr_speed": torch.tensor([10.0]),
                "extras": {"closest_lane_point": lane},
            },
            strategy_depth=0,
        )
        expected_x = torch.arange(1, 11, dtype=torch.float32)
        self.assertTrue(
            torch.allclose(action.positions[0, :, 0], expected_x, atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(action.positions[0, :, 1], torch.zeros(10))
        )
        self.assertAlmostEqual(float(info["idm_acceleration"][0]), 0.0, places=5)

    def test_native_idm_drops_behind_route_point_and_limits_yaw(self):
        planner = IDMPlannerPolicy(torch.device("cpu"), horizon=5)
        lane = torch.zeros((1, 1, 8, 4))
        lane[0, 0, :, 0] = torch.tensor(
            [-0.8, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        )
        lane[0, 0, :, 1] = 0.5
        action, _ = planner.get_action(
            {
                "curr_speed": torch.tensor([5.0]),
                "extras": {"closest_lane_point": lane},
            },
            strategy_depth=0,
        )
        self.assertTrue(torch.all(action.positions[0, :, 0] > 0.0))
        yaw = action.yaws[0, :, 0]
        yaw_step = torch.diff(torch.cat((torch.zeros(1), yaw)))
        self.assertLessEqual(float(yaw_step.abs().max()), 1.2 * 0.1 + 1e-6)

        # The reported orientation and the executed displacement must describe
        # the same motion after yaw-rate limiting.
        displacement = torch.diff(
            torch.cat((torch.zeros((1, 2)), action.positions[0]), dim=0),
            dim=0,
        )
        motion_yaw = torch.atan2(displacement[:, 1], displacement[:, 0])
        heading_error = torch.atan2(
            torch.sin(motion_yaw - yaw), torch.cos(motion_yaw - yaw)
        )
        self.assertLessEqual(float(heading_error.abs().max()), 1e-5)

    def test_unified_wrapper_replaces_only_ego_row(self):
        wrapper = PlannerOverrideRolloutWrapper(
            ego_policy=_MockEgoPlanner(),
            agents_policy=_MockScenePolicy(),
            strategy_depth=0,
        )
        obs = {
            "agents": {
                "target_positions": torch.zeros((3, 6, 2)),
                "agent_name": ["vehicle-a", "ego", "vehicle-b"],
                "scene_index": torch.zeros(3, dtype=torch.long),
            }
        }
        rollout = wrapper.get_action(obs, step_index=0)
        self.assertTrue(
            torch.allclose(
                rollout.agents.positions[1, :, 0],
                torch.arange(1, 7, dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                rollout.agents.positions[[0, 2], :, 0],
                torch.full((2, 6), 50.0),
            )
        )

    def test_strive_temporal_resampling_does_not_double_speed(self):
        # STRIVE emits 0.2 s poses. A 10 m/s path should become 1 m increments
        # at CCDiff's 0.1 s action rate, not 2 m increments.
        plan = torch.zeros((1, 5, 4))
        plan[0, :, 0] = torch.arange(1, 6, dtype=torch.float32) * 2.0
        plan[0, :, 2] = 1.0
        resampled = STRIVEPlannerPolicy._resample_plan(
            plan,
            source_dt=0.2,
            target_dt=0.1,
            target_horizon=10,
            initial_speed=10.0,
            max_speed=20.0,
            max_accel=4.0,
        )
        expected_x = torch.arange(1, 11, dtype=torch.float32)
        self.assertTrue(torch.allclose(resampled[:, 0], expected_x, atol=1e-5))
        speed = torch.diff(
            torch.cat((torch.zeros(1), resampled[:, 0]))
        ) / 0.1
        acceleration = torch.diff(speed) / 0.1
        self.assertLessEqual(float(speed.max()), 20.0)
        self.assertLessEqual(float(acceleration.abs().max()), 4.0)


if __name__ == "__main__":
    unittest.main()
