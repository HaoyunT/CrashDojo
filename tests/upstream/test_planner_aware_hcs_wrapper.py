import hashlib
import json
import math
import tempfile
import unittest

import numpy as np
import torch

from tbsim.policies.common import Action, RolloutAction
from tbsim.policies.wrappers import PlannerOverrideRolloutWrapper


class _FakeModel:
    def __init__(self):
        self.config = None
        self.example_batch = None
        self.flow_velocity_guidance_config = None
        self.clear_calls = 0

    def set_guidance(self, config, example_batch=None):
        self.config = config
        self.example_batch = example_batch

    def clear_guidance(self):
        self.config = None
        self.example_batch = None
        self.clear_calls += 1

    def set_flow_velocity_guidance(self, config=None):
        self.flow_velocity_guidance_config = config


class _FakeTrafficPolicy:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = _FakeModel()
        self.last_observation = None

    def get_action(self, observation, step_index):
        self.last_observation = observation
        batch = observation["target_positions"].shape[0]
        horizon = observation["target_positions"].shape[1]
        return Action(
            positions=torch.zeros((batch, horizon, 2)),
            yaws=torch.zeros((batch, horizon, 1)),
        ), {}


class _FakeEgoPolicy:
    def get_action(
        self,
        observation,
        step_index,
        candidate_agents_future=None,
        strategy_depth=0,
    ):
        batch = observation["target_positions"].shape[0]
        horizon = observation["target_positions"].shape[1]
        return Action(
            positions=torch.zeros((batch, horizon, 2)),
            yaws=torch.zeros((batch, horizon, 1)),
        ), {}


class PlannerAwareHCSWrapperTest(unittest.TestCase):
    @staticmethod
    def _observation():
        transforms = torch.eye(3).repeat(3, 1, 1)
        transforms[0, :2, 2] = torch.tensor([0.0, 0.0])
        transforms[1, :2, 2] = torch.tensor([2.0, 2.0])
        transforms[2, :2, 2] = torch.tensor([4.0, -4.0])
        return {
            "scene_index": torch.zeros(3, dtype=torch.long),
            "track_id": torch.tensor([0, 11, 22], dtype=torch.long),
            "world_from_agent": transforms,
            "curr_speed": torch.tensor([0.0, 1.0, 1.0]),
            "yaw": torch.tensor([0.0, math.pi / 2.0, math.pi / 2.0]),
            "extent": torch.tensor(
                [[4.5, 2.0, 1.5], [4.5, 2.0, 1.5], [4.5, 2.0, 1.5]]
            ),
        }

    @staticmethod
    def _coalition_observation():
        transforms = torch.eye(3).repeat(5, 1, 1)
        transforms[:, :2, 2] = torch.tensor(
            [
                [0.0, 0.0],
                [20.0, 20.0],
                [4.0, -4.0],
                [4.0, 2.0],
                [4.0, -2.0],
            ]
        )
        return {
            "scene_index": torch.zeros(5, dtype=torch.long),
            "track_id": torch.tensor([0, 11, 22, 33, 44], dtype=torch.long),
            "world_from_agent": transforms,
            "curr_speed": torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]),
            "yaw": torch.tensor(
                [0.0, 0.0, math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0]
            ),
            "extent": torch.tensor([[4.5, 2.0, 1.5]] * 5),
        }

    def test_installs_causal_target_on_traffic_model(self):
        traffic = _FakeTrafficPolicy()
        wrapper = PlannerOverrideRolloutWrapper(
            ego_policy=object(),
            agents_policy=traffic,
            safesim_guidance=True,
            planner_aware_hcs=True,
            planner_aware_hcs_config={
                "dt": 1.0,
                "reference_agent_idx": 0,
                "severity_mode": "collision",
                "cause_weight": 1.0,
                "collateral_weight": 1.0,
                "arrival_weight": 0.3,
                "start_step": 0,
                "end_step": 8,
            },
        )
        # Keep this focused test permissive about collateral so that it tests
        # the planner/HCS/guidance interface rather than the screening gate.
        wrapper._planner_aware_hcs_selector.min_bystander_gap_m = -10.0
        plan = torch.stack(
            (torch.arange(1.0, 9.0), torch.zeros(8)), dim=-1
        )[None]
        ego_action = Action(
            positions=plan,
            yaws=torch.zeros((1, 8, 1)),
        )
        observation = self._observation()
        event = wrapper._configure_planner_aware_hcs_guidance(
            observation, ego_action, step_index=0
        )
        self.assertEqual(event["target_local_index"], 2)
        self.assertEqual(event["collision_type"], "side")
        guide = traffic.model.config[0][0]
        self.assertEqual(guide["name"], "safesim_target_pair")
        self.assertEqual(guide["agents"], [2])
        self.assertEqual(guide["params"]["target_ind"], 2)
        self.assertEqual(guide["params"]["collision_type"], "side")
        self.assertEqual(guide["params"]["conflict_step"], 3)
        self.assertEqual(guide["params"]["loss_timesteps"], 4)
        self.assertEqual(
            tuple(guide["params"]["conflict_point_world"]), (4.0, 0.0)
        )
        self.assertEqual(guide["params"]["arrival_weight"], 0.3)
        self.assertGreater(guide["params"]["marginal_weight"], 0.0)
        self.assertIs(traffic.model.example_batch, observation)
        self.assertEqual(
            traffic.model.flow_velocity_guidance_config[
                "trust_max_rms"
            ],
            0.05,
        )

    def test_schedule_and_episode_target_lock_fail_closed(self):
        traffic = _FakeTrafficPolicy()
        wrapper = PlannerOverrideRolloutWrapper(
            ego_policy=object(),
            agents_policy=traffic,
            safesim_guidance=True,
            planner_aware_hcs=True,
            planner_aware_hcs_config={
                "dt": 1.0,
                "reference_agent_idx": 0,
                "severity_mode": "collision",
                "cause_weight": 1.0,
                "collateral_weight": 1.0,
                "start_step": 2,
                "end_step": 4,
            },
        )
        wrapper._planner_aware_hcs_selector.min_bystander_gap_m = -10.0
        plan = torch.stack(
            (torch.arange(1.0, 9.0), torch.zeros(8)), dim=-1
        )[None]
        ego_action = Action(
            positions=plan,
            yaws=torch.zeros((1, 8, 1)),
        )
        observation = self._observation()

        self.assertIsNone(
            wrapper._configure_planner_aware_hcs_guidance(
                observation, ego_action, step_index=0
            )
        )
        self.assertIsNone(traffic.model.config)
        self.assertIsNone(traffic.model.flow_velocity_guidance_config)
        self.assertEqual(traffic.model.clear_calls, 1)
        self.assertIsNone(wrapper._planner_aware_hcs_locked_target_local)

        event = wrapper._configure_planner_aware_hcs_guidance(
            observation, ego_action, step_index=2
        )
        self.assertEqual(event["target_local_index"], 2)
        self.assertTrue(event["target_locked"])
        self.assertEqual(wrapper._planner_aware_hcs_locked_target_local, 2)
        self.assertEqual(wrapper._planner_aware_hcs_locked_target_track_id, 22)

        # Reordering rows may occur as active support changes. The identity
        # lock follows resolved track_id=22 rather than the old local row 2.
        reordered = self._observation()
        order = torch.tensor([0, 2, 1])
        for key in (
            "scene_index", "track_id", "world_from_agent", "curr_speed",
            "yaw", "extent",
        ):
            reordered[key] = reordered[key][order]
        reordered_event = wrapper._configure_planner_aware_hcs_guidance(
            reordered, ego_action, step_index=3
        )
        self.assertEqual(reordered_event["target_local_index"], 1)
        self.assertEqual(reordered_event["target_track_id"], 22)
        self.assertEqual(wrapper._planner_aware_hcs_locked_target_local, 1)

        # The locked target becomes invalid. The selector must fail closed;
        # it may not switch to a different actor within the same episode.
        unavailable = self._observation()
        unavailable["curr_speed"][2] = 0.0
        self.assertIsNone(
            wrapper._configure_planner_aware_hcs_guidance(
                unavailable, ego_action, step_index=3
            )
        )
        self.assertIsNone(traffic.model.config)
        self.assertIsNone(traffic.model.flow_velocity_guidance_config)
        self.assertEqual(wrapper._planner_aware_hcs_locked_target_local, 2)
        self.assertEqual(wrapper._planner_aware_hcs_locked_target_track_id, 22)

        self.assertIsNone(
            wrapper._configure_planner_aware_hcs_guidance(
                observation, ego_action, step_index=4
            )
        )
        self.assertIsNone(traffic.model.config)
        self.assertIsNone(traffic.model.flow_velocity_guidance_config)

    def test_event_action_info_obeys_to_numpy_contract(self):
        traffic = _FakeTrafficPolicy()
        wrapper = PlannerOverrideRolloutWrapper(
            ego_policy=object(),
            agents_policy=traffic,
            safesim_guidance=True,
            planner_aware_hcs=True,
            planner_aware_hcs_config={"dt": 1.0},
        )
        event = {
            "collision_type": "rear_end",
            "conflict_point_world": [1.25, -2.5],
            "conflict_step": 7,
            "score": 0.75,
            "target_locked": True,
            "details": {"timing_score": 0.5},
        }
        serialized = wrapper._planner_aware_hcs_event_to_action_info(event)
        action = RolloutAction(
            agents=Action(
                positions=torch.zeros((1, 2, 2)),
                yaws=torch.zeros((1, 2, 1)),
            ),
            agents_info={"planner_aware_hcs_event": serialized},
        ).to_numpy()
        restored = action.agents_info["planner_aware_hcs_event"]
        self.assertEqual(restored["collision_type"], "rear_end")
        self.assertEqual(restored["conflict_step"].item(), 7)
        self.assertAlmostEqual(restored["score"].item(), 0.75)
        self.assertTrue(restored["target_locked"].item())
        self.assertEqual(restored["conflict_point_world"].shape, (2,))

    def test_inactive_window_uses_exact_native_observation(self):
        traffic = _FakeTrafficPolicy()
        wrapper = PlannerOverrideRolloutWrapper(
            ego_policy=_FakeEgoPolicy(),
            agents_policy=traffic,
            safesim_guidance=True,
            planner_aware_hcs=True,
            planner_aware_hcs_config={
                "dt": 1.0,
                "start_step": 2,
                "end_step": 4,
            },
        )
        observation = {
            "target_positions": torch.zeros((3, 8, 2)),
            "target_yaws": torch.zeros((3, 8, 1)),
            "scene_index": torch.zeros(3, dtype=torch.long),
        }
        wrapper.get_action({"agents": observation}, step_index=0)
        self.assertIs(traffic.last_observation, observation)
        self.assertNotIn(
            "planner_reference_positions", traffic.last_observation
        )
        self.assertIsNone(traffic.model.config)
        self.assertIsNone(traffic.model.flow_velocity_guidance_config)

    def test_matched_shuffled_semantics_never_changes_executed_action(self):
        donor_positions = np.stack(
            (np.zeros(8), np.arange(1.0, 9.0)), axis=-1
        ).astype(np.float32)
        donor_yaws = np.full((8, 1), math.pi / 2.0, dtype=np.float32)
        plan_sha = hashlib.sha256(
            donor_positions.tobytes(order="C")
            + donor_yaws.tobytes(order="C")
        ).hexdigest()
        payload = {
            "schema": "m2c_b3a_r2_semantic_plan_v1",
            "planner": "strive",
            "seed": 1,
            "source_hdf5": "/immutable/donor/data.hdf5",
            "source_frame": 20,
            "plan_sha256": plan_sha,
            "positions": donor_positions.tolist(),
            "yaws": donor_yaws.tolist(),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=True
        ) as stream:
            json.dump(payload, stream)
            stream.flush()
            wrapper = PlannerOverrideRolloutWrapper(
                ego_policy=object(),
                agents_policy=_FakeTrafficPolicy(),
                safesim_guidance=True,
                planner_aware_hcs=True,
                planner_aware_hcs_config={
                    "semantic_control": "matched_shuffled",
                    "semantic_plan_json": stream.name,
                },
            )

        executed = Action(
            positions=torch.full((1, 8, 2), 17.0),
            yaws=torch.full((1, 8, 1), -0.25),
        )
        executed_positions = executed.positions.clone()
        executed_yaws = executed.yaws.clone()
        semantic = wrapper._planner_aware_hcs_semantic_action(
            executed, step_index=20
        )
        self.assertTrue(torch.equal(executed.positions, executed_positions))
        self.assertTrue(torch.equal(executed.yaws, executed_yaws))
        self.assertTrue(
            torch.equal(
                semantic.positions[0], torch.from_numpy(donor_positions)
            )
        )
        self.assertTrue(
            torch.equal(semantic.yaws[0], torch.from_numpy(donor_yaws))
        )
        self.assertIsNot(semantic, executed)


if __name__ == "__main__":
    unittest.main()
