import unittest

import torch

from tbsim.policies.common import Action
from tbsim.policies.wrappers import GenerativeEgoReplayRolloutWrapper


class _MockFlowScenePolicy:
    def __init__(self):
        self.device = torch.device("cpu")

    def eval(self):
        return self

    def get_action(self, obs, step_index=None):
        batch_size, horizon = obs["target_positions"].shape[:2]
        positions = torch.full((batch_size, horizon, 2), 40.0)
        yaws = torch.zeros((batch_size, horizon, 1))
        return Action(positions=positions, yaws=yaws), {}


class GenerativeEgoWrapperTest(unittest.TestCase):
    def test_flow_ego_keeps_generated_ego_and_replays_traffic(self):
        wrapper = GenerativeEgoReplayRolloutWrapper(
            _MockFlowScenePolicy(), "flow_matching"
        )
        target_positions = torch.zeros((2, 4, 2))
        target_positions[0, :, 0] = 5.0
        target_positions[1, :, 0] = 9.0
        rollout = wrapper.get_action(
            {
                "agents": {
                    "target_positions": target_positions,
                    "target_yaws": torch.zeros((2, 4, 1)),
                    "agent_name": ["ego", "vehicle-a"],
                    "scene_index": torch.zeros(2, dtype=torch.long),
                }
            },
            step_index=0,
        )
        self.assertTrue(
            torch.allclose(
                rollout.agents.positions[0], torch.full((4, 2), 40.0)
            )
        )
        self.assertTrue(
            torch.allclose(
                rollout.agents.positions[1, :, 0], torch.full((4,), 9.0)
            )
        )
        self.assertTrue(
            bool(
                rollout.agents_info[
                    "generative_ego_transport_is_flow"
                ].all()
            )
        )


if __name__ == "__main__":
    unittest.main()
