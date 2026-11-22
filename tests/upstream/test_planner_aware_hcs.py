import math
import unittest

import torch

from ccdiff.models.planner_aware_hcs import (
    PlannerAwareHCS,
    PlannerAwareHCSSelectionError,
)


class PlannerAwareHCSTest(unittest.TestCase):
    @staticmethod
    def _observation(speeds=(0.0, 1.0, 1.0)):
        transforms = torch.eye(3).repeat(3, 1, 1)
        transforms[0, :2, 2] = torch.tensor([0.0, 0.0])
        transforms[1, :2, 2] = torch.tensor([2.0, 2.0])
        transforms[2, :2, 2] = torch.tensor([4.0, -4.0])
        return {
            "scene_index": torch.zeros(3, dtype=torch.long),
            "world_from_agent": transforms,
            "curr_speed": torch.tensor(speeds),
            "yaw": torch.tensor([0.0, math.pi / 2.0, math.pi / 2.0]),
            "extent": torch.tensor(
                [[4.5, 2.0, 1.5], [4.5, 2.0, 1.5], [4.5, 2.0, 1.5]]
            ),
            # The selector must never inspect this logged/future field.
            "target_positions": torch.full((3, 8, 2), 9999.0),
        }

    @staticmethod
    def _ego_plan():
        return torch.stack(
            (torch.arange(1.0, 9.0), torch.zeros(8)), dim=-1
        )

    def test_selects_actual_plan_intersection_not_nearest_agent(self):
        selector = PlannerAwareHCS(
            dt=1.0,
            min_conflict_step=0,
            min_bystander_gap_m=-10.0,
        )
        event = selector.select(self._observation(), self._ego_plan())
        self.assertEqual(event.target_local_index, 2)
        self.assertEqual(event.collision_type, "side")
        self.assertEqual(event.conflict_step, 3)
        self.assertAlmostEqual(event.conflict_time_s, 4.0)
        self.assertAlmostEqual(event.conflict_point_world[0], 4.0, places=5)
        self.assertAlmostEqual(event.conflict_point_world[1], 0.0, places=5)
        self.assertEqual(event.details["planner_reference_used"], 1.0)

    def test_logged_future_cannot_change_selection(self):
        selector = PlannerAwareHCS(
            dt=1.0,
            min_conflict_step=0,
            min_bystander_gap_m=-10.0,
        )
        observation = self._observation()
        first = selector.select(observation, self._ego_plan())
        observation["target_positions"].normal_(mean=-1000.0, std=50.0)
        second = selector.select(observation, self._ego_plan())
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_fails_closed_without_moving_candidate(self):
        selector = PlannerAwareHCS(dt=1.0, min_conflict_step=0)
        with self.assertRaises(PlannerAwareHCSSelectionError):
            selector.select(
                self._observation(speeds=(0.0, 0.0, 0.0)),
                self._ego_plan(),
            )

    @staticmethod
    def _coalition_observation(blocker_speeds=(1.0, 1.0)):
        transforms = torch.eye(3).repeat(5, 1, 1)
        transforms[:, :2, 2] = torch.tensor(
            [
                [0.0, 0.0],   # tested ego
                [20.0, 20.0], # irrelevant bystander
                [4.0, -4.0],  # primary crossing actor
                [4.0, 2.0],   # reaches the left escape corridor
                [4.0, -2.0],  # reaches the right escape corridor
            ]
        )
        return {
            "scene_index": torch.zeros(5, dtype=torch.long),
            "world_from_agent": transforms,
            "curr_speed": torch.tensor(
                [0.0, 1.0, 1.0, blocker_speeds[0], blocker_speeds[1]]
            ),
            "yaw": torch.tensor(
                [0.0, 0.0, math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0]
            ),
            "extent": torch.tensor([[4.5, 2.0, 1.5]] * 5),
            # Poisoned logged futures prove the selector is current-state only.
            "target_positions": torch.full((5, 8, 2), 7777.0),
        }

    def test_selects_two_escape_blockers_from_actual_plan(self):
        selector = PlannerAwareHCS(
            dt=1.0,
            min_conflict_step=0,
            min_bystander_gap_m=-10.0,
        )
        observation = self._coalition_observation()
        primary = selector.select(
            observation, self._ego_plan(), candidate_local_indices=[2]
        )
        event = selector.select_escape_coalition(
            observation,
            self._ego_plan(),
            primary,
            min_blockers=2,
            max_blockers=2,
            corridor_offset_m=6.0,
        )
        self.assertEqual(event.controlled_local_indices, (2, 3, 4))
        self.assertEqual(
            {member.role for member in event.members},
            {"block_left", "block_right"},
        )
        self.assertEqual(event.details["planner_reference_used"], 1.0)
        self.assertEqual(event.details["logged_future_used"], 0.0)

        observation["target_positions"].normal_(mean=-2000.0, std=100.0)
        repeated = selector.select_escape_coalition(
            observation,
            self._ego_plan(),
            primary,
            candidate_local_indices=[3, 4],
            corridor_offset_m=6.0,
        )
        self.assertEqual(event.to_dict(), repeated.to_dict())

    def test_coalition_fails_closed_when_locked_blocker_stops(self):
        selector = PlannerAwareHCS(
            dt=1.0,
            min_conflict_step=0,
            min_bystander_gap_m=-10.0,
        )
        observation = self._coalition_observation()
        primary = selector.select(
            observation, self._ego_plan(), candidate_local_indices=[2]
        )
        observation["curr_speed"][3] = 0.0
        with self.assertRaises(PlannerAwareHCSSelectionError):
            selector.select_escape_coalition(
                observation,
                self._ego_plan(),
                primary,
                candidate_local_indices=[3, 4],
                corridor_offset_m=6.0,
            )


if __name__ == "__main__":
    unittest.main()
