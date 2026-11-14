"""Execute each external planner's returned trajectory without a second tracker."""
import json
from pathlib import Path

import torch
from ccdiff.planners.base import EgoPlannerPolicy


class DirectEgoPolicy(EgoPlannerPolicy):
    supports_candidate_future = False

    def __init__(self, planner, device, directory):
        super().__init__(device)
        self.planner = planner
        self.name = getattr(planner, 'name', type(planner).__name__)
        self.directory = Path(directory)

    def plan(self, ego_observation, candidate_agents_future=None, strategy_depth=0,
             agent_observation=None, step_index=0, **kwargs):
        action, info = self.planner.plan(
            ego_observation, candidate_agents_future=candidate_agents_future,
            strategy_depth=strategy_depth, agent_observation=agent_observation,
            step_index=step_index, **kwargs)
        if action.positions.shape[0] != 1 or not all(
                bool(torch.isfinite(value).all()) for value in (action.positions, action.yaws)):
            raise ValueError('Expected one finite supplied ego trajectory')
        record = dict(step_index=int(step_index), planner=self.name,
                      positions=action.positions.detach().cpu().tolist(),
                      yaws=action.yaws.detach().cpu().tolist(),
                      world_from_agent=ego_observation['world_from_agent'].detach().cpu().tolist(),
                      raw_planner_reference_executed_directly=True,
                      execution_protocol='supplied_planner_trajectory_exact_v5')
        with (self.directory / 'direct_ego_actions.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        return action, dict(info, execution_protocol='supplied_planner_trajectory_exact_v5')

    def eval(self):
        self.planner.eval()
        return self

    def close(self):
        if hasattr(self.planner, 'close'):
            self.planner.close()
