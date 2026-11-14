"""Common interface for rule-based and learned ego planners."""

from abc import abstractmethod

from tbsim.policies.base import Policy


class EgoPlannerPolicy(Policy):
    """tbsim policy with an explicit planner-facing API.

    ``candidate_agents_future`` is deliberately part of the contract even for
    K=0 planners that ignore it. A run may only be labelled strategy-depth 1
    when the selected planner advertises and exercises candidate-future use.
    """

    supports_candidate_future = False

    def eval(self):
        return self

    @abstractmethod
    def plan(
        self,
        ego_observation,
        candidate_agents_future=None,
        strategy_depth=0,
        agent_observation=None,
        **kwargs,
    ):
        raise NotImplementedError

    def get_action(
        self,
        obs_dict,
        candidate_agents_future=None,
        strategy_depth=0,
        agent_obs=None,
        **kwargs,
    ):
        if strategy_depth > 0 and not self.supports_candidate_future:
            raise ValueError(
                "strategy_depth>0 requested for a planner that does not use "
                "candidate_agents_future"
            )
        return self.plan(
            obs_dict,
            candidate_agents_future=candidate_agents_future,
            strategy_depth=strategy_depth,
            agent_observation=agent_obs,
            **kwargs,
        )
