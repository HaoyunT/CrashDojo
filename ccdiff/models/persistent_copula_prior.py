"""Marginal-preserving persistent copula source for frozen flow matching.

The state stores the *independent* Gaussian innovations, not the correlated
output. Every source coordinate therefore keeps an exact N(0,1) marginal
while optional copulas add within-horizon, cross-replan, or cross-actor
dependence. No decoded endpoint or trajectory is modified. The joint source
is only white N(0,I) when every correlation parameter is zero.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PersistentCopulaConfig:
    persistence: float
    execute_steps: int
    actor_pairs: Tuple[Tuple[int, int], ...] = ()
    copula_rho: float = 0.0
    action_dims: Optional[Tuple[int, ...]] = None
    proposal_start_index: int = 1
    temporal_rho: float = 0.0
    temporal_action_dims: Optional[Tuple[int, ...]] = None

    def validate(self, shape: Sequence[int]) -> None:
        if len(shape) != 5:
            raise ValueError("source shape must be [B,K,M,T,A]")
        _, samples, actors, horizon, action_dim = [int(value) for value in shape]
        if not 0.0 <= float(self.persistence) < 1.0:
            raise ValueError("persistence must be in [0,1)")
        if not -1.0 < float(self.copula_rho) < 1.0:
            raise ValueError("copula_rho must be strictly inside (-1,1)")
        if not -1.0 < float(self.temporal_rho) < 1.0:
            raise ValueError("temporal_rho must be strictly inside (-1,1)")
        if not 1 <= int(self.execute_steps) < horizon:
            raise ValueError("execute_steps must be in [1,horizon)")
        if not 0 <= int(self.proposal_start_index) <= samples:
            raise ValueError("proposal_start_index exceeds K")
        receivers = [int(receiver) for receiver, _ in self.actor_pairs]
        anchors = [int(anchor) for _, anchor in self.actor_pairs]
        if len(set(receivers)) != len(receivers) or set(receivers).intersection(anchors):
            raise ValueError("receivers must be unique and cannot be anchors")
        if any(
            receiver < 0 or receiver >= actors or anchor < 0 or anchor >= actors
            or receiver == anchor
            for receiver, anchor in self.actor_pairs
        ):
            raise ValueError("invalid actor pair")
        dims = range(action_dim) if self.action_dims is None else self.action_dims
        if not tuple(dims) or any(int(dim) < 0 or int(dim) >= action_dim for dim in dims):
            raise ValueError("invalid action_dims")
        temporal_dims = (
            range(action_dim)
            if self.temporal_action_dims is None
            else self.temporal_action_dims
        )
        if not tuple(temporal_dims) or any(
            int(dim) < 0 or int(dim) >= action_dim for dim in temporal_dims
        ):
            raise ValueError("invalid temporal_action_dims")


class PersistentCopulaSource:
    """Stateful source sampler with exact per-call Gaussian marginal contract."""

    def __init__(self, config: PersistentCopulaConfig):
        self.config = config
        self._previous_independent = None
        self._previous_step = None

    @property
    def initialized(self) -> bool:
        return self._previous_independent is not None

    def reset(self) -> None:
        self._previous_independent = None
        self._previous_step = None

    def sample(
        self,
        shape: Sequence[int],
        *,
        global_replan_step: int,
        device=None,
        dtype=None,
        generator=None,
    ) -> torch.Tensor:
        self.config.validate(shape)
        step = int(global_replan_step)
        if step < 0:
            raise ValueError("global_replan_step must be non-negative")
        if self._previous_step is not None and step <= self._previous_step:
            raise ValueError("replan steps must be strictly increasing; reset per episode")

        independent = torch.randn(
            tuple(int(value) for value in shape),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        start = int(self.config.proposal_start_index)
        execute = int(self.config.execute_steps)
        persistence = float(self.config.persistence)
        if self._previous_independent is not None:
            if tuple(self._previous_independent.shape) != tuple(independent.shape):
                raise ValueError("source shape changed within one episode")
            overlap = independent.shape[3] - execute
            if start < independent.shape[1] and overlap > 0 and persistence > 0.0:
                innovation_scale = math.sqrt(1.0 - persistence * persistence)
                independent[:, start:, :, :overlap] = (
                    persistence
                    * self._previous_independent[:, start:, :, execute:]
                    + innovation_scale * independent[:, start:, :, :overlap]
                )

        # Save independent innovations.  Reusing correlated output here would
        # compound rho across calls and violate the exact per-call covariance.
        self._previous_independent = independent.detach().clone()
        self._previous_step = step
        output = independent.clone()

        # Add within-horizon dependence without changing any one-step marginal.
        # This is an AR(1) Gaussian copula on the FM source, not a decoded-action
        # filter or trajectory projection. Sample zero remains the native factual
        # draw so proposal comparisons keep an unmodified reference arm.
        temporal_rho = float(self.config.temporal_rho)
        if temporal_rho != 0.0 and start < output.shape[1]:
            innovation_scale = math.sqrt(1.0 - temporal_rho * temporal_rho)
            temporal_dims = (
                tuple(range(output.shape[-1]))
                if self.config.temporal_action_dims is None
                else tuple(int(dim) for dim in self.config.temporal_action_dims)
            )
            for dim in temporal_dims:
                for time_index in range(1, output.shape[3]):
                    output[:, start:, :, time_index, dim] = (
                        temporal_rho
                        * output[:, start:, :, time_index - 1, dim]
                        + innovation_scale
                        * independent[:, start:, :, time_index, dim]
                    )

        rho = float(self.config.copula_rho)
        if self.config.actor_pairs and rho != 0.0 and start < output.shape[1]:
            residual_scale = math.sqrt(1.0 - rho * rho)
            dims = (
                tuple(range(output.shape[-1]))
                if self.config.action_dims is None
                else tuple(int(dim) for dim in self.config.action_dims)
            )
            # Freeze the temporally transformed bases so multiple receivers do
            # not recursively compound actor correlation.
            copula_base = output.clone()
            for receiver, anchor in self.config.actor_pairs:
                receiver_base = copula_base[:, start:, int(receiver)]
                anchor_base = copula_base[:, start:, int(anchor)]
                for dim in dims:
                    output[:, start:, int(receiver), :, dim] = (
                        rho * anchor_base[..., dim]
                        + residual_scale * receiver_base[..., dim]
                    )
        return output

    def normalized_kl_contract(self, shape: Sequence[int], replan_calls: int):
        """Analytic Gaussian KL accounting, reported per modified coordinate."""
        self.config.validate(shape)
        batch, samples, actors, horizon, action_dim = [int(value) for value in shape]
        proposal_samples = max(0, samples - int(self.config.proposal_start_index))
        transitions = max(0, int(replan_calls) - 1)
        overlap = horizon - int(self.config.execute_steps)
        persistence_coordinates = (
            batch * proposal_samples * actors * overlap * action_dim * transitions
        )
        alpha = float(self.config.persistence)
        persistence_kl_per_coordinate = (
            0.0 if alpha == 0.0 else -0.5 * math.log(1.0 - alpha * alpha)
        )

        dims = action_dim if self.config.action_dims is None else len(self.config.action_dims)
        copula_coordinates = (
            batch * proposal_samples * horizon * dims * int(replan_calls)
        )
        rho = float(self.config.copula_rho)
        copula_kl_per_coordinate_per_receiver = (
            0.0 if rho == 0.0 else -0.5 * math.log(1.0 - rho * rho)
        )
        copula_receiver_coordinates = copula_coordinates * len(self.config.actor_pairs)
        temporal_dims = (
            action_dim
            if self.config.temporal_action_dims is None
            else len(self.config.temporal_action_dims)
        )
        temporal_transition_coordinates = (
            batch
            * proposal_samples
            * actors
            * max(0, horizon - 1)
            * temporal_dims
            * int(replan_calls)
        )
        temporal_rho = float(self.config.temporal_rho)
        temporal_kl_per_transition_coordinate = (
            0.0
            if temporal_rho == 0.0
            else -0.5 * math.log(1.0 - temporal_rho * temporal_rho)
        )
        return {
            "persistence_modified_coordinates": persistence_coordinates,
            "persistence_kl_per_coordinate": persistence_kl_per_coordinate,
            "persistence_total_kl": persistence_coordinates * persistence_kl_per_coordinate,
            "copula_receiver_coordinates": copula_receiver_coordinates,
            "copula_kl_per_receiver_coordinate": copula_kl_per_coordinate_per_receiver,
            "copula_total_kl": copula_receiver_coordinates * copula_kl_per_coordinate_per_receiver,
            "temporal_transition_coordinates": temporal_transition_coordinates,
            "temporal_kl_per_transition_coordinate": temporal_kl_per_transition_coordinate,
            "temporal_total_kl": (
                temporal_transition_coordinates
                * temporal_kl_per_transition_coordinate
            ),
            "note": (
                "These are component source-law KL terms. They are not a "
                "guarantee of decoded trajectory naturalness and are not "
                "blindly additive when mechanisms share coordinates."
            ),
        }
