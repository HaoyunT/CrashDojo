"""Persist the selected actor's native endpoint speed across executed prefixes.

The observation/conditioning history is preserved. Only the decoder's current
dynamic state uses its previously executed endpoint speed. The next observed
pose must agree with the committed prefix before that state is reused.
"""
import math
from types import MethodType

import torch

from causal_native_initial import initial_endpoint_speed
from signed_candidate_kinematics import signed_candidate_trajectory


class PersistentSelectedState:
    def __init__(self, dt=.1, prefix=5):
        if dt <= 0 or prefix < 1:
            raise ValueError('Positive time step and executed prefix required')
        self.dt, self.prefix = dt, prefix
        self.target = self.step = self.initial_state = self.pending = None
        self.records, self.commits = [], []
        self.transition_verified = False

    def begin(self, observation, target, step):
        if type(target) is not int or type(step) is not int:
            raise ValueError('Integer actor and step required')
        if self.target is not None and self.target != target:
            raise ValueError('Selected actor changed within a rollout')
        if self.step is not None and self.pending is None:
            raise ValueError('Previous action was not committed')
        self.transition_verified = False
        if self.pending is None:
            if step != 0:
                raise ValueError('First observed step must be zero')
        else:
            if step != self.pending['next_step']:
                raise ValueError('Unexpected number of executed steps')
            xy = observation['centroid'][target].detach().cpu().reshape(-1)
            yaw = float(observation['yaw'][target].detach().cpu().reshape(-1)[0])
            expected = torch.tensor(self.pending['world_xy'], dtype=xy.dtype)
            error = float((xy-expected).abs().max())
            difference = yaw-self.pending['world_yaw']
            angle_error = abs(math.atan2(math.sin(difference),math.cos(difference)))
            if error > .002 or angle_error > .002:
                raise ValueError('Observed pose does not match the committed executed prefix')
            self.transition_verified = True
        self.target, self.step, self.initial_state = target, step, None

    def install(self, policy):
        owner = getattr(policy, '_persistent_selected_state_owner', None)
        if owner is self:
            return
        if owner is not None:
            raise ValueError('Policy already belongs to another state tracker')
        original = policy.get_aux_info
        tracker = self

        def get_aux_info(_policy, batch, *args, **kwargs):
            aux = original(batch, *args, **kwargs)
            if tracker.target is None:
                raise ValueError('State tracker was not bound to an observation')
            states = aux['curr_states']
            if states.ndim != 3 or states.shape[0] != 1 or states.shape[-1] != 4:
                raise ValueError('Expected one scene with native unicycle states')
            observed = float(states[0,tracker.target,2])
            if tracker.pending is None:
                history=batch['history_positions'][0,tracker.target,-3:]
                available=batch['history_availabilities'][0,tracker.target,-3:]
                if not bool(available.all()):raise ValueError('Three real observed positions required')
                initial=initial_endpoint_speed(history,tracker.dt)
                if not -10. <= float(initial) <= 30.:raise ValueError('Observed endpoint estimate outside native bounds')
                result=dict(aux);result['curr_states']=states.clone()
                result['curr_states'][0,tracker.target,2]=initial
                origin = 'causal_endpoint_speed_from_three_unchanged_observed_positions'
            else:
                if not tracker.transition_verified:
                    raise ValueError('Cannot reuse a state before verifying execution')
                result = dict(aux)
                result['curr_states'] = states.clone()
                result['curr_states'][0,tracker.target,2] = tracker.pending['endpoint_speed']
                origin = 'previous_native_endpoint_after_verified_execution'
            used = result['curr_states'][0,tracker.target].detach().clone()
            if not bool(torch.isfinite(used).all()):
                raise ValueError('Nonfinite native initial state')
            tracker.initial_state = used
            tracker.records.append(dict(step=tracker.step,target=tracker.target,
                observed_interval_speed_mps=observed,native_endpoint_speed_mps=float(used[2]),
                speed_difference_mps=float(used[2])-observed,source=origin,
                previous_prefix_execution_verified=tracker.transition_verified,
                conditioning_history_modified=False))
            return result

        policy.get_aux_info = MethodType(get_aux_info,policy)
        policy._persistent_selected_state_owner = self

    def commit(self, observation, positions, yaws):
        if self.initial_state is None:
            raise ValueError('No native decoder state captured for this action')
        if self.commits and self.commits[-1]['step'] == self.step:
            raise ValueError('Action committed twice')
        if positions.ndim != 2 or positions.shape[0] < self.prefix or positions.shape[-1] != 2:
            raise ValueError('Incomplete selected action prefix')
        angles=yaws.reshape(-1)
        trajectory=signed_candidate_trajectory(positions[None,:self.prefix],angles[None,:self.prefix],
                                               self.initial_state[None],self.dt)
        endpoint_speed=float(trajectory[0,-1,2])
        if not -10.001 <= endpoint_speed <= 30.001:
            raise ValueError('Reconstructed speed exceeds unchanged native model bounds')
        # Match the executor's measured pose; a float32 inverse transform can
        # drift by millimeters at large world coordinates.
        origin=observation['centroid'][self.target]
        heading=observation['yaw'][self.target].reshape(())
        c,s=torch.cos(heading),torch.sin(heading)
        rotation=torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
        endpoint=positions[self.prefix-1]@rotation.T+origin
        yaw=float(angles[self.prefix-1])+float(heading)
        self.pending=dict(next_step=self.step+self.prefix,endpoint_speed=endpoint_speed,
                          world_xy=endpoint.detach().cpu().tolist(),world_yaw=yaw)
        self.commits.append(dict(step=self.step,target=self.target,
            native_start_speed_mps=float(self.initial_state[2]),native_end_speed_mps=endpoint_speed,
            expected_next_step=self.pending['next_step'],expected_next_world_xy=self.pending['world_xy'],
            scope='state_committed_for_prefix_execution_next_pose_must_verify'))

    def report(self):
        return dict(records=self.records,commits=self.commits,
            acceptance_thresholds_changed=False,observation_or_conditioning_history_changed=False,
            selected_native_state_only=True,unexecuted_future_used=False)
