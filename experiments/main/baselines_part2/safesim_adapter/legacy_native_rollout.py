"""Preserve native selection/guidance under the common execution contract."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from tbsim.policies.common import Action, RolloutAction
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils import tensor_utils as TensorUtils
from causal_observation import CausalPlannerObservation, observed_source, array
from lane_client import CanonicalLaneClient
from legacy_runtime_utils import configure_budget, state_digest, legacy_metadata
from common_execution import compose
from execution_audit import record


class NativeLegacyRollout(RolloutWrapper):
    def __init__(self, base, env, method, control_config, directory, reference, job, steps):
        super().__init__(ego_policy=base.ego_policy, agents_policy=base.agents_policy)
        self.env, self.method, self.control_config = env, method, control_config
        self.directory = Path(directory)
        self.job = job
        self.steps = int(steps)
        if self.steps <= 0 or self.steps % 5:
            raise ValueError('Execution steps must be positive and divisible by five')
        self.model = self.agents_policy
        while hasattr(self.model, 'policy'):
            self.model = self.model.policy
        self.active = self.model.nets['policy']
        self.budget = configure_budget(self.active)
        self.control_config.guide_config.params.num_samples = 5
        self.active.do_guidance = True
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.before = state_digest(self.model)
        self.lanes = CanonicalLaneClient(self.directory, reference)
        self.reference = reference
        self.view = None
        self.calls = 0
        self.execution_audits = []

    def get_action(self, obs, step_index):
        from numeric_policy import configure
        configure()
        names = list(self.env.current_agent_names)
        if step_index != self.calls * 5 or names.count('ego') != 1:
            raise ValueError('Unexpected native rollout step/ego identity')
        ego = names.index('ego')
        if list(np.flatnonzero(self.env.batch_ego_indices)) != [ego]:
            raise ValueError('Native victim ego differs from the external planner ego')
        if self.view is None:
            common_names = self.reference['names']
            if names != common_names:
                raise ValueError('Shared initial actor population/order differs from the common scene')
            indices = [common_names.index(name) for name in names]

            def subset(value):
                if isinstance(value, dict):
                    return {key: subset(item) for key, item in value.items()}
                if (isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim > 0
                        and value.shape[0] == len(common_names)):
                    return value[indices]
                if isinstance(value, list) and len(value) == len(common_names):
                    return [value[index] for index in indices]
                return copy.deepcopy(value)

            reference = subset(self.reference)
            reference['names'] = names
            initial_errors = {}
            for key in ['centroid', 'extent']:
                actual = array(obs['agents'][key])
                expected = array(reference['observed_reference'][key])
                if actual.shape != expected.shape:
                    raise ValueError('Common initial ' + key + ' shape differs')
                error = float(np.max(np.abs(actual - expected)))
                if not np.isfinite(error) or error > .01:
                    raise ValueError('Common initial ' + key + ' differs: ' + str(error))
                initial_errors[key + '_max_abs_error'] = error
            delta = array(obs['agents']['yaw']).reshape(-1) - array(reference['observed_reference']['yaw']).reshape(-1)
            error = float(np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta)))))
            if not np.isfinite(error) or error > .001:
                raise ValueError('Common initial heading differs')
            initial_errors['yaw_max_abs_error_rad'] = error
            self.view = CausalPlannerObservation(reference, self.lanes, reference['neighbor_distance'])
            (self.directory / 'native_population.json').write_text(json.dumps(dict(
                native_agent_names=names, common_agent_names=common_names,
                common_population_order_matches=names == common_names,
                population_protocol='common_initial_conditions_before_method_owned_selection',
                missing_common_agent_ids=[name for name in common_names if name not in names],
                external_history_source='real_initial_common_observed_history_subset_by_actor_id',
                initial_state_consistency=initial_errors,
            ), indent=2))
        source = observed_source(self.env.get_observation(return_raw=True), names,
                                 self.reference['start_frame'] + step_index, 11, self.env)
        external_ego, external_agents = self.view.build(obs['agents'], names, step_index, source)
        external_ego = TensorUtils.to_torch(external_ego, device=self.device, ignore_if_unspecified=True)
        external_agents = TensorUtils.to_torch(external_agents, device=self.device, ignore_if_unspecified=True)
        with torch.no_grad():
            ego_action, ego_info = self.ego_policy.get_action(
                external_ego, step_index=step_index, agent_obs=external_agents,
                strategy_depth=0, candidate_agents_future=None,
            )
            if ego_info.get('execution_protocol') != 'native_nuplan_lqr_bicycle_v1':
                raise ValueError('Common native tracker was not attested')
            obs['agents']['ego_plan'] = ego_action.positions
            generated, info = self.agents_policy.get_action(obs['agents'], step_index=step_index)
        samples = info.get('action_samples')
        if not isinstance(samples, dict) or samples['positions'].shape[1] != 5:
            raise ValueError('Native generator did not use K5 candidates')
        if not torch.isfinite(samples['positions']).all() or not torch.isfinite(samples['yaws']).all():
            raise FloatingPointError('Nonfinite native legacy candidates')
        selected = list(map(int, np.flatnonzero(self.env.batch_ctrl_indices)))
        selection_mode = (getattr(self.env, '_native_selection_mode', 'official_preset')
                          if self.method == 'safesim' else 'native_HCS')
        adapted = selection_mode == 'user_authorized_v6_fallback'
        configs = getattr(self.env, 'current_collision_configs', None)
        audit = dict(step_index=step_index, method=self.method,
            runtime_agent_names=names, selected_actor_ids=[names[i] for i in selected],
            selection_source=selection_mode, selection_mode=selection_mode,
            selection_adaptation_authorized=adapted, native_selection=not adapted,
            original_guidance_enabled=bool(self.active.do_guidance), native_collision_configs=configs,
            background_execution='exact_available_logged_replay', reused_ours_target=adapted,
            reused_ours_no_action=False, candidates=5, external_planner_observation=self.view.audits[-1])
        audit.update(start_frame=self.reference['start_frame'], scene_id=self.reference['scene_id'],
            native_preset_assignment=getattr(self.env, '_native_preset_assignment', None),
            selected_actor_observed_history_world=np.where(
                np.isfinite(self.view.world_history[selected]), self.view.world_history[selected], None).tolist(),
            observed_history_layout=['x', 'y', 'vx', 'vy', 'ax', 'ay', 'sin_yaw', 'cos_yaw'],
            observed_past_only=True, external_ego_info=ego_info)
        with (self.directory / 'native_selection_steps.jsonl').open('a') as stream:
            stream.write(json.dumps(audit, default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value)) + '\n')
        positions, yaws, execution = compose(obs['agents'], generated, ego_action, selected, ego)
        execution['selected_prefixes'] = [dict(actor_index=index,
            history_local_last3=external_agents['agent_hist'][index, -3:, :2].detach().cpu().numpy().tolist(),
            selected_local_first5=positions[index, :5].detach().cpu().numpy().tolist())
            for index in execution['selected_non_ego_indices']]
        # The legacy simulator combines split ego/agent actions into float64.
        record(self.directory, step_index, names, obs['agents'], positions, yaws, execution,
               action_dtype=np.float64)
        self.execution_audits.append(execution)
        generated = Action(positions=positions, yaws=yaws)
        # Keep the native split-action logging layout after composing all rows.
        if 'ego_idx' in obs['agents']:
            generated.eliminate_ego_action(obs['agents']['ego_idx'])
            self.eliminate_ego_action_in_dict(info, obs['agents']['ego_idx'])
        self.calls += 1
        return RolloutAction(ego_action, legacy_metadata(ego_info), generated, info)

    def finalize(self):
        after = state_digest(self.model)
        selection_mode = (getattr(self.env, '_native_selection_mode', 'official_preset')
                          if self.method == 'safesim' else 'native_HCS')
        adapted = selection_mode == 'user_authorized_v6_fallback'
        result = dict(calls=self.calls, expected_calls=self.steps // 5, weights_and_buffers_unchanged=self.before == after,
            before_sha256=self.before, after_sha256=after, budget=self.budget,
            common_execution_calls=len(self.execution_audits),
            common_execution_protocol='supplied_ego_native_selected_logged_background_v5',
            supplied_ego_executed_exactly=all(a['supplied_ego_executed_exactly'] for a in self.execution_audits),
            background_exact_replay=all(a['background_exact_replay'] for a in self.execution_audits),
            comparison_track=('native_guidance_with_user_authorized_v6_selection_adapter' if adapted
                              else 'method_owned_selection_and_guidance'), physical_acceptance_not_implied=True,
            passed=self.calls == self.steps // 5 and self.before == after,
            optimizer_steps=0, method=self.method, dataset=self.job['dataset'], planner=self.job['planner'],
            steps=self.steps, native_selection=not adapted, native_guidance=True, reused_ours_target=adapted,
            selection_mode=selection_mode, selection_adaptation_authorized=adapted,
            selection_fallback_reason=getattr(self.env, '_native_selection_reason', None),
            algorithm_label='SafeSim_selection_adapted' if adapted else self.method)
        (self.directory / 'native_inference_validation.json').write_text(json.dumps(result, indent=2))
        if self.calls != self.steps // 5 or self.before != after:
            raise AssertionError('Native legacy frozen inference contract failed')

    def close(self):
        self.lanes.close()
        if hasattr(self.ego_policy, 'close'):
            self.ego_policy.close()
