"""CCDiff native CIG/guide with exact supplied ego and logged background."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ccdiff.policies.hierarchical import CCDiffHybridPolicyControl
from tbsim.policies.common import Action, RolloutAction
from tbsim.policies.wrappers import PlannerOverrideRolloutWrapper, RolloutWrapper

from common_execution import compose
from execution_audit import record
from native_cig import rank_initial_history


def digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


class NativeCCDiffRollout150(RolloutWrapper):
    _slice_batch = PlannerOverrideRolloutWrapper._slice_batch

    def __init__(self, env, policy, model, job, directory, repo, steps):
        super().__init__(ego_policy=policy.ego_policy, agents_policy=policy.agents_policy)
        self.env, self.model, self.job = env, model, job
        self.directory, self.repo = Path(directory), Path(repo)
        self.steps, self.calls = steps, 0
        self.names = list(env.current_agent_names)
        if env.current_scene_names != [job['scene']] or self.names.count('ego') != 1:
            raise ValueError('Unexpected admitted scene or ego identity')
        self.ego = self.names.index('ego')
        self.selected = None
        self.hybrid = CCDiffHybridPolicyControl(self.device)
        self.hybrid.add_policy(self.agents_policy)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.before = digest(model)
        self.execution_audits = []
        self.guide_calls = 0

    def select(self, observation):
        ranking, provenance = rank_initial_history(
            observation, self.names, self.job['scene'], self.repo, self.directory)
        count = min(5, len(self.names))
        selected = ranking[:count]
        fill = []
        if len(selected) < count:
            # Preserve the released seeded fill, including its replacement default.
            fill = list(map(int, np.random.choice(
                [i for i in range(len(self.names)) if i not in selected], size=count-len(selected))))
            selected += fill
        self.selected = selected
        effective = list(dict.fromkeys(selected))
        self.hybrid.set_controllable_set(selected)
        active = self.model.ema_policy if self.model.use_ema else self.model.nets['policy']
        active.set_guidance_dim(selected, 5)
        value = dict(provenance, selected_indices=selected,
                     selected_actor_ids=[self.names[i] for i in selected],
                     effective_execution_indices=effective,
                     raw_selection_contains_duplicates=len(effective) != len(selected),
                     seeded_fill_indices=fill, released_control_budget=5,
                     selection_schedule='once_at_reset', externally_controlled_ego_id='ego',
                     executed_non_ego_actor_ids=[self.names[i] for i in effective if i != self.ego],
                     native_selected_ego_overridden_by_external_planner=self.ego in selected,
                     reused_ours_no_action=False, runtime_agent_names=self.names)
        (self.directory / 'native_selection.json').write_text(json.dumps(value, indent=2))

    def get_action(self, obs, step_index):
        if list(self.env.current_agent_names) != self.names or step_index != self.calls*5:
            raise ValueError('Native actor order or five-step clock changed')
        agents = obs['agents']
        if self.selected is None:
            self.select(agents)
        active = self.model.ema_policy if self.model.use_ema else self.model.nets['policy']
        guidance = active.current_perturbation_guidance
        if guidance is None or list(guidance.available_idx) != self.selected:
            raise ValueError('Native partial guidance is not bound to the CIG-selected actors')
        if not getattr(guidance, '_native150_audit_installed', False):
            original_perturb = guidance.perturb

            def perturb(*args, **kwargs):
                self.guide_calls += 1
                return original_perturb(*args, **kwargs)

            guidance.perturb = perturb
            guidance._native150_audit_installed = True
        calls_before = self.guide_calls
        ego_obs = self._slice_batch(agents, [self.ego], len(self.names))
        with torch.no_grad():
            ego_action, ego_info = self.ego_policy.get_action(
                ego_obs, step_index=step_index, strategy_depth=0,
                candidate_agents_future=None, agent_obs=agents)
            generated, info = self.agents_policy.get_action(agents, step_index=step_index)
        if self.guide_calls == calls_before:
            raise ValueError('Native perturbation guidance was not actually invoked')
        samples = info.get('action_samples')
        if not isinstance(samples, dict) or samples['positions'].shape[1] != 5:
            raise ValueError('Actual model candidate budget differs from K5')
        if not torch.isfinite(samples['positions']).all() or not torch.isfinite(samples['yaws']).all():
            raise FloatingPointError('Nonfinite native model candidates')
        # Native seeded fill may repeat an index; execution acts on actor identities.
        effective = list(dict.fromkeys(self.selected))
        positions, yaws, execution = compose(agents, generated, ego_action, effective, self.ego)
        execution.update(native_raw_selected_indices=list(self.selected),
                         native_raw_selection_contains_duplicates=len(effective) != len(self.selected),
                         effective_execution_indices=effective)
        controlled = execution['selected_non_ego_indices']
        prefixes = []
        for index in controlled:
            history = agents['history_positions'][index, -3:]
            if len(history) != 3 or not agents['history_availabilities'][index, -3:].all():
                raise ValueError('Native selected actor lacks the observed dynamics prefix')
            prefixes.append(dict(actor_index=index, actor_id=self.names[index],
                                 history_local_last3=history.detach().cpu().tolist(),
                                 selected_local_first5=positions[index, :5].detach().cpu().tolist()))
        record(self.directory, step_index, self.names, agents, positions, yaws, execution,
               action_dtype=positions.detach().cpu().numpy().dtype, selected_prefixes=prefixes)
        self.execution_audits.append(execution)
        active = self.model.ema_policy if self.model.use_ema else self.model.nets['policy']
        row = dict(step_index=int(step_index), selected_actor_ids=[self.names[i] for i in self.selected],
                   native_raw_selected_indices=list(self.selected),
                   effective_execution_indices=effective,
                   executed_non_ego_actor_ids=[self.names[i] for i in controlled],
                   executed_prefixes=prefixes, observed_candidates=5,
                   original_partial_guidance_class=type(active.current_perturbation_guidance).__name__,
                   custom_fixed_actor_guidance_installed=False,
                   native_guide_calls=self.guide_calls-calls_before,
                   ego_execution=ego_info.get('execution_protocol'))
        with (self.directory / 'native_selection_steps.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.calls += 1
        return RolloutAction(agents=Action(positions=positions, yaws=yaws), agents_info={})

    def finalize(self):
        after = digest(self.model)
        passed = self.calls == self.steps//5 and self.before == after
        value = dict(calls=self.calls, expected_calls=self.steps//5,
                     weights_and_buffers_unchanged=after == self.before,
                     before_sha256=self.before, after_sha256=after,
                     common_execution_calls=len(self.execution_audits),
                     common_execution_protocol='supplied_ego_native_selected_logged_background_v5',
                     supplied_ego_executed_exactly=all(x['supplied_ego_executed_exactly'] for x in self.execution_audits),
                     background_exact_replay=all(x['background_exact_replay'] for x in self.execution_audits),
                     comparison_track='method_owned_selection_and_guidance',
                     native_cig_recomputed=True, optimizer_steps=0, training=False,
                     native_guide_calls=self.guide_calls,
                     method=self.job['method'], dataset=self.job['dataset'],
                     planner=self.job['planner'], steps=self.steps,
                     native_selection=True, native_guidance=True,
                     initial_population_matched=True,
                     initial_reference_sha256=self.job['_initial_reference_sha256'],
                     reused_ours_target=False,
                     physical_acceptance_not_implied=True, passed=passed)
        (self.directory / 'native_inference_validation.json').write_text(json.dumps(value, indent=2))
        if not passed:
            raise AssertionError('Frozen native inference integrity failed')

    def close(self):
        if hasattr(self.ego_policy, 'close'):
            self.ego_policy.close()
