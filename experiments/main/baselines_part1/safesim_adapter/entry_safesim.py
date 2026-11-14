"""SafeSim native selection and guidance on the fixed mixed 150-scene cohort."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO = ROOT / 'code/CCDiff_M2c_planner'
BASE = Path('/root/path/to/workspace/home')
HERE = Path(__file__).resolve().parent
CONTEXTUAL = ROOT / 'launch/causal_guidance_v26_contextual_val150_20261030'


def load(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.splitlines()[0])


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True))


def official_preset(project, dataset, scene):
    module = 'selected_scene_config_nuplan_mini_val' if dataset == 'nuplan' else 'selected_scene_config'
    source = project / 'tbsim/configs' / (module + '.py')
    presets = runpy.run_path(str(source)).get('PREDEFINED_SCENE_ALL_INIT', {})
    return presets.get(scene), source


def main():
    os.environ.update(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', NVIDIA_TF32_OVERRIDE='0',
                      NPY_DISABLE_CPU_FEATURES='AVX512F', OPENBLAS_CORETYPE='Haswell',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    args.job_file = args.job_file.resolve()
    args.output = args.output.resolve()
    outer = load(args.job_file)
    job = dict(outer.get('source_job', outer))
    method = outer['method'].lower()
    planner = outer.get('planner', job['planner'])
    job['planner'] = planner
    if method != 'safesim':
        raise ValueError('Only native SafeSim is supported')
    steps = args.steps or int(job['steps'])
    if steps <= 0 or steps % 5 or steps > int(job['steps']):
        raise ValueError('Invalid five-frame execution budget')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'native_inference_validation.json').exists():
        raise FileExistsError('Refusing to overwrite an existing native inference result')
    manifest = load(job['manifest'])
    scene, start = manifest['scene_id'], int(manifest['start_step_10hz'])
    project = ROOT / 'code' / 'safe-sim_legacy100'
    preset, preset_source = official_preset(project, job['dataset'], scene)
    reference_path = Path(outer.get('initial_reference', args.output / 'common_initial_reference.pt'))
    if not reference_path.exists():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_CORETYPE='Haswell',
            OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        producer = (HERE.parent / 'cache_adapter/prepare_reference_isolated.py'
                    if job['dataset'] == 'nuplan' else HERE / 'prepare_reference.py')
        subprocess.run([str(BASE / 'miniconda3/envs/ccdiff/bin/python'),
            str(producer), '--job-file', str(args.job_file.resolve()),
            '--output', str(reference_path)], check=True, env=env)
    os.environ.update(OPENBLAS_CORETYPE='Haswell', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled', WANDB_DISABLED='true',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', RULE_OUTPUT=str(args.output / 'rule_adapter'),
                      V26_DATASET=job['dataset'], V26_NUPLAN_SCENE=scene,
                      V26_COORDINATE_AUDIT=str(args.output / 'coordinate_audit'))
    paths = [HERE, project, REPO / 'legacy_common_adapter_v20_20261027',
             BASE / 'Friday/safe-sim/trajdata/src', REPO,
             BASE / 'CCDiff/third_party/trajdata/src', ROOT / 'launch/rule_eval_v1',
             BASE / 'planner_sources_20260810/nuplan_source/nuplan-devkit-master']
    sys.path[:0] = list(map(str, paths))
    os.chdir(project)
    os.environ.setdefault('CRASHDOJO_RULE_ADAPTER_DIR', str(CONTEXTUAL / 'rule_adapter'))
    import numpy as np
    import torch
    from ccdiff.evaluation.native_tracker_client import NativeTrackingEgoPolicy
    from ccdiff.evaluation.five_class_planners import build_selected_ego
    from observation_validity import install
    from legacy_native_rollout import NativeLegacyRollout
    import tbsim.evaluation.policy_composers as composers
    import tbsim.evaluation.env_builders as builders
    import tbsim.utils.env_utils as env_utils
    from tbsim.envs.env_trajdata import EnvUnifiedSimulation
    from trajdata.data_structures.data_index import DataIndex
    torch.set_num_threads(1)
    reference = torch.load(reference_path, map_location='cpu')
    if reference['scene_id'] != scene or reference['start_frame'] != start:
        raise ValueError('Common initial reference scene/start mismatch')
    if job.get('nuplan_cache') and Path(reference['cache_location']).resolve() != Path(job['nuplan_cache']).resolve():
        raise ValueError('Common initial reference does not use the job private nuPlan cache')
    if job['dataset'] == 'nuplan':
        from cached_nuplan import install as install_cached_nuplan
        save(args.output / 'legacy_dataset_compatibility.json', install_cached_nuplan(reference))
    save(args.output / 'observation_adapter.json', install(legacy=True))
    if method == 'safesim':
        from safesim_broadcast import install as install_broadcast
        save(args.output / 'native_guidance_compatibility.json', install_broadcast())
    original_index = DataIndex.__init__

    def utf8_index(self, data_index, verbose=False):
        rows = [((str(row[0]).encode('utf-8') if isinstance(row[0], (str, Path)) else row[0]), *row[1:])
                for row in data_index]
        return original_index(self, rows, verbose=verbose)

    DataIndex.__init__ = utf8_index

    class NativeExternalPlanner(composers.PolicyComposer):
        def get_policy(self, **kwargs):
            from numeric_policy import configure
            configure()
            raw = build_selected_ego(planner, self.device)
            return NativeTrackingEgoPolicy(raw, self.device, args.output), None

    composers.NativeExternalPlanner = NativeExternalPlanner
    original_dataset = builders.UnifiedDataset

    def shared_dataset(*dataset_args, **kwargs):
        desired = reference['desired_data']
        if job['dataset'] == 'nuplan':
            import dill
            records = dill.load((Path(reference['cache_location']) / 'nuplan_mini/scenes_list.dill').open('rb'))
            split = next(row.split for row in records if row.name == scene)
            desired = ['nuplan_mini-' + split if split.startswith('mini_') else split]
        kwargs.update(cache_location=Path(reference['cache_location']), desired_data=desired,
                      data_dirs=reference['data_dirs'], num_workers=0, rebuild_cache=False,
                      rebuild_maps=False, ego_only=False)
        return original_dataset(*dataset_args, **kwargs)

    builders.UnifiedDataset = shared_dataset
    original_filter = EnvUnifiedSimulation._disable_offroad_agents
    assignment = {}

    def use_authorized_fallback(env, reason):
        target = manifest.get('adversary_vehicle_id')
        if not target or target == 'ego' or target not in reference['names']:
            raise ValueError('Authorized v6 fallback target is invalid or outside the common population')
        assignment.update(ego_name='ego', adv_names=[target])
        env._native_selection_mode = 'user_authorized_v6_fallback'
        env._native_selection_reason = reason
        save(args.output / 'native_assignment_source.json', dict(scene_id=scene, start_frame=start,
            preset=preset, preset_source=str(preset_source), selected_actor_ids=[target],
            native_ego_actor_ids=['ego'], selection_valid=True,
            selection_mode='user_authorized_v6_fallback', native_selection=False,
            reused_ours_target=True, selection_adaptation_authorized=True,
            fallback_manifest=str(job['manifest']), fallback_reason=reason,
            native_guidance_preserved=True))

    def common_admission(self, simscene, *filter_args, **filter_kwargs):
        if simscene.scene_name != scene:
            raise ValueError('Unexpected scene in common admission')
        original_agents = list(simscene.agents)
        if method == 'safesim' and preset is None:
            use_authorized_fallback(self, 'No published native SafeSim predefined assignment for this scene')
        elif method == 'safesim':
            original_filter(self, simscene, *filter_args, **filter_kwargs)
            native_names = [actor.name for actor in simscene.agents]
            ego_name = preset.get('ego_name')
            selected_names = preset.get('adv_names')
            if ego_name is None:
                indices = preset.get('adv_indices', preset.get('ctrl_indices'))
                try:
                    ego_name = native_names[int(preset['ego_idx'])]
                    selected_names = [native_names[int(index)] for index in indices]
                except (IndexError, TypeError, KeyError):
                    ego_name, selected_names = None, []
            absent = [name for name in selected_names if name not in reference['names']]
            if ego_name != 'ego' or not selected_names or absent:
                reason = ('Native preset indices cannot bind to the live native population' if ego_name is None
                          else 'Native SafeSim victim is not protected ego' if ego_name != 'ego'
                          else 'Native SafeSim actor outside common pool: ' + repr(absent))
                use_authorized_fallback(self, reason)
            else:
                assignment.update(ego_name=ego_name, adv_names=selected_names)
                self._native_selection_mode = 'official_preset'
                self._native_selection_reason = None
                save(args.output / 'native_assignment_source.json', dict(scene_id=scene,
                    start_frame=start, preset=preset, preset_source=str(preset_source),
                    original_native_actor_names=native_names, selected_actor_ids=selected_names,
                    native_ego_actor_ids=[ego_name], selection_valid=True, reused_ours_target=False,
                    selection_mode='official_preset', native_selection=True,
                    selection_adaptation_authorized=False,
                    native_admission_applied_before_index_to_identity_binding=True,
                    native_admission_uses_recorded_future_for_parked_mask=True))
        by_name = {actor.name: actor for actor in original_agents}
        missing = [name for name in reference['names'] if name not in by_name]
        if missing:
            raise ValueError('Common initial population is absent from legacy scene: ' + repr(missing))
        simscene.agents = [by_name[name] for name in reference['names']]

    EnvUnifiedSimulation._disable_offroad_agents = common_admission

    def build_environment(eval_cfg, exp_config, device, agent_policy, *unused, **unused_kw):
        eval_cfg.unlock()
        eval_cfg.start_frame_index_each_episode = [start]
        eval_cfg.num_episode_repeats = 1
        eval_cfg.num_simulation_steps = steps
        # The nusc builder has the complete legacy lane/full-future extras;
        # the dataset wrapper supplies the actual nuPlan split and centered data.
        env = builders.EnvNuscBuilder(eval_cfg, exp_config, device).get_env(split_ego=True, parse_obs=True)
        indices = [index for index in range(env._num_total_scenes)
                   if env.dataset.get_scene(index).name == scene]
        if len(indices) != 1:
            raise ValueError('Native scene identity is ambiguous: ' + repr(indices))
        env._native_preset_assignment = assignment
        return env, indices, None

    env_utils.build_environment_and_scenes = build_environment
    original_rollout = env_utils.rollout_episodes

    def rollout(env, policy, **kwargs):
        kwargs.update(start_frame_index_each_episode=[start], num_episodes=1,
                      horizon=steps, n_step_action=5, init_recipe=dict(predefined_scene_init=None))
        if method == 'safesim':
            original_init = env.init_ego_and_target_agents

            def initialize(recipe):
                return original_init(dict(predefined_scene_init={scene: dict(assignment)}))

            env.init_ego_and_target_agents = initialize
            # This marker selects native initialization after reset has bound IDs.
            kwargs['init_recipe'] = dict(predefined_scene_init={scene: {'deferred_native_binding': True}})
        retained = {key: metric for key, metric in env._metrics.items()
                    if metric.__class__.__name__ == 'OffRoadRate'}
        save(args.output / 'native_metric_collection.json', dict(retained_native_metrics=list(retained),
            disabled_native_step_metrics=[key for key in env._metrics if key not in retained],
            native_ade_fde_retained_in_get_metrics=True,
            reason='NaN-unsafe vector geometry metrics disabled; center raster ORR retained'))
        env._metrics = retained
        lane = env.dataset.extras['closest_lane_point']

        def available_lane(element, **lane_kwargs):
            if not np.isfinite(element.agent_from_world_tf).all():
                lane_kwargs['ref_polyline_ids'] = 'None'
            return lane(element, **lane_kwargs)

        env.dataset.extras['closest_lane_point'] = available_lane
        wrapper = NativeLegacyRollout(policy, env, method, kwargs['control_config'], args.output,
                                      reference, job, steps)
        try:
            result = original_rollout(env, wrapper, **kwargs)
            wrapper.finalize()
            return result
        finally:
            wrapper.close()

    env_utils.rollout_episodes = rollout
    native_cfg = args.output / 'legacy_eval_config.json'
    save(native_cfg, dict(policy=dict(pos_to_yaw=False), start_frame_index_each_episode=[start]))
    sys.argv = [str(project / 'scripts/run_adv_simulation.py'), '--config_file', str(native_cfg),
        '--results_root_dir', str(args.output / 'native_results'),
        '--dataset_path', next(iter(reference['data_dirs'].values())),
        '--env', 'nuplan' if job['dataset'] == 'nuplan' else 'nusc',
        '--eval_class', 'NativeExternalPlanner', '--agent_eval_class', 'Diffusion' if method == 'safesim' else 'FM',
        '--ckpt_yaml', str(ROOT / 'configs' / (method + '_checkpoints.yaml')),
        '--scene_select_mode', 'no_collision', '--sim-steps', str(steps), '--n_step_action', '5',
        '--seed', '1', '--num_scenes_per_batch', '1', '--num_scenes_to_evaluate', '1',
        '--prefix', 'native_selection_']
    sys.argv += ['--guidance', '--guidance_fn', 'route_collision_ttc_causecollision',
                 '--guidance_params', 'combine_loss,ctrl_weights,[0.5,0.5,0.5,0.5]']
    save(args.output / 'command.json', dict(command=sys.argv, method=method, planner=planner,
         dataset=job['dataset'], scene_id=scene, start_frame=start, steps=steps,
         shared_initial_reference=str(reference_path), native_guidance=True,
         selection_policy='official_preset_with_authorized_v6_fallback' if method == 'safesim' else 'native_HCS',
         selection_adaptation_authorized=method == 'safesim',
         common_ego_execution='native_nuplan_lqr_bicycle_v1'))
    runpy.run_path(sys.argv[0], run_name='__main__')
    from audit_execution import audit
    execution_audit = audit(args.output, steps)
    save(args.output / 'common_execution_validation.json', execution_audit)
    validation_path = args.output / 'native_inference_validation.json'
    validation = load(validation_path)
    validation['executed_simulator_poses_audited'] = True
    validation['explicit_frame_availability_audited'] = True
    save(validation_path, validation)


if __name__ == '__main__':
    main()
