"""Isolated native CCDiff DDPM/FM evaluation on the fixed mixed cohort."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO = ROOT / 'code/CCDiff_M2c_planner'
CONTEXT = ROOT / 'launch/causal_guidance_v26_contextual_full_eval_20261101'
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    envelope = json.loads(args.job_file.read_text())
    job = copy.deepcopy(envelope.get('source_job', envelope))
    for key in ('method', 'planner'):
        if key in envelope:
            job[key] = envelope[key]
    if job.get('method') not in ('ccdiff_ddpm', 'ccdiff_fm'):
        raise ValueError('Expected an explicit CCDiff DDPM/FM method')
    steps = job['steps'] if args.steps is None else args.steps
    if not 0 < steps <= job['steps'] or steps % 5:
        raise ValueError('Steps must be a positive five-frame block within the frozen duration')
    args.output.mkdir(parents=True, exist_ok=True)
    attestation = args.output / 'native_entry_contract.json'
    if attestation.exists() or (args.output / 'native_results').exists():
        raise FileExistsError('Refuse to overwrite a native rollout')
    required = dict(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', NVIDIA_TF32_OVERRIDE='0',
                    OPENBLAS_CORETYPE='Haswell', NPY_DISABLE_CPU_FEATURES='AVX512F',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    os.environ.update(required)
    os.environ['V26_DATASET'] = job['dataset']
    os.environ['V26_NUPLAN_SCENE'] = job['scene']
    os.environ['V26_COORDINATE_AUDIT'] = str(args.output / 'coordinate_audit')
    os.environ['RULE_OUTPUT'] = str(args.output / 'rule_adapter')
    os.environ['CRASHDOJO_RULE_ADAPTER_DIR'] = str(CONTEXT / 'rule_adapter')
    os.environ['CCDIFF_PLANNER_GPU'] = os.environ.get('CUDA_VISIBLE_DEVICES', '') if job['planner'] in ('diffusion_planner', 'flow_planner') else ''
    sys.path[:0] = list(map(str, [HERE, CONTEXT, CONTEXT / 'map_deps', CONTEXT / 'overlay', REPO,
        REPO / 'third_party/tbsim',
        Path('/path/to/workspace/home/CCDiff/third_party/trajdata/src'),
        Path('/path/to/workspace/home/planner_sources_20260810/nuplan_source/nuplan-devkit-master'),
        CONTEXT / 'rule_adapter']))
    os.chdir(REPO)
    from numeric_policy import configure
    configure()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU is required')
    reference_path = envelope.get('initial_reference', job.get('initial_reference'))
    if not reference_path:
        raise ValueError('Native admission requires an explicit frozen initial_reference')
    reference_path = Path(reference_path)
    reference = torch.load(reference_path, map_location='cpu')
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    job['_initial_reference_sha256'] = reference_sha
    from tbsim.evaluation import env_builders
    original_dataset = env_builders.UnifiedDataset

    def serial_dataset(*values, **kwargs):
        kwargs['num_workers'] = 0
        if job['dataset'] == 'nuplan':
            kwargs.update(cache_location=Path(job['nuplan_cache']), rebuild_cache=False,
                          rebuild_maps=False)
        return original_dataset(*values, **kwargs)

    env_builders.UnifiedDataset = serial_dataset
    domain = json.loads(Path(job['config']).read_text())
    cfg = json.loads((ROOT / 'configs/ccdiff_paper_eval.json').read_text())
    for key in ('dataset_path', 'trajdata_cache_location', 'trajdata_rebuild_cache',
                'trajdata_source_test', 'trajdata_data_dirs', 'eval_scenes'):
        if key in domain:
            cfg[key] = copy.deepcopy(domain[key])
        if key in domain.get('trajdata', {}):
            cfg.setdefault('trajdata', {})[key] = copy.deepcopy(domain['trajdata'][key])
    cfg['policy']['num_action_samples'] = 5
    cfg['apply_guidance'] = True
    cfg['num_simulation_steps'] = steps
    cfg.setdefault('trajdata', {})['num_simulation_steps'] = steps
    if job['dataset'] == 'nuplan':
        cache = job.get('nuplan_cache', domain.get('trajdata', {}).get('trajdata_cache_location'))
        if not cache:
            raise ValueError('nuPlan requires an explicit centered runtime cache')
        if Path(reference['cache_location']).resolve() != Path(cache).resolve():
            raise ValueError('Common initial reference does not use the job private nuPlan cache')
        job['nuplan_cache'] = str(cache)
        os.environ['V26_NUPLAN_CACHE'] = str(cache)
        cfg['trajdata_cache_location'] = str(cache)
        cfg['trajdata']['trajdata_cache_location'] = str(cache)
        from nuplan_adapter import install
        install()
        sys.path.insert(0, str(HERE.parent / 'cache_adapter'))
        from readonly_maps import install as install_readonly_maps
        from trajdata.dataset_specific.nuplan.nuplan_dataset import NuplanDataset
        map_guard = install_readonly_maps(NuplanDataset, cache)
        (args.output / 'readonly_maps_validation.json').write_text(json.dumps(map_guard, indent=2))
    from availability_export import install as install_availability
    install_availability()
    # Retain raw logged future masks as well as the shared availability logger.
    import importlib.util
    validity_path = ROOT / 'launch/benchmark50_native_common_execution_v5_20261029/observation_validity.py'
    validity_spec = importlib.util.spec_from_file_location('native150_observation_validity', validity_path)
    validity = importlib.util.module_from_spec(validity_spec)
    validity_spec.loader.exec_module(validity)
    validity.install(legacy=False)
    import tbsim.utils.trajdata_utils as trajdata_utils
    original_map_loader = trajdata_utils.load_vec_map
    map_cache = cfg.get('trajdata', {}).get('trajdata_cache_location', cfg.get('trajdata_cache_location'))
    if not map_cache:
        raise ValueError('The native guide requires an explicit domain map cache')

    def load_map(map_name, cache_path=None):
        return original_map_loader(map_name, cache_path=map_cache)

    trajdata_utils.load_vec_map = load_map
    from ccdiff.evaluation.selection_manifest import resolve_scene
    record = json.loads(next(line for line in Path(job['manifest']).read_text().splitlines() if line.strip()))
    start = record['start_step_10hz']
    if type(start) is not int or start < 1:
        raise ValueError('An explicit positive start frame is required')
    if record['scene_id'] != job['scene'] or record['dataset'] != job['dataset']:
        raise ValueError('Source scene identity differs')
    if reference['scene_id'] != job['scene'] or reference['start_frame'] != start:
        raise ValueError('Common initial reference has a different scene/start clock')
    original_get_env = env_builders.EnvUnifiedBuilder.get_env

    def get_env(builder, *values, **kwargs):
        env = original_get_env(builder, *values, **kwargs)
        index = resolve_scene(env.dataset, record)
        builder.eval_cfg.eval_scenes = [index]
        original_reset = env.reset

        def reset(scene_indices=None, start_frame_index=None):
            if list(scene_indices or []) != [index]:
                raise ValueError('Native evaluation tried to change the frozen scene')
            if start_frame_index is not None and list(start_frame_index) != [start]:
                raise ValueError('Native evaluation tried to change the frozen start frame')
            result = original_reset(scene_indices=[index], start_frame_index=[start])
            if len(result) != 1 or not all(result):
                raise ValueError('The exact selected scene is not admissible; no replacement allowed')
            if list(env.current_agent_names) != reference['names']:
                raise ValueError('Native initial actor population/order differs from the common reference')
            import numpy as np
            actual_history = np.asarray(env.get_observation()['agents']['agent_hist'])
            expected_history = np.asarray(reference['observed_reference']['agent_hist'])
            if actual_history.shape != expected_history.shape or not np.allclose(
                    actual_history, expected_history, rtol=0, atol=1e-4, equal_nan=True):
                raise ValueError('Native initial observed history differs from the common reference')
            (args.output / 'native_admission_validation.json').write_text(json.dumps(dict(
                passed=True, scene=job['scene'], start_frame=start,
                runtime_agent_names=list(env.current_agent_names),
                initial_reference=str(reference_path), initial_reference_sha256=reference_sha,
                actor_order_exact=True, observed_history_equal_nan_atol=1e-4,
                reused_ours_target=False), indent=2))
            return result

        env.reset = reset
        return env

    env_builders.EnvUnifiedBuilder.get_env = get_env
    from ccdiff.evaluation.native_tracker_client import NativeTrackingEgoPolicy
    from ccdiff.evaluation.five_class_planners import build_selected_ego
    import ccdiff.planners as public
    import tbsim.utils.scene_edit_utils as edit
    from native_rollout import NativeCCDiffRollout150
    original_factory, original_rollout = public.build_ego_planner, edit.guided_rollout

    def build(name, device, strive_profile=None):
        # The shared factory dynamically delegates learned planners back here.
        # Restore its original delegate only for the synchronous construction.
        installed = public.build_ego_planner
        public.build_ego_planner = original_factory
        try:
            raw = build_selected_ego(name, device, strive_profile)
        finally:
            public.build_ego_planner = installed
        return NativeTrackingEgoPolicy(raw, device, args.output)

    def rollout(env, policy, model, **kwargs):
        if kwargs['start_frames'] != [start] or kwargs['n_step_action'] != 5 or kwargs['horizon'] != steps:
            raise ValueError('Native clock or execution budget differs from the frozen cohort')
        wrapper = NativeCCDiffRollout150(env, policy, model, job, args.output, REPO, steps)
        try:
            result = original_rollout(env, wrapper, model, **kwargs)
            wrapper.finalize()
            return result
        finally:
            wrapper.close()

    public.build_ego_planner, edit.guided_rollout = build, rollout
    config_path = args.output / 'native_config.json'
    config_path.write_text(json.dumps(cfg, indent=2))
    fm = job['method'] == 'ccdiff_fm'
    checkpoint = ROOT / 'models/original150k_bnonly_v1' / ('flow' if fm else 'diffusion') / 'run0'
    contract = dict(method=job['method'], dataset=job['dataset'], scene=job['scene'],
                    planner=job['planner'], steps=steps, full_steps=job['steps'], start_frame=start,
                    candidates=5, n_step_action=5, seed=1, native_selection=True,
                    native_guidance=True, reused_ours_target=False, training=False,
                    guidance_map_cache=str(map_cache),
                    ego_execution='same_source_NativeTrackingEgoPolicy_as_contextual150',
                    selection='released_CIG_recomputed_on_admitted_initial_history')
    contract['initial_reference_sha256'] = reference_sha
    attestation.write_text(json.dumps(contract, indent=2))
    sys.argv = [str(REPO / 'ccdiff/examples/scene_editor.py'), '--config_file', str(config_path),
        '--env', 'trajdata', '--eval_class', 'CCFlowMatching' if fm else 'CCDiff',
        '--registered_name', 'trajdata_nusc_ccflow_matching' if fm else 'trajdata_nusc_ccdiff',
        '--policy_ckpt_dir', str(checkpoint), '--policy_ckpt_key', 'last.ckpt',
        '--results_root_dir', str(args.output / 'native_results'),
        '--num_scenes_to_evaluate', '1', '--num_scenes_per_batch', '1',
        '--num_simulation_steps', str(steps), '--n_step_action', '5', '--seed', '1',
        '--ego_planner', job['planner'], '--strategy_depth', '0', '--scene_source', 'm2_guided',
        '--guidance_method', 'ccdiff', '--w_collision', '-50.0', '--w_offroad', '1.0',
        '--editing_source', 'config', 'heuristic', '--deterministic_pairing']
    # The native editor otherwise derives start from model history length.
    # Replace this one clock assignment in memory; model history stays intact.
    editor = Path(sys.argv[0])
    source = editor.read_text()
    clock_line = 'start_frame_index = [[exp_config.algo.history_num_frames+1]] * len(scene_indices)'
    if source.count(clock_line) != 1:
        raise ValueError('Unsupported scene editor start-clock assignment')
    patched = source.replace(clock_line, 'start_frame_index = [[%d]] * len(scene_indices)' % start)
    (args.output / 'native_start_clock_adapter.json').write_text(json.dumps(dict(
        source=str(editor), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        patched_source_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        replacement_count=1, start_frame=start, model_history_frames_unchanged=True,
        repository_modified=False), indent=2))
    exec(compile(patched, str(editor), 'exec'), dict(__name__='__main__', __file__=str(editor)))


if __name__ == '__main__':
    main()
