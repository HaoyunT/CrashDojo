"""Freeze the modern runtime's initial population and observed planner history."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO = ROOT / 'code/CCDiff_M2c_planner'
BASE = Path('/root/path/to/workspace/home')
CONTEXTUAL = ROOT / 'launch/causal_guidance_v26_contextual_full_eval_20261101'


def prepare(job_file, output):
    outer = json.loads(Path(job_file).read_text())
    job = outer.get('source_job', outer)
    manifest = json.loads(Path(job['manifest']).read_text().splitlines()[0])
    os.environ.update(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', NVIDIA_TF32_OVERRIDE='0',
                      NPY_DISABLE_CPU_FEATURES='AVX512F', OPENBLAS_CORETYPE='Haswell',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    os.environ.update(V26_DATASET=job['dataset'], V26_NUPLAN_SCENE=manifest['scene_id'],
                      V26_COORDINATE_AUDIT=str(Path(output).parent / 'coordinate_audit'))
    sys.path[:0] = list(map(str, [CONTEXTUAL, CONTEXTUAL / 'overlay', CONTEXTUAL / 'map_deps', REPO,
        REPO / 'third_party/tbsim', BASE / 'CCDiff/third_party/trajdata/src',
        BASE / 'planner_sources_20260810/nuplan_source/nuplan-devkit-master']))
    import numpy as np
    import torch
    from numeric_policy import configure
    configure()
    torch.set_num_threads(1)
    from tbsim.configs.scene_edit_config import SceneEditingConfig
    from tbsim.utils.config_utils import get_experiment_config_from_file, translate_pass_trajdata_cfg
    from tbsim.utils.batch_utils import set_global_batch_type
    from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
    from tbsim.evaluation import env_builders
    from causal_observation import observed_source
    if job['dataset'] == 'nuplan':
        from nuplan_adapter import install
        install()
    config = json.loads(Path(job['config']).read_text())
    cfg = SceneEditingConfig(registered_name='trajdata_nusc_ccflow_matching')
    cfg.update(**config)
    if job['dataset'] == 'nuplan' and job.get('nuplan_cache'):
        cfg.trajdata_cache_location = str(job['nuplan_cache'])
    cfg.num_simulation_steps = int(job['steps'])
    cfg.num_scenes_per_batch = 1
    cfg.metrics.compute_analytical_metrics = False
    exp = get_experiment_config_from_file(ROOT / 'models/original150k_bnonly_v1/flow/run0/config.json')
    set_global_batch_type('trajdata')
    env_name = next(iter(cfg.trajdata_data_dirs))
    set_global_trajdata_batch_env(env_name)
    set_global_trajdata_batch_raster_cfg(exp.env.rasterizer)
    original = env_builders.UnifiedDataset

    def serial(*args, **kwargs):
        kwargs['num_workers'] = 0
        return original(*args, **kwargs)

    env_builders.UnifiedDataset = serial
    env = env_builders.EnvUnifiedBuilder(cfg, exp, torch.device('cpu')).get_env(
        split_ego=False, parse_obs=True, planner_override=True)
    indices = [i for i in range(env._num_total_scenes)
               if env.dataset.get_scene(i).name == manifest['scene_id']]
    if len(indices) != 1:
        raise ValueError('Expected one exact native scene, got ' + repr(indices))
    start = int(manifest['start_step_10hz'])
    env.reset(scene_indices=indices, start_frame_index=start)
    names = list(env.current_agent_names)
    if names.count('ego') != 1 or len(names) != len(set(names)):
        raise ValueError('Invalid shared initial actor population')
    obs = env.get_observation()['agents']
    needed = ['agent_hist', 'agent_hist_extent', 'scene_index', 'track_id', 'type',
              'map_names', 'centroid', 'yaw', 'world_from_agent', 'agent_from_world', 'extent']
    reference = {key: copy.deepcopy(obs[key]) for key in needed}
    source = observed_source(env.get_observation(return_raw=True), names, start, 31, env)
    if np.asarray(reference['agent_hist']).shape != (len(names), 31, 8):
        raise ValueError('Modern reference must have 31 real observed frames')
    row = dict(scene_id=manifest['scene_id'], start_frame=start, names=names,
               observed_reference=reference, observed_source=source,
               cache_location=str(cfg.trajdata_cache_location),
               lane_params=dict(cfg.evaluation_vec_map_params),
               config=str(job['config']), scene_index=indices[0], dataset=job['dataset'],
               data_dirs=dict(cfg.trajdata_data_dirs),
               desired_data=list(cfg.trajdata_source_test),
               neighbor_distance=float(translate_pass_trajdata_cfg(exp).max_agents_distance),
               generated_from='modern_EnvUnifiedBuilder_real_initial_observation',
               used_recorded_future_for_planner_history=False)
    torch.save(row, output)
    audit = dict(scene_id=row['scene_id'], start_frame=start, actor_names=names,
                 history_frames=31, observed_past_only=True,
                 common_population_source=row['generated_from'], cache_location=row['cache_location'])
    Path(output).with_suffix('.json').write_text(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.job_file, arguments.output)
