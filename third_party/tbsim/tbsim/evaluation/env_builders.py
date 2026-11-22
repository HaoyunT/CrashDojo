#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

from copy import deepcopy
import numpy as np
from collections import defaultdict
import os
import fcntl
from pathlib import Path

from trajdata import AgentType, UnifiedDataset

from tbsim.configs.eval_config import EvaluationConfig
from tbsim.configs.base import ExperimentConfig
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation
from tbsim.envs.env_trajdata import EnvUnifiedSimulation, EnvSplitUnifiedSimulation
from tbsim.utils.trajdata_utils import TRAJDATA_AGENT_TYPE_MAP, get_closest_lane_point_wrapper, get_full_fut_traj, get_full_fut_valid
from tbsim.utils.config_utils import translate_pass_trajdata_cfg
import tbsim.envs.env_metrics as EnvMetrics

class EnvironmentBuilder(object):
    """Builds an simulation environment for evaluation."""
    def __init__(self, eval_config: EvaluationConfig, exp_config: ExperimentConfig, device):
        self.eval_cfg = eval_config
        self.exp_cfg = exp_config
        self.device = device

    def _get_analytical_metrics(self):
        metrics = dict(
            all_off_road_rate=EnvMetrics.OffRoadRate(),
            all_disk_off_road_rate=EnvMetrics.DiskOffRoadRate(),
            # all_sem_layer_rate=EnvMetrics.SemLayerRate(),
            all_collision_rate=EnvMetrics.CollisionRate(),
            all_disk_collision_rate=EnvMetrics.DiskCollisionRate(),
            # agents_collision_rate=EnvMetrics.CollisionRate(),
            all_coverage=EnvMetrics.OccupancyCoverage(
                gridinfo={"offset": np.zeros(2), "step": 2.0*np.ones(2)},
            ),
            all_diversity=EnvMetrics.OccupancyDiversity(
                gridinfo={"offset": np.zeros(2), "step": 4.0*np.ones(2)},
            ),
            all_failure=EnvMetrics.CriticalFailure(num_offroad_frames=2),
            all_comfort=EnvMetrics.Comfort(sim_dt=self.exp_cfg.algo.step_time, stat_dt=0.5),
        )
        return metrics

    def get_env(self):
        raise NotImplementedError

class EnvUnifiedBuilder(EnvironmentBuilder):
    def get_env(
        self,
        split_ego=False,
        parse_obs=True,
        planner_override=False,
    ):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        exp_cfg.lock()

        # the config used at training time
        # print('exp_cfg', exp_cfg)
        data_cfg = translate_pass_trajdata_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        agent_only_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_only_types]
        agent_predict_types = None
        if data_cfg.trajdata_predict_types is not None:
            agent_predict_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_predict_types]
        # TBD: hack to accomodate drivesim rollout data when not enough future data is available
        if 'drivesim' in self.eval_cfg.trajdata_data_dirs.keys():
            future_sec_min = 0.1
        else:
            future_sec_min = future_sec
        cache_location = Path(self.eval_cfg.trajdata_cache_location).expanduser()
        rebuild_cache = self.eval_cfg.trajdata_rebuild_cache
        rebuild_maps = self.eval_cfg.trajdata_rebuild_cache
        auto_rebuild = False
        desired_data = self.eval_cfg.trajdata_source_test
        if os.environ.get('V26_DATASET') == 'nuplan':
            selected_scene = os.environ.get('V26_NUPLAN_SCENE')
            # The v6 manifest mixes mini_train, mini_val, and mini_test records.
            # Select the matching trajdata split instead of trusting the generic config.
            if selected_scene:
                import dill
                scenes_path = cache_location / 'nuplan_mini' / 'scenes_list.dill'
                records = dill.load(scenes_path.open('rb'))
                split = next((r.split for r in records if r.name == selected_scene), None)
                if split is None:
                    raise KeyError('nuPlan scene is absent from scenes_list: ' + selected_scene)
                desired_data = ['nuplan_mini-' + split if split.startswith('mini_') else split]
                # Build only the selected missing scene; completed cache entries stay frozen.
                if not (cache_location / 'nuplan_mini' / selected_scene).is_dir():
                    rebuild_cache = True
                    # Map caches are already frozen; only the selected scene data needs rebuilding.
                    rebuild_maps = False
                    auto_rebuild = True
        kwargs = dict(
            cache_location=cache_location,
            desired_data=desired_data,
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec_min, future_sec),#(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs=self.eval_cfg.trajdata_data_dirs,
            only_types=agent_only_types,
            only_predict=agent_predict_types,
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=data_cfg.trajdata_incl_map,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "no_map_fill_value": data_cfg.no_map_fill_value,
            },
            incl_vector_map=True,
            vector_map_params = {
                "incl_road_lanes": True,                
                "incl_road_areas": False,                 
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,                 
                # Collation can be quite slow if vector maps are included,
                "no_collate": True,
            },
            # centric=data_cfg.trajdata_centric,
            # hardcode to agent-centric and then use a wrapper to support scene-centric
            # TBD: support scene-centric simulation
            centric='agent',
            scene_description_contains=data_cfg.trajdata_scene_desc_contains,
            standardize_data=data_cfg.trajdata_standardize_data,
            verbose=True,
            num_workers=os.cpu_count(),
            rebuild_cache=rebuild_cache,
            rebuild_maps=rebuild_maps,
            # A dictionary that contains functions that generate our custom data.
            # Can be any function and has access to the batch element.
            extras={
                "closest_lane_point": get_closest_lane_point_wrapper(self.eval_cfg.evaluation_vec_map_params),
                "full_fut_traj": get_full_fut_traj,
                "full_fut_valid": get_full_fut_valid,
            },
        )

        cache_lock = None
        if auto_rebuild:
            lock_path = cache_location / 'nuplan_mini' / ('.' + selected_scene + '.lock')
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            cache_lock = lock_path.open('w')
            fcntl.flock(cache_lock, fcntl.LOCK_EX)
            if (cache_location / 'nuplan_mini' / selected_scene).is_dir():
                kwargs['rebuild_cache'] = False
                kwargs['rebuild_maps'] = False
        try:
            env_dataset = UnifiedDataset(**kwargs)
        finally:
            if cache_lock is not None:
                fcntl.flock(cache_lock, fcntl.LOCK_UN)
                cache_lock.close()

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
            if split_ego or planner_override:
                metrics.update(
                    ego_collision_rate=EnvMetrics.EgoCollisionRate(),
                    ego_disk_collision_rate=EnvMetrics.EgoDiskCollisionRate(),
                    ego_off_road_rate=EnvMetrics.OffRoadRate(),
                    ego_disk_off_road_rate=EnvMetrics.DiskOffRoadRate(),
                    ego_comfort=EnvMetrics.Comfort(
                        sim_dt=self.exp_cfg.algo.step_time,
                        stat_dt=0.5,
                    ),
                    ego_progress=EnvMetrics.EgoProgress(),
                )

        # if we don't have a map, can't compute map-based metrics
        if not data_cfg.trajdata_incl_map:
            metrics.pop("all_off_road_rate", None)
            metrics.pop("all_sem_layer_rate", None)
            metrics.pop("all_coverage", None)
            metrics.pop("all_diversity", None)
            metrics.pop("all_failure", None)
            metrics.pop("ego_off_road_rate", None)
            metrics.pop("ego_disk_off_road_rate", None)

        if split_ego or planner_override:
            env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
                split_ego=split_ego,
                parse_obs=parse_obs,
                save_action_samples=self.eval_cfg.save_action_samples,
            )
        else:
            env = EnvUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
                save_action_samples=self.eval_cfg.save_action_samples,
            )

        return env
