"""Read prepared nuPlan records through legacy trajdata without raw-map dependencies."""
from pathlib import Path
import dill


def install(reference):
    from trajdata.data_structures.environment import EnvMetadata
    from trajdata.data_structures.scene_metadata import Scene
    from trajdata.dataset_specific.raw_dataset import RawDataset
    from trajdata.utils import env_utils

    cache = Path(reference['cache_location'])
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cache_adapter'))
    from readonly_maps import install as install_readonly_maps
    records = dill.load((cache / 'nuplan_mini/scenes_list.dill').open('rb'))
    selected = next(row for row in records if row.name == reference['scene_id'])

    class CachedNuplanDataset(RawDataset):
        def compute_metadata(self, env_name, data_dir):
            return EnvMetadata(name=env_name, data_dir=data_dir, dt=.05,
                parts=[(selected.split,), (selected.location,)],
                scene_split_map={row.name.split('=')[0]: row.split for row in records},
                map_locations=(selected.location,))

        def load_dataset_obj(self, verbose=False):
            return None

        def _get_matching_scenes_from_cache(self, scene_tag, scene_desc_contains, env_cache):
            if scene_desc_contains is not None:
                raise ValueError('Scene-description filtering is unsupported for the fixed cached scene')
            if selected.location not in scene_tag or selected.split not in scene_tag:
                return []
            return [Scene(self.metadata, selected.name, selected.location, selected.split,
                          selected.length, selected.data_idx, None)]

        def _get_matching_scenes_from_obj(self, scene_tag, scene_desc_contains, env_cache):
            return self._get_matching_scenes_from_cache(scene_tag, scene_desc_contains, env_cache)

        def get_scene(self, scene_info):
            if isinstance(scene_info, Scene):
                return scene_info
            raise RuntimeError('Shared nuPlan scene cache is missing; prepare it with the modern CPU runtime')

        def get_agent_info(self, *args, **kwargs):
            raise RuntimeError('Shared nuPlan agent cache is missing; raw-data rebuilding is forbidden here')

    readonly = install_readonly_maps(CachedNuplanDataset, cache)

    original = env_utils.get_raw_dataset

    def get_raw_dataset(dataset_name, data_dir):
        if dataset_name == 'nuplan_mini':
            return CachedNuplanDataset(dataset_name, data_dir, parallelizable=False, has_maps=True)
        return original(dataset_name, data_dir)

    env_utils.get_raw_dataset = get_raw_dataset
    return dict(adapter='legacy_trajdata_cache_only_nuplan', cache_location=str(cache),
                scene_id=selected.name, centered_cache_reused_without_transform=True,
                raw_data_or_map_rebuilding_allowed=False, readonly_maps=readonly)
