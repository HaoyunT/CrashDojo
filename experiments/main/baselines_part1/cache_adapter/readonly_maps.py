"""Require the warmed private nuPlan maps and suppress implicit rebuilds."""
from pathlib import Path

from isolate_cache import PRIVATE, ready


def verify(cache):
    if Path(cache).resolve() != PRIVATE.resolve():
        raise ValueError('nuPlan evaluation must use the queue private cache')
    return ready()


def install(dataset_class, cache):
    verify(cache)

    def cache_maps(self, cache_path, map_cache_class, map_params):
        if float(map_params.get('px_per_m', 2.0)) != 2.0:
            raise ValueError('Private nuPlan maps are frozen at 2 pixels per meter')
        verify(cache_path)

    def cache_all_scenes_list(self, env_cache, all_scenes_list):
        # UnifiedDataset may request cache writes even with rebuild_maps=False.
        verify(cache)

    dataset_class.cache_maps = cache_maps
    dataset_class.cache_all_scenes_list = cache_all_scenes_list
    return dict(cache_location=str(PRIVATE), maps_read_only=True,
                scene_registry_read_only=True, readiness_marker=str(PRIVATE / 'maps_ready.json'))
