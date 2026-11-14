"""Run the legacy reference producer with every nuPlan cache path isolated."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from isolate_cache import ROOT, PRIVATE, cache_lock, paths, ready, sha


def prepare(job_file, output):
    outer = json.loads(Path(job_file).read_text())
    job = outer.get('source_job', outer)
    if job['dataset'] != 'nuplan':
        raise ValueError('This isolated producer accepts nuPlan jobs only')
    paths()
    adapter = ROOT / 'launch/baselines_native_fixed150_20261030/safesim_adapter'
    sys.path.append(str(adapter))
    from tbsim.evaluation import env_builders
    original_init = env_builders.EnvUnifiedBuilder.__init__
    original_dataset = env_builders.UnifiedDataset
    from trajdata.dataset_specific.nuplan.nuplan_dataset import NuplanDataset
    original_maps = NuplanDataset.cache_maps
    original_metadata = NuplanDataset.cache_all_scenes_list

    def isolated_init(self, eval_cfg, *args, **kwargs):
        eval_cfg.unlock()
        eval_cfg.trajdata_cache_location = str(PRIVATE)
        eval_cfg.trajdata_rebuild_cache = False
        if 'trajdata' in eval_cfg:
            eval_cfg.trajdata.trajdata_cache_location = str(PRIVATE)
            eval_cfg.trajdata.trajdata_rebuild_cache = False
        return original_init(self, eval_cfg, *args, **kwargs)

    def isolated_dataset(*args, **kwargs):
        if args:
            raise ValueError('Expected keyword-only dataset construction for cache isolation')
        kwargs.update(cache_location=PRIVATE, num_workers=0, rebuild_maps=False)
        if Path(kwargs['cache_location']).resolve() != PRIVATE.resolve():
            raise ValueError('Reference attempted to leave its isolated cache')
        return original_dataset(**kwargs)

    def frozen_maps(self, cache_path, map_cache_class, map_params):
        if Path(cache_path).resolve() != PRIVATE.resolve() or float(map_params['px_per_m']) != 2.0:
            raise ValueError('Implicit map build attempted outside the frozen private maps')
        ready()

    def frozen_metadata(self, env_cache, all_scenes_list):
        if self.name != 'nuplan_mini':
            raise ValueError('Unexpected dataset metadata rewrite')
        ready()

    with cache_lock():
        marker = ready()
        env_builders.EnvUnifiedBuilder.__init__ = isolated_init
        env_builders.UnifiedDataset = isolated_dataset
        NuplanDataset.cache_maps = frozen_maps
        NuplanDataset.cache_all_scenes_list = frozen_metadata
        try:
            spec = importlib.util.spec_from_file_location('native_reference_producer', adapter / 'prepare_reference.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.prepare(job_file, output)
            metadata = Path(output).with_suffix('.json')
            audit = json.loads(metadata.read_text())
            if Path(audit['cache_location']).resolve() != PRIVATE.resolve():
                raise ValueError('Produced reference points outside its isolated cache')
            ready()
            audit.update(private_cache=True, shared_cache_written=False,
                         private_maps_marker_sha256=sha(PRIVATE / 'maps_ready.json'),
                         reference_adapter_sha256=sha(adapter / 'prepare_reference.py'))
            metadata.write_text(json.dumps(audit, indent=2) + '\n')
        finally:
            env_builders.EnvUnifiedBuilder.__init__ = original_init
            env_builders.UnifiedDataset = original_dataset
            NuplanDataset.cache_maps = original_maps
            NuplanDataset.cache_all_scenes_list = original_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.job_file, args.output)


if __name__ == '__main__':
    main()
