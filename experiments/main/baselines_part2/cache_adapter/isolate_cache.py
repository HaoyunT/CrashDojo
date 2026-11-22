"""Build and verify a private centered nuPlan cache without shared writes."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time


ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
QUEUE = ROOT / 'eval/baselines_native_v6_remaining225_20261101'
PRIVATE = QUEUE / 'inputs/cache/nuplan_centered_private'
CONTEXTUAL = ROOT / 'launch/causal_guidance_v26_contextual_full_eval_20261101'
REPO = ROOT / 'code/CCDiff_M2c_planner'
BASE = Path('/path/to/workspace/home')
MARKER = PRIVATE / 'maps_ready.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def paths():
    sys.path[:0] = list(map(str, [CONTEXTUAL, CONTEXTUAL / 'overlay',
        CONTEXTUAL / 'map_deps', REPO, REPO / 'third_party/tbsim',
        BASE / 'CCDiff/third_party/trajdata/src',
        BASE / 'planner_sources_20260810/nuplan_source/nuplan-devkit-master']))
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Haswell',
        NPY_DISABLE_CPU_FEATURES='AVX512F', CUDA_VISIBLE_DEVICES='',
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', NVIDIA_TF32_OVERRIDE='0',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONUTF8='1', LC_ALL='C.UTF-8')


def validate_path():
    parent = QUEUE / 'inputs/cache'
    parent.mkdir(parents=True, exist_ok=True)
    if PRIVATE.is_symlink() or PRIVATE.resolve().parent != parent.resolve():
        raise ValueError('Private cache must be a real directory inside this queue')
    PRIVATE.mkdir(exist_ok=True)
    for item in PRIVATE.rglob('*'):
        if item.is_symlink():
            raise ValueError('Private cache contains a symlink: ' + str(item))
    return PRIVATE.resolve()


@contextmanager
def cache_lock():
    validate_path()
    with (PRIVATE.parent / '.nuplan_centered_private.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ready():
    validate_path()
    value = json.loads(MARKER.read_text())
    if value.get('status') != 'validated' or value.get('cache') != str(PRIVATE.resolve()):
        raise ValueError('Private map cache lacks a valid readiness record')
    for name, expected in value['files_sha256'].items():
        path = PRIVATE / name
        if sha(path) != expected:
            raise ValueError('Private warmed input changed: ' + str(path))
    return value


def warm(config_file, source_cache):
    paths()
    with cache_lock():
        if MARKER.exists():
            try:
                return ready()
            except (OSError, ValueError):
                pass
        config = json.loads(Path(config_file).read_text())
        source = Path(source_cache).resolve()
        if source == PRIVATE.resolve() or PRIVATE.resolve() in source.parents:
            raise ValueError('Metadata source must be separate from the private cache')
        metadata = source / 'nuplan_mini/scenes_list.dill'
        destination = PRIVATE / 'nuplan_mini/scenes_list.dill'
        source_maps = source / 'nuplan_mini/maps'
        source_files = sorted(p for p in source_maps.rglob('*') if p.is_file())
        if not source_files or any(p.is_symlink() for p in source_maps.rglob('*')):
            raise ValueError('Source maps must be complete ordinary files')
        before = {str(p.relative_to(source)): sha(p) for p in source_files + [metadata]}
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata, destination)
        shutil.copytree(source_maps, PRIVATE / 'nuplan_mini/maps', dirs_exist_ok=True)
        after = {name: sha(source / name) for name in before}
        copied = {name: sha(PRIVATE / name) for name in before}
        if before != after or before != copied:
            raise ValueError('Source map snapshot changed or private copy differs')

        # Reuse the active evaluation's exact centered maps, then load all four serially.
        from numeric_policy import configure
        configure()
        from trajdata.dataset_specific.nuplan import nuplan_utils
        from trajdata.caching.df_cache import DataFrameCache
        from trajdata.maps.map_api import MapAPI
        import numpy as np
        api = MapAPI(PRIVATE)
        checks, hashes = [], {}
        for city in nuplan_utils.NUPLAN_LOCATIONS:
            if not DataFrameCache.is_map_cached(PRIVATE, 'nuplan_mini', city, 2.0):
                raise ValueError('Incomplete private map cache: ' + city)
            vector = api.get_map('nuplan_mini:' + city)
            elements = list(vector.iter_elems())
            if vector.map_id != 'nuplan_mini:' + city or not elements:
                raise ValueError('Invalid private map identity/elements: ' + city)
            if not np.isfinite(vector.extent).all():
                raise ValueError('Nonfinite private map extent: ' + city)
            for path in DataFrameCache.get_map_paths(PRIVATE, 'nuplan_mini', city, 2.0)[1:]:
                if path.is_file():
                    if path.stat().st_size == 0:
                        raise ValueError('Empty private map file: ' + str(path))
                    hashes[str(path.relative_to(PRIVATE))] = sha(path)
                else:
                    import zarr
                    raster = zarr.open(str(path), mode='r')
                    if len(raster.shape) != 3 or not all(raster.shape):
                        raise ValueError('Invalid private raster array: ' + str(path))
                    for starts in itertools.product(*(range(0, n, c) for n, c in zip(raster.shape, raster.chunks))):
                        slices = tuple(slice(start, min(start + chunk, size))
                                       for start, chunk, size in zip(starts, raster.chunks, raster.shape))
                        if not np.isfinite(raster[slices]).all():
                            raise ValueError('Nonfinite private raster chunk: ' + str(path))
                    for item in sorted(path.rglob('*')):
                        if item.is_file():
                            hashes[str(item.relative_to(PRIVATE))] = sha(item)
            checks.append(dict(city=city, map_id=vector.map_id, elements=len(elements),
                               extent=np.asarray(vector.extent).tolist()))
        if len(checks) != 4:
            raise ValueError('Expected all four nuPlan city maps')
        hashes[str(destination.relative_to(PRIVATE))] = sha(destination)
        result = dict(status='validated', cache=str(PRIVATE.resolve()),
            source_metadata=str(metadata), source_metadata_sha256=sha(metadata),
            source_cache=str(source), source_snapshot_sha256=before,
            source_snapshot_stable_before_after=True,
            copy_mode='ordinary_files_no_symlinks_or_hardlinks',
            source_config=str(config_file), source_config_sha256=sha(config_file),
            host=socket.gethostname(), created_unix=time.time(),
            coordinate_adapter=str(CONTEXTUAL / 'nuplan_adapter.py'),
            coordinate_adapter_sha256=sha(CONTEXTUAL / 'nuplan_adapter.py'),
            shared_cache_written=False, serial_map_warmup=True,
            checks=checks, files_sha256=hashes)
        save(MARKER, result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONTEXTUAL / 'inputs/nuplan_train_config.json')
    parser.add_argument('--source-cache', type=Path, default=CONTEXTUAL / 'nuplan_centered_runtime_cache')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    value = ready() if args.verify_only else warm(args.config, args.source_cache)
    print(json.dumps(dict(status=value['status'], cache=value['cache'],
        marker=str(MARKER), maps=value['checks'],
        boston_pb_sha256=value['files_sha256']['nuplan_mini/maps/boston.pb'],
        files_validated=len(value['files_sha256'])), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
