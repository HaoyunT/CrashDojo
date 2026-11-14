"""Paths and deterministic runtime policy for isolated v24 evaluation."""
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
HERE = ROOT / 'launch/crashdojo_v26_nuplan_mini_3090_20261102'
QUEUE = ROOT / 'eval/crashdojo_v26_nuplan_mini_3090_20261102'
REPO = ROOT / 'code/CCDiff_M2c_planner'
OVERLAY = HERE / 'overlay'
BASE = Path('/root/path/to/workspace/home')
PY = BASE / 'miniconda3/envs/ccdiff/bin/python'
MAP_PY = BASE / 'miniconda3/envs/safesim/bin/python'
spec = importlib.util.spec_from_file_location('v24_frozen_runtime', ROOT / 'launch/road_scene_strength_v20_20261026/runtime_common.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
sha, load, save, HOSTS = runtime.sha, runtime.load, runtime.save, runtime.HOSTS


def environment(gpu):
    env = runtime.environment(gpu)
    env['PYTHONPATH'] = ':'.join(map(str, [HERE, HERE / 'map_deps',
        BASE / 'planner_sources_20260810/nuplan_source/nuplan-devkit-master',
        REPO, REPO / 'third_party/tbsim', BASE / 'CCDiff/third_party/trajdata/src']))
    env['CRASHDOJO_RULE_ADAPTER_DIR'] = str(HERE/'rule_adapter')
    return env


def verify(protocol):
    for group in ['source_sha256', 'input_sha256']:
        for name, digest in protocol[group].items():
            if sha(name) != digest:
                raise RuntimeError('Frozen v24 input/source changed: ' + name)
