"""Isolated evaluation entry with causal pressure and optional centered nuPlan cache."""
import importlib.util
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = HERE / 'overlay'
SUPPORT = HERE.parents[1] / 'support'
sys.path[:0] = [str(HERE), str(OVERLAY), str(SUPPORT)]
from numeric_policy import configure
configure()
# Dataset preprocessing is serial: worker fan-out per rollout otherwise exceeds
# the host CPU budget and the upstream parallel adapter rejects Unicode paths.
from tbsim.evaluation import env_builders
dataset_class = env_builders.UnifiedDataset


def serial_dataset(*args, **kwargs):
    kwargs['num_workers'] = 0
    return dataset_class(*args, **kwargs)


env_builders.UnifiedDataset = serial_dataset
if os.environ.get('V26_DATASET') == 'nuplan':
    from nuplan_adapter import install
    install()
from availability_export import install
install()
import no_action_binding
out = Path(sys.argv[sys.argv.index('--results_root_dir') + 1])
for leaf in ['five_class_guidance', 'selected_planner_rollout']:
    name = 'ccdiff.evaluation.' + leaf
    spec = importlib.util.spec_from_file_location(name, OVERLAY / (leaf + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if leaf == 'five_class_guidance':
        from request_loader_compat import install as install_request_loader
        install_request_loader(module, out, HERE / 'compatibility_sources.json')
        from dynamic_strength import install
        install(module)
import ccdiff.evaluation.five_class_planners as planners
from ccdiff.evaluation.native_tracker_client import NativeTrackingEgoPolicy
original = planners.build_selected_ego


def build_tracked(name, device, strive_profile=None):
    return NativeTrackingEgoPolicy(original(name, device, strive_profile), device, out)


planners.build_selected_ego = build_tracked
runpy.run_path(str(OVERLAY / 'scene_editor_benchmark.py'), run_name='__main__')
