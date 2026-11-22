import importlib.util
import runpy
import sys
from pathlib import Path
overlay=Path(__file__).parent
repo=overlay.parent
sys.path.insert(0,str(overlay))
from numeric_policy import configure
configure()
from availability_export import install
install()
import no_action_binding
for leaf in ['five_class_guidance','selected_planner_rollout']:
    name='ccdiff.evaluation.'+leaf
    spec=importlib.util.spec_from_file_location(name,overlay/(leaf+'.py'))
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
import ccdiff.evaluation.five_class_planners as planners
from ccdiff.evaluation.native_tracker_client import NativeTrackingEgoPolicy
out=Path(sys.argv[sys.argv.index('--results_root_dir')+1])
original=planners.build_selected_ego
def build_tracked(name,device,strive_profile=None):
    return NativeTrackingEgoPolicy(original(name,device,strive_profile),device,out)
planners.build_selected_ego=build_tracked
runpy.run_path(str(overlay/'scene_editor_benchmark.py'),run_name='__main__')
