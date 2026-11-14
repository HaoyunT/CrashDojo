import importlib.util, json, os, runpy, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;OVERLAY=HERE/'overlay'
sys.path[:0]=[str(HERE),str(OVERLAY)]
from numeric_policy import configure
configure()
import ccdiff.models
spec=importlib.util.spec_from_file_location('ccdiff.models.ccdiff',HERE/'model_override.py')
model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model);ccdiff.models.ccdiff=model
from tbsim.evaluation import env_builders
dataset_class=env_builders.UnifiedDataset
def serial(*a,**kw):kw['num_workers']=0;return dataset_class(*a,**kw)
env_builders.UnifiedDataset=serial
job=json.loads(Path(os.environ['ABLATION_JOB_FILE']).read_text())
out=Path(sys.argv[sys.argv.index('--results_root_dir')+1])
from runtime_bindings import install
install(job,out)
from safe_planner_shutdown import install
install(out)
from availability_export import install
install()
import no_action_binding
for leaf in ['five_class_guidance','selected_planner_rollout']:
    name='ccdiff.evaluation.'+leaf;spec=importlib.util.spec_from_file_location(name,OVERLAY/(leaf+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if leaf=='five_class_guidance':
        from compat_v6_loader import install
        install(module,out)
        from dynamic_strength import install
        install(module)
        from ablation_hooks import install_guidance
        install_guidance(module)
import ccdiff.evaluation.five_class_planners as planners
from ccdiff.evaluation.native_tracker_client import NativeTrackingEgoPolicy
from native_tracker_history_padding import install as install_tracker_transport
install_tracker_transport(out)
original=planners.build_selected_ego
def build(name,device,strive_profile=None):
    if os.environ.get('ABLATION_A8_PHASE')=='nominal':
        from a8_policies import NominalEgoPolicy
        return NominalEgoPolicy(device)
    return NativeTrackingEgoPolicy(original(name,device,strive_profile),device,out)
if os.environ.get('ABLATION_A8_PHASE')=='fixed':
    from a8_policies import FixedTargetRollout
    sys.modules['ccdiff.evaluation.selected_planner_rollout'].SelectedPlannerRollout=FixedTargetRollout
planners.build_selected_ego=build
runpy.run_path(str(OVERLAY/'scene_editor_benchmark.py'),run_name='__main__')
