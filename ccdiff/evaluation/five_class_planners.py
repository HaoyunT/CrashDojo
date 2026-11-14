"""Explicitly reuse the report's unified rule-planner adapter, not same-name substitutes."""
import os,sys
from pathlib import Path

def build_selected_ego(name,device,strive_profile=None):
    if name in ('idm','strive','pdm_closed'):
        directory=Path(os.environ.get('CRASHDOJO_RULE_ADAPTER_DIR',
            '/path/to/workspace/Friday/safesim_planner_20261012_01/launch/rule_eval_v1'))
        if not (directory/'rule_client.py').is_file():raise FileNotFoundError('Missing report-compatible rule adapter: '+str(directory))
        sys.path.insert(0,str(directory))
        from rule_client import RulePlannerClient
        print('FIVE_CLASS_PLANNER_ADAPTER',name,str(directory/'rule_client.py'),flush=True)
        return RulePlannerClient(device,name)
    from ccdiff.planners import build_ego_planner
    return build_ego_planner(name,device=device)
