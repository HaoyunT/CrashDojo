"""Independent executed-trajectory and actor-specific deviation audit."""
import csv
import hashlib
import json
from pathlib import Path
from runtime_status import completed_runtime_ok
import sys
import h5py
import numpy as np

sys.path.insert(0,'/path/to/storage/external_scripts')
from q16_map_events_v2_20261020 import load_tracks
ROOT=Path('/path/to/workspace/Friday/safesim_planner_20261012_01/eval')
CACHE=Path('/path/to/storage/data/outputs/q11_nuscenes_val_full_20261017/cache')
rows=[];tracks={}

def transform(points,centroid,yaw):
    points=np.asarray(points,dtype=centroid.dtype)
    rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    return np.asarray([point@rotation+centroid for point in points])

for arm in ['guided','noguide']:
    for parent in sorted((ROOT/('benchmark50_'+arm+'_20261021_v3')).glob('scene-*')):
        if not parent.is_dir():continue
        state=parent/'status.json'
        if not completed_runtime_ok(parent):continue
        path=next(parent.rglob('data.hdf5'));r=json.loads((parent/'request.json').read_text())
        binding=json.loads((path.parent/'selection_binding.json').read_text());ego=binding['runtime_ego_index'];adv=binding['runtime_target_index']
        with h5py.File(path,'r') as f:
            g=f[next(iter(f))];position=np.asarray(g['centroid']);yaw=np.asarray(g['yaw'])
        scene=r['scene_id']
        if scene not in tracks:tracks[scene]=load_tracks(CACHE/(scene+'.npz'))
        out=dict(arm=arm,run=parent.name,scene=scene,family=r['family'])
        roles=[('E',ego,'ego')]+([('A',adv,r['adversary_vehicle_id'])] if adv is not None else [])
        for role,index,aid in roles:
            a=tracks[scene][aid];ts=range(r['start_step_10hz'],r['start_step_10hz']+len(position[index]))
            available=np.array([t in a['lookup'] for t in ts],dtype=bool)
            reference=np.array([a['xy'][a['lookup'][t]] for t in ts if t in a['lookup']])
            error=np.linalg.norm(position[index][available].astype(float)-reference,axis=-1)
            out[role+'_reference_frames']=int(available.sum())
            out[role+'_reference_fraction']=float(available.mean())
            out[role+'_final_reference_available']=bool(available[-1])
            out[role+'_ADE_m']=float(error.mean()) if len(error) else None
            out[role+'_FDE_m']=float(error[-1]) if available[-1] else None
            out[role+'_last_available_reference_error_m']=float(error[-1]) if len(error) else None
            if role=='E':assert available.all(),'Ego reference must cover the full evaluation'
        steps=[json.loads(s) for s in (path.parent/'selected_planner_steps.jsonl').read_text().splitlines()]
        if adv is not None:
            errors=[]
            for step in steps:
                k=step['step_index'];n=min(5,len(position[adv])-k-1)
                if n>0:errors.append(float(np.abs(transform(step['selected_local_first5'][:n],position[adv,k],yaw[adv,k])-position[adv,k+1:k+1+n]).max()))
            out['A_execution_error_max_m']=max(errors)
            out['replans_without_feasible_prefix']=sum(s.get('feasible_execution_candidates')==0 for s in steps)
        tracker=[json.loads(s) for s in (parent/'native_tracker_execution.jsonl').read_text().splitlines()]
        errors=[]
        for step in tracker:
            k=step['step_index'];n=min(5,len(position[ego])-k-1)
            if n>0:errors.append(float(np.abs(transform(step['tracked_positions'][:n],position[ego,k],yaw[ego,k])-position[ego,k+1:k+1+n]).max()))
        out['E_execution_error_max_m']=max(errors)
        out['controller_state_persisted']=all(s['provenance']['initialization']=='persisted_native_state_at_last_executed_prefix' for s in tracker[1:])
        out['execution_matches_logged_actions']=out['E_execution_error_max_m']<.002 and out.get('A_execution_error_max_m',0)<.002
        rows.append(out)
outdir=ROOT/'benchmark50_run_20261021_v3';outdir.mkdir(exist_ok=True)
result=dict(audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='completed_rollouts_only_no_quality_success_implied',rows=rows,
            all_execution_checks_pass=all(r['execution_matches_logged_actions'] and r['controller_state_persisted'] for r in rows))
(outdir/'actor_metrics_and_execution.json').write_text(json.dumps(result,indent=2))
if rows:
    with (outdir/'actor_metrics_and_execution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
print(json.dumps({'completed':len(rows),'all_execution_checks_pass':result['all_execution_checks_pass']}))
