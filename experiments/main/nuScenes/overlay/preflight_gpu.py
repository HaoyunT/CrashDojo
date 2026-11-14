"""All-request GPU loss/gradient checks; no training or model-weight updates."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from types import SimpleNamespace

REPO=Path('/path/to/workspace/Friday/safesim_planner_20261012_01/code/CCDiff_M2c_planner')
INPUT=REPO/'benchmark50_inputs_20261021_v3'
sys.path[:0]=[str(REPO/'benchmark50_overlay_20261021_v3'),'/path/to/storage/external_scripts']
from five_class_guidance import load_request,FiveClassGuide,loss_terms
from q16_map_events_v2_20261020 import load_tracks

torch.set_num_threads(1)
records=[json.loads(s) for s in (INPUT/'manifest.jsonl').read_text().splitlines()]
assert len(records)==50
results=[]
for record in records:
    name=record['scene_id'];r=load_request(INPUT/(name+'.json'),record)
    if r['family']=='NONE':
        results.append(dict(scene=name,family='NONE',schema_pass=True));continue
    tr=load_tracks(Path('/path/to/storage/data/outputs/q11_nuscenes_val_full_20261017/cache')/(name+'.npz'))
    actors=[tr[record['adversary_vehicle_id']],tr['ego']]
    transforms=[];sizes=[];histories=[];speeds=[]
    for a in actors:
        j=a['lookup'][31];yaw=float(a['yaw'][j]);c,s=math.cos(yaw),math.sin(yaw)
        mat=np.array([[c,-s,a['xy'][j,0]],[s,c,a['xy'][j,1]],[0.,0.,1.]])
        transforms.append(mat);sizes.append([a['size'][1],a['size'][0],a['size'][2]])
        hist=np.array([a['xy'][a['lookup'][t]] for t in range(1,32)])
        histories.append((hist-a['xy'][j])@mat[:2,:2]);speeds.append(float(a['v'][j]))
    tensor=lambda x:torch.tensor(np.asarray(x),device='cuda',dtype=torch.float64)
    obs=dict(world_from_agent=tensor(transforms),extent=tensor(sizes),history_positions=tensor(histories),
             history_availabilities=torch.ones((2,31),device='cuda',dtype=torch.bool))
    ts=torch.arange(1,61,device='cuda',dtype=torch.float64)*.1
    ego=SimpleNamespace(positions=torch.stack((ts*speeds[1],ts*0),-1)[None],yaws=ts[None,:,None]*0)
    guide=FiveClassGuide(r,Path('/tmp/unused_benchmark50_preflight_audit.jsonl'))
    guide.prepare(obs,ego,0,1,0.,.1)
    state=torch.stack((ts*speeds[0],ts*0,ts*0+speeds[0],ts*0,ts*0,ts*0),-1)[None].requires_grad_(True)
    try:
        val,terms,dist=loss_terms(state,state.detach(),guide.context,r)
        gradient=torch.autograd.grad(val.sum(),state)[0]
        assert torch.isfinite(val).all() and torch.isfinite(gradient).all()
        results.append(dict(scene=name,family=r['family'],schema_pass=True,finite_loss_and_gradient=True))
    except Exception as exc:
        results.append(dict(scene=name,family=r['family'],schema_pass=True,finite_loss_and_gradient=False,error=repr(exc)))
report=dict(device=torch.cuda.get_device_name(),training=False,requests=results,passed=all(r.get('finite_loss_and_gradient',True) for r in results))
(INPUT/'gpu_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['passed'] else 1)
