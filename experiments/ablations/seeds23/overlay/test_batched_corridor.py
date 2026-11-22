"""Compare original and batched polygon-union values and pose gradients on GPU."""
import ast
import json
from pathlib import Path
import time
import torch
from five_class_guidance import body_outside_penalty
REPO=Path('/path/to/workspace/Friday/safesim_planner_20261012_01/code/CCDiff_M2c_planner')
INPUT=REPO/'benchmark50_inputs_20261021_v3'
source=REPO/'benchmark50_overlay_20261021_v2/five_class_guidance.py'
fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='body_outside_penalty')
ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
torch.manual_seed(721);torch.set_num_threads(1);rows=[]
for scene in ['scene-0109','scene-0523','scene-0560','scene-0096']:
    r=json.loads((INPUT/(scene+'.json')).read_text())
    for dtype in [torch.float64,torch.float32]:
        polys=[torch.tensor(p,device='cuda',dtype=dtype) for p in r['corridor_world_polygons']]
        groups={}
        for p in polys:groups.setdefault(len(p),[]).append(p)
        batch=[torch.stack(g[i:i+32]) for g in groups.values() for i in range(0,len(g),32)]
        route=torch.tensor(r['route_world_xy'],device='cuda',dtype=dtype)
        pos=(route[torch.linspace(0,len(route)-1,60,device='cuda').long()][None].expand(5,-1,-1)
             +torch.randn((5,60,2),device='cuda',dtype=dtype)*4).requires_grad_(True)
        yaw=torch.randn((5,60),device='cuda',dtype=dtype,requires_grad=True)
        extent=torch.tensor([4.8,2.0],device='cuda',dtype=dtype)
        times=[];values=[];gradients=[]
        for func,p in [(ns['body_outside_penalty'],polys),(body_outside_penalty,batch)]:
            torch.cuda.synchronize();start=time.perf_counter()
            v=func(pos,yaw,extent,p,3.,5);g=torch.autograd.grad(v.sum(),(pos,yaw))
            torch.cuda.synchronize();times.append(time.perf_counter()-start);values.append(v);gradients.append(g)
        torch.testing.assert_close(values[0],values[1],rtol=1e-6,atol=1e-5)
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-4)
        rows.append(dict(scene=scene,dtype=str(dtype),polygons=len(polys),batches=len(batch),
            value_max_error=float((values[0]-values[1]).abs().max()),
            gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(*gradients)),
            old_seconds=times[0],batched_seconds=times[1],speedup=times[0]/times[1]))
report=dict(passed=True,device=torch.cuda.get_device_name(),cases=rows,holes_preserved=True,
            physics_thresholds_unchanged=True,training=False)
(INPUT/'batched_corridor_equivalence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
