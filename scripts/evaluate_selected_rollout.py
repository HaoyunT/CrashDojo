"""Read-only independent executed-trajectory diagnostics; no model/simulator.

Interpolated rectangles reduce missed contacts between frames but do not certify
unobserved physical trajectories. Never substitutes for official native metrics.
"""
import argparse,json,math
from pathlib import Path
import h5py
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

def box(x,yaw,extent):
 u=np.array([np.cos(yaw),np.sin(yaw)])*extent[0]/2
 v=np.array([-np.sin(yaw),np.cos(yaw)])*extent[1]/2
 return Polygon([x+u+v,x+u-v,x-u-v,x-u+v])

def evaluate(directory,request):
 directory=Path(directory);r=json.loads(Path(request).read_text());bind=json.loads((directory/'selection_binding.json').read_text())
 a=bind['runtime_target_index'];e=bind['runtime_ego_index'];dt=.1
 with h5py.File(directory/'data.hdf5','r') as f:
  if len(f)!=1:raise ValueError('Expected one fixed selected scene')
  g=f[next(iter(f))];xy=g['centroid'][:];yaw=g['yaw'][:];ext=g['extent'][:]
 if not np.isfinite(xy).all() or not np.isfinite(yaw).all():raise ValueError('Nonfinite executed trajectory')
 corr=unary_union([Polygon(p) for p in r['corridor_world_polygons']]);T=xy.shape[1];target_hit=[];ego_hit=[];any_hit=[];first_other=None;initial_overlap=False;min_d=float('inf');coverage=[]
 def inspect(t,pos,angles,sizes):
  nonlocal first_other,initial_overlap,min_d
  bb=[box(p,y,sz) for p,y,sz in zip(pos,angles,sizes)]
  min_d=min(min_d,bb[a].distance(bb[e]));coverage.append(bb[a].intersection(corr).area/max(bb[a].area,1e-8))
  for i in range(len(bb)):
   for j in range(i+1,len(bb)):
    if bb[i].intersects(bb[j]):
     any_hit.append(t)
     if e in (i,j):ego_hit.append(t)
     if set((i,j))==set((a,e)):target_hit.append(t)
     elif a in (i,j) and first_other is None:first_other=t
     if t==0 and set((i,j))==set((a,e)):initial_overlap=True
 for k in range(T):
  inspect(k*dt,xy[:,k],yaw[:,k],ext[:,k])
  if k+1<T:
   displacement=np.linalg.norm(xy[:,k+1]-xy[:,k],axis=-1).max();angle=(yaw[:,k+1]-yaw[:,k]+np.pi)%(2*np.pi)-np.pi
   n=max(1,int(math.ceil(displacement/.05)),int(math.ceil(np.abs(angle).max()/.02)))
   if n>200:raise ValueError('Unreasonable between-frame motion')
   for z in range(1,n):
    u=z/n;inspect((k+u)*dt,(1-u)*xy[:,k]+u*xy[:,k+1],yaw[:,k]+u*angle,(1-u)*ext[:,k]+u*ext[:,k+1])
 velocity=np.diff(xy[a],axis=0)/dt;speed=np.linalg.norm(velocity,axis=-1)
 acc=np.diff(velocity,axis=0)/dt;jerk=np.diff(acc,axis=0)/dt
 result=dict(scope='executed_linear_interpolation_geometry_diagnostic',time_origin='first_recorded_frame',frames=int(T),duration_between_recorded_endpoints_s=(T-1)*dt,
  target_contact=bool(target_hit),ego_any_contact=bool(ego_hit),scene_any_contact=bool(any_hit),first_recorded_target_overlap=initial_overlap,
  first_target_contact_s=min(target_hit) if target_hit else None,first_non_target_contact_s=first_other,
  min_target_body_distance_m=min_d,target_route_body_coverage_min=min(coverage),
  speed_max_mps=float(speed.max()) if len(speed) else None,
  vector_acceleration_max_mps2=float(np.linalg.norm(acc,axis=-1).max()) if len(acc) else None,
  vector_jerk_max_mps3=float(np.linalg.norm(jerk,axis=-1).max()) if len(jerk) else None,
  capability_status=r['capability_status'],mechanism_completion='not_certified_by_short_smoke',
  qualified_conflict_rate=None,formal_validated=False,
  caveat='Finite-difference metrics and linearly interpolated contact diagnostics are not native SCR/ORR or certified continuous physics')
 (directory/'executed_geometry_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--request',required=True);a=p.parse_args();print(json.dumps(evaluate(a.directory,a.request),indent=2))
