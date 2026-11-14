import json,sys
from pathlib import Path
from runtime_status import completed_runtime_ok
import numpy as np,h5py
from shapely.geometry import Point,LineString,Polygon
from shapely import wkt
from shapely.ops import unary_union,linemerge
repo=Path('/path/to/workspace/Friday/safesim_planner_20261012_01/code/CCDiff_M2c_planner');sys.path.insert(0,str(repo/'scripts'))
from safe_geometry import evaluate,box,dynamic_diagnostic
sys.path.insert(0,'/path/to/storage/external_scripts')
from q16_map_events_v2_20261020 import NativeMap
maps={}
root=Path(sys.argv[1])
def coords(points,route):
 a=np.asarray(route.coords[:-1]);b=np.asarray(route.coords[1:]);delta=b-a;length=np.linalg.norm(delta,axis=-1);valid=length>1e-6;a=a[valid];delta=delta[valid];length=length[valid];u=delta/length[:,None];prefix=np.r_[0,np.cumsum(length)[:-1]]
 rel=points[...,None,:]-a;along=np.clip(np.sum(rel*u,axis=-1),0,length);closest=a+along[...,None]*u;ix=np.argmin(np.sum((points[...,None,:]-closest)**2,axis=-1),axis=-1);q=np.take_along_axis(closest,ix[...,None,None],axis=-2)[...,0,:];normal=np.stack([-u[ix,1],u[ix,0]],axis=-1)
 return prefix[ix]+np.take_along_axis(along,ix[...,None],axis=-1)[...,0],np.sum((points-q)*normal,axis=-1)
results=[]
for parent in sorted(root.glob('*')):
 if not completed_runtime_ok(parent):continue
 p=next(parent.rglob('data.hdf5')).parent
 effective=p/'effective_request.json'
 request_path=effective if effective.exists() else parent/'request.json'
 r=json.loads(request_path.read_text())
 if r['family']=='NONE':
  result=dict(run=parent.name,family='NONE',action_applicable=False,A_only_qualified_contact=False,qualified_contact_with_ego_dynamic_check=False,formal_evaluation=True,reason=r['no_action_reason'])
  (p/'complete_action_diagnostic.json').write_text(json.dumps(result,indent=2));results.append(result);continue
 g=evaluate(p,request_path);binding=json.loads((p/'selection_binding.json').read_text());a=binding['runtime_target_index'];e=binding['runtime_ego_index']
 with h5py.File(p/'data.hdf5') as f:
  group=f[next(iter(f))];xy=group['centroid'][a].astype(float);yaw=group['yaw'][a].astype(float);extent=group['extent'][a].astype(float);ego_xy=group['centroid'][e].astype(float);ego_yaw=group['yaw'][e].astype(float);ego_extent=group['extent'][e].astype(float)
 T=len(xy);t=np.arange(T)*.1;s,d=coords(xy,LineString(r['route_world_xy']));par=r['parameters'];family=r['family'];start=par.get('t0',0)
 v=np.diff(xy,axis=0)/.1;speed=np.linalg.norm(v,axis=-1);acc=np.diff(v,axis=0)/.1;heading=np.stack([np.cos(yaw[1:-1]),np.sin(yaw[1:-1])],-1);along=(acc*heading).sum(-1);lat=(acc*np.stack([-heading[:,1],heading[:,0]],-1)).sum(-1);j=np.diff(along)/.1;limits=r['engineering_limits']
 dynamics=dict(longitudinal_acceleration_max=float(along.max()),longitudinal_acceleration_min=float(along.min()),lateral_acceleration_abs_max=float(abs(lat).max()),longitudinal_jerk_abs_max=float(abs(j).max()))
 dynamics['provisional_bounds_pass']=bool(speed.max()<=limits['max_speed_mps'] and along.max()<=limits['max_acceleration_mps2'] and along.min()>=-limits['max_braking_mps2'] and abs(lat).max()<=limits['max_lateral_mps2'] and abs(j).max()<=limits['max_jerk_mps3'])
 mech=dict(final_offset_m=float(d[-1]),initial_offset_m=float(d[0]),final_speed_mps=float(speed[-1]),route_progress_m=float(s[-1]-s[0]),completed=False)
 if family=='C1':
  end=par.get('speed_t0',start)+par['speed_transition_s'];mask=t[1:]>=end;mech['speed_template_max_error_after_transition']=float(abs(speed[mask]-par['target_speed_mps']).max()) if mask.any() else None
  mech['completed']=bool(mask.sum()>=5 and mech['speed_template_max_error_after_transition']<=.5)
 if family=='C2':
  end=start+par['T_lc_s'];mask=t>=end;desired=par['initial_offset_m']+par['delta_d_m'];mech['desired_offset_m']=desired;mech['completed']=bool(mask.sum()>=5 and np.max(abs(d[mask]-desired))<=.3)
 if family in ('C3A','C3B'):
  boundary=wkt.loads(r['geometry_evidence']['shared_boundary_wkt']);boundary=linemerge(boundary) if boundary.geom_type=='MultiLineString' else boundary
  if boundary.geom_type!='LineString':raise ValueError('Disconnected native shared boundary')
  _,sd=coords(xy[0],boundary);sign=np.sign(sd);corner=np.array([np.asarray(box(pos,angle,sz).exterior.coords)[:4] for pos,angle,sz in zip(xy,yaw,extent)]);_,side=coords(corner,boundary);outward=-sign*side;depth=outward.max(-1);end=start+par['T_in_s']+par['T_hold_s']+par['T_out_s'];mask=t>=end
  ret=par['initial_offset_m'] if family=='C3A' else par['return_offset_m'];mech.update(peak_body_intrusion_m=float(depth.max()),fully_crossed=bool((outward.min(-1)>0).any()),return_hold_frames=int(mask.sum()),returned_body=bool(mask.sum()>=5 and depth[mask].max()<=.03))
  mech['completed']=bool(depth.max()>=par.get('z_min_m',.2) and depth.max()<=par.get('z_max_m',extent[0,1]/2) and not mech['fully_crossed'] and mech['returned_body'] and (family=='C3B' or np.max(abs(d[mask]-ret))<=.3))
 if family in ('C4','C5'):
  remaining=par['arrival_route_s_m']-s[-1];mech['event_route_distance_remaining_m']=float(remaining);mech['passed_event_with_rear_body']=bool(remaining<-extent[-1,0]/2)
  records=[json.loads(line) for line in (parent/'selection_manifest.jsonl').read_text().splitlines()];record=next(z for z in records if z['scene_id']==r['scene_id'] and z['adversary_vehicle_id']==r['adversary_vehicle_id']);city=record['city']
  if city not in maps:maps[city]=NativeMap('nuscenes',city)
  native_map=maps[city]
  if family=='C5':
   receiving=native_map.poly[r['geometry_evidence']['common_lane']].buffer(.15);tail=[box(x,y,z) for x,y,z in zip(xy[-6:],yaw[-6:],extent[-6:])];tail_cov=min(b.intersection(receiving).area/b.area for b in tail);mech['receiving_lane_last_half_second_coverage_min']=tail_cov;mech['completed']=bool(mech['passed_event_with_rear_body'] and tail_cov>=.98)
  else:
   ca,ce=r['geometry_evidence']['crossing_connectors'];zone=native_map.poly[ca].intersection(native_map.poly[ce]);visit=[]
   for px,py,sz in [(xy,yaw,extent),(ego_xy,ego_yaw,ego_extent)]:
    inside=np.array([box(x,y,z).intersects(zone) for x,y,z in zip(px,py,sz)]);visit.append(bool(inside.any() and not inside[:6].any() and not inside[-6:].any()))
   mech['both_vehicles_local_zone_complete']=visit;mech['completed']=bool(all(visit))
  mech['completion_status']='native_geometry_completion_pass' if mech['completed'] else 'native_geometry_completion_incomplete'
 contact_time=g['first_target_contact_s'];pre_count=T if contact_time is None else int(np.floor(contact_time/.1))+1
 pre_xy=xy[:pre_count];pre_yaw=yaw[:pre_count];pre_v=np.diff(pre_xy,axis=0)/.1;pre_acc=np.diff(pre_v,axis=0)/.1
 h=np.stack([np.cos(pre_yaw[1:-1]),np.sin(pre_yaw[1:-1])],-1);pre_long=(pre_acc*h).sum(-1);pre_lat=(pre_acc*np.stack([-h[:,1],h[:,0]],-1)).sum(-1);pre_j=np.diff(pre_long)/.1
 corridor=unary_union([Polygon(poly) for poly in r['corridor_world_polygons']]);coverage=[box(x,y,sz).intersection(corridor).area/box(x,y,sz).area for x,y,sz in zip(pre_xy,pre_yaw,extent[:pre_count])]
 pre_dynamic=dynamic_diagnostic(pre_xy,pre_yaw,limits)
 pre=dict(frames=pre_count,sampled_body_coverage_min=min(coverage) if coverage else None,longitudinal_acceleration_min=pre_dynamic.get('long_accel_min'),longitudinal_acceleration_max=pre_dynamic.get('long_accel_max'),lateral_acceleration_abs_max=pre_dynamic.get('lat_accel_abs_max'),longitudinal_jerk_abs_max=pre_dynamic.get('long_jerk_max_mps3'),vector_jerk_max_mps3=pre_dynamic.get('vector_jerk_max_mps3'),sampling='recorded_10hz_positions_unsmoothed_finite_difference',provisional_bounds_pass=pre_dynamic['common_dynamic_envelope_pass'],evaluable=pre_dynamic['evaluable'])
 passed=bool(g['geometry_evaluable'] and g['target_route_body_coverage_min']>=.98 and pre['provisional_bounds_pass'] and mech['completed'])
 result=dict(evaluator_version="jerk_v2_contiguous_native_boundary",run=parent.name,family=family,geometry=g,dynamics=dynamics,precontact_diagnostics=pre,mechanism=mech,provisional_reasonableness_pass=passed,qualified_contact_under_provisional_checks=bool(passed and g['target_contact'] and not g['first_recorded_target_overlap'] and (g['first_non_target_contact_s'] is None or g['first_non_target_contact_s']>g['first_target_contact_s'])),formal_evaluation=True)
 result['ego_precontact_diagnostics']=dynamic_diagnostic(ego_xy[:pre_count],ego_yaw[:pre_count],limits)
 temporal=bool(contact_time is not None and contact_time>=start)
 if temporal and family in ('C2','C3A','C3B'):
  bd=wkt.loads(r['geometry_evidence']['shared_boundary_wkt']);bd=linemerge(bd) if bd.geom_type=='MultiLineString' else bd
  _,initial_side=coords(xy[0],bd);source_sign=np.sign(initial_side)
  fraction=contact_time/.1;i=min(len(xy)-2,int(fraction));alpha=min(1.,fraction-i)
  heading=yaw[i]+alpha*((yaw[i+1]-yaw[i]+np.pi)%(2*np.pi)-np.pi)
  corners=np.asarray(box(xy[i]*(1-alpha)+xy[i+1]*alpha,heading,extent[i]).exterior.coords)[:4]
  _,sides=coords(corners,bd);temporal=bool((-source_sign*sides).max()>.03)
 result['contact_after_maneuver_and_body_entry']=temporal
 result['A_only_qualified_contact']=bool(result['qualified_contact_under_provisional_checks'] and temporal)
 result['qualified_contact_with_ego_dynamic_check']=bool(result['A_only_qualified_contact'] and result['ego_precontact_diagnostics']['common_dynamic_envelope_pass'])
 result['validation_scope']='A mechanism and engineering checks; E common envelope separately audited; no certified full-system validity'
 print('EVALUATED',parent.name,'geometry',g['geometry_evaluable'],'A_pass',passed,'E_pass',result['ego_precontact_diagnostics']['common_dynamic_envelope_pass'],flush=True)
 (p/'complete_action_diagnostic.json').write_text(json.dumps(result,indent=2));results.append(result)
(root/'complete_action_summary.json').write_text(json.dumps(results,indent=2));print('RESULTS',len(results))
