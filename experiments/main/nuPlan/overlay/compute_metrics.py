#!/usr/bin/env python3
"""Read completed rollouts. Reproduce the released CFD for the original150k BN-only runs (not paper-scale CFD).
Keep native masked collision metrics separate from unmasked geometry audits.
"""
import ast, argparse, datetime, hashlib, json, math
from typing import Dict
from pathlib import Path
from runtime_status import completed_runtime_ok
import h5py, numpy as np, torch
from pyemd import emd
from scipy.stats import wasserstein_distance

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x, dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [plain(v) for v in x]
    if isinstance(x,np.generic): return plain(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x

def run(root, base, family_name, output):
    torch.set_num_threads(1)
    src=root/'code/CCDiff_M2c_planner'
    stats_src=base/'CCDiff/third_party/trajdata/src/trajdata/simulation/sim_stats.py'
    # Execute the real released function, retaining its histogram and masking semantics.
    tree=ast.parse(stats_src.read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calc_stats')
    ns={'torch':torch,'np':np,'Tensor':torch.Tensor,'Dict':Dict,'angle_wrap_torch':lambda x:(x+np.pi)%(2*np.pi)-np.pi}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(stats_src),'exec'),ns)
    official_calc=ns['calc_stats']
    bins={}
    for k,lo,hi in [('velocity',0,4),('lon_accel',-2,2),('lat_accel',-2,2),('jerk',-1,1),('rel_pos_norm',0,50),('rel_lon_pos_abs',0,50),('rel_lat_pos_abs',0,50),('rel_yaw_abs',0,np.pi),('rel_yaw_vel_abs',0,np.pi),('rel_vel_norm',0,4),('rel_lon_vel',-4,4),('rel_lat_vel',-4,4),('rel_accel_norm',0,2),('rel_lon_accel',-2,2),('rel_lat_accel',-2,2),('rel_jerk',-1,1)]:bins[k]=torch.linspace(lo,hi,21)
    gtpath=src/'gt_files/trajdata_nusc_new/GroundTruth/hist_stats_new_scene_current_speed.json';gt=json.loads(gtpath.read_text()); keys=['lon_accel','lat_accel','jerk']
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':{'sim_stats':str(stats_src),'sim_stats_sha256':sha(stats_src),'parser_sha256':sha(src/'ccdiff/examples/parse_scene_edit_results.py'),'gt_histogram':str(gtpath),'gt_sha256':sha(gtpath)},'cfd_protocol':{'stat_dt':.5,'rollout_dt':.1,'stride':5,'normalization':'scene_then_global_mass','stationary_filter':'current_speed > 0.5 m/s at first downsampled interval','components':keys,'n_bins':20,'bin_ranges':{'lon_accel':[-2,2],'lat_accel':[-2,2],'jerk':[-1,1]},'out_of_range':'discarded, matching released implementation','paper_scale_verified':False},'runs':{}}
    for family in [family_name]:
      for path in sorted((root/'eval'/family).rglob('data.hdf5')):
        status=path.parent.parent/'status.json'
        if not completed_runtime_ok(status.parent):continue
        cell=path.parent.parent.name; print('START',cell,flush=True)
        stats=json.loads(path.with_name('stats.json').read_text()); total={k:torch.zeros(20) for k in keys}; scenes=[]; checks=0
        with h5py.File(path,'r') as f:
          for name,g in f.items():
            pos=torch.Tensor(g['centroid'][:,::5]);yaw=torch.Tensor(g['yaw'][:,::5,None]) if False else torch.Tensor(g['yaw'][:][:,::5,None])
            hist=official_calc(positions=pos,heading=yaw,dt=.5,bins=bins,disable_control_on_stationary='current_speed',vec_map=None)
            # Independently check absolute features against the real function; track dropped mass.
            vel=torch.diff(pos,dim=1,prepend=pos[:,[0]]-(pos[:,[1]]-pos[:,[0]]))/.5
            p0=torch.where(torch.isnan(pos),torch.zeros_like(pos),pos)
            valid=(p0.sum(-1)!=0)&(p0[:,1].sum(-1)!=0)[:,None]&(torch.linalg.vector_norm(vel,dim=-1)[:,0]>.5)[:,None]
            acc=torch.diff(vel,dim=1,prepend=vel[:,[0]]-(vel[:,[1]]-vel[:,[0]]))/.5
            norm=torch.linalg.vector_norm(acc,dim=-1)
            vals={'lon_accel':norm*torch.cos(yaw.squeeze(-1)),'lat_accel':norm*torch.sin(yaw.squeeze(-1)),'jerk':torch.diff(norm,dim=1,prepend=norm[:,[0]]-(norm[:,[1]]-norm[:,[0]]))/.5}
            row={'scene':name,'n_agents':int(pos.shape[0]),'n_steps':int(g['centroid'].shape[1]),'cfd_hist_counts':{},'cfd_coverage':{}}
            for k in keys:
              h=hist[k].hist;assert torch.equal(h,torch.histogram(vals[k][valid],bins[k]).hist),(cell,name,k)
              row['cfd_hist_counts'][k]=h.tolist();cnt=int(valid.sum());finite=int(torch.isfinite(vals[k][valid]).sum())
              row['cfd_coverage'][k]={'eligible':cnt,'finite':finite,'in_bins':int(h.sum())}
              total[k]+=h/h.sum() if h.sum()>0 else h
            checks+=1
            if 'drivable_map' in g:
              # Released centerpoint offroad operation: round then clamp to raster edge.
              p=np.asarray(g['centroid'],dtype=np.float32);tf=np.asarray(g['raster_from_world'],dtype=np.float32)
              xy=np.einsum('ntij,ntj->nti',tf[...,:2,:2],p)+tf[...,:2,2]
              validp=np.isfinite(xy).all(-1);dm=np.asarray(g['drivable_map'],dtype=bool)
              uv=np.rint(np.nan_to_num(xy)).astype(int);u=np.clip(uv[...,0],0,dm.shape[-1]-1);v=np.clip(uv[...,1],0,dm.shape[-2]-1)
              off=~dm[np.arange(p.shape[0])[:,None],np.arange(p.shape[1])[None,:],v,u]
              track=np.asarray(g['track_id']);ego=np.asarray(g['is_ego'],dtype=bool) if 'is_ego' in g else track==0
              row['raster_offroad']={key:float(off[mask&validp].mean()) if np.any(mask&validp) else None for key,mask in [('all',np.ones(validp.shape,bool)),('ego',ego),('non_ego',~ego)]}
            scenes.append(row)
        n=len(scenes);dists={};hjson={'stats':{},'ticks':{}};coverage={}
        for k in keys:
          sim=(total[k]/n).numpy().astype(np.float64);ref=np.array(gt['stats'][k],dtype=np.float64);edges=np.array(gt['ticks'][k],dtype=np.float64);assert np.allclose(edges,bins[k].numpy(),rtol=0,atol=0)
          if sim.sum()>0:
            a=ref/ref.sum();b=sim/sim.sum();x=edges[1:];cost=np.abs(x[:,None]-x[None,:]);value=emd(a,b,cost)
            alt=wasserstein_distance(x,x,u_weights=a,v_weights=b);assert abs(value-alt)<1e-5,(cell,k,value,alt)
          else:value=None
          dists[k]=value;hjson['stats'][k]=sim.tolist();hjson['ticks'][k]=edges.tolist()
          counts={z:sum(s['cfd_coverage'][k][z] for s in scenes) for z in ['eligible','finite','in_bins']};counts['in_bins_fraction']=counts['in_bins']/counts['eligible'] if counts['eligible'] else None;counts['empty_scenes']=sum(sum(s['cfd_hist_counts'][k])==0 for s in scenes);coverage[k]=counts
        native={k:{'mean':float(np.nanmean(v)),'n':len(v),'finite':int(np.isfinite(v).sum()),'positive':int(np.sum(np.asarray(v)>0))} for k,v in stats.items() if k!='scene_index' and not k.startswith('guide')}
        row={'family':family,'cell':cell,'hdf5':str(path),'stats':str(path.with_name('stats.json')),'stats_sha256':sha(path.with_name('stats.json')),'n_scenes':n,'CFD_release':float(np.mean(list(dists.values()))) if all(v is not None for v in dists.values()) else None,'CFD_components':dists, 'emd_scipy_check_atol':1e-5,'CFD_histograms':hjson,'CFD_coverage':coverage,'official_function_crosschecks':checks,'ADE_m':native['ade']['mean'],'FDE_m':native['fde']['mean'],'native':native,'scenes':scenes}
        if 'all_collision_rate_coll_any' in stats:row['SCR_native']=native['all_collision_rate_coll_any']['positive']/n
        if 'all_off_road_rate_rate' in stats:row['ORR_native_percent']=100*native['all_off_road_rate_rate']['mean']
        if 'ego_collision_rate_coll_any' in stats:row['ego_collision_native']=native['ego_collision_rate_coll_any']['positive']/n
        elif 'ego_failure_failure_collision' in stats:row['ego_collision_native']=native['ego_failure_failure_collision']['positive']/n
        if 'raster_offroad' in scenes[0]:row['ORR_hdf5_percent']={key:100*float(np.mean([s['raster_offroad'][key] for s in scenes])) for key in ['all','ego','non_ego']}
        report['runs'][cell]=row
        out=Path(output);out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print('DONE',cell,'CFD',row['CFD_release'],'SCR',row.get('SCR_native'),'ORR',row.get('ORR_native_percent'),row.get('ORR_hdf5_percent'),flush=True)
    return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.base,a.family,a.output)
