"""Frozen target-only metrics on the declared horizon, including its endpoint."""
import argparse,csv,hashlib,json,math,os,time,traceback
from collections import defaultdict,Counter
from pathlib import Path
import h5py,numpy as np,shapely
from common import *
from metrics import contacts
VERSION='ablation_target_metrics_v3_h7_20261108'
FEATURES=('lon_accel','lat_accel','jerk')
EDGES={'lon_accel':np.arange(-100.,100.0001,.5),'lat_accel':np.arange(-100.,100.0001,.5),'jerk':np.arange(-1000.,1000.0001,1.)}
ROADS={};TRACKS={}

def tracks(path):
    if path not in TRACKS:
        z=np.load(path,allow_pickle=False);out={}
        for i,aid in enumerate(z['ids']):
            a=z['data'][z['agent_index']==i];out[str(aid)]={int(x[0]):x for x in a}
        TRACKS[path]=out
    return TRACKS[path]

def reference(job,actor,times):
    t=tracks(job['cache'])[actor];a=np.full((len(times),2),np.nan);y=np.full(len(times),np.nan);v=np.zeros(len(times),bool)
    for i,k in enumerate(times):
        if int(k) in t:a[i]=t[int(k)][1:3];y[i]=t[int(k)][5];v[i]=True
    return a,y,v

def features(x,y,valid):
    # Original target-protocol backward differences, without derivative padding.
    x=x[::5];y=y[::5];valid=valid[::5]
    velocity=np.diff(x,axis=0)/.5;accel=np.diff(velocity,axis=0)/.5
    good=valid[2:]&valid[1:-1]&valid[:-2]
    forward=np.c_[np.cos(y[2:]),np.sin(y[2:])];left=np.c_[-np.sin(y[2:]),np.cos(y[2:])]
    lon=(accel*forward).sum(-1);lat=(accel*left).sum(-1);jerk=np.diff(lon)/.5
    return {'lon_accel':lon[good],'lat_accel':lat[good],'jerk':jerk[good[1:]&good[:-1]]}

def histograms(f):
    result={}
    for k,values in f.items():
        v=values[np.isfinite(values)];edges=EDGES[k]
        # Preserve denominator and tail mass; do not silently drop outliers.
        clipped=np.clip(v,edges[0],np.nextafter(edges[-1],edges[0]))
        counts=np.histogram(clipped,bins=edges)[0]
        result[k]=dict(counts=counts.tolist(),n=len(v),below=int((v<edges[0]).sum()),above=int((v>=edges[-1]).sum()))
    return result

def freeze_reference():
    dest=QUEUE/'metric_protocol.json'
    if dest.exists():
        previous=load(dest)
        if previous['version']==VERSION:return previous
        dest.rename(QUEUE/(previous['version']+'_protocol.json'))
    cohort=load(QUEUE/'cohort.json');steps=load(QUEUE/'run_manifest.json')['steps'];H=steps*.1
    assert steps==70 and all(j['steps']==steps for j in cohort)
    pool={k:np.zeros(len(EDGES[k])-1,dtype=np.int64) for k in FEATURES};included=[]
    for job in cohort:
        start=load(job['manifest'])['start_step_10hz']
        for actor in job['candidate_ids']:
            xy,yaw,av=reference(job,actor,np.arange(start,start+steps+1));h=histograms(features(xy,yaw,av))
            for k in FEATURES:pool[k]+=h[k]['counts']
            included.append(dict(scene=job['scene'],actor=actor,available_frames=int(av.sum()),histogram_support={k:{x:h[k][x] for x in ['n','below','above']} for k in FEATURES}))
    protocol=dict(version=VERSION,frozen_unix=time.time(),source_sha256=sha(__file__),reference='Observed nuScenes validation logs for each unique frozen scene/candidate actor once; same reference for every planner and variant; no generated outcome used.',
        reference_actors=included,reference_histograms={k:v.tolist() for k,v in pool.items()},edges={k:v.tolist() for k,v in EDGES.items()},
        trajectory_times_s=dict(initial=0.,first_future=.1,last_future=H,dt=.1,frames_including_initial=steps+1),
        pose_source='Recorded simulator states through the penultimate time; only the declared final endpoint is reconstructed from the last executed action using the recorded centroid and yaw of its five-frame replan origin, exactly as env_trajdata._step. Stored inverted world_from_agent matrices are not used. Earlier actions are crosschecked against recorded poses within the unchanged 1 cm tolerance.',
        ADE=f'Single generated target; mean over paired valid future timestamps 0.1--{H:g} s. FDE requires the {H:g} s endpoint, otherwise missing.',
        CR=f'Oriented target/ego box overlap at any of {steps+1} times including initial state, target pair only; initial contact is separately recorded.',
        MS='Norm of the relative velocity vector at first target/ego contact; backward 0.1s velocity, observed preceding raw frame if contact is initial; no contact gives null.',
        Adv_OR='Target centroid ever outside frozen nuScenes vector drivable union, including initial; initial offroad separately recorded.',
        RM='Mean of three normalized-histogram W1 distances for signed body-longitudinal acceleration, signed body-lateral acceleration, and longitudinal jerk. 0.5s grid, backward differences without padding. Fixed broad signed bins retain clipped tail mass with separate overflow counts.',
        legacy_RM_compatibility='The archived positive-only accel-norm-times-world-yaw reconstruction is not reused; this new ablation uses the existing target-coordinate derivative definition. Values are not directly pooled with legacy RM.',
        source_cache_sha256={j['cache']:sha(j['cache']) for j in cohort})
    save(dest,protocol);return protocol

def executed(directory,expected_steps):
    fs=list(Path(directory).rglob('data.hdf5'));assert len(fs)==1
    path=fs[0];b=load(path.with_name('selection_binding.json'));ids=[b['runtime_ego_index'],b['runtime_target_index']]
    with h5py.File(path) as h:
        g=h[next(iter(h))];xy=np.asarray(g['centroid'],dtype=float)[ids];yaw=np.asarray(g['yaw'],dtype=float)[ids].reshape(2,-1);ext=np.asarray(g['extent'],dtype=float)[ids];av=np.asarray(g['availability'],dtype=bool)[ids].reshape(2,-1)
        local=np.asarray(g['action_positions'],dtype=float)[ids,:,0];dyaw=np.asarray(g['action_yaws'],dtype=float)[ids,:,0,0]
    t=xy.shape[1];assert t==expected_steps
    # Match the simulator's actual execution transform, not the separately
    # exported inverse matrix, whose float32 translation can drift by cm.
    origin_index=(np.arange(t)//5)*5;origin_yaw=yaw[:,origin_index]
    c=np.cos(origin_yaw);s=np.sin(origin_yaw)
    rotation=np.stack([np.stack([c,-s],axis=-1),np.stack([s,c],axis=-1)],axis=-2)
    future=np.einsum('ntij,ntj->nti',rotation,local)+xy[:,origin_index]
    fyaw=origin_yaw+dyaw
    disagreement=np.linalg.norm(future[:,:-1]-xy[:,1:],axis=-1);finite=av[:,1:]&np.isfinite(disagreement)
    alignment=float(disagreement[finite].max()) if finite.any() else None
    assert alignment is not None and alignment<.01,alignment
    # State exports provide availability; the final commanded target/ego pose
    # must also be finite. No background population enters these metrics.
    return path,b,np.concatenate([xy,future[:,-1:]],axis=1),np.concatenate([yaw,fyaw[:,-1:]],axis=1),np.concatenate([ext,ext[:,-1:]],axis=1),np.concatenate([av,av[:,-1:]],axis=1),alignment

def score(dest,protocol):
    job=load(dest/'job.json');request=load(dest/'request.json');actual=dest
    steps=protocol['trajectory_times_s']['frames_including_initial']-1
    path,b,xy,yaw,extent,av,alignment=executed(actual,steps);finite=np.isfinite(xy).all(-1)&np.isfinite(yaw);av &=finite
    pairs,hits=contacts(xy,yaw,extent,av);hit=hits[0];contact=int(np.flatnonzero(hit)[0]) if hit.any() else None
    start=b['start_step_10hz'];actor=request['adversary_vehicle_id'];ref,ry,rv=reference(job,actor,np.arange(start,start+steps+1))
    assert np.linalg.norm(ref[0]-xy[1,0])<.05
    pair=av[1]&rv;pair[0]=False;errors=np.linalg.norm(xy[1]-ref,axis=-1)
    ade=float(errors[pair].mean()) if pair.any() else None;fde=float(errors[-1]) if pair[-1] else None
    speed=None
    if contact is not None:
        if contact>0:
            velocities=(xy[:,contact]-xy[:,contact-1])/.1
        else:
            previous=[]
            for name in ['ego',actor]:
                r=tracks(job['cache'])[name];previous.append(r[start-1][1:3])
            velocities=(xy[:,0]-np.asarray(previous))/.1
        if np.isfinite(velocities).all():speed=float(np.linalg.norm(velocities[0]-velocities[1]))
    city=b['city']
    if city not in ROADS:
        roadpath=ROOT/'eval/baselines_native_fixed150_20261030/inputs/roads'/f'nuscenes__{city}.wkb'
        road=shapely.from_wkb(roadpath.read_bytes());shapely.prepare(road);ROADS[city]=(road,str(roadpath),sha(roadpath))
    road,roadpath,roadhash=ROADS[city];outside=np.zeros(steps+1,bool);outside[av[1]]=~shapely.covers(road,shapely.points(xy[1,av[1]]))
    generated=histograms(features(xy[1],yaw[1],av[1]));distances={}
    for k in FEATURES:
        p=np.asarray(generated[k]['counts'],float);q=np.asarray(protocol['reference_histograms'][k],float)
        distances[k]=float(np.abs(np.cumsum(p/p.sum()-q/q.sum())[:-1]).sum()*(EDGES[k][1]-EDGES[k][0])) if p.sum() and q.sum() else None
    rm=float(np.mean(list(distances.values()))) if all(x is not None for x in distances.values()) else None
    result=dict(version=VERSION,job_id=job['job_id'],scene=job['scene'],seed=job['stream'],planner=job['planner'],variant_id=job['variant'],target_id=actor,
        CR=bool(hit.any()),MS=speed,Adv_OR=bool(outside.any()),RM=rm,ADE=ade,FDE=fde,Adv_ADE=ade,Adv_FDE=fde,
        initial_contact=bool(hit[0]),initial_offroad=bool(outside[0]),first_contact_s=None if contact is None else contact*.1,
        valid_generated_trajectories=int(av[1].any()),valid_ADE_trajectories=int(pair.any()),valid_FDE_trajectories=int(pair[-1]),valid_paired_frames=int(pair.sum()),valid_target_frames=int(av[1].sum()),
        rm_distances=distances,rm_support={k:{x:generated[k][x] for x in ['n','below','above']} for k in FEATURES},hdf5=str(path),reference_cache=job['cache'],start_step_10hz=start,road_path=roadpath,road_sha256=roadhash,action_frame_alignment_max_m=alignment,
        metric_protocol_sha256=sha(QUEUE/'metric_protocol.json'))
    save(dest/'final_metrics.json',result);return result

def writecsv(path,rows,fields):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    tmp.replace(path)

def interval(rows,key,rng):
    if key=='MS':
        grouped={r['scene']:[] for r in rows}
        for r in rows:
            if r.get(key) is not None:grouped[r['scene']].append(float(r[key]))
        values=list(grouped.values());n=len(values);counts=np.array([len(x) for x in values]);totals=np.array([sum(x) for x in values])
        if not counts.sum():return None,None,None,0
        value=float(totals.sum()/counts.sum())
        if n<2:return value,None,None,n
        indices=rng.integers(n,size=(5000,n));den=counts[indices].sum(axis=1);ok=den>0
        boot=totals[indices].sum(axis=1)[ok]/den[ok];lo,hi=np.quantile(boot,[.025,.975])
        return value,float(lo),float(hi),int((counts>0).sum())
    grouped=defaultdict(list)
    for r in rows:
        if r.get(key) is not None:grouped[r['scene']].append(float(r[key]))
    v=np.array([np.mean(x) for x in grouped.values()]);n=len(v)
    if not n:return None,None,None,0
    if n<2:return float(v.mean()),None,None,n
    means=v[rng.integers(n,size=(5000,n))].mean(axis=1);lo,hi=np.quantile(means,[.025,.975])
    return float(v.mean()),float(lo),float(hi),n

def summarize_once(protocol):
    jobs=load(QUEUE/'run_manifest.json')['jobs'];rows=[];index=[];summaries=[];records=[];counts=Counter();errors=[]
    for j in jobs:
        d=QUEUE/'runs'/j['job_id'];sp=d/'status.json';status=load(sp) if sp.exists() else {'status':'pending'}
        row=dict(job_id=j['job_id'],scene=j['scene'],seed=j['stream'],planner=j['planner'],variant_id=j['variant'],status=status['status'],error=status.get('error'),wall_seconds=status.get('wall_seconds'),host=status.get('host'))
        counts[status['status']]+=1
        if status['status']=='completed':
            try:
                metric=load(d/'final_metrics.json') if (d/'final_metrics.json').exists() else score(d,protocol)
                if metric['version']!=VERSION:
                    (d/'final_metrics.json').rename(d/(metric['version']+'.json'))
                    metric=score(d,protocol)
                assert metric['version']==VERSION;row.update({k:v for k,v in metric.items() if not isinstance(v,(dict,list))});records.append(row)
                index.append({k:metric[k] for k in ['job_id','scene','seed','planner','variant_id','target_id','hdf5','reference_cache','start_step_10hz']})
            except Exception as e:
                row['metric_error']=repr(e);errors.append(dict(job_id=j['job_id'],error=repr(e),traceback=traceback.format_exc()))
        rows.append(row)
    rng=np.random.default_rng(20261109);keys=['CR','MS','Adv_OR','RM','ADE','FDE','Adv_ADE','Adv_FDE']
    for planner in ['idm','diffusion_planner']:
        for variant in sorted({j['variant'] for j in jobs}):
            allr=[r for r in rows if r['planner']==planner and r['variant_id']==variant];rr=[r for r in records if r['planner']==planner and r['variant_id']==variant]
            s=dict(planner=planner,variant_id=variant,planned=len(allr),attempted=sum(r['status']!='pending' for r in allr),completed=sum(r['status']=='completed' for r in allr),failed=sum(r['status']=='failed' for r in allr),metric_valid=len(rr),collisions=sum(r['CR'] for r in rr),valid_trajectories=sum(r['valid_generated_trajectories'] for r in rr),valid_FDE=sum(r['valid_FDE_trajectories'] for r in rr),partial=len(rr)<len(allr))
            for k in keys:
                value,lo,hi,n=interval(rr,k,rng);scale=100 if k in ['CR','Adv_OR'] else 1
                s[k]=None if value is None else value*scale;s[k+'_lo95']=None if lo is None else lo*scale;s[k+'_hi95']=None if hi is None else hi*scale;s[k+'_scenes']=n
            # MS is conditional on collision, so also retain the requested mean
            # over all collision episodes rather than substituting zeros.
            valid_ms=[r['MS'] for r in rr if r['MS'] is not None];s['MS']=float(np.mean(valid_ms)) if valid_ms else None;s['MS_collision_rollouts']=len(valid_ms)
            summaries.append(s)
    contrasts=[]
    bykey={(r['scene'],r['seed'],r['planner'],r['variant_id']):r for r in records}
    for planner in ['idm','diffusion_planner']:
        for v in sorted({j['variant'] for j in jobs} - {'A0'}):
            paired=[]
            for r in records:
                if r['planner']!=planner or r['variant_id']!=v:continue
                full=bykey.get((r['scene'],r['seed'],planner,'A0'))
                if not full:continue
                paired.append(dict(scene=r['scene'],**{k:float(full[k])-float(r[k]) if full[k] is not None and r[k] is not None else None for k in keys}))
            c=dict(planner=planner,contrast='A0-'+v,paired_rollouts=len(paired),paired_scenes=len({r['scene'] for r in paired}))
            for k in keys:
                x,lo,hi,n=interval(paired,k,rng);scale=100 if k in ['CR','Adv_OR'] else 1
                c[k]=None if x is None else x*scale;c[k+'_lo95']=None if lo is None else lo*scale;c[k+'_hi95']=None if hi is None else hi*scale;c[k+'_paired_scenes']=n;c[k+'_paired_rollouts']=sum(r[k] is not None for r in paired)
            contrasts.append(c)
    fields=['job_id','scene','seed','planner','variant_id','status','host','wall_seconds',*keys,'initial_contact','initial_offroad','first_contact_s','target_id','valid_generated_trajectories','valid_ADE_trajectories','valid_FDE_trajectories','valid_paired_frames','error','metric_error','hdf5','reference_cache']
    writecsv(QUEUE/'per_rollout_metrics.csv',rows,fields);writecsv(QUEUE/'ablation_summary.csv',summaries,list(summaries[0]));writecsv(QUEUE/'paired_contrasts.csv',contrasts,list(contrasts[0]));writecsv(QUEUE/'trajectory_index.csv',index,['job_id','scene','seed','planner','variant_id','target_id','hdf5','reference_cache','start_step_10hz'])
    save(QUEUE/'metric_errors.json',errors);save(QUEUE/'progress.json',dict(updated_unix=time.time(),counts=dict(counts),metric_valid=len(records),metric_errors=len(errors),total=len(jobs),metric_protocol=VERSION,summary_partial=counts['completed']<len(jobs)))
    print(json.dumps(dict(counts=dict(counts),metric_valid=len(records),metric_errors=len(errors))),flush=True)
    return counts['pending']==0 and counts['running']==0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();protocol=freeze_reference()
    while True:
        done=summarize_once(protocol)
        if done or not a.watch:break
        time.sleep(180)
