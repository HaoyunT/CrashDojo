"""Three selectors on one frozen, technically eligible candidate pool."""
import copy,hashlib,importlib.util,json,math,sys,time,collections
import numpy as np
from pathlib import Path
from common import ROOT,HERE,QUEUE,load,save,sha
from selector_rules import (SELECTOR_REVISION, canonical_candidates,
                            matches_full_request, selection_seeds)
sys.path.insert(0,'/path/to/storage/external_scripts')
from q16_map_events_v2_20261020 import load_tracks

def full_key(x):
    j=x['judgement'];e=x['evidence']
    return (j['n_fail'],j['n_insufficient'],{'type_statistics_and_similar_cases':0,'type_statistics_only':1}.get(j['crash_level'],2),
        0 if e['contact'] else 1 if e['minimum_body_gap_in_effective_window_m']<=2. else 2,
        {'natural_L1':0,'within_crash_precrash_P90':1}.get(j['strength_tier'],2),
        round(e['minimum_body_gap_in_effective_window_m'],3),round(e['peak_braking_mps2']+e['peak_acceleration_mps2'],2),x['actor'])
def ttc(tracks,actor,start):
    a,e=tracks[actor],tracks['ego'];ia,ie=a['lookup'][start],e['lookup'][start]
    av=(a['xy'][ia]-a['xy'][a['lookup'][start-5]])/.5
    ev=(e['xy'][ie]-e['xy'][e['lookup'][start-5]])/.5
    p=a['xy'][ia]-e['xy'][ie];v=av-ev;r=.5*(np.linalg.norm(a['size'][:2])+np.linalg.norm(e['size'][:2]))
    aa=float(v@v);bb=2*float(p@v);cc=float(p@p-r*r);disc=bb*bb-4*aa*cc
    if cc<=0:return 0.
    if aa<1e-10 or bb>=0 or disc<0:return None
    val=(-bb-math.sqrt(disc))/(2*aa);return float(val) if val>=0 else None

def main():
    cohort=load(QUEUE/'cohort.json');outputs=[];pending_context=[];summaries=[]
    seeds=selection_seeds(load(QUEUE/'run_manifest.json'))
    controls=['version','mode','engineering_limits','gradient_steps','step_size','normalized_action_trust','last_flow_steps','body_corridor_guidance',
        'executed_jerk_guidance','select_guided_candidate','execution_phase_tracking','tracking_scales','trajectory_solver','exact_body_prefix_gate',
        'corridor_execution_prefix_weight','corridor_lookahead_frames','prefix_feasibility_repair_steps','contact_phase_window','contact_margin_m',
        'deviation_guidance','executed_acceleration_guidance','non_target_avoidance','qualified_contact_v2','multimodal_obstacle_seed']
    for job in cohort:
        scene=job['scene'];q0=load(job['base_request']);m0=load(job['manifest']);ctx0=load(job['context_record']);dest=QUEUE/'inputs'/scene
        capture=load(QUEUE/'candidate_capture/00/candidate_pools'/(scene+'.json'))
        pool=[x for x in capture['pool'] if x['actor'] in job['candidate_ids']]
        full_actor=q0['adversary_vehicle_id']
        assert any(x['actor']==full_actor for x in pool), (scene,'Full absent from native pool')
        assert all(x.get('source')=='native_search' and x['request'].get('selection_mode')!='random_fallback' and x['request'].get('mechanism_type')!='random_actor_speed_control' for x in pool)
        actors=sorted({x['actor'] for x in pool});tracks=load_tracks(Path(job['cache']));start=m0['start_step_10hz']
        values={a:ttc(tracks,a,start) for a in actors};canonical=canonical_candidates(pool)
        chosen_ttc=min(actors,key=lambda a:(values[a] if values[a] is not None else math.inf,a))
        save(dest/'eligible_candidates.json',dict(actors=actors,tuple_count=len(pool),ttc_seconds=values,
            selector_revision=SELECTOR_REVISION,
            within_actor_rule='Canonical family/parameter/geometry identity only; no Eq. (2) judgement or outcome ranking.',
            eligibility='Observed-history and native-geometry eligible, non-parked same-height actors with evaluable mechanism templates; no random/speed-only fallback. Same pool in all selection arms.',
            ttc_rule='Earliest nonnegative constant-observed-velocity disc contact; half-diagonal radii; 0 for initial disc overlap; null means infinity; stable actor-ID ties.',
            capture_sha256=sha(QUEUE/'candidate_capture/00/candidate_pools'/(scene+'.json'))))
        scene_selection={}
        for seed in seeds:
            random_seed=int.from_bytes(hashlib.sha256(f'CrashDojo_random_actor_v1|{scene}|{seed}'.encode()).digest()[:8],'little')
            random_actor=actors[int(np.random.Generator(np.random.PCG64(random_seed)).integers(len(actors)))]
            for variant,item in [('A1',canonical[random_actor]),('A2',canonical[chosen_ttc])]:
                same=matches_full_request(item,q0)
                if same:
                    overrides={k:job[k] for k in ['base_request','manifest','context_record','family']}
                else:
                    q=copy.deepcopy(item['request'])
                    for k in controls:
                        if k in q0:q[k]=copy.deepcopy(q0[k])
                        else:q.pop(k,None)
                    q['weights']=dict(q0['weights'],action=5. if q['family']=='C1' else 20.)
                    if q.get('selection_mode')!='random_fallback':q['selection_mode']='ablation_frozen_native_candidate'
                    q.update(adversary_vehicle_id=item['actor'],scene_id=q0['scene_id'],start_step_10hz=start)
                    token=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()[:16]
                    qp=dest/f'candidate_{token}.json';mp=dest/f'manifest_{token}.jsonl';cp=dest/f'context_{token}.json'
                    if not qp.exists():
                        save(qp,q);m=dict(m0,adversary_vehicle_id=item['actor'],selection_policy=variant)
                        mp.write_text(json.dumps(m)+'\n')
                        pending_context.append((job,qp,mp,cp))
                    overrides=dict(base_request=str(qp),manifest=str(mp),context_record=str(cp),family=q['family'])
                result=dict(variant=variant,scene=scene,seed=seed,selected_actor=item['actor'],family=item['family'],
                    selector_revision=SELECTOR_REVISION,
                    same_actor_as_full=item['actor']==full_actor,candidate_ids=actors,random_selection_seed=random_seed if variant=='A1' else None,
                    ttc_seconds=values[item['actor']],job_overrides=overrides)
                save(dest/f'selection_{variant}_s{seed}.json',result);outputs.append(result);scene_selection[f'{variant}_s{seed}']=item['actor']
        summaries.append(dict(scene=scene,candidate_actors=len(actors),full_actor=full_actor,selections=scene_selection))
    # Recompute the selected actor's causal context with the unchanged frozen
    # classifier and per-scene prior table. No evaluation outcome is accessed.
    old=ROOT/'launch/full_corpus_nonpass_20261102/ours/build_context_inputs.py'
    spec=importlib.util.spec_from_file_location('frozen_context_builder',old);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    records={j['scene']:dict(b.read(b.SOURCES['nuscenes']/'scene_meta.json')[j['scene']],split='train') for j in cohort}
    nusc=b.NuScenesSource(records);snapshot,_=b.event_snapshot();predictor=b.Predictor(device='cpu');maps={};tracks_cache={}
    for i,(job,qp,mp,cp) in enumerate(pending_context):
        q=load(qp);m=load(mp);ctx0=load(job['context_record']);city=m['city'];scene=job['scene']
        if city not in maps:maps[city]=b.RoadMap('nuscenes',city)
        if scene not in tracks_cache:tracks_cache={scene:nusc.scene(scene)}
        tracks=tracks_cache[scene];cutoff=m.get('start_time_s',m['start_step_10hz']*.1);table=load(ctx0['prior_table'])
        result=b.derive(tracks,q['adversary_vehicle_id'],cutoff,maps[city],predictor,snapshot,q,table,'nuscenes')
        check=b.derive(b.alter_future(tracks,cutoff,truncate=True),q['adversary_vehicle_id'],cutoff,maps[city],predictor,snapshot,q,table,'nuscenes')
        assert b.canonical(result)==b.canonical(check)
        context=dict(ctx0,actor=q['adversary_vehicle_id'],family=q['family'],request_path=str(qp),manifest_path=str(mp),base_request_sha256=sha(qp),manifest_sha256=sha(mp))
        context.update(result);save(cp,context)
        print(json.dumps(dict(context=i+1,total=len(pending_context),scene=scene,factor=context['prior']['factor'])),flush=True)
    save(QUEUE/'selection_summary.json',dict(scenes=summaries,request_count=len(outputs),changed_contexts=len(pending_context),
        same_actor_counts={v:sum(x['same_actor_as_full'] for x in outputs if x['variant']==v) for v in ['A1','A2']}))
    save(QUEUE/'selectors_ready.json',dict(ready=True,selector_revision=SELECTOR_REVISION,seeds=seeds,finished_unix=time.time(),scenes=len(cohort),requests=len(outputs),source_sha256=sha(__file__)))
    print('SELECTORS_READY',flush=True)
if __name__=='__main__':main()
