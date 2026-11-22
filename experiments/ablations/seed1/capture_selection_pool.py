#!/usr/bin/env python3
"""v6 selection: every candidate judged on the seven requirements, then ranked.

Candidate generation is the v5 one (same anchor, filters, native constructor,
template grid, unified first-contact event). New here:
- each (vehicle, mechanism, parameter) candidate gets seven separate verdicts
  (criteria_v6.judge), including crash-typology support from CRSS/CISS;
- ranking: fewest failed requirements, then fewest insufficient, then crash
  support level, then conflict quality (contact before near), then strength
  tier, gap, effort (contact is not required: a near conflict is acceptable);
- the previous default (v2 random fallback) and the v5 pick are judged with the
  same function so the change is visible scene by scene.
Offline screening against the logged ego; not a closed-loop result.
"""
import argparse,collections,copy,json,sys,time,traceback
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
ROOT=Path('/path/to/workspace/CrashDojo_pipeline')
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src/selection_v5_review'));sys.path.insert(0,str(ROOT/'src/selection_v6_seven_criteria'))
from select_against_logged_ego_v4 import load_tracks,observed_prefix,contextual_build,height_evidence
from audit_selection_v3 import Map,read_actors,sha,clean,save
from build_runtime_packets_v2 import NATIVE,V20,TEMPLATE,OVERLAY,import_file,template_controls
from scene_execution_policy import digest
from select_v5 import variants,parked_at
import core_v5 as core
import criteria_v6 as crit

V5=ROOT/'artifacts/selection_v5_review_20261029'
V2=ROOT/'artifacts/scene_execution_v2_final/execution_index.json'
AUDITS=ROOT/'artifacts/selection_review_v3_20261029'
OUT=ROOT/'artifacts/selection_v6_seven_criteria_20261030/selection_shards'


def slim(item):
    j=item['judgement']
    return dict(actor=item['actor'],family=item['family'],source=item['source'],parameters=item['request']['parameters'],interaction_window_s=item['request']['interaction_window_s'],
        overall=j['overall'],n_fail=j['n_fail'],n_insufficient=j['n_insufficient'],statuses={k:v['status'] for k,v in j['criteria'].items()},
        mechanism=j['mechanism']['key'],najm=j['criteria']['3_crash_basis'].get('najm_id'),contact=item['evidence']['contact'],
        min_gap_m=round(item['evidence']['minimum_body_gap_in_effective_window_m'],2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--scene-id',action='append');ap.add_argument('--output',type=Path,default=OUT);args=ap.parse_args()
    out=args.output/('%02d'%args.shard);out.mkdir(parents=True,exist_ok=True)
    v2idx=json.loads(V2.read_text());v5idx=json.loads((V5/'execution_index.json').read_text())
    v5sel={json.loads(l)['scene_id']:json.loads(l) for l in (V5/'selections.jsonl').read_text().splitlines()}
    units={r['canonical_id']:r for r in json.loads((ROOT/'artifacts/step2/split_manifest.json').read_text())['units']}
    audits={r['scene_id']:r for r in [json.loads(line) for line in (AUDITS/'all_evidence.jsonl').read_text().splitlines()]}
    builder=import_file('v6_native_builder',NATIVE);base_template=template_controls(json.loads(TEMPLATE.read_text()));prior=json.loads((V20/'strength_prior_r2.json').read_text())
    ref=crit.Reference()
    items_all=sorted(v2idx['scenes'],key=lambda sid:(units[sid]['dataset'],units[sid]['city'],sid))
    n=len(items_all);mine=items_all[args.shard*n//args.shards:(args.shard+1)*n//args.shards]
    if args.scene_id:mine=[s for s in mine if s in args.scene_id]
    if args.limit:mine=mine[:args.limit]
    signals=core.SignalLog();metas={};mk=None;m=None;ctl=None;results=[];packets={};scenes={};start=time.time()
    for sid in mine:
        oldp=v2idx['packets'][v2idx['scenes'][sid]['default_packet_id']];row=units[sid];audit=audits[sid];v5r=v5sel[sid]
        scene=copy.deepcopy(v5idx['scenes'][sid])
        result=dict(number=v5r['number'],scene_id=sid,scene_name=scene['scene_name'],dataset=row['dataset'],city=row['city'],v2_actor=v5r['v2_actor'],v5_actor=v5r['selected_actor'],
            v5_decision=v5r['decision'],v5_packet_id=v5r['packet_id'],decision='no_candidate_keep_v5',recorded_ego_future_used=True,closed_loop_validated=False)
        chosen=None;reject=collections.Counter();pool=[]
        try:
            if oldp.get('derived_scene'):
                result.update(decision='derived_scene_separate',reason='synthetic source; reported separately')
            else:
                record=json.loads(Path(oldp['manifest']).read_text().splitlines()[0]);step=record['start_step_10hz'];steps=oldp['steps'];t=np.arange(steps+1)*.1;H=steps/10
                cache=Path(oldp['source_cache']['path']);tracks=load_tracks(cache);ego_tr=tracks['ego'];elog=core.logged_states(ego_tr,step,steps)
                if not elog['valid'][0] or np.sum(elog['valid'])<2:raise ValueError('ego_log_has_insufficient_execution_frames')
                observed=observed_prefix(tracks,step);actors=read_actors(oldp,step);by={x['id']:x for x in actors}
                heights=height_evidence(oldp,record,row);parked=parked_at(cache,step)
                if mk!=(row['dataset'],row['city']):m=None;mk=(row['dataset'],row['city']);m=Map(*mk);ctl=crit.Control(m,*mk)
                log=anchor=None
                if row['dataset']=='nuplan':
                    mp=cache.parent.parent/'scene_meta.json'
                    if str(mp) not in metas:metas[str(mp)]=json.loads(mp.read_text())
                    meta=metas[str(mp)][cache.stem];log=meta['log'];anchor=meta['timestamps_us'][step]
                m.point_match(actors);builder.START=step;e=by['ego']
                background={k:core.logged_states(tr,step,steps) for k,tr in tracks.items() if k!='ego'}
                ctx=crit.Context(m,ctl,ref,row['dataset'],row['city'],log,elog,t,actors,e,tracks,step,scene.get('situation'),parked)
                def judge_all(q,a,source,reason,fallback=False):
                    ev,tr=core.evaluate(q,a,elog,t,background)
                    if ev is None:reject['%s:%s'%(source,tr)]+=1;return None
                    ev['signals']=core.signal_evidence(q,m,tr,t,a['length'],ev['encounter_s'],row['dataset'],log,anchor,signals)
                    ev['fine_0p02s']=core.fine_check(q,a,elog,t) if ev['contact'] else None
                    j=crit.judge(q,ev,tr,a,ctx,reason,fallback=fallback)
                    return dict(actor=a['id'],family=q['family'],source=source,request=q,actor_state=a,evidence=ev,judgement=j)
                # previous default (v2) judged as-is
                v2q=json.loads(Path(oldp['request']).read_text());v2a=by.get(oldp.get('adversary_vehicle_id') or v2q.get('adversary_vehicle_id'))
                v2item=judge_all(copy.deepcopy(v2q),v2a,'v2_default','v2 默认配置（v5 中为兜底）',fallback=v5r['decision'].startswith('fallback')) if v2a else None
                if v2item:pool.append(v2item)
                for a in actors:
                    if a['id']=='ego':continue
                    if a['history_s']<1:reject['target_history_under_1s']+=1;continue
                    if a['speed']>25:reject['target_speed_outside_guide_budget']+=1;continue
                    if a['lane'] is None or e['lane'] is None:reject['ambiguous_current_lane']+=1;continue
                    if heights.get(a['id'],0)>3:reject['current_height_difference_over_3m']+=1;continue
                    distance=float(np.min(np.linalg.norm(elog['xy'][elog['valid']]-[a['x'],a['y']],axis=1)))
                    if distance>a['speed']*H+1.2*H*H+10:reject['outside_displacement_bound']+=1;continue
                    if a['stationary_observed'] or a['speed']<.5:reject['no_observed_moving_target']+=1;continue
                    base=copy.deepcopy(base_template);base.update(scene_id=record['scene_id'],adversary_vehicle_id=a['id'],start_step_10hz=step)
                    try:native,reason=builder.construct(m,observed[a['id']],observed['ego'],base)
                    except (ValueError,KeyError,IndexError,AssertionError) as ex:reject['constructor:'+str(ex)]+=1;continue
                    if native is None:reject['constructor:'+str(reason)]+=1;continue
                    if native['family']=='C1' and native['geometry_evidence'].get('longitudinal_role')!='adversary_ahead':reject['C1_requires_lead_vehicle']+=1;continue
                    if native['parameters']['initial_speed_mps']<.5:reject['causal_history_indicates_stationary']+=1;continue
                    if native['family'] in ('C4','C5'):
                        route=LineString(native['route_world_xy']);cp=route.interpolate(native['parameters']['arrival_route_s_m'])
                        d=np.linalg.norm(elog['xy']-np.array(cp.coords[0]),axis=1);d[~elog['valid']]=np.inf;j=int(np.argmin(d))
                        if d[j]>max(e['width'],e['length']) or j<1:reject['logged_ego_does_not_reach_native_conflict_point']+=1;continue
                        native['parameters']['arrival_s']=float(t[j]);native['interaction_window_s']=[max(.5,float(t[j])-1.5),min(H,float(t[j])+1.5)]
                    native['weights']['pressure']=prior['config']['base_pressure'][native['family']]
                    for q in variants(native,H):
                        it=judge_all(q,a,'search','')
                        if it:pool.append(it)
                result['candidates_judged']=len(pool);result['rejection_counts']=dict(reject)
                save(out/'candidate_pools'/(scene['scene_name']+'.json'),dict(scene=scene['scene_name'],scene_id=sid,anchor=step,pool=pool,rejections=dict(reject)))
                # v5 pick for comparison: same actor/params as the v5 selection
                v5p=v5idx['packets'][v5r['packet_id']]
                same=lambda x:all(round(x['request']['parameters'].get(k,0),4)==round((v5p.get('parameters') or {}).get(k,0),4) for k in ('target_speed_mps','speed_t0','t0'))
                v5item=next((x for x in pool if x['source']=='search' and x['actor']==v5r['selected_actor'] and x['family']==v5r.get('family') and same(x)),None) if v5r['decision'] not in ('fallback_unverified_previous_selection',) else v2item
                result['v5_pick_judgement']=None if v5item is None else slim(v5item)
                result['v2_default_judgement']=None if v2item is None else slim(v2item)
                if pool:
                    pool.sort(key=crit.rank_key);chosen=pool[0]
                    result['alternatives']=[slim(x) for x in pool[1:6]]
                    j=chosen['judgement']
                    reason=(f"{len(pool)} 个候选中七条{('全部通过' if j['overall']==crit.PASS else '不通过 %d 条、证据不足 %d 条'%(j['n_fail'],j['n_insufficient']))}且排序第一："
                            f"机制 {j['criteria']['3_crash_basis'].get('mechanism_label')}（Najm {j['criteria']['3_crash_basis'].get('najm_id')}），"
                            f"{'窗口内接触' if chosen['evidence']['contact'] else '最小间距 %.2f m'%chosen['evidence']['minimum_body_gap_in_effective_window_m']}，"
                            f"强度 {j['strength_tier']}")
                    j['criteria']['7_config_reviewable']['selection_reason']=reason
                    result.update(decision={crit.PASS:'all_seven_pass',crit.INSUF:'no_fail_but_insufficient',crit.FAIL:'best_available_has_fail'}[j['overall']],
                        selected_actor=chosen['actor'],family=chosen['family'],source=chosen['source'],judgement=j,
                        evidence={k:v for k,v in chosen['evidence'].items() if k not in ('signals',)},signals=chosen['evidence'].get('signals'),
                        parameters=chosen['request']['parameters'],interaction_window_s=chosen['request']['interaction_window_s'])
                    if chosen['source']=='search':
                        aid=chosen['actor'];q=copy.deepcopy(chosen['request']);ev=chosen['evidence']
                        ev['logged_pair']=core.logged_pair(tracks[aid],elog,step,steps,t,ev['effective_window_s'])
                        try:q,ca=contextual_build(q,tracks,m,prior,dataset=row['dataset'],scale=1.,calibration_status='v6_seven_criteria_logged_ego_screened_engineering_prior_not_calibrated')
                        except (ValueError,KeyError,IndexError,AssertionError) as ex:ca=dict(context=dict(eligible_context=False,reason=str(ex)),state='base_engineering_weights_retained')
                        pid='packet_v6_'+digest(['selection_v6',sid,step,aid,q['parameters'],q['interaction_window_s']])[:20];dest=out/'packets'/pid
                        q.update(request_path=str(dest/'request.json'),selection_manifest_row=0,strength_level='medium',selection_mode='seven_criteria_v6',purpose='offline_seven_criteria_v6',
                            log_conditioned_selection=dict(recorded_ego_future_used=True,recorded_background_future_used=True,online_causal_selection=False,closed_loop_verified=False),
                            physical_reference_binding=dict(source='seven_criteria_v6_bounded_nominal_search_against_logged_ego',calibrated=False,validated_risk_level=False,
                                crash_reference=j['criteria']['6_action_strength'].get('crash_reference'),strength_tier=j['strength_tier']),
                            seven_criteria=dict(overall=j['overall'],statuses={k:v['status'] for k,v in j['criteria'].items()},mechanism=j['mechanism']['key'],najm_id=j['criteria']['3_crash_basis'].get('najm_id'),selection_reason=reason))
                        rec=copy.deepcopy(record);rec.update(adversary_vehicle_id=aid,selection_policy='seven_criteria_v6',closed_loop_validated=False,recorded_ego_future_used=True)
                        full=all(step-k in tracks[aid]['lookup'] and step-k in ego_tr['lookup'] for k in range(31))
                        overlay=OVERLAY if full else ROOT/'src/random_execution_overlay_v2'
                        packet=dict(packet_id=pid,scene_id=sid,dataset=row['dataset'],option_id='seven_criteria_v6_'+pid,level='medium',request=str(dest/'request.json'),manifest=str(dest/'manifest.jsonl'),config=oldp['config'],steps=steps,
                            configuration_ready=True,runtime_verified=False,calibrated=False,selection_mode='seven_criteria_v6',history_policy='full_history' if full else 'partial_history_with_original_availability_masks',
                            weights=q['weights'],parameters=q['parameters'],family=q['family'],interaction_window_s=q['interaction_window_s'],physical_reference_binding=q['physical_reference_binding'],source_cache=oldp['source_cache'],
                            guidance_function=oldp['guidance_function'],guidance_source=str(overlay/'five_class_guidance.py'),executor_entry=str(overlay/'entry.py'))
                        save(dest/'request.json',q);save(dest/'seven_criteria.json',j);save(dest/'selection_evidence.json',ev);save(dest/'context_audit.json',ca)
                        (dest/'manifest.jsonl').write_text(json.dumps(clean(rec),ensure_ascii=False)+'\n')
                        packet['input_sha256']={str(pp):sha(pp) for pp in [dest/'request.json',dest/'manifest.jsonl',Path(packet['config'])]};save(dest/'packet.json',packet)
                        scene.update(default_packet_id=pid,packets_by_level={'medium':pid},selected_vehicle=dict(dataset=row['dataset'],scene_id=sid,source_token=aid,namespace='instance_token' if row['dataset']=='nuscenes' else 'track_token',
                            identity_source='original_track_cache_at_frozen_anchor',runtime_step_10hz=step),family=q['family'],selection_mode='seven_criteria_v6',runtime_verified=False,target_context=ca['context'])
                        packets[pid]=packet
                    else:
                        pid=v2idx['scenes'][sid]['default_packet_id'];pk=copy.deepcopy(v2idx['packets'][pid]);pk['runtime_verified']=False;packets[pid]=pk
                        scene=copy.deepcopy(v2idx['scenes'][sid]);scene.update(default_packet_id=pid,packets_by_level={'medium':pid},selection_mode='v2_default_judged_v6',runtime_verified=False)
                else:result['reason']='no judgeable candidate (constructor / template rejections); keep v5 configuration, unreviewed'
        except Exception as ex:
            chosen=None;result.update(decision='selection_error_keep_v5',error=repr(ex),traceback=traceback.format_exc())
        if chosen is None:
            pid=scene['default_packet_id'];pk=copy.deepcopy(v5idx['packets'][pid]);packets[pid]=pk
            result.update(selected_actor=v5r['selected_actor'],family=scene.get('family'),source='v5_kept_unjudged')
        scene['selection_v6']=result['decision'];scene['seven_criteria_overall']=(result.get('judgement') or {}).get('overall');scenes[sid]=scene
        result['packet_id']=scene['default_packet_id'];v5p=v5idx['packets'][v5r['packet_id']]
        result['changed_vs_v5']=dict(vehicle=result.get('selected_actor')!=v5r['selected_actor'],family=result.get('family')!=v5p.get('family'),
            parameters=any(round((result.get('parameters') or v5p.get('parameters') or {}).get(k,0),4)!=round((v5p.get('parameters') or {}).get(k,0),4) for k in ('target_speed_mps','speed_t0','t0','speed_transition_s')))
        results.append(clean(result));save(out/'decisions'/('%04d.json'%result['number']),clean(result))
    save(out/'shard_index.json',dict(scenes=scenes,packets=packets))
    with (out/'selections.jsonl').open('w') as f:
        for r in results:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(shard=args.shard,scenes=len(results),counts=collections.Counter(r['decision'] for r in results),elapsed_s=round(time.time()-start,1))),flush=True)
if __name__=='__main__':main()
