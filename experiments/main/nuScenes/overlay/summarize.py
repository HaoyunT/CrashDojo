"""Aggregate frozen cohorts without dropping failed or no-action scenes."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance

ROOT=Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO=ROOT/'code/CCDiff_M2c_planner'
P=json.loads((REPO/'benchmark50_protocol_20261021_v3/protocol.json').read_text())
OUT=ROOT/'eval/benchmark50_run_20261021_v3'
reports={};diagnostics={}
for arm in ['guided','noguide']:
    stage=ROOT/'eval'/('benchmark50_'+arm+'_20261021_v3')
    reports[arm]=json.loads((stage/'native_and_cfd.json').read_text())['runs']
    diagnostics[arm]={x['run']:x for x in json.loads((stage/'complete_action_summary.json').read_text())}
gt=json.loads((REPO/'gt_files/trajdata_nusc_new/GroundTruth/hist_stats_new_scene_current_speed.json').read_text())
rows=[];episodes=[]
actor_audit=json.loads((OUT/'actor_metrics_and_execution.json').read_text())

def avg(items,key):
    vals=[r[key] for r in items if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else None

def cfd(items):
    components={}
    for key in ['lon_accel','lat_accel','jerk']:
        h=np.zeros(20)
        for r in items:
            for scene in r['scenes']:
                x=np.asarray(scene['cfd_hist_counts'][key],float)
                if x.sum()>0:h+=x/x.sum()
        ref=np.asarray(gt['stats'][key],float);ticks=np.asarray(gt['ticks'][key],float)[1:]
        components[key]=float(wasserstein_distance(ticks,ticks,u_weights=h,v_weights=ref)) if h.sum()>0 else None
    return float(np.mean(list(components.values()))) if all(x is not None for x in components.values()) else None

for planner in P['planners']:
    cohort=[j for j in P['jobs'] if j['planner']==planner and j['arm']=='guided']
    assert len(cohort)==50
    successful=[reports['guided'][j['job_id']] for j in cohort if j['job_id'] in reports['guided']]
    native_hits=sum(r.get('ego_collision_native',0)>0 for r in successful)
    count=dict(action_applicable=0,action_complete=0,A_reasonable=0,target_contacts=0,new_raw_target_contacts=0,
               new_qualified_A=0,new_qualified_AE=0,matched_pairs_completed=0,geometry_unknown=0,
               matched_geometry_pairs_evaluable=0)
    for j in cohort:
        key=j['job_id'];g=diagnostics['guided'].get(key);n=diagnostics['noguide'].get(key)
        episode=dict(scene=j['scene'],planner=planner,family=j['family'],guided_completed=g is not None,
                     control_completed=n is not None,new_qualified_A=False,new_qualified_AE=False)
        if g is not None and n is not None:count['matched_pairs_completed']+=1
        if j['family']!='NONE':
            count['action_applicable']+=1
            if g:
                contact=g['geometry']['target_contact'];count['target_contacts']+=int(contact is True)
                count['geometry_unknown']+=int(contact is None)
                count['action_complete']+=int(g['mechanism']['completed'])
                count['A_reasonable']+=int(g['provisional_reasonableness_pass'])
                episode.update(action_complete=g['mechanism']['completed'],A_reasonable=g['provisional_reasonableness_pass'],
                               target_contact=contact,A_qualified=g['A_only_qualified_contact'],AE_qualified=g['qualified_contact_with_ego_dynamic_check'])
                if n and contact is not None and n['geometry']['target_contact'] is not None:
                    count['matched_geometry_pairs_evaluable']+=1
                    fresh=contact and not n['geometry']['target_contact']
                    qa=bool(fresh and g['A_only_qualified_contact']);qe=bool(fresh and g['qualified_contact_with_ego_dynamic_check'])
                    count['new_raw_target_contacts']+=int(fresh);count['new_qualified_A']+=int(qa);count['new_qualified_AE']+=int(qe)
                    episode.update(new_raw_target_contact=fresh,new_qualified_A=qa,new_qualified_AE=qe)
        episodes.append(episode)
    n_complete=len(successful)
    row=dict(planner=planner,planned_scenes=50,completed_scenes=n_complete,missing_or_failed=50-n_complete,
             native_ego_CR_completed_percent=100*native_hits/n_complete if n_complete else None,
             native_ego_CR_lower_bound_all50_percent=100*native_hits/50,
             native_ego_CR_upper_bound_all50_percent=100*(native_hits+50-n_complete)/50,
             SCR_completed_percent=100*avg(successful,'SCR_native') if successful else None,
             ORR_completed_percent=avg(successful,'ORR_native_percent'),CFD=cfd(successful),
             ADE_m=avg(successful,'ADE_m'),FDE_m=avg(successful,'FDE_m'),**count,
             new_qualified_A_rate_all50_percent=2*count['new_qualified_A'],
             new_qualified_AE_rate_all50_percent=2*count['new_qualified_AE'])
    controls=[reports['noguide'][j['job_id']] for j in cohort if j['job_id'] in reports['noguide']]
    paired=[(reports['guided'][j['job_id']],reports['noguide'][j['job_id']]) for j in cohort
            if j['job_id'] in reports['guided'] and j['job_id'] in reports['noguide']]
    row['control_completed_scenes']=len(controls)
    row['control_native_ego_CR_completed_percent']=100*avg(controls,'ego_collision_native') if controls else None
    for key in ['ORR_native_percent','ADE_m','FDE_m']:
        row['control_'+key]=avg(controls,key)
        differences=[g[key]-n[key] for g,n in paired if g.get(key) is not None and n.get(key) is not None]
        row['paired_guided_minus_control_'+key]=float(np.mean(differences)) if differences else None
    row['control_CFD']=cfd(controls)
    rows.append(row)
    actor_rows=[x for x in actor_audit['rows'] if x['arm']=='guided' and x['run'].endswith('_'+planner+'_s1')]
    for key in ['A_ADE_m','A_FDE_m','E_ADE_m','E_FDE_m']:
        row[key]=avg(actor_rows,key)
        row[key+'_n']=sum(x.get(key) is not None for x in actor_rows)
    row['A_reference_frame_fraction_mean']=avg(actor_rows,'A_reference_fraction')
    row['execution_audit_pass_count']=sum(x['execution_matches_logged_actions'] and x['controller_state_persisted'] for x in actor_rows)
(OUT/'summary.json').write_text(json.dumps(dict(protocol=str(REPO/'benchmark50_protocol_20261021_v3/protocol.json'),
    rows=rows,episodes=episodes,SOTA_claim=False,comparability=P['comparability']),ensure_ascii=False,indent=2))
with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fmt=lambda x:'—' if x is None else f'{x:.2f}'
text=['# CrashDojo-FM：历史50场景首轮评测','',
      '固定50场景、第31帧、100步、seed 1、5个planner；每个planner 50个guided + 50个无guide对照。未训练。',
      '碰撞与合理性失败均保留。无适用动作场景计入50个分母。能力标准仍为工程筛查，不代表车辆物理认证。',
      '历史算法的执行层、特征适配器、模型版本和候选预算存在差异；本轮不构成SOTA结论。','',
      '| Planner | 完成/50 | 新增合格A碰撞/50 | 新增合格AE碰撞/50 | 原生Ego CR（已完成） | SCR | ORR | CFD | ADE/FDE m |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    text.append(f"| {r['planner']} | {r['completed_scenes']}/50 | {r['new_qualified_A']}/50 | {r['new_qualified_AE']}/50 | {fmt(r['native_ego_CR_completed_percent'])}% | {fmt(r['SCR_completed_percent'])}% | {fmt(r['ORR_completed_percent'])}% | {fmt(r['CFD'])} | {fmt(r['ADE_m'])}/{fmt(r['FDE_m'])} |")
text+=['','如有运行失败，原生碰撞率的全50场景上下界见CSV；不能把缺失回合解释为真实无碰撞。',
       '目标车ADE仅使用存在真实记录的帧，帧覆盖率和样本数单独报告；最终帧缺少真实标注时FDE缺失，不填零，也不以最后可用帧替代。',
       '下一轮依据完整失败分类调整全局推理参数与几何模板，并完整重跑所报告的版本；不拼接各场景最好结果。']
(OUT/'50场景评测结果.md').write_text('\n'.join(text)+'\n')
historical=[r for r in csv.DictReader(Path(__file__).with_name('historical_reference.csv').open(encoding='utf-8-sig'))
            if r['sample_scope']=='50 个场景']
assert len(historical)==20
display=dict(idm='IDM',strive='STRIVE（规则适配器）',pdm_closed='PDM-Closed',
             diffusion_planner='Diffusion Planner',flow_planner='Flow Planner')
comparison=[dict(r,comparison_status='historical_reference_protocol_not_fully_aligned') for r in historical]
for r in rows:
    comparison.append(dict(planner=display[r['planner']],method='CrashDojo-FM v3',
        sample_scope=f"{r['completed_scenes']}/50 个场景完成",Ego_CR_percent=r['native_ego_CR_completed_percent'],
        SCR_percent=r['SCR_completed_percent'],ORR_percent=r['ORR_completed_percent'],CFD=r['CFD'],
        ADE_m=r['ADE_m'],FDE_m=r['FDE_m'],comparable_as_ranking=False,source_version='benchmark50_20261021_v3',
        comparison_status='current_inference_protocol_full_cohort_required'))
with (OUT/'五算法完整对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
print(json.dumps(rows,ensure_ascii=False,indent=2))
