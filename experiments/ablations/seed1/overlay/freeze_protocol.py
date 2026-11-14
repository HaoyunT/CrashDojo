import collections
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO=ROOT/'code/CCDiff_M2c_planner'
INPUT=REPO/'benchmark50_inputs_20261021_v3'
OVERLAY=REPO/'benchmark50_overlay_20261021_v3'
DEST=REPO/'benchmark50_protocol_20261021_v3'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

assert not DEST.exists(),'Never overwrite a frozen protocol'
assert json.loads((INPUT/'gpu_preflight.json').read_text())['passed']
assert json.loads((INPUT/'runtime_cache_preflight.json').read_text())['all_initial_history_identity_and_ego_future_pass']
assert json.loads((INPUT/'partial_reference_runtime_probe.json').read_text())['passed']
assert json.loads((INPUT/'batched_corridor_equivalence.json').read_text())['passed']
DEST.mkdir();rows=[json.loads(s) for s in (INPUT/'manifest.jsonl').read_text().splitlines()]
assert len(rows)==50 and len({r['scene_id'] for r in rows})==50
planners=['idm','strive','pdm_closed','diffusion_planner','flow_planner']
source_paths=list(OVERLAY.glob('*.py'))+[OVERLAY/'historical_reference.csv']+[REPO/p for p in [
 'scripts/native_tracking_entry.py','scripts/evaluate_selected_rollout.py','ccdiff/evaluation/native_tracker_client.py',
 'ccdiff/evaluation/selection_manifest.py','ccdiff/evaluation/five_class_planners.py',
 'ccdiff/examples/scene_editor.py','ccdiff/models/ccdiff.py',
 'selection_inputs_20261020/nuscenes_selection_config.json']]
source_hashes={str(p):sha(p) for p in source_paths}
jobs=[];request_hashes=[];manifest_hashes=[]
for arm in ['guided','noguide']:
    out=DEST/arm;out.mkdir();shutil.copy(INPUT/'manifest.jsonl',out/'manifest.jsonl')
    manifest_hashes.append(sha(out/'manifest.jsonl'))
    for i,row in enumerate(rows):
        request=json.loads((INPUT/(row['scene_id']+'.json')).read_text())
        if arm=='noguide':
            request.update(gradient_steps=0,select_guided_candidate=False,prefix_feasibility_repair_steps=0)
        path=out/(row['scene_id']+'.json');path.write_text(json.dumps(request,ensure_ascii=False,indent=2))
        request_hashes.append(sha(path))
        for planner in planners:
            jobs.append(dict(job_id=row['scene_id']+'_'+planner+'_s1',scene=row['scene_id'],family=request['family'],
                             arm=arm,planner=planner,seed=1,row=i,steps=100,samples=5,
                             request=str(path),manifest=str(out/'manifest.jsonl'),
                             stage='benchmark50_'+arm+'_20261021_v3'))
gate=[]
for family in ['NONE','C1','C2','C3A','C3B','C4','C5']:
    found=[j for j in jobs if j['family']==family and j['arm']=='guided' and j['planner']=='idm']
    if found:gate.append(found[0]['job_id'])
for planner in planners[1:]:
    gate.append(next(j['job_id'] for j in jobs if j['family']=='C1' and j['arm']=='guided' and j['planner']==planner))
protocol=dict(version=1,status='frozen',execution_authorized=True,
    revision='v3_actual_history_partial_future_reference; C1_shared_route_origin_fix; equivalent_batched_polygon_union_loss; earlier_versions_preserved',
    authorization='User explicitly requested CrashDojo-FM evaluation and optimization on the historical 50 val scenes on 2026-10-21.',
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='first_complete_50_scene_inference_round_not_a_SOTA_claim',training=False,
    planners=planners,seeds=[1],steps=100,samples=5,start_frame=31,dt=.1,replan_steps=5,
    cohort=[r['scene_id'] for r in rows],jobs=jobs,planned_rollouts=len(jobs),
    startup_gate_job_ids=gate,request_sha256=request_hashes,selection_manifest_sha256=list(set(manifest_hashes)),
    source_sha256=source_hashes,
    validity_policy=dict(body_corridor_min=.98,speed_max=25.,longitudinal_acceleration=[-6.,3.],
                         lateral_acceleration_abs_max=4.,longitudinal_and_vector_jerk_abs_max=10.,
                         raw_10hz_finite_differences=True,capability_status='provisional',
                         require_complete_mechanism=True,require_no_initial_target_overlap=True,
                         exclude_earlier_adversary_non_target_contact=True,
                         require_after_action_and_body_entry=True,primary_denominator_per_planner=50,
                         no_action_scenes_remain_in_denominator=True,technical_failures_retained=True),
    evaluator_version='benchmark50_native_metrics_and_native_map_complete_action_v1',
    objective='Increase matched newly caused qualified A-E contact while retaining constraints and reporting native CR, SCR, ORR, CFD, ADE/FDE plus actor-specific errors.',
    inference_optimization=dict(pressure_weight=2.,no_weight_training=True,
        solver='L-BFGS quasi-Newton; not Gauss-Newton',optimization_variable='selected_actor_sampled_actions',
        last_flow_steps=10,max_LBFGS_iterations_per_flow_step=30,max_prefix_repair_iterations=20,
        guidance='five_class_geometry_and_clock_conditioned; prefix feasibility repair; frozen unguided predicted endpoint deviation anchor',
        no_recorded_future_input_to_runtime_guide=True),
    reference_policy=dict(required_actual_history_frames=31,require_full_adversary_recorded_future=False,
        simulation='freeze_initial_agents_and_append_generated_states',
        adversary_ADE='available_recorded_frames_only_with_count_and_fraction',
        adversary_FDE='reported_only_if_actual_reference_exists_at_final_frame',
        missing_reference_imputation=False),
    comparability=dict(same_historical_scene_set=True,
        historical_results_are_reference_only=True,
        unresolved=['historical planner feature adapters and execution differ from current nuPlan native tracking',
                    'historical generator checkpoints and candidate budgets differ across cells',
                    'website candidate ranking is an offline process and can use recorded reference future',
                    'scene-0096 was used in earlier development; this cohort is not entirely untouched'],
        sotaclaim_allowed=False),
    candidate_coverage=json.loads((INPUT/'coverage.json').read_text())['family_counts'],
    resource_policy=dict(gpus=[0,1],max_owned_rollouts_per_gpu=2,timeout_seconds_per_rollout=3600))
(DEST/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
for name in ['coverage.json','runtime_cache_preflight.json','gpu_preflight.json','partial_reference_runtime_probe.json','batched_corridor_equivalence.json']:
    shutil.copy(INPUT/name,DEST/name)
shutil.copytree(OVERLAY,DEST/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
print(json.dumps(dict(protocol=str(DEST/'protocol.json'),jobs=len(jobs),gates=gate,families=protocol['candidate_coverage']),ensure_ascii=False))
