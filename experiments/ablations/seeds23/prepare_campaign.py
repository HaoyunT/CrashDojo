"""Freeze a prospective cohort using input metadata, never rollout outcomes."""
import copy, datetime, hashlib, json, sys, collections
import numpy as np
from pathlib import Path
from common import ROOT, HERE, QUEUE, load, save, sha

PIPE = Path('/path/to/workspace/CrashDojo_pipeline')
sys.path[:0] = [str(PIPE/'src'), str(ROOT/'launch/road_scene_strength_v20_20261026')]
sys.path.insert(0,'/path/to/storage/external_scripts')
from q16_map_events_v2_20261020 import load_tracks

def main():
    QUEUE.mkdir(exist_ok=True)
    assert not (QUEUE/'run_manifest.json').exists(), 'Preserve frozen cohort'
    sources = [ROOT/'eval/causal_guidance_v26_contextual_val150_20261030/protocol.json',
               ROOT/'eval/full_corpus_nonpass_20261102/ours/protocol.json']
    jobs = {}
    for source in sources:
        for j in load(source)['jobs']:
            if j['dataset']=='nuscenes' and j['planner']=='idm': jobs[j['scene']] = j
    assert len(jobs)==150
    index_path=PIPE/'artifacts/selection_v6_seven_criteria_20261030/execution_index.json'
    index=load(index_path)
    packets={s['scene_name']:index['packets'][s['default_packet_id']] for s in index['scenes'].values() if s['dataset']=='nuscenes'}
    eligible=[]; rejected=[]
    for scene,j in sorted(jobs.items()):
        q=load(j['base_request']); m=load(j['manifest']); pk=packets[scene]
        tracks=load_tracks(Path(pk['source_cache']['path'])); start=m['start_step_10hz']; aid=m['adversary_vehicle_id']
        ok=q['family']!='NONE' and all(k in tracks and all(start-t in tracks[k]['lookup'] for t in range(6)) for k in [aid,'ego'])
        if not ok: rejected.append(dict(scene=scene,reason='missing target or six contiguous observed positions')); continue
        if not Path(j['context_record']).exists(): rejected.append(dict(scene=scene,reason='missing frozen context'));continue
        ego_xy=tracks['ego']['xy'][tracks['ego']['lookup'][start]]
        admission_ids=sorted(k for k,tr in tracks.items() if k!='ego' and all(start-t in tr['lookup'] for t in range(6)) and
            (k==aid or np.linalg.norm(tr['xy'][tr['lookup'][start]]-ego_xy)<=60.))
        assert aid in admission_ids
        r=copy.deepcopy(j);r.update(cache=pk['source_cache']['path'],packet=pk,steps=100,selected_actor=aid,candidate_ids=admission_ids)
        eligible.append(r)
    eligible.sort(key=lambda j:hashlib.sha256(('CrashDojo_A0_A8_20261107_v1|'+j['scene']).encode()).hexdigest())
    assert len(eligible)>=100,(len(eligible),rejected)
    cohort=eligible[:100]
    template=load(ROOT/'launch/causal_guidance_v26_contextual_val150_20261030/inputs/nuscenes_val_config.json')
    template['num_simulation_steps']=template['trajdata']['num_simulation_steps']=100
    template['policy']['num_action_samples']=5
    cfg=QUEUE/'inputs/nuscenes_val_config.json';save(cfg,template)
    matrix=[]
    for j in cohort:
        scene=j['scene']; dest=QUEUE/'inputs'/scene;dest.mkdir(parents=True,exist_ok=True)
        q=load(j['base_request']);m=load(j['manifest']);c=load(j['context_record'])
        save(dest/'request_A0.json',q);save(dest/'manifest_A0.jsonl',m)
        save(dest/'context_A0.json',c)
        j.update(original_request=j['base_request'],original_manifest=j['manifest'],original_context=j['context_record'],
                 base_request=str(dest/'request_A0.json'),manifest=str(dest/'manifest_A0.jsonl'),context_record=str(dest/'context_A0.json'),config=str(cfg))
        for seed in [1,2,3]:
            for planner in ['idm','diffusion_planner']:
                for variant in range(9):
                    v='A'+str(variant)
                    matrix.append(dict(job_id=f'{scene}_s{seed}_{planner}_{v}',scene=scene,stream=seed,planner=planner,variant=v))
    save(QUEUE/'cohort.json',cohort)
    release=load(ROOT/'eval/causal_guidance_v26_contextual_val150_20261030/release.json')['datasets']['nuscenes']
    checkpoint=load(sources[0])['generator_checkpoint']
    assert sha(checkpoint['path'])==checkpoint['sha256']
    manifest=dict(version='CrashDojo_A0_A8_nusc100_s123_20261107_v1',paper_revision='48164d5',
        paper_revision_is_not_runtime_git_sha=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scene_selection='SHA256 fixed order of 150 nuScenes val input configurations; require a target and six contiguous observed positions for target and ego; first 100. No rollout metrics read.',
        source_scene_count=150,input_eligible_count=len(eligible),rejected_inputs=rejected,
        selected_scenes=[j['scene'] for j in cohort],seeds=[1,2,3],planners=['idm','diffusion_planner'],
        variants=['A'+str(v) for v in range(9)],planned_rollouts=5400,jobs=matrix,
        dt=.1,steps=100,replan_steps=5,action_samples=5,background='logged_replay',
        generator_checkpoint=checkpoint,pressure_scale=release['scale'],pressure_beta=release['beta'],
        A0_reused=False,source_protocols={str(p):sha(p) for p in sources},selection_index_sha256=sha(index_path),
        cohort_sha256=sha(QUEUE/'cohort.json'),config_sha256=sha(cfg),
        limitations=['Eligibility is executable-input eligibility, not a claim that all seven historical contextual criteria pass.',
                     'Existing Full selection is offline and may use logged ego continuation; no new outcome-based scene screening.'])
    save(QUEUE/'run_manifest.json',manifest)
    print(json.dumps(dict(queue=str(QUEUE),scenes=100,eligible=len(eligible),jobs=len(matrix),families=dict(collections.Counter(load(j['base_request'])['family'] for j in cohort)))),flush=True)

if __name__=='__main__':main()
