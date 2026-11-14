"""Prepare the authorized native-only cohort without reading rollout outcomes."""
import argparse
import collections
import copy
import datetime
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from common import ROOT, HERE, QUEUE, load, save, sha

INTAKE = ROOT / 'eval/crashdojo_nofallback_intake_20261108'
PIPE = Path('/path/to/workspace/CrashDojo_pipeline')
VERSION = HERE.name + '_v1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inventory',type=Path,default=INTAKE/'strict_nofallback82_inventory/input_inventory.json')
    parser.add_argument('--expected-scenes',type=int,default=82)
    parser.add_argument('--horizon-steps',type=int,choices=[70,100],default=100)
    args=parser.parse_args();steps=args.horizon_steps
    assert not (QUEUE / 'run_manifest.json').exists(), 'Preserve frozen campaign'
    inventory_path = args.inventory
    inventory = load(inventory_path)
    rows = inventory['scenes']
    assert len(rows) == len({x['scene'] for x in rows}) == args.expected_scenes
    rows.sort(key=lambda x: hashlib.sha256((VERSION + '|' + x['scene']).encode()).hexdigest())
    metadata_path = Path('/path/to/storage/data/outputs/q11_nuscenes_val_full_20261017/scene_meta.json')
    metadata = load(metadata_path)
    index_path = PIPE / 'artifacts/selection_v6_seven_criteria_20261030/execution_index.json'
    index = load(index_path)
    scene_index = {x['scene_name']: x for x in index['scenes'].values() if x['dataset'] == 'nuscenes'}
    sys.path.insert(0, '/path/to/storage/external_scripts')
    from q16_map_events_v2_20261020 import load_tracks
    config = load(ROOT / 'launch/causal_guidance_v26_contextual_val150_20261030/inputs/nuscenes_val_config.json')
    config['num_simulation_steps'] = config['trajdata']['num_simulation_steps'] = steps
    config['policy']['num_action_samples'] = 5
    config_path = QUEUE / 'inputs/nuscenes_val_config.json'
    save(config_path, config)
    prior_source = ROOT / 'eval/full_corpus_nonpass_20261102/ours/context_inputs_v1/observational_prior_eval_logs_excluded.json'
    prior_path = QUEUE / 'inputs/observational_prior_eval_logs_excluded.json'
    prior_path.write_bytes(prior_source.read_bytes())
    table = load(prior_path)
    assert table['fit_only'] and table['evaluation_logs_excluded']
    excluded = {(x['dataset'], x['log']) for x in table['excluded_evaluation_logs']}
    prepared = []
    for row in rows:
        scene = row['scene']
        assert metadata[scene]['split'] == 'val'
        q = load(row['request_path'])
        m = load(row['manifest_path'])
        pool_path = INTAKE / row['source_stage'] / 'scenes' / scene / 'candidate_pool.json'
        capture = load(pool_path)
        pool = capture['pool']
        assert pool and all(x['source'] == 'native_search' for x in pool)
        assert q['selection_mode'] == 'seven_criteria_nofallback_candidate'
        assert not q.get('random_fallback') and q.get('mechanism_type') != 'random_actor_speed_control'
        actor = str(q['adversary_vehicle_id'])
        start = q['start_step_10hz']
        assert actor == str(row['selected_actor']) == str(m['adversary_vehicle_id'])
        assert q['scene_id'] == m['scene_id'] == scene
        assert start == row['start_step'] == m['start_step_10hz']
        assert ('nuscenes', m['log_id']) in excluded
        packet = index['packets'][scene_index[scene]['default_packet_id']]
        cache = Path(packet['source_cache']['path'])
        tracks = load_tracks(cache)
        candidates = sorted({x['actor'] for x in pool})
        assert actor in candidates
        assert all(t in tracks['ego']['lookup'] for t in range(start-5, start+steps+1))
        for aid in candidates:
            assert all(start-d in tracks[aid]['lookup'] for d in range(6)), (scene, aid)
            assert all(x['request'].get('selection_mode') != 'random_fallback' for x in pool if x['actor'] == aid)
        dest = QUEUE / 'inputs' / scene
        dest.mkdir(parents=True)
        q.update(selection_mode='native_mechanism_candidates_scored_v6', selection_manifest_row=0,
                 purpose='Frozen Full request for the authorized native-only ablation cohort')
        m.update(selection_policy='native_mechanism_candidates_scored_v6', closed_loop_validated=False)
        m.pop('source_option_id', None)
        qp, mp, cp = dest / 'request_A0.json', dest / 'manifest_A0.jsonl', dest / 'context_A0.json'
        save(qp, q)
        mp.write_text(json.dumps(m, ensure_ascii=False, allow_nan=False) + '\n')
        save(QUEUE / 'candidate_capture/00/candidate_pools' / (scene + '.json'), capture)
        prepared.append(dict(dataset='nuscenes', partition='evaluation', source_split='val', scene=scene,
            log=m['log_id'], family=q['family'], planner='idm', arm='calibrated', steps=steps,
            selected_actor=actor, candidate_ids=candidates, cache=str(cache), packet=packet,
            base_request=str(qp), manifest=str(mp), context_record=str(cp), config=str(config_path),
            original_request=row['request_path'], original_manifest=row['manifest_path'],
            native_candidate_pool=str(pool_path), native_candidate_pool_sha256=sha(pool_path),
            input_criteria_overall=row['criteria_overall'], input_criteria_fail_count=row['criteria_fail_count'],
            initial_state='offline_input_checks_passed_runtime_preflight_pending'))
    save(QUEUE / 'cohort_prepared_inputs.json', prepared)

    builder_path = ROOT / 'launch/full_corpus_nonpass_20261102/ours/build_context_inputs.py'
    spec = importlib.util.spec_from_file_location('native82_context_builder', builder_path)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.torch.set_num_threads(1)
    # Here "train" is solely the existing raw trainval-table loader mode.
    # Every scene's actual val membership was asserted above.
    source = b.NuScenesSource({j['scene']: dict(metadata[j['scene']], split='train') for j in prepared})
    snapshot, snapshot_source = b.event_snapshot()
    predictor = b.Predictor(device='cpu')
    maps = {}
    started = time.time()
    for number, job in enumerate(prepared, 1):
        q, m = load(job['base_request']), load(job['manifest'])
        scene, city, cutoff = job['scene'], m['city'], m['start_step_10hz'] * .1
        if city not in maps:
            maps[city] = b.RoadMap('nuscenes', city)
        tracks = source.scene(scene)
        variants = [tracks, b.alter_future(tracks, cutoff, truncate=True), b.alter_future(tracks, cutoff)]
        values = [b.derive(t, job['selected_actor'], cutoff, maps[city], predictor, snapshot, q, table, 'nuscenes') for t in variants]
        assert b.canonical(values[0]) == b.canonical(values[1]) == b.canonical(values[2]), scene
        context = dict(dataset='nuscenes', scene=scene, scene_id=scene, actor=job['selected_actor'],
            family=q['family'], city=city, log=m['log_id'], start_step_10hz=m['start_step_10hz'],
            start_time_s=cutoff, observation_cutoff_s=cutoff, cutoff=cutoff, cutoff_s=cutoff,
            request_path=job['base_request'], manifest_path=job['manifest'],
            base_request_sha256=sha(job['base_request']), manifest_sha256=sha(job['manifest']),
            future_invariance_passed=True, future_invariance_modes=['full','future_truncated','future_polluted_and_future_actor_injected'],
            prior_table=str(prior_path), prior_table_sha256=sha(prior_path),
            source_refs=dict(context_builder=str(builder_path),context_builder_sha256=sha(builder_path),
                             semantic_model_sha256=sha(b.MODEL/'semantic_classifier.pt')),
            **values[0])
        save(job['context_record'], context)
        prior = context['prior']
        job.update(prior_factor=prior['factor'], prior_reason=prior['reason'],
                   prior_support_level=prior['support_level'], prior_events=prior['events'],
                   prior_scenes=prior['scenes'], prior_logs=prior['logs'], prior_fallback_reason=prior.get('fallback_reason'))
        save(QUEUE / 'preparation_progress.json', dict(stage='causal_contexts',done=number,total=len(prepared),scene=scene,elapsed_s=time.time()-started))
        print(json.dumps(dict(context=number,total=len(prepared),scene=scene,prior_factor=prior['factor'])),flush=True)
    save(QUEUE / 'cohort.json', prepared)
    old = load(ROOT / 'eval/crashdojo_A0_A8_nusc100_s123_20261107/run_manifest.json')
    assert sha(old['generator_checkpoint']['path']) == old['generator_checkpoint']['sha256']
    matrix = [dict(job_id=f'{j["scene"]}_s{seed}_{planner}_A{v}',scene=j['scene'],stream=seed,planner=planner,variant=f'A{v}')
              for j in prepared for seed in [1,2,3] for planner in ['idm','diffusion_planner'] for v in range(9)]
    manifest = dict(version=VERSION,paper_revision='48164d5',paper_revision_is_not_runtime_git_sha=True,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scene_selection=f'{len(prepared)} unique val scenes from the supplied frozen input inventory; construction and ordering are documented in that inventory. No generated collision outcome read. No random/speed-only target fallback.',
        source_scene_count=150,selected_scene_count=len(prepared),selected_scenes=[j['scene'] for j in prepared],
        seeds=[1,2,3],planners=['idm','diffusion_planner'],variants=[f'A{v}' for v in range(9)],planned_rollouts=len(matrix),jobs=matrix,
        dt=.1,steps=steps,replan_steps=5,action_samples=5,background='logged_replay',
        generator_checkpoint=old['generator_checkpoint'],pressure_scale=old['pressure_scale'],pressure_beta=old['pressure_beta'],
        A0_reused=False,cohort_sha256=sha(QUEUE/'cohort.json'),config_sha256=sha(config_path),
        selection_index_sha256=sha(index_path),inventory_sha256=sha(inventory_path),
        input_inventory=str(inventory_path),input_construction_protocol=inventory.get('search_protocol'),
        prior_table=str(prior_path),prior_table_sha256=sha(prior_path),prior_source=str(prior_source),
        prior_policy='Frozen full-corpus prior excluding all 150 validation-scene logs; no refitting or retuning.',
        original_campaign=str(ROOT/'eval/crashdojo_A0_A8_nusc100_s123_20261107'),
        limitations=['Native-only target selection does not imply all seven contextual criteria pass.',
                     'Offline selection can use the recorded ego continuation; pressure contexts use observations up to the start time only.',
                     f'{sum(len(j["candidate_ids"])>1 for j in prepared)} of {len(prepared)} scenes contain multiple eligible target actors; selection ablations may coincide in single-candidate scenes.',
                     'Prior lookup table is uniformly the existing full-corpus heldout-log table, whereas the previous cohort inherited two tables. These runs are a separate experiment.'])
    save(QUEUE / 'run_manifest.json', manifest)
    save(QUEUE / 'input_preflight.json',dict(passed=True,scenes=len(prepared),native_only=True,causal_contexts_verified=True,
        future_invariance_modes=['truncate','pollute'],generated_outcomes_used=False,runtime_preflight_pending=True))
    print(json.dumps(dict(stage='prepared',scenes=len(prepared),planned_rollouts=len(matrix))),flush=True)


if __name__ == '__main__':
    main()
