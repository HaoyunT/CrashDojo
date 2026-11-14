"""Freeze the original scene cohort and method-owned baseline integrations."""
import collections
import copy
import json
from pathlib import Path
import time

from common import ROOT, HERE, QUEUE, SOURCE, REPO, METHODS, PLANNERS, load, save, sha
from metrics import CONTRACT, FIELDS


def main():
    if (QUEUE / 'protocol.json').exists():
        raise FileExistsError('Refuse to overwrite a frozen baseline protocol')
    source = load(SOURCE / 'protocol.json')
    assert set(source['planners']) == set(PLANNERS) and len(source['jobs']) == 225 * len(PLANNERS)
    assert source['scenes'] == 225
    assert sha(source['selection_manifest']) == source['selection_manifest_sha256']
    support = load(HERE / 'safesim_adapter/support225.json')
    support_rows = support['rows'] if isinstance(support, dict) else support
    by_scene = {r.get('scene', r.get('scene_id')): r for r in support_rows}
    assert len(by_scene) == len(support_rows) == 225
    static_inputs = list((QUEUE / 'inputs/roads').glob('*.wkb')) + [
        HERE / 'safesim_adapter/support225.json', ROOT / 'configs/safesim_checkpoints.yaml',
        ROOT / 'configs/ccdiff_paper_eval.json',
        ROOT / 'configs/rule_planner_eval_v1.json']
    static_hashes = {str(p): sha(p) for p in static_inputs}
    base = Path('/root/path/to/workspace/home')
    checkpoint_paths = [
        ROOT / 'models/original150k_bnonly_v1/diffusion/run0/checkpoints/last.ckpt',
        ROOT / 'models/original150k_bnonly_v1/flow/run0/checkpoints/last.ckpt',
        base / 'Friday/safe-sim/models/safesim_checkpoints/safesim_fut32/test/run0/checkpoints/iter70000.ckpt',
        base / 'Friday/safe-sim/models/safesim_checkpoints/nusc_dynUnicycle_gl0_yrl0_tfTrue_4130553/run0/checkpoints/iter94000_ep6_valLoss0.06.ckpt']
    checkpoint_identity = {str(p): dict(sha256=sha(p), size=p.stat().st_size,
        mtime_ns=p.stat().st_mtime_ns) for p in checkpoint_paths}
    for checkpoint in checkpoint_paths:
        config = checkpoint.parent.parent / 'config.json'
        if config.exists():
            static_hashes[str(config)] = sha(config)
    jobs = []
    for original in source['jobs']:
        assert original['dataset'] == 'nuplan' and original['steps'] == 70
        manifest = json.loads(Path(original['manifest']).read_text().splitlines()[0])
        for method in METHODS:
            inputs = {str(original[k]): sha(original[k]) for k in ['manifest', 'base_request', 'config']}
            inputs.update(static_hashes)
            adaptation = False
            if method == 'safesim':
                row = by_scene.get(original['scene'])
                if row is None:
                    raise ValueError('SafeSim support unaccounted for: ' + original['scene'])
                adaptation = (row.get('planned_selection_mode') == 'user_authorized_v6_fallback'
                              or row.get('status', row.get('state')) in ['blocked_selection', 'adapted_selection'])
            source_job = copy.deepcopy(original)
            if original['dataset'] == 'nuplan':
                source_job['nuplan_cache'] = str(QUEUE / 'inputs/cache/nuplan_centered_private')
            job = dict(job_id=method + '_' + original['job_id'], method=method,
                planner=original['planner'], source_job=source_job,
                dataset=original['dataset'], scene=original['scene'],
                steps=original['steps'], start_frame=int(manifest['start_step_10hz']), seed=1,
                initial_state=('blocked_input' if original['packet_duration_shortfall'] else 'queued'),
                blocked_reason=('source_packet_shorter_than_70_frames' if original['packet_duration_shortfall'] else None),
                source_split=original['source_split'],
                selection_adaptation_authorized=method == 'safesim',
                selection_adaptation_expected=adaptation,
                display_method='SafeSim-selection-adapted' if method == 'safesim' else method,
                initial_reference=str(QUEUE / 'inputs/references' / (original['block'] + '.pt')),
                input_sha256=inputs)
            jobs.append(job)
            save(QUEUE / 'jobs' / (job['job_id'] + '.json'), job)
    runtime = sorted(HERE.rglob('*.py'))
    dependencies = sorted(set(
        list((ROOT / 'launch/causal_guidance_v26_contextual_full_eval_20261101').rglob('*.py'))
        + list((REPO / 'ccdiff').rglob('*.py'))
        + list((REPO / 'third_party/tbsim/tbsim').rglob('*.py'))
        + list((REPO / 'legacy_common_adapter_v20_20261027').glob('*.py'))
        + list((ROOT / 'launch/benchmark50_native_common_execution_v5_20261029').glob('*.py'))))
    for project in ['safe-sim_legacy100']:
        dependencies += list((ROOT / 'code' / project / 'tbsim').rglob('*.py'))
    assert len(jobs) == 225 * len(PLANNERS) * len(METHODS)
    value = dict(version=QUEUE.name, status='frozen_preflight', training=False,
        methods=METHODS, planners=PLANNERS, total=len(jobs), jobs=jobs,
        unique_scenes=225, created_unix=time.time(),
        comparison_track='method_owned_selection_and_guidance_v6_remaining225',
        source_protocol=str(SOURCE / 'protocol.json'), source_protocol_sha256=sha(SOURCE / 'protocol.json'),
        selection_manifest=source['selection_manifest'], selection_manifest_sha256=source['selection_manifest_sha256'],
        dataset_counts=dict(collections.Counter(j['dataset'] for j in jobs)),
        selected_actors='CCDiff preserves native selection; SafeSim uses its official preset when available and otherwise the request target',
        actor_budgets='native actor counts and reselection schedules preserved; not equalized',
        ego_execution='same five_class_planners.build_selected_ego and NativeTrackingEgoPolicy as contextual750',
        background_execution='exact available logged replay; explicit missing availability',
        metric_contract=CONTRACT, metric_fields=FIELDS,
        safe_sim_support=str(HERE / 'safesim_adapter/support225.json'),
        safe_sim_unsupported_jobs=0,
        short_packet_blocked_jobs=sum(j['initial_state'] == 'blocked_input' for j in jobs),
        safe_sim_expected_adapted_jobs=sum(j['selection_adaptation_expected'] for j in jobs),
        unsupported_is_neither_completed_nor_zero_collision=True,
        runtime_sha256={str(p): sha(p) for p in runtime},
        dependency_sha256={str(p): sha(p) for p in sorted(set(dependencies))},
        static_input_sha256=static_hashes, checkpoint_identity=checkpoint_identity,
        road_audit_sha256=sha(QUEUE / 'inputs/roads/audit.json'))
    save(QUEUE / 'protocol.json', value)
    save(QUEUE / 'state.json', dict(state='preflight', total=len(jobs),
        completed=0, running=0, failed=0, blocked_selection=value['safe_sim_unsupported_jobs'],
        blocked_input=value['short_packet_blocked_jobs'],
        queued=len(jobs)-value['safe_sim_unsupported_jobs']-value['short_packet_blocked_jobs']))
    print(json.dumps({k: v for k, v in value.items() if k in ['total', 'dataset_counts', 'safe_sim_unsupported_jobs']}))


if __name__ == '__main__':
    main()
