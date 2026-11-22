"""Freeze the remaining v6 scenes for an isolated contextual v26 rollout."""
import collections
import hashlib
import json
import time
from pathlib import Path

from common import HERE, QUEUE, ROOT, load, save, sha
from contextual_contract import bind_context


SOURCE = ROOT / 'eval/v6_allpass_remaining225_20261101'
SELECTION = SOURCE / 'selection_v6_manifest.jsonl'
CONTEXT_DIR = SOURCE / 'context_inputs_v1'
OLD = ROOT / 'eval/causal_guidance_v26_contextual_val150_20261030'
PLANNERS = ['idm', 'strive', 'pdm_closed', 'diffusion_planner', 'flow_planner']


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    assert not (QUEUE / 'protocol.json').exists()
    selected = rows(SELECTION)
    contexts = rows(CONTEXT_DIR / 'contexts.jsonl')
    audit = load(CONTEXT_DIR / 'audit.json')
    assert len(selected) == len(contexts) == 225
    assert audit['completed'] and audit['scenes'] == 225
    assert audit['future_invariance_passed'] and audit['max_input_time_lte_cutoff']
    assert audit['fixed_selection_manifest_sha256'] == sha(SELECTION)
    assert audit['contexts_sha256'] == sha(CONTEXT_DIR / 'contexts.jsonl')
    context_by_index = {int(row['eval_index']): row for row in contexts}
    assert len(context_by_index) == 225
    assert [row['eval_index'] for row in selected] == list(range(225))
    assert [row['eval_index'] for row in selected if row['packet_duration_shortfall']] == [222]

    old_release = OLD / 'release.json'
    release = load(old_release)
    assert release['datasets']['nuplan']['scale'] == 4.0
    assert release['datasets']['nuplan']['beta'] == 1.0
    save(QUEUE / 'release.json', release)
    record_dir = HERE / 'inputs/context_records'
    record_dir.mkdir(exist_ok=True)
    inputs = {str(SELECTION): sha(SELECTION), str(CONTEXT_DIR / 'contexts.jsonl'): sha(CONTEXT_DIR / 'contexts.jsonl'),
              str(CONTEXT_DIR / 'audit.json'): sha(CONTEXT_DIR / 'audit.json'),
              str(QUEUE / 'release.json'): sha(QUEUE / 'release.json')}
    jobs = []
    factors = []
    for selected_row in selected:
        index = selected_row['eval_index']
        context = context_by_index[index]
        request_path = Path(selected_row['request'])
        manifest_path = Path(selected_row['manifest'])
        assert sha(request_path) == selected_row['request_sha256']
        assert sha(manifest_path) == selected_row['manifest_sha256']
        request = load(request_path)
        manifest = rows(manifest_path)[0]
        assert context['request_path'] == str(request_path)
        assert context['manifest_path'] == str(manifest_path)
        assert context['scene_id'] == selected_row['source_scene_id']
        assert context['fixed_manifest_scene_id'] == selected_row['scene_id']
        assert context['start_step_10hz'] == request['start_step_10hz'] == manifest['start_step_10hz']
        assert context['family'] == request['family'] == selected_row['family']
        assert str(context['actor']) == str(request['adversary_vehicle_id']) == selected_row['selected_actor']
        assert context['future_invariance_passed']
        assert context['max_input_time_s'] <= context['cutoff_s'] + 1e-8
        table_path = ROOT / context['prior_table']
        assert sha(table_path) == context['prior_table_sha256']
        table = load(table_path)
        assert table['fit_only'] and table['evaluation_logs_excluded']
        record = dict(
            eval_index=index, dataset='nuplan', scene=selected_row['source_scene_id'],
            family=request['family'], actor=str(request['adversary_vehicle_id']),
            start_step_10hz=request['start_step_10hz'], start_time_s=manifest['start_time_s'],
            observation_cutoff_s=context['cutoff_s'], max_input_time_s=context['max_input_time_s'],
            base_request_sha256=sha(request_path), manifest_sha256=sha(manifest_path),
            resolver_request=context['resolver_request'], resolver_context=context['resolver_context'],
            raw_context=context['context'], prior=context['prior'],
            prior_table=str(table_path), prior_table_sha256=sha(table_path),
            future_invariance_passed=True, preparation_record=context,
        )
        block = f'remaining225_{index:03d}'
        record_path = record_dir / f'{block}.json'
        save(record_path, record)
        inputs[str(record_path)] = sha(record_path)
        inputs[str(table_path)] = sha(table_path)
        inputs[str(request_path)] = sha(request_path)
        inputs[str(manifest_path)] = sha(manifest_path)
        prior = context['prior']
        factors.append(prior['factor'])
        for planner in PLANNERS:
            job = dict(
                job_id=f'{block}_{planner}', block=block, dataset='nuplan',
                partition=selected_row['isolated_role'], source_split=selected_row['source_split'],
                scene=selected_row['source_scene_id'], log=selected_row['log_id'],
                family=selected_row['family'], planner=planner, arm='calibrated',
                scale=None, beta=None, prior_factor=prior['factor'], prior_reason=prior['reason'],
                prior_support_level=prior['support_level'], prior_events=prior['events'],
                prior_scenes=prior['scenes'], prior_logs=prior['logs'],
                prior_fallback_reason=prior.get('fallback_reason'),
                base_request=str(request_path), manifest=str(manifest_path),
                config=str(HERE / 'inputs/nuplan_train_config.json'), steps=70,
                packet_steps=selected_row['packet_steps'],
                packet_duration_shortfall=selected_row['packet_duration_shortfall'],
                selected_actor=selected_row['selected_actor'], context_record=str(record_path),
                native_main_table_eligible=False,
            )
            bind_context(job, request)
            jobs.append(job)
    config_path = HERE / 'inputs/nuplan_train_config.json'
    inputs[str(config_path)] = sha(config_path)
    checkpoint = ROOT / 'models/original150k_bnonly_v1/flow/run0/checkpoints/last.ckpt'
    stat = checkpoint.stat()
    summary = dict(scenes=225, jobs=1125, runnable_full_length_scenes=224,
                   short_packet_index=222, split_counts=dict(collections.Counter(
                       x['source_split'] for x in selected)),
                   family_counts=dict(collections.Counter(x['family'] for x in selected)),
                   factors_nonunit=sum(abs(x - 1.) > 1e-12 for x in factors),
                   factor_range=[min(factors), max(factors)],
                   selection_unchanged=True, native_main_table_eligible=False)
    save(QUEUE / 'preparation_audit.json', summary)
    inputs[str(QUEUE / 'preparation_audit.json')] = sha(QUEUE / 'preparation_audit.json')
    protocol = dict(
        version=HERE.name, status='frozen', execution_authorized=True,
        authorization='User requested remaining 225 v6 all-pass scenes and all planners.',
        created_unix=time.time(), scope='v6_remaining225_contextual_pressure_five_planners',
        training=False, no_retraining=True, planners=PLANNERS, seeds=[1], scenes=225,
        selection_count=225, selection_manifest=str(SELECTION), selection_manifest_sha256=sha(SELECTION),
        selection_policy='v6 seven-criteria all-pass; offline selection; closed_loop_validated=false',
        fixed_scene_set=True, no_future_in_dynamic_inputs=True,
        evaluation_outcomes_excluded_from_selection=True, jobs=jobs,
        source_sha256={str(path): sha(path) for path in sorted(HERE.rglob('*.py'))},
        input_sha256=inputs, reference_protocol=str(OLD / 'protocol.json'),
        reference_protocol_sha256=sha(OLD / 'protocol.json'),
        pressure_rule='base_family * frozen_dataset_scale * hierarchical_context_factor * exp(beta*(1-2*live_CV_risk)); cap 16*base',
        context_update_policy='episode-start causal context; live pair risk every 0.5 s',
        context_prior_scope='observed natural motion intensity, not collision probability or an optimal guidance weight',
        context_statistics_evaluation_logs_excluded=True,
        tuning_scope='inherited IDM scale/beta; no retuning on new evaluation outcomes',
        selection_scope='offline v6 selected actors retained; not an online causal selection claim',
        packet_shortfall_policy='index 222 has only 55 source frames and is blocked from 70-frame protocol until independently resolved',
        native_main_table_eligible=False,
        generator_checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint),
                                  size=stat.st_size, mtime_ns=stat.st_mtime_ns),
    )
    save(QUEUE / 'protocol.json', protocol)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
