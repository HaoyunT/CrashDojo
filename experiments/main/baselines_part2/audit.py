"""Audit every requested cell; unsupported scenes never become successful runs."""
import collections
import math
from pathlib import Path

from common import QUEUE, METHODS, PLANNERS, load, save, sha, verify_sources
from metrics import FIELDS, CONTRACT
from run_job import frozen_job, validate_inference, validate_reference


def main():
    protocol = load(QUEUE / 'protocol.json')
    verify_sources(protocol)
    for path, expected in protocol['static_input_sha256'].items():
        assert sha(path) == expected, 'Frozen comparison input changed: ' + path
    for path, expected in protocol['checkpoint_identity'].items():
        assert sha(path) == expected['sha256'], 'Frozen checkpoint changed: ' + path
    for path, expected in protocol['dependency_sha256'].items():
        assert sha(path) == expected, 'Frozen dependency changed: ' + path
    assert protocol['total'] == len(protocol['jobs'])
    assert len({j['job_id'] for j in protocol['jobs']}) == len(protocol['jobs'])
    errors, unsupported, blocked_input, results = [], [], [], []
    for job in protocol['jobs']:
        if job['initial_state'] == 'blocked_selection':
            unsupported.append(dict(job_id=job['job_id'], reason=job['blocked_reason']))
            continue
        if job['initial_state'] == 'blocked_input':
            blocked_input.append(dict(job_id=job['job_id'], reason=job['blocked_reason']))
            continue
        try:
            assert frozen_job(protocol, job['job_id']) == job
            state = load(QUEUE / 'status' / (job['job_id'] + '.json'))
            assert state['state'] == 'completed' and state['returncode'] == 0
            assert state['job_id'] == job['job_id'] and state['job'] == job
            assert state['formal'] is True and state['training'] is False
            directory = Path(state['output'])
            assert directory.resolve() == (QUEUE / 'runs' / job['job_id']).resolve()
            for path, expected in job['input_sha256'].items():
                assert sha(path) == expected, 'Frozen scene input changed: ' + path
            assert validate_reference(job) == {key: state[key] for key in
                ['initial_reference_sha256', 'initial_reference_metadata_sha256']}
            metrics = load(directory / 'metrics.json')
            validation = load(directory / 'native_inference_validation.json')
            validate_inference(validation, job, job['steps'])
            assert metrics['frames'] == job['steps']
            for key in ['method', 'planner', 'dataset', 'scene']:
                assert metrics[key] == job[key], 'Metric identity differs: ' + key
            assert sha(directory / 'metrics.json') == state['metrics_sha256']
            assert sha(directory / 'native_inference_validation.json') == state['validation_sha256']
            assert metrics['contract'] == CONTRACT
            assert directory.resolve() in Path(metrics['hdf5']).resolve().parents
            assert sha(metrics['hdf5']) == metrics['hdf5_sha256']
            for field in FIELDS[:5]:
                assert type(metrics[field]) is bool, 'Invalid collision flag: ' + field
            for field in FIELDS[5:]:
                assert type(metrics[field]) in [int, float] and math.isfinite(metrics[field])
                assert metrics[field] >= 0, 'Negative native metric: ' + field
            assert metrics['ORR_native'] <= 1
            assert not metrics['qualified_selected_contact_10Hz'] or metrics['target_contact_10Hz']
            assert not metrics['target_contact_10Hz'] or metrics['ego_collision_10Hz']
            assert not metrics['ego_collision_10Hz'] or metrics['scene_collision_10Hz']
            assert not metrics['target_background_contact_10Hz'] or metrics['scene_collision_10Hz']
            metrics['_audit_selection_mode'] = validation.get('selection_mode', 'native')
            results.append((job, metrics))
        except (OSError, KeyError, TypeError, ValueError, AssertionError) as exc:
            errors.append(dict(job_id=job['job_id'], error=repr(exc)))
    def aggregate(items, expected=225):
        row = dict(completed=len(items), expected=expected,
                   selection_modes=dict(collections.Counter(m['_audit_selection_mode'] for _, m in items)))
        for field in FIELDS:
            values = [m[field] for _, m in items if isinstance(m.get(field), (bool, int, float))
                      and math.isfinite(m[field])]
            row[field] = dict(mean=sum(values)/len(values) if values else None,
                              available=len(values), expected=expected)
        return row
    audit = dict(status='completed_audited' if not errors and not unsupported and not blocked_input else 'incomplete_or_unsupported',
                 passed=not errors and not unsupported and not blocked_input, total=protocol['total'], completed=len(results),
                 errors=errors, unsupported=unsupported, blocked_input=blocked_input, training=False,
                 protocol_sha256=sha(QUEUE / 'protocol.json'))
    save(QUEUE / 'completion_audit.json', audit)
    save(QUEUE / 'provisional_metrics.json', dict(status=audit['status'], contract=CONTRACT,
         by_method_planner={method: {planner: aggregate([(j, m) for j, m in results
             if j['method'] == method and j['planner'] == planner]) for planner in PLANNERS}
             for method in METHODS},
         by_method_planner_domain={method: {planner: {dataset: aggregate(
             [(j, m) for j, m in results if j['method'] == method and j['planner'] == planner
              and j['dataset'] == dataset], expected=sum(j['method'] == method
                  and j['planner'] == planner and j['dataset'] == dataset for j in protocol['jobs']))
             for dataset in sorted({j['dataset'] for j in protocol['jobs']})}
             for planner in PLANNERS} for method in METHODS},
         by_method_planner_split={method: {planner: {split: aggregate(
             [(j, m) for j, m in results if j['method'] == method and j['planner'] == planner
              and j['source_split'] == split], expected=sum(j['method'] == method
                  and j['planner'] == planner and j['source_split'] == split for j in protocol['jobs']))
             for split in ['train', 'val', 'test']} for planner in PLANNERS} for method in METHODS}))
    if audit['passed']:
        save(QUEUE / 'finalized.json', audit)
    elif (QUEUE / 'finalized.json').exists():
        (QUEUE / 'finalized.json').unlink()
    print(audit['status'], audit['completed'], 'unsupported', len(unsupported), 'errors', len(errors))
    if not audit['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
