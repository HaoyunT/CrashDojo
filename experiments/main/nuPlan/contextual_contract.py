"""Bind a frozen scene context to the selected actor and runtime pressure."""
import hashlib
import json
import math
from pathlib import Path

from hierarchical_prior import resolve


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_context(job, request):
    record = read(job['context_record'])
    assert record['dataset'] == job['dataset']
    assert record['scene'] == job['scene']
    assert record['family'] == request['family'] == job['family']
    assert record['resolver_request']['family'] == request['family']
    for key, value in request['geometry_evidence'].items():
        assert record['resolver_request']['geometry_evidence'][key] == value
    assert str(record['actor']) == str(request['adversary_vehicle_id'])
    assert record['start_step_10hz'] == request['start_step_10hz']
    assert record['base_request_sha256'] == sha(job['base_request'])
    assert record['manifest_sha256'] == sha(job['manifest'])
    assert record['future_invariance_passed']
    cutoff = record['observation_cutoff_s']
    assert math.isfinite(cutoff) and cutoff <= record['start_time_s'] + 1e-8
    assert record['max_input_time_s'] <= cutoff + 1e-8
    table = read(record['prior_table'])
    assert record['prior_table_sha256'] == sha(record['prior_table'])
    assert table['fit_only'] and table['evaluation_logs_excluded']
    prior = resolve(table['rows'], job['dataset'], record['resolver_request'],
                    record['resolver_context'])
    for key in ['factor', 'reason', 'support_level', 'events', 'scenes', 'logs']:
        assert prior[key] == record['prior'][key], (key, prior, record['prior'])
    assert math.isfinite(prior['factor']) and .75 <= prior['factor'] <= 1.35
    assert math.isclose(prior['factor'], job['prior_factor'], rel_tol=0, abs_tol=1e-12)
    assert prior['reason'] == job['prior_reason']
    context = record['resolver_context']
    metadata = dict(
        record=str(job['context_record']), record_sha256=sha(job['context_record']),
        observation_cutoff_s=cutoff, max_input_time_s=record['max_input_time_s'],
        update_policy='episode_start_context; live_pair_risk_every_replan',
        road_type=context.get('road', {}).get('road_type', 'UNKNOWN'),
        semantic_label=context.get('semantic_label', 'UNKNOWN'),
        traffic_state=context.get('traffic_state', 'UNKNOWN'),
        actor_role=prior.get('actor_role', 'UNKNOWN'),
        support_scope=prior.get('support_scope', 'none'),
        matched_cell_actor_role=prior.get('matched_cell_actor_role'),
        matched_source_roles=prior.get('matched_source_roles'),
        prior_table_sha256=record['prior_table_sha256'],
        future_invariance_passed=True, evaluation_logs_excluded=True,
    )
    return prior, metadata


def audit_updates(updates, prior, context):
    for update in updates:
        assert update['context_conditioning_v26'] == context
        assert update['prior_factor'] == prior['factor']
        assert update['prior_reason'] == prior['reason']
        assert update['prior_support_level'] == prior['support_level']
        expected = min(update['base_pressure'] * 16., max(0.,
            update['base_pressure'] * update['calibration_scale'] * prior['factor']
            * update['temporal_factor']))
        assert math.isclose(update['pressure'], expected, rel_tol=1e-12, abs_tol=1e-12)
        assert not update['recorded_future_used']
