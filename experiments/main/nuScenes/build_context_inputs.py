"""Freeze causal contexts for an explicit evaluation cohort, on CPU."""
import argparse
import ast
import collections
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
HERE = Path(__file__).resolve().parent
DEST = None
V22 = ROOT / 'launch/semantic_mining_v22_20261028'
V21 = ROOT / 'launch/semantic_mining_v21_20261028'
V20 = ROOT / 'launch/road_scene_strength_v20_20261026'
MINING = Path('/path/to/storage/data')
MODEL = MINING / 'outputs/q20_semantic_causal_v2_20261028'
sys.path[:0] = [str(V22), str(HERE), str(V20), str(HERE / 'map_deps')]

import numpy as np
import torch
from causal_common import SOURCES, clean, resample
from native_features import FEATURES, RoadMap, alter_future, compute
from raw_sources import NuScenesSource, load_nuplan_log
from semantic_model import Predictor
from context_features import road_context
import hierarchical_prior as hp


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def event_snapshot():
    # Keep the event geometry rule, with the later v23 variable-length speed fix.
    source = V21 / 'mine_semantic_context.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'snapshot')
    body = ast.get_source_segment(source.read_text(), node)
    own = "float(np.linalg.norm(primary['xy'][-1]-primary['xy'][-6])/.5)"
    peer = "float(np.linalg.norm(p['xy'][-1]-p['xy'][-6])/.5)"
    assert body.count(own) == body.count(peer) == 1
    body = body.replace(own, 'snapshot_speed(primary)').replace(peer, 'snapshot_speed(p)')
    def snapshot_speed(pre):
        n = min(5, len(pre['xy']) - 1)
        if n < 1:
            raise ValueError('No observed displacement for causal speed')
        return float(np.linalg.norm(pre['xy'][-1] - pre['xy'][-1-n]) / (n * .1))
    namespace = dict(np=np, resample=resample, road_context=road_context,
                     snapshot_speed=snapshot_speed)
    exec(compile(body, str(source), 'exec'), namespace)
    return namespace['snapshot'], source


def build_prior(events, excluded):
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    exclusions = collections.Counter()
    fit_logs = collections.defaultdict(set)
    for r in events:
        if r['split'] != 'fit':
            continue
        ds = r['dataset']
        counts[ds + '|fit_before'] += 1
        if (ds, r['log_id']) in excluded:
            counts[ds + '|evaluation_log_excluded'] += 1
            continue
        counts[ds + '|fit_after_log_exclusion'] += 1
        intensity = r['metrics'].get('intensity')
        if intensity is None or not np.isfinite(intensity) or intensity <= 0:
            exclusions['no_positive_finite_intensity'] += 1
            continue
        if r['applicability'] != 'main':
            exclusions['semantic_unknown_excluded_or_optional'] += 1
            continue
        if not r['context'].get('eligible') or not r['context'].get('complete_neighbor_history'):
            exclusions['incomplete_causal_context'] += 1
            continue
        if r['road_type'] == 'UNKNOWN':
            exclusions['unknown_native_road'] += 1
            continue
        counts[ds + '|eligible_fit_events'] += 1
        fit_logs[ds].add(r['log_id'])
        role = r['context']['actor_role_relative_to_recorded_ego']
        for level, semantic, road, traffic in [
            ('family_role', 'ALL', 'ALL', 'ALL'),
            ('semantic', r['semantic_label'], 'ALL', 'ALL'),
            ('road', 'ALL', r['road_type'], 'ALL'),
            ('semantic_road', r['semantic_label'], r['road_type'], 'ALL'),
            ('joint', r['semantic_label'], r['road_type'], r['traffic_state'])]:
            groups[(ds, r['family'], role, level, semantic, road, traffic)].append(r)
            if level != 'joint':
                groups[(ds, r['family'], 'ALL', level, semantic, road, traffic)].append(r)
    cells = []
    for key, rows in sorted(groups.items()):
        ds, family, role, level, semantic, road, traffic = key
        values = np.array([r['metrics']['intensity'] for r in rows])
        logs = collections.defaultdict(list)
        for r in rows:
            logs[r['log_id']].append(np.log(r['metrics']['intensity']))
        assert not {(ds, log) for log in logs} & excluded
        scenes = len({r['scene_id'] for r in rows})
        cells.append(dict(split='fit', dataset=ds, family=family, actor_role=role,
            level=level, semantic_label=semantic, road_type=road, traffic_state=traffic,
            events=len(rows), scenes=scenes, logs=len(logs),
            supported=len(rows) >= 30 and scenes >= 10 and len(logs) >= 3,
            intensity_p05=float(np.quantile(values, .05)),
            intensity_p50=float(np.median(values)),
            intensity_p95=float(np.quantile(values, .95)),
            log_equal_mean_log_intensity=float(np.mean([np.median(a) for a in logs.values()])),
            source_logs=sorted(logs),
            source_roles=dict(collections.Counter(
                r['context']['actor_role_relative_to_recorded_ego'] for r in rows)),
            role_pooling='all_observed_roles' if role == 'ALL' else 'exact_role'))
    table = hp.build_table(cells)
    by_scope = collections.defaultdict(lambda: dict(cells=0, supported=0))
    for cell in cells:
        key = '|'.join((cell['dataset'], cell['family'], cell['role_pooling'], cell['level']))
        by_scope[key]['cells'] += 1
        by_scope[key]['supported'] += int(hp.supported(cell, strict=cell['level'] in ['joint', 'semantic_road']))
    table.update(fit_only=True, evaluation_logs_excluded=True,
        excluded_evaluation_logs=[dict(dataset=ds, log=log) for ds, log in sorted(excluded)],
        source_events=str(MODEL / 'events_v22.jsonl'),
        source_events_sha256=sha(MODEL / 'events_v22.jsonl'),
        grouping_policy_source=str(V22 / 'summarize_semantic_mining.py'),
        grouping_policy_source_sha256=sha(V22 / 'summarize_semantic_mining.py'),
        fit_event_counts=dict(counts), fitting_exclusions=dict(exclusions),
        source_fit_logs={ds: sorted(logs) for ds, logs in fit_logs.items()},
        statistical_support_thresholds_unchanged=True,
        role_pooled_levels=['semantic_road', 'semantic', 'road', 'family_role'],
        role_pooled_joint_cells=False,
        role_pooling_uses_unique_logs_and_log_equal_medians=True)
    table['support_summary_by_role_scope'] = dict(sorted(by_scope.items()))
    return table


def classifier_overlap(excluded):
    training = read(MODEL / 'semantic_training.json')
    result = {}
    for partition, details in training['partitions'].items():
        by_domain = {}
        for ds in ['nuplan', 'nuscenes']:
            evaluation = {log for domain, log in excluded if domain == ds}
            present = set(details['logs'])
            present |= {x[len(ds) + 1:] for x in present if x.startswith(ds + '|')}
            overlap = sorted(evaluation & present)
            by_domain[ds] = dict(count=len(overlap), logs=overlap)
        result[partition] = by_domain
    return dict(partitions=result, classifier_frozen_not_retrained=True,
        strict_log_heldout_classifier_claim=False,
        scope='fixed_mixed_development_cohort_with_disclosed_semantic_model_overlap',
        training_path=str(MODEL / 'semantic_training.json'),
        training_sha256=sha(MODEL / 'semantic_training.json'))


def raw_prefix_digest(tracks, cutoff):
    digest = hashlib.sha256()
    for actor, track in sorted(tracks.items()):
        keep = track['time_s'] <= cutoff + 1e-8
        if not keep.any():
            continue
        digest.update(actor.encode())
        for key in ['time_s', 'xy', 'yaw']:
            value = np.asarray(track[key])[keep]
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(np.ascontiguousarray(track['size']).tobytes())
        digest.update(str(track['max_gap_s']).encode())
    return digest.hexdigest()


def derive(tracks, actor, cutoff, road, predictor, snapshot, request, table, dataset):
    errors = []
    try:
        context = snapshot(tracks, actor, cutoff, road)
    except (IndexError, KeyError, ValueError) as error:
        errors.append(type(error).__name__ + ': ' + str(error))
        context = dict(eligible=False, complete_neighbor_history=False,
                       reason='event_snapshot_unavailable', road={'road_type': 'UNKNOWN'})
        primary = resample(tracks[actor], cutoff, 1.) if actor in tracks else None
        if primary is not None:
            try:
                context['road'] = road_context(dict(xy=primary['xy'], yaw=primary['yaw'],
                    ts=np.arange(len(primary['xy']))), road)
                context['max_source_time_s'] = primary['last_observation_s']
            except (IndexError, KeyError, ValueError):
                pass
    if not context.get('eligible'):
        context.setdefault('complete_neighbor_history', False)
    features, metadata = compute(tracks['ego'], tracks, cutoff, road)
    if features is None:
        semantic = dict(semantic_label='UNKNOWN', raw_prediction=None, confidence=None,
            accepted=False, ground_truth=False,
            acceptance_reason='insufficient_observed_ego_history')
    else:
        semantic = predictor.describe([[features[k] for k in FEATURES]])[0]
    context.update(semantic_label=semantic['semantic_label'], semantic=semantic,
                   semantic_input_metadata=metadata,
                   observation_cutoff_s=cutoff, extraction_errors=errors)
    role = context.get('actor_role_relative_to_recorded_ego', 'unknown')
    resolver_request = dict(family=request['family'],
                            geometry_evidence=copy.deepcopy(request['geometry_evidence']))
    resolver_request['geometry_evidence']['actor_role'] = role
    resolver_context = copy.deepcopy(context)
    masks = []
    if not context.get('eligible') or not context.get('complete_neighbor_history'):
        for key in ['traffic_state', 'semantic_label']:
            resolver_context[key] = 'UNKNOWN'
            masks.append(key)
    if not context.get('eligible'):
        resolver_request['geometry_evidence']['actor_role'] = 'UNKNOWN'
    resolver_context['untrusted_axes_masked'] = masks
    prior = hp.resolve(table['rows'], dataset, resolver_request, resolver_context)
    times = [context.get('max_source_time_s', 0.)]
    if metadata:
        times.append(metadata['max_input_time_s'])
    maximum = max(times)
    assert np.isfinite(maximum) and maximum <= cutoff + 1e-8
    return clean(dict(context=context, resolver_request=resolver_request,
        resolver_context=resolver_context, prior=prior, max_input_time_s=maximum,
        causal_raw_prefix_sha256=raw_prefix_digest(tracks, cutoff),
        semantic_features_sha256=hashlib.sha256(canonical(features).encode()).hexdigest()))


def resolution_trace(table, dataset, request, context, prior):
    role = hp.request_role(request)
    result = []
    searches = [('exact_role', role, level) for level in hp.LEVELS[:-2]]
    searches += [('pooled_roles', 'ALL', level) for level in ['semantic_road', 'semantic', 'road']]
    searches += [('exact_role', role, 'family_role')]
    for scope, match_role, level in searches:
        matches = [r for r in table['rows'] if r['level'] == level and hp._matches(r,
            dataset, request['family'], match_role, context.get('semantic_label', 'UNKNOWN'),
            context.get('road', {}).get('road_type', 'UNKNOWN'), context.get('traffic_state', 'UNKNOWN'))]
        eligible = [r for r in matches if hp.supported(r, strict=level in ['joint', 'semantic_road'])]
        label = 'role_pooled_' + level if scope == 'pooled_roles' else level
        result.append(dict(level=label, scope=scope, matching_actor_role=match_role,
            matching_cells=len(matches), supported_matching_cells=len(eligible),
            selected=prior['support_level'] == label,
            supported_cells=[{k: r.get(k) for k in ['actor_role', 'source_roles', 'events', 'scenes', 'logs']}
                             for r in eligible]))
    parent = hp._family_row(table['rows'], dataset, request['family']) if prior['support_level'].startswith('role_pooled_') else hp._parent(table['rows'], dataset, request['family'], role)
    return dict(requested_axes=dict(family=request['family'], actor_role=role,
        road_type=context.get('road', {}).get('road_type', 'UNKNOWN'),
        semantic_label=context.get('semantic_label', 'UNKNOWN'),
        traffic_state=context.get('traffic_state', 'UNKNOWN')), levels=result,
        selected_support_level=prior['support_level'], fallback_reason=prior.get('fallback_reason'),
        matched_source_roles=prior.get('matched_source_roles'),
        support_scope=prior.get('support_scope'),
        parent_scope='pooled_roles' if parent and parent.get('actor_role') == 'ALL' else 'exact_role',
        parent={k: parent.get(k) for k in ['level', 'actor_role', 'source_roles', 'events', 'scenes', 'logs']} if parent else None)


def main():
    global DEST
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-scenes', type=int, required=True)
    args = parser.parse_args()
    DEST = args.out
    torch.set_num_threads(1)
    DEST.mkdir(parents=True, exist_ok=False)
    started = time.time()
    fixed_path = args.selection_manifest
    fixed = [json.loads(s) for s in fixed_path.read_text().splitlines()]
    assert len(fixed) == args.expected_scenes
    assert {r['eval_index'] for r in fixed} == set(range(args.expected_scenes))
    records = []
    for row in sorted(fixed, key=lambda x: x['eval_index']):
        request = read(row['request'])
        manifest_rows = [json.loads(s) for s in Path(row['manifest']).read_text().splitlines()]
        manifest = manifest_rows[request.get('selection_manifest_row', 0)]
        assert str(manifest['adversary_vehicle_id']) == str(request['adversary_vehicle_id']) == str(row['selected_actor'])
        assert manifest['start_step_10hz'] == request['start_step_10hz']
        assert manifest['scene_id'] == request['scene_id']
        cutoff = float(manifest.get('start_time_s', request['start_step_10hz'] * .1))
        assert abs(cutoff - request['start_step_10hz'] * .1) < 1e-6
        records.append(dict(row=row, request=request, manifest=manifest, cutoff=cutoff))
    excluded = {(r['row']['dataset'], r['manifest']['log_id']) for r in records}
    events = [json.loads(s) for s in (MODEL / 'events_v22.jsonl').open()]
    table = build_prior(events, excluded)
    table_path = DEST / 'observational_prior_eval_logs_excluded.json'
    save(table_path, table)
    overlap = classifier_overlap(excluded)
    save(DEST / 'preparation_support.json', dict(support_summary=table['support_summary'],
        fit_event_counts=table['fit_event_counts'], classifier_log_overlap=overlap))
    print(json.dumps(dict(stage='prior_ready', cells=len(table['rows']),
        supported=sum(v['supported'] for v in table['support_summary'].values()),
        fit_event_counts=table['fit_event_counts']), ensure_ascii=False), flush=True)
    metadata = {ds: read(path / 'scene_meta.json') for ds, path in SOURCES.items()}
    snapshot, snapshot_source = event_snapshot()
    predictor = Predictor(device='cpu')
    groups = collections.defaultdict(list)
    for record in records:
        m = record['manifest']
        groups[(record['row']['dataset'], m['city'], m['log_id'])].append(record)
    # NuScenesSource uses this value as the raw trainval-table loader mode;
    # val membership must be proved by the frozen selection manifest.
    nusc_records = {r['manifest']['scene_id']: dict(metadata['nuscenes'][r['manifest']['scene_id']], split='train')
                    for r in records if r['row']['dataset'] == 'nuscenes'}
    nusc = NuScenesSource(nusc_records) if nusc_records else None
    maps = {}
    output = []
    for (dataset, city, log), cohort in sorted(groups.items()):
        key = (dataset, city)
        if key not in maps:
            maps[key] = RoadMap(dataset, city)
        road = maps[key]
        wanted = {r['manifest']['scene_id']: dict(metadata[dataset].get(r['manifest']['scene_id'], {}),
                    scene_token=r['manifest']['scene_token']) for r in cohort}
        source = load_nuplan_log(log, wanted) if dataset == 'nuplan' else None
        for record in cohort:
            row, request, manifest, cutoff = (record[k] for k in ['row', 'request', 'manifest', 'cutoff'])
            scene = manifest['scene_id']
            tracks = source[scene] if source is not None else nusc.scene(scene)
            actor = str(manifest['adversary_vehicle_id'])
            variants = [tracks, alter_future(tracks, cutoff, truncate=True), alter_future(tracks, cutoff)]
            results = [derive(t, actor, cutoff, road, predictor, snapshot, request, table, dataset) for t in variants]
            assert canonical(results[0]) == canonical(results[1]) == canonical(results[2]), (row['eval_index'], 'future input invariance failed')
            result = results[0]
            item = dict(eval_index=row['eval_index'], dataset=dataset,
                scene_id=scene, fixed_manifest_scene_id=row['scene_id'], scene=scene,
                actor=actor, family=request['family'], city=city, log=log,
                start_step_10hz=request['start_step_10hz'], start_time_s=cutoff,
                cutoff=cutoff, cutoff_s=cutoff, observation_cutoff_s=cutoff,
                request_path=row['request'], manifest_path=row['manifest'],
                base_request_sha256=sha(row['request']), manifest_sha256=sha(row['manifest']),
                future_invariance_passed=True, future_invariance_modes=['full', 'future_truncated', 'future_polluted_and_future_actor_injected'],
                prior_table=str(table_path), prior_table_sha256=sha(table_path),
                source_refs=dict(raw_loader=str(V22 / 'raw_sources.py'),
                    metadata=str(SOURCES[dataset] / 'scene_meta.json'),
                    metadata_sha256=sha(SOURCES[dataset] / 'scene_meta.json'),
                    event_context_function=str(snapshot_source) + ':snapshot',
                    snapshot_speed_patch='v23 n=min(5,len(history)-1), observed n*0.1s denominator',
                    semantic_model=str(MODEL / 'semantic_classifier.pt'),
                    semantic_model_sha256=sha(MODEL / 'semantic_classifier.pt')),
                **result)
            item['resolution_trace'] = resolution_trace(table, dataset,
                item['resolver_request'], item['resolver_context'], item['prior'])
            output.append(item)
            with (DEST / 'contexts.partial.jsonl').open('a') as f:
                f.write(canonical(item) + '\n')
            print(json.dumps(dict(stage='context_ready', done=len(output), index=row['eval_index'],
                dataset=dataset, factor=item['prior']['factor'], reason=item['prior']['reason'],
                elapsed_s=round(time.time() - started, 1)), ensure_ascii=False), flush=True)
    output.sort(key=lambda x: x['eval_index'])
    assert len(output) == args.expected_scenes
    assert {r['eval_index'] for r in output} == set(range(args.expected_scenes))
    final = DEST / 'contexts.jsonl'
    final.write_text(''.join(canonical(item) + '\n' for item in output))
    audit = dict(completed=True, scenes=args.expected_scenes, cpu_only=True,
        original_selection_and_actions_unchanged=True,
        decision_counts=dict(collections.Counter(r['decision'] for r in fixed)),
        selection_source_counts=dict(collections.Counter(r['source'] for r in fixed)),
        event_role_geometry_unchanged=True,
        short_history_speed_fix='v23 variable-length causal speed; no fixed [-6] access',
        fixed_selection_manifest=str(fixed_path), fixed_selection_manifest_sha256=sha(fixed_path),
        future_invariance_passed=True, max_input_time_lte_cutoff=True,
        contexts_sha256=sha(final), prior_table=str(table_path), prior_table_sha256=sha(table_path),
        prior_fit_only=True, evaluation_logs_excluded=True,
        fit_event_counts=table['fit_event_counts'], support_summary=table['support_summary'],
        classifier_log_overlap=overlap,
        factor_counts=dict(collections.Counter(str(x['prior']['factor']) for x in output)),
        prior_reason_counts=dict(collections.Counter(x['prior']['reason'] for x in output)),
        factor_not_one=sum(abs(x['prior']['factor'] - 1.) > 1e-12 for x in output),
        dataset_counts=dict(collections.Counter(x['dataset'] for x in output)),
        context_complete_counts=dict(collections.Counter(str(bool(x['context'].get('complete_neighbor_history'))) for x in output)),
        source_hashes={str(p): sha(p) for p in [Path(__file__), V22 / 'raw_sources.py',
            V22 / 'native_features.py', V22 / 'semantic_model.py', V22 / 'causal_common.py',
            V22 / 'summarize_semantic_mining.py', V20 / 'context_features.py',
            snapshot_source, HERE / 'hierarchical_prior.py', MODEL / 'events_v22.jsonl',
            MODEL / 'semantic_classifier.pt', MODEL / 'semantic_training.json']},
        elapsed_s=time.time() - started)
    save(DEST / 'audit.json', audit)
    print(json.dumps(dict(stage='completed', contexts=str(final), scenes=args.expected_scenes,
        factor_not_one=audit['factor_not_one'], prior_reason_counts=audit['prior_reason_counts'],
        elapsed_s=audit['elapsed_s']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
