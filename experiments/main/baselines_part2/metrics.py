"""Eight shared metrics with explicit native-selection and quality semantics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import shapely
from shapely.affinity import translate

from common import ROOT, QUEUE, load, save, sha

CONTRACT = dict(
    version='v6_remaining225_native_selection_metrics_v1', dt_s=.1,
    collision='10Hz oriented boxes, boundaries included, jointly available finite poses',
    selected_contact='selection active in execution block that produced frame; initial frame uses first selection',
    selected_background_contact='at least one active selected actor in a non-ego contacting pair',
    qualified_selected_contact='post-reset contacting selected actor passes strictly precontact executed-prefix dynamics and >=98% native lane/connector union body coverage',
    quality_limit='discrete engineering screen; not responsibility, avoidability, or continuous physical certification',
    original_ours_corridor_qualification_interchangeable=False,
    qualified_ours_recomputation_required=True,
    ade_fde='native trajectory logger, equal actor mean, includes replay actors',
    offroad='native all-agent raster center offroad rate, distinct from body coverage',
    aggregation='equal available-scene mean per method/planner/domain; missing is not zero')
FIELDS = ['ego_collision_10Hz', 'scene_collision_10Hz', 'target_contact_10Hz',
          'qualified_selected_contact_10Hz', 'target_background_contact_10Hz',
          'ADE_native_m', 'FDE_native_m', 'ORR_native']


def shared_geometry():
    path = ROOT / 'launch/causal_guidance_v26_contextual_full_eval_20261101/metrics.py'
    spec = importlib.util.spec_from_file_location('frozen_context_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.contacts, module.prefix_dynamics


def evaluate(output, job, steps):
    output = Path(output)
    files = list(output.rglob('data.hdf5'))
    if len(files) != 1:
        raise ValueError('Expected exactly one native rollout HDF5')
    path = files[0]
    execution = [json.loads(s) for s in (output / 'common_execution_steps.jsonl').read_text().splitlines()]
    if [r['step_index'] for r in execution] != list(range(0, steps, 5)):
        raise ValueError('Incomplete native execution blocks')
    names = execution[0]['names']
    if any(r['names'] != names for r in execution):
        raise ValueError('Actor population changed without a lifecycle mapping')
    ids = execution[0].get('track_ids', list(range(len(names))))
    with h5py.File(path) as handle:
        if len(handle) != 1:
            raise ValueError('Expected one scene')
        group = handle[next(iter(handle))]
        track = np.asarray(group['track_id'])
        if not (track == track[:, :1]).all():
            raise ValueError('Unstable actor identity in rollout log')
        mapping = {int(t): i for i, t in enumerate(track[:, 0])}
        order = [mapping[int(t)] for t in ids]
        position, yaw, extent = [np.asarray(group[k])[order] for k in ['centroid', 'yaw', 'extent']]
        if 'availability' not in group:
            raise ValueError('Missing explicit frame availability')
        available = np.asarray(group['availability'])[order].astype(bool)
    if position.shape != (len(names), steps, 2):
        raise ValueError('Frozen native actor/frame budget differs')
    yaw = yaw.reshape(position.shape[:2])
    available = available.reshape(position.shape[:2])
    available &= np.isfinite(position).all(-1) & np.isfinite(yaw) & np.isfinite(extent).all(-1)
    verified_actor_frames = 0
    for record in execution:
        expected_xy = np.asarray(record['expected_world_positions'], dtype=float)
        expected_yaw = np.asarray(record['expected_world_yaws'], dtype=float)
        for offset in range(5):
            frame = record['step_index'] + offset + 1
            if frame >= steps:
                continue
            valid = available[:, frame]
            if (not np.isfinite(expected_xy[valid, offset]).all()
                    or not np.isfinite(expected_yaw[valid, offset]).all()):
                raise ValueError('An executed available pose lacks a finite composed action')
            if not np.allclose(position[valid, frame], expected_xy[valid, offset], rtol=0, atol=1e-6):
                raise ValueError('Actual executed positions differ from composed actions')
            angle_error = (yaw[valid, frame] - expected_yaw[valid, offset] + np.pi) % (2*np.pi) - np.pi
            if not (np.abs(angle_error) <= 1e-6).all():
                raise ValueError('Actual executed headings differ from composed actions')
            verified_actor_frames += int(valid.sum())
    ego = names.index('ego')
    if not available[ego].all():
        raise ValueError('The planner ego is not available at every frame')
    selected = np.zeros_like(available)
    for frame in range(steps):
        selected[execution[max(0, frame - 1) // 5]['transport']['selected_non_ego_indices'], frame] = True
    selected[ego] = False
    contact_function, prefix_dynamics = shared_geometry()
    pairs, hit = contact_function(position, yaw, extent, available)
    ego_pairs = (pairs == ego).any(-1)
    partners = np.where(pairs[ego_pairs, 0] == ego, pairs[ego_pairs, 1], pairs[ego_pairs, 0])
    selected_hits = hit[ego_pairs] & selected[partners]
    background = hit[~ego_pairs] & (selected[pairs[~ego_pairs, 0]] | selected[pairs[~ego_pairs, 1]])
    source = job['source_job']
    manifest = json.loads(Path(source['manifest']).read_text().splitlines()[0])
    request = load(source['base_request'])
    road_path = QUEUE / 'inputs/roads' / (source['dataset'] + '__' + manifest['city'] + '.wkb')
    road = shapely.from_wkb(road_path.read_bytes())
    if source['dataset'] == 'nuplan':
        context = ROOT / 'launch/causal_guidance_v26_contextual_full_eval_20261101'
        sys.path.insert(0, str(context))
        from nuplan_adapter import origin
        offset = origin(manifest['city'])
        road = translate(road, xoff=-offset[0], yoff=-offset[1])
    qualified = False
    quality_diagnostics = []
    for row, actor in enumerate(partners):
        contacts = np.flatnonzero(selected_hits[row])
        if not len(contacts):
            continue
        first = int(contacts[0])
        actor = int(actor)
        before = available[actor, :first + 1]
        angles = yaw[actor, :first + 1]
        forward = np.c_[np.cos(angles), np.sin(angles)] * extent[actor, :first + 1, 0, None] / 2
        left = np.c_[-np.sin(angles), np.cos(angles)] * extent[actor, :first + 1, 1, None] / 2
        corners = position[actor, :first + 1, None] + np.stack(
            [forward + left, forward - left, -forward - left, -forward + left], 1)
        boxes = shapely.polygons(corners[before])
        coverage = shapely.area(shapely.intersection(boxes, road)) / shapely.area(boxes)
        road_valid = bool(len(coverage) and np.isfinite(coverage).all() and (coverage >= .98 - 1e-7).all())
        diagnostics = []
        expected_prefixes = 0
        for block_index, record in enumerate(execution):
            if block_index > max(0, first - 1) // 5 or first == 0:
                break
            if actor not in record['transport']['selected_non_ego_indices']:
                continue
            expected_prefixes += 1
            prefixes = record.get('selected_prefixes', record['transport'].get('selected_prefixes', []))
            prefix = next((p for p in prefixes if p['actor_index'] == actor), None)
            if prefix is None:
                raise ValueError('Selected contact lacks executed-prefix dynamics evidence')
            diagnostics.append(prefix_dynamics(prefix, request['engineering_limits'],
                               min(5, first - record['step_index'])))
        dynamics_valid = bool(expected_prefixes and len(diagnostics) == expected_prefixes
                              and all(d['valid'] for d in diagnostics))
        passed = dynamics_valid and road_valid
        qualified |= passed
        quality_diagnostics.append(dict(actor=names[actor], first_contact_frame=first,
            dynamics_valid=dynamics_valid, road_valid=road_valid, qualified=passed,
            prefixes=expected_prefixes, min_body_lane_surface_coverage=float(coverage.min()) if len(coverage) else None))
    stats = load(path.with_name('stats.json'))
    def mean(key):
        value = np.asarray(stats.get(key, []), float)
        return float(np.nanmean(value)) if np.isfinite(value).any() else None
    result = dict(contract=CONTRACT, method=job['method'], planner=job['planner'],
        dataset=source['dataset'], scene=source['scene'], frames=steps, actors=len(names),
        hdf5=str(path), hdf5_sha256=sha(path), road_sha256=sha(road_path),
        execution_log_sha256=sha(output / 'common_execution_steps.jsonl'),
        independent_executed_actor_frames=verified_actor_frames,
        ego_collision_10Hz=bool(hit[ego_pairs].any()), scene_collision_10Hz=bool(hit.any()),
        target_contact_10Hz=bool(selected_hits.any()),
        qualified_selected_contact_10Hz=bool(qualified),
        target_background_contact_10Hz=bool(background.any()),
        ADE_native_m=mean('ade'), FDE_native_m=mean('fde'), ORR_native=mean('all_off_road_rate_rate'),
        selected_contact_quality_diagnostics=quality_diagnostics,
        native_selected_counts=[len(r['transport']['selected_non_ego_indices']) for r in execution])
    for field in ['ADE_native_m', 'FDE_native_m', 'ORR_native']:
        if result[field] is None:
            raise ValueError('Missing required native metric: ' + field)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    result = evaluate(args.output, load(args.job_file), args.steps)
    save(args.output / 'metrics.json', result)
    print(json.dumps({key: result[key] for key in FIELDS}), flush=True)


if __name__ == '__main__':
    main()
