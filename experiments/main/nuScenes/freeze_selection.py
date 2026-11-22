"""Freeze v6 requests for the nuPlan mini val/test and nuScenes val sweep."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


ARTIFACTS = Path('/path/to/workspace/CrashDojo_pipeline/artifacts')
INDEX = ARTIFACTS / 'selection_v6_seven_criteria_20261030/execution_index.json'
INVENTORY = ARTIFACTS / 'scene_execution_v2_fm/scene_inventory.json'
ALLOWED = {
    ('nuplan', 'val', 'calibration'): 197,
    ('nuplan', 'test', 'evaluation'): 224,
    ('nuscenes', 'val', 'evaluation'): 150,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    index = json.loads(INDEX.read_text())
    inventory = json.loads(INVENTORY.read_text())
    grouped = collections.defaultdict(list)
    for scene_id, item in inventory['scenes'].items():
        key = (item['dataset'], item['source_split'], item['isolated_role'])
        if key in ALLOWED:
            grouped[key].append((scene_id, item))
    for key, expected in ALLOWED.items():
        assert len(grouped[key]) == expected, (key, len(grouped[key]), expected)

    rows = []
    for key in ALLOWED:
        for scene_id, item in sorted(grouped[key]):
            selected = index['scenes'][scene_id]
            packet = index['packets'][selected['default_packet_id']]
            request_path = Path(packet['request'])
            manifest_path = Path(packet['manifest'])
            config_path = Path(packet['config'])
            request = json.loads(request_path.read_text())
            manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            manifest = manifest_rows[int(request.get('selection_manifest_row', 0))]
            assert request['scene_id'] == manifest['scene_id'] == scene_id
            assert str(request['adversary_vehicle_id']) == str(manifest['adversary_vehicle_id'])
            assert request['start_step_10hz'] == manifest['start_step_10hz']
            assert request['family'] == packet['family'] == selected['family']
            assert selected['scene_name'] == item['scene_name']
            target_steps = 100 if item['dataset'] == 'nuscenes' else 70
            source = 'v6_search' if '/selection_shards/' in str(request_path) else 'existing_packet'
            rows.append(dict(
                eval_index=len(rows), dataset=item['dataset'], source_split=item['source_split'],
                isolated_role=item['isolated_role'], scene_id=scene_id,
                scene_name=item['scene_name'], city=manifest['city'],
                log_id=manifest['log_id'], family=request['family'],
                selected_actor=str(request['adversary_vehicle_id']),
                packet_id=packet['packet_id'], packet_steps=packet['steps'],
                target_steps=target_steps, packet_duration_shortfall=packet['steps'] < target_steps,
                decision=selected.get('selection_v6'),
                seven_criteria_overall=selected.get('seven_criteria_overall'),
                source=source, selection_mode=packet.get('selection_mode'),
                closed_loop_validated=False, recorded_ego_future_used=True,
                request=str(request_path), request_sha256=sha(request_path),
                manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
                config=str(config_path), config_sha256=sha(config_path),
            ))
    assert len(rows) == sum(ALLOWED.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                                   for row in rows))
    summary = dict(scenes=len(rows), cohorts={
        '|'.join(key): len(grouped[key]) for key in ALLOWED},
        selection_decisions=dict(collections.Counter(row['decision'] for row in rows)),
        packet_duration_shortfall=sum(row['packet_duration_shortfall'] for row in rows),
        index=str(INDEX), index_sha256=sha(INDEX), inventory=str(INVENTORY),
        inventory_sha256=sha(INVENTORY), output=str(args.output), output_sha256=sha(args.output),
    )
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
