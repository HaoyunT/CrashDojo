"""Freeze native lane surfaces for a method-independent contact quality screen."""
import json
from pathlib import Path
import sys

from common import ROOT, QUEUE, SOURCE, load, save, sha


def main():
    context = ROOT / 'launch/causal_guidance_v26_contextual_val150_20261030'
    pipeline = Path('/path/to/workspace/CrashDojo_pipeline/src')
    sys.path[:0] = [str(context / 'map_deps'), str(pipeline)]
    from native_map_source import NativeMap
    from shapely.ops import unary_union
    import shapely
    import native_map_source
    records = {}
    for job in load(SOURCE / 'protocol.json')['jobs']:
        if job['planner'] != 'idm':
            continue
        manifest = json.loads(Path(job['manifest']).read_text().splitlines()[0])
        key = job['dataset'] + '__' + manifest['city']
        records[key] = (job['dataset'], manifest['city'])
    destination = QUEUE / 'inputs/roads'
    destination.mkdir(parents=True, exist_ok=True)
    audit = dict(source_sha256=sha(native_map_source.__file__), maps={},
                 screen='common_native_lane_and_connector_union_body_coverage',
                 min_body_coverage=.98,
                 distinct_from_original_target_specific_corridor=True,
                 recompute_for_ours_before_qualified_contact_comparison=True)
    for key, (dataset, city) in sorted(records.items()):
        path = destination / (key + '.wkb')
        metadata = destination / (key + '.json')
        if path.exists() and metadata.exists():
            audit['maps'][key] = load(metadata)
            continue
        native = NativeMap(dataset, city)
        road = unary_union(list(native.poly.values()))
        if road.is_empty or not road.is_valid:
            raise ValueError('Invalid native lane surface ' + key)
        path.write_bytes(shapely.to_wkb(road))
        entry = dict(dataset=dataset, city=city, path=str(path), sha256=sha(path),
                     polygons=len(native.poly), source_sha256={p: sha(p) for p in native.map_sources})
        save(metadata, entry)
        audit['maps'][key] = entry
        print(key, 'prepared', flush=True)
    save(destination / 'audit.json', audit)


if __name__ == '__main__':
    main()
