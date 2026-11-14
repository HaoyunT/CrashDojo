"""Record SafeSim preset coverage before the independent 225-scene baseline run."""
import json
from pathlib import Path
import runpy

from common import HERE, QUEUE, ROOT, load, save, sha


def main():
    output = HERE / 'safesim_adapter/support225.json'
    if output.exists():
        raise FileExistsError(output)
    source = load(ROOT / 'eval/causal_guidance_v26_contextual_full_eval_20261101/protocol.json')
    jobs = [job for job in source['jobs'] if job['planner'] == 'idm']
    assert len(jobs) == 225
    preset_source = ROOT / 'code/safe-sim_legacy100/tbsim/configs/selected_scene_config_nuplan_mini_val.py'
    presets = runpy.run_path(str(preset_source)).get('PREDEFINED_SCENE_ALL_INIT', {})
    rows = []
    for job in jobs:
        manifest = json.loads(Path(job['manifest']).read_text().splitlines()[0])
        available = job['scene'] in presets
        rows.append(dict(eval_index=int(job['block'].rsplit('_', 1)[-1]), dataset='nuplan',
            scene_id=job['scene'], source_split=job['source_split'],
            start_step_10hz=manifest['start_step_10hz'], supported=True,
            preset=presets.get(job['scene']), preset_source=str(preset_source),
            preset_source_sha256=sha(preset_source), official_preset_available=available,
            planned_selection_mode='official_preset' if available else 'user_authorized_v6_fallback',
            selection_adaptation_authorized=not available,
            fallback_reason=None if available else 'No published native SafeSim predefined assignment'))
    assert len({row['scene_id'] for row in rows}) == 225
    save(output, dict(method='safesim', total=225, rows=rows,
        preset_available=sum(row['official_preset_available'] for row in rows),
        planned_official_preset=sum(row['official_preset_available'] for row in rows),
        planned_user_authorized_v6_fallback=sum(not row['official_preset_available'] for row in rows),
        selection_adaptation_authorized=True,
        note='Report native-preset and v6-target selection-adapted results separately.'))
    print(json.dumps(dict(output=str(output), sha256=sha(output),
        official=sum(row['official_preset_available'] for row in rows), adapted=sum(
            not row['official_preset_available'] for row in rows))))


if __name__ == '__main__':
    main()
