"""A complete 100-step run must retain A beyond the last recorded future frame."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
ROOT=Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO=ROOT/'code/CCDiff_M2c_planner'
INPUT=REPO/'benchmark50_inputs_20261021_v3'
p=ROOT/'eval/benchmark50_future_reference_probe_20261021/scene-0109_idm_s1_batched'
assert json.loads((p/'status.json').read_text())['returncode']==0
r=json.loads((p/'request.json').read_text());f=next(p.rglob('data.hdf5'));d=f.parent
b=json.loads((d/'selection_binding.json').read_text());v=json.loads((d/'selected_planner_validation.json').read_text())
assert v['weights_and_buffers_unchanged'] and v['optimizer_steps']==0 and not v['formal_evaluation']
assert b['adversary_vehicle_id']==r['adversary_vehicle_id'] and b['start_step_10hz']==31
steps=[json.loads(s) for s in (d/'selected_planner_steps.jsonl').read_text().splitlines()]
assert [s['step_index'] for s in steps]==list(range(0,100,5))
assert all(s['non_target_non_ego_replayed'] for s in steps)
with h5py.File(f) as h:
    assert len(h)==1;g=h[next(iter(h))];a=b['runtime_target_index'];xy=np.asarray(g['centroid'][a]);yaw=np.asarray(g['yaw'][a])
assert xy.shape==(100,2) and np.isfinite(xy).all() and np.isfinite(yaw).all()
assert r['reference_availability']==dict(recorded_reference_frames=50,recorded_final_frame_available=False)
assert v['guide_calls']>0
stats=json.loads((d/'stats.json').read_text());assert all(np.isfinite(stats[k]).all() for k in ['ade','fde'])
report=dict(passed=True,scene=r['scene_id'],actor=r['adversary_vehicle_id'],simulation_frames=100,
            original_reference_frames=50,simulated_frames_after_reference_ends=50,planner_calls=20,
            guide_calls=v['guide_calls'],model_unchanged=True,training=False,formal=False,
            quality_acceptance_not_implied=True,evidence_hdf5=str(f),hdf5_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
(INPUT/'partial_reference_runtime_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
