"""Runtime integrity only; a completed run may still fail every quality metric."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np

parent=Path(sys.argv[1]);request=json.loads((parent/'request.json').read_text())
paths=list(parent.rglob('data.hdf5'));assert len(paths)==1,paths
p=paths[0].parent
binding=json.loads((p/'selection_binding.json').read_text())
validation=json.loads((p/'selected_planner_validation.json').read_text())
assert validation['weights_and_buffers_unchanged'] and validation['optimizer_steps']==0
assert validation['formal_evaluation']
assert binding['scene_id']==request['scene_id'] and binding['start_step_10hz']==31
assert binding['adversary_vehicle_id']==request['adversary_vehicle_id']
steps=[json.loads(s) for s in (p/'selected_planner_steps.jsonl').read_text().splitlines()]
assert [x['step_index'] for x in steps]==list(range(0,100,5))
assert all(x['non_target_non_ego_replayed'] for x in steps)
stats=json.loads((p/'stats.json').read_text())
assert len(stats['scene_index'])==1 and stats['scene_index'][0]==request['scene_id']
assert all(np.isfinite(stats[k]).all() for k in ['ade','fde'])
with h5py.File(paths[0],'r') as f:
    assert len(f)==1
    g=f[next(iter(f))];positions=np.asarray(g['centroid']);yaw=np.asarray(g['yaw'])
    assert positions.shape[1:]==(100,2)
    ego=binding['runtime_ego_index'];assert np.isfinite(positions[ego]).all() and np.isfinite(yaw[ego]).all()
    if request['family']=='NONE':
        assert binding['runtime_target_index'] is None and validation['no_action_calls']==20 and validation['guide_calls']==0
    else:
        a=binding['runtime_target_index'];assert a!=ego and np.isfinite(positions[a]).all() and np.isfinite(yaw[a]).all()
        assert validation['guide_calls']>0
out=dict(passed=True,scene=request['scene_id'],family=request['family'],frames=100,planner_calls=20,
         model_unchanged=True,training=False,formal=True,quality_acceptance_not_implied=True)
(parent/'runtime_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
