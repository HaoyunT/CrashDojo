"""Verify identity and actual history; never fabricate missing recorded futures."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,'/path/to/storage/external_scripts')
from q16_map_events_v2_20261020 import load_tracks
REPO=Path('/path/to/workspace/Friday/safesim_planner_20261012_01/code/CCDiff_M2c_planner')
INPUT=REPO/'benchmark50_inputs_20261021_v3'
CACHE=Path('/root/path/to/workspace/home/.unified_data_cache/nusc_trainval')
TRACKS=Path('/path/to/storage/data/outputs/q11_nuscenes_val_full_20261017/cache')
records=[]
for line in (INPUT/'manifest.jsonl').read_text().splitlines():
    r=json.loads(line);scene=r['scene_id'];a=load_tracks(TRACKS/(scene+'.npz'))
    df=pd.read_feather(CACHE/scene/'agent_data_dt0.10.feather').set_index(['agent_id','scene_ts'])
    for aid in ['ego']+([] if r['no_action'] else [r['adversary_vehicle_id']]):
        ts=list(range(1,32));history=np.array([a[aid]['xy'][a[aid]['lookup'][t]] for t in ts])
        runtime=df.loc[[(aid,t) for t in ts],['x','y']].to_numpy()
        error=float(np.linalg.norm(history-runtime,axis=-1).max());assert error<.001,(scene,aid,error)
        available=[t in a[aid]['lookup'] for t in range(31,131)]
        runtime_available=[(aid,t) in df.index for t in range(31,131)]
        assert available==runtime_available,(scene,aid)
        if aid=='ego':assert all(available)
        else:
            assert sum(available)==r['reference_availability']['recorded_reference_frames']
            assert available[-1]==r['reference_availability']['recorded_final_frame_available']
        records.append(dict(scene=scene,actor=aid,history_frames=31,history_position_error_max_m=error,
                            recorded_reference_frames=sum(available),recorded_final_frame_available=available[-1]))
report=dict(records=records,scenes=50,actor_records=len(records),
            max_history_position_error_m=max(x['history_position_error_max_m'] for x in records),
            all_initial_history_identity_and_ego_future_pass=True,missing_reference_imputed=False)
(INPUT/'runtime_cache_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'}))
