"""CPU lane queries against the same centered cache used by the scene."""
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import torch
from trajdata.maps.map_api import MapAPI
from tbsim.utils.trajdata_utils import get_closest_lane_point_for_one_agent


def main():
    torch.set_num_threads(1)
    config = json.loads(Path(sys.argv[1]).read_text())
    with contextlib.redirect_stdout(sys.stderr):
        api = MapAPI(Path(config['cache_location']))
    source = Path(inspect.getfile(get_closest_lane_point_for_one_agent))
    provenance = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      params=config['params'], cache_location=config['cache_location'])
    print(json.dumps(dict(ready=True, provenance=provenance)), flush=True)
    for line in sys.stdin:
        try:
            data = json.loads(line)
            lanes = []
            with contextlib.redirect_stdout(sys.stderr):
                for i, name in enumerate(data['map_names']):
                    vector = api.get_map(name, incl_road_lanes=True, incl_road_areas=False,
                                         incl_ped_crosswalks=False, incl_ped_walkways=False)
                    points = get_closest_lane_point_for_one_agent(np.asarray(data['agent_hist'])[i], vector,
                        np.asarray(data['world_from_agent'])[i], np.asarray(data['agent_from_world'])[i],
                        config['params'])
                    lanes.append(points.numpy()[..., [0, 1, 3]])
            value = np.stack(lanes).astype(object)
            value[~np.isfinite(np.stack(lanes))] = None
            print(json.dumps(dict(ok=True, lanes=value.tolist()), allow_nan=False), flush=True)
        except BaseException as error:
            print(json.dumps(dict(ok=False, error=repr(error), traceback=traceback.format_exc())), flush=True)


if __name__ == '__main__':
    main()
