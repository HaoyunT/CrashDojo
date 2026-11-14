"""Keep the production adapter intact; center ego boxes in an isolated cache."""
import json
import os
from pathlib import Path

import numpy as np

OFFSET_M = 1.461
ORIGINS = {'las_vegas': (660000., 3990000.), 'boston': (330000., 4690000.),
           'singapore': (360000., 140000.), 'pittsburgh': (580000., 4470000.)}
CITY_NAMES = {'us-nv-las-vegas-strip': 'las_vegas', 'us-ma-boston': 'boston',
              'sg-one-north': 'singapore', 'us-pa-pittsburgh-hazelwood': 'pittsburgh'}


def origin(city):
    return np.asarray(ORIGINS[CITY_NAMES.get(city, city)],dtype=np.float64)


def shift_request(request, city):
    import copy
    result = copy.deepcopy(request)
    offset = origin(city)
    for key, value in result.items():
        if key.endswith('_world_xy') or key.endswith('_world_polygons') or key=='crossing_point_world':
            if key.endswith('_world_polygons'):
                result[key] = [(np.asarray(p,dtype=np.float64)-offset).tolist() for p in value]
            else:
                result[key] = (np.asarray(value,dtype=np.float64)-offset).tolist()
    evidence = result.get('geometry_evidence',{})
    if evidence.get('crossing_point'):
        evidence['crossing_point'] = (np.asarray(evidence['crossing_point'])-offset).tolist()
    if evidence.get('shared_boundary_wkt'):
        from shapely import wkt
        from shapely.affinity import translate
        evidence['shared_boundary_wkt'] = translate(wkt.loads(evidence['shared_boundary_wkt']),
            xoff=-offset[0],yoff=-offset[1]).wkt
    result['runtime_coordinate_origin_utm_m'] = offset.tolist()
    return result


def center_ego(data, dt):
    result = data.copy()
    mask = result.index.get_level_values('agent_id') == 'ego'
    rows = result.loc[mask]
    yaw = np.unwrap(rows['heading'].to_numpy())
    omega = np.gradient(yaw, dt)
    alpha = np.gradient(omega, dt)
    c, s = np.cos(yaw), np.sin(yaw)
    result.loc[mask, 'x'] += OFFSET_M * c
    result.loc[mask, 'y'] += OFFSET_M * s
    result.loc[mask, 'vx'] -= OFFSET_M * s * omega
    result.loc[mask, 'vy'] += OFFSET_M * c * omega
    result.loc[mask, 'ax'] -= OFFSET_M * (c * omega**2 + s * alpha)
    result.loc[mask, 'ay'] += OFFSET_M * (-s * omega**2 + c * alpha)
    return result


def install():
    from trajdata.dataset_specific.nuplan.nuplan_dataset import NuplanDataset
    from trajdata.dataset_specific.nuplan import nuplan_utils
    populate = nuplan_utils.populate_vector_map

    def populate_centered(vector_map, nuplan_map, lane_connector_idxs):
        populate(vector_map, nuplan_map, lane_connector_idxs)
        offset = origin(nuplan_map.map_name)
        vector_map.extent[[0,1,3,4]] -= np.tile(offset,2)
        seen = set()
        for element in vector_map.iter_elems():
            for value in vars(element).values():
                for line in value if isinstance(value,list) else [value]:
                    if hasattr(line,'points') and id(line) not in seen:
                        line.points = np.asarray(line.points,dtype=np.float64).copy()
                        line.points[:,:2] -= offset
                        seen.add(id(line))

    nuplan_utils.populate_vector_map = populate_centered
    for name in ['_get_matching_scenes_from_obj','_get_matching_scenes_from_cache']:
        original_match = getattr(NuplanDataset,name)

        def matching(self,*args,_original=original_match,**kwargs):
            scenes = _original(self,*args,**kwargs)
            selected = os.environ.get('V26_NUPLAN_SCENE') or os.environ.get('V24_NUPLAN_SCENE')
            if not selected:
                raise KeyError('V26_NUPLAN_SCENE')
            return [s for s in scenes if s.name==selected]

        setattr(NuplanDataset,name,matching)
    original = NuplanDataset.get_agent_info

    def get_agent_info(self, scene, cache_path, cache_class):
        class CenteredCache(cache_class):
            @staticmethod
            def save_agent_data(data, cache_path, scene):
                result = center_ego(data, scene.dt)
                raw_xy = result[['x','y']].to_numpy(dtype=np.float64)
                offset = origin(scene.location)
                result[['x','y']] = raw_xy-offset
                cache_class.save_agent_data(result, cache_path, scene)
                audit_path = os.environ.get('V26_COORDINATE_AUDIT') or os.environ.get('V24_COORDINATE_AUDIT')
                if not audit_path:
                    raise KeyError('V26_COORDINATE_AUDIT')
                directory = Path(audit_path)
                directory.mkdir(parents=True, exist_ok=True)
                name = scene.name.replace('/', '_')
                (directory / (name + '.json')).write_text(json.dumps(dict(
                    scene=scene.name, source_reference='rear_axle', cache_reference='vehicle_box_center',
                    offset_m=OFFSET_M, velocities_and_accelerations_transformed=True,
                    coordinate_origin_utm_m=offset.tolist(),map_data_and_request_shifted_together=True,
                    raw_world_float32_max_error_m=float(np.abs(raw_xy.astype(np.float32)-raw_xy).max()),
                    shifted_float32_max_error_m=float(np.abs((raw_xy-offset).astype(np.float32)-(raw_xy-offset)).max()),
                    original_source_modified=False, native_dt_s=scene.dt), indent=2))
        return original(self, scene, cache_path, CenteredCache)

    NuplanDataset.get_agent_info = get_agent_info
