"""Recompute discrete oriented-box contacts and executed-prefix validity."""
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union


def contacts(position, yaw, extent, available):
    n, frames = available.shape
    pairs = np.asarray(list(itertools.combinations(range(n), 2)), dtype=int).reshape(-1, 2)
    if not len(pairs):
        return pairs, np.zeros((0,frames), dtype=bool)
    i, j = pairs[:,0], pairs[:,1]
    forward = np.stack([np.cos(yaw), np.sin(yaw)], -1)
    lateral = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    axes = np.stack([forward[i],lateral[i],forward[j],lateral[j]], -2)
    displacement = position[j]-position[i]
    distance = np.abs((axes*displacement[:,:,None,:]).sum(-1))
    radius = np.zeros_like(distance)
    for actors in [i,j]:
        radius += (np.abs((axes*forward[actors,:,None,:]).sum(-1))*extent[actors,:,0,None]/2
                   +np.abs((axes*lateral[actors,:,None,:]).sum(-1))*extent[actors,:,1,None]/2)
    hit = (distance <= radius+1e-8).all(-1) & available[i] & available[j]
    return pairs, hit


def prefix_dynamics(step, limit, frames=5):
    history = np.asarray(step['history_local_last3'], dtype=float)
    future = np.asarray(step['selected_local_first5'], dtype=float)[:frames]
    xy = np.concatenate([history,future])
    velocity = np.diff(xy,axis=0)/.1
    acceleration = np.diff(velocity,axis=0)/.1
    jerk = np.diff(acceleration,axis=0)/.1
    # Resolve acceleration along the interval's mean velocity; retain a vector
    # jerk bound across the replan boundary. This is a discrete engineering screen.
    v = np.linalg.norm(velocity[2:],axis=-1)
    a = np.linalg.norm(acceleration[1:],axis=-1)
    middle = (velocity[1:-1] + velocity[2:]) / 2
    direction = middle / np.maximum(np.linalg.norm(middle,axis=-1,keepdims=True),1e-8)
    longitudinal = (acceleration[1:]*direction).sum(-1)
    lateral = np.abs(acceleration[1:,0]*direction[:,1]-acceleration[1:,1]*direction[:,0])
    jj = np.linalg.norm(jerk,axis=-1)
    valid = ((v<=limit['max_speed_mps']+1e-6).all()
             and (longitudinal<=limit['max_acceleration_mps2']+1e-6).all()
             and (longitudinal>=-limit['max_braking_mps2']-1e-6).all()
             and (lateral<=limit['max_lateral_mps2']+1e-6).all()
             and (jj<=limit['max_jerk_mps3']+1e-6).all())
    return dict(valid=bool(valid),max_speed_mps=float(v.max()) if len(v) else 0.,
                max_acceleration_norm_mps2=float(a.max()) if len(a) else 0.,
                max_longitudinal_acceleration_mps2=float(longitudinal.max()) if len(a) else 0.,
                max_longitudinal_braking_mps2=float(-longitudinal.min()) if len(a) else 0.,
                max_lateral_acceleration_mps2=float(lateral.max()) if len(a) else 0.,
                max_jerk_norm_mps3=float(jj.max()) if len(jj) else 0.)


def evaluate(directory, request):
    directory = Path(directory)
    paths = list(directory.rglob('data.hdf5'))
    if len(paths)!=1:
        raise ValueError('Exactly one rollout artifact required')
    path = paths[0]
    binding = json.loads(path.with_name('selection_binding.json').read_text())
    ego, target = binding['runtime_ego_index'], binding['runtime_target_index']
    with h5py.File(path) as h:
        if len(h)!=1:
            raise ValueError('Exactly one scene required')
        g = h[next(iter(h))]
        position,yaw,extent,available = [np.asarray(g[k]) for k in ['centroid','yaw','extent','availability']]
    yaw = yaw.reshape(position.shape[:2])
    available = available.reshape(position.shape[:2]).astype(bool)
    finite = np.isfinite(position).all(-1) & np.isfinite(yaw) & np.isfinite(extent).all(-1)
    selected = [ego]+([target] if target is not None else [])
    if not finite[selected].all():
        raise FloatingPointError('Nonfinite controlled trajectory')
    pairs, hit = contacts(position,yaw,extent,available & finite)
    ego_hit = hit[(pairs==ego).any(-1)].any(0)
    target_hit = np.zeros(len(ego_hit),bool)
    background = np.zeros(len(ego_hit),bool)
    if target is not None:
        target_pair = ((pairs==ego).any(-1)&(pairs==target).any(-1))
        target_hit = hit[target_pair].any(0)
        background = hit[(pairs==target).any(-1)&~(pairs==ego).any(-1)].any(0)
    contact_frame = int(np.flatnonzero(target_hit)[0]) if target_hit.any() else None
    steps = [json.loads(s) for s in path.with_name('selected_planner_steps.jsonl').read_text().splitlines()]
    dynamic = []
    for step in steps:
        if 'history_local_last3' not in step:
            continue
        d = prefix_dynamics(step,request['engineering_limits'])
        d['step_index']=step['step_index']
        dynamic.append(d)
    all_valid = all(d['valid'] for d in dynamic)
    precontact = [prefix_dynamics(s,request['engineering_limits'],
                    min(5,contact_frame-s['step_index']+1) if contact_frame is not None else 5)
                  for s in steps if 'history_local_last3' in s
                  and (contact_frame is None or s['step_index']<=contact_frame)]
    coverage = None
    if target is not None:
        corridor = unary_union([Polygon(p) for p in request['corridor_world_polygons']])
        angles = yaw[target]
        forward = np.c_[np.cos(angles),np.sin(angles)]*extent[target,:,0,None]/2
        left = np.c_[-np.sin(angles),np.cos(angles)]*extent[target,:,1,None]/2
        corners = position[target,:,None,:]+np.stack([forward+left,forward-left,-forward-left,-forward+left],1)
        boxes = shapely.polygons(corners)
        coverage = np.asarray(shapely.area(shapely.intersection(boxes,corridor))/shapely.area(boxes))
        road_before = bool((coverage[:contact_frame+1 if contact_frame is not None else len(coverage)]>=.98-1e-7).all())
    else:
        road_before = True
    valid_contact = bool(target_hit.any() and precontact and all(d['valid'] for d in precontact) and road_before)
    stats = json.loads(path.with_name('stats.json').read_text())
    mean = lambda key: float(np.nanmean(stats[key])) if key in stats and np.isfinite(stats[key]).any() else None
    return dict(hdf5=str(path),frames=position.shape[1],ego_collision_10Hz=bool(ego_hit.any()),
        scene_collision_10Hz=bool(hit.any()),target_contact_10Hz=bool(target_hit.any()),
        first_target_contact_s=None if contact_frame is None else contact_frame*.1,
        target_background_contact_10Hz=bool(background.any()),
        target_all_executed_prefixes_within_engineering_screen=all_valid,
        invalid_target_prefixes=sum(not d['valid'] for d in dynamic),target_prefixes=len(dynamic),
        target_prefix_diagnostics=dynamic,target_body_corridor_min=None if coverage is None else float(coverage.min()),
        qualified_target_contact_10Hz=valid_contact,
        quality_scope='discrete 10Hz geometry plus velocity-resolved acceleration and vector jerk executed-prefix screen; not continuous collision or full physical certification',
        ADE_native_m=mean('ade'),FDE_native_m=mean('fde'),ORR_native=mean('all_off_road_rate_rate'),
        ego_collision_native=bool(any(np.any(np.asarray(stats[k])>0) for k in ['ego_collision_rate_coll_any','ego_failure_failure_collision'] if k in stats)),
        no_action=target is None)
