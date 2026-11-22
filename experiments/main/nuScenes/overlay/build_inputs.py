"""Prepare the historical 50-scene cohort at frame 31, without running FM.

Use the published website's per-window candidate ranking, with explicit FM
history and native-map eligibility. Keep every scene, including no-action cases.
Targets use initial states and map topology; recorded future is not a guide input.
"""
import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, linemerge, triangulate

sys.path.insert(0, '/path/to/storage/external_scripts')
from q16_map_events_v2_20261020 import NativeMap, load_tracks, wrap, boxes

ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO = ROOT / 'code/CCDiff_M2c_planner'
SITE = Path('/path/to/storage/data/outputs/q11_nuscenes_val_full_20261017')
DEST = REPO / 'benchmark50_inputs_20261021_v3'
START = 31
STEPS = 100


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    return x


def write(path, data):
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False))


def truth(x): return str(x).lower() == 'true'


def rank(r):
    status = r['status']
    tier = 3 if status in ('selected', 'label_review') else 2 if status == 'semantic_pending' else 1 if status == 'vehicle_only' else 0
    risk = float(r.get('risk_score') or 0.)
    if tier >= 2: return tier, 1, risk, int(status == 'selected'), -START, r['adv_id']
    return tier, int(truth(r.get('role_matches_syllabus'))), risk, int(truth(r.get('has_full_vehicle_future'))), -START, r['adv_id']


def path_line(m, keys):
    xy = np.concatenate([np.asarray(m.line[k].coords) for k in keys])
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-5]
    return LineString(xy[keep])


def paths(m, start, depth=7, budget=20):
    queue = [[start]]; result = []
    while queue and len(result) < budget:
        p = queue.pop(0)
        nxt = sorted(k for k in m.out[p[-1]] if k not in p)
        if len(p) >= depth or not nxt: result.append(p)
        else: queue.extend(p + [k] for k in nxt)
    def cost(p):
        turns = sum(abs(wrap(m.tangent(k, m.line[k].length) - m.tangent(k, 0))) for k in p)
        jumps = sum(abs(wrap(m.tangent(b, 0) - m.tangent(a, m.line[a].length))) for a, b in zip(p, p[1:]))
        return float(turns + jumps), tuple(p)
    return sorted(result, key=cost)


def match(m, tr):
    j = tr['lookup'][START]; pt = Point(tr['xy'][j]); out = []
    for k in m.keys[m.tree.query(pt.buffer(.5))]:
        angle = abs(wrap(m.tangent(k, m.line[k].project(pt)) - tr['yaw'][j]))
        if angle < .6 and m.poly[k].distance(pt) <= .5:
            out.append((m.line[k].distance(pt) + 2 * angle, str(k)))
    return min(out)[1] if out else None


def offset(route, xy):
    s = route.project(Point(xy)); q = np.asarray(route.interpolate(s).coords[0])
    d = np.asarray(route.interpolate(min(route.length, s+.1)).coords[0]) - np.asarray(route.interpolate(max(0, s-.1)).coords[0])
    d /= max(np.linalg.norm(d), 1e-8)
    return float((xy-q) @ np.array([-d[1], d[0]])), s


def speed_parameters(v0, target):
    target = float(np.clip(target, 0., 25.))
    # Quintic speed transition: max a=1.875*dv/T and max jerk=5.774*dv/T^2.
    dv = abs(target-v0)
    a_margin = 2.4 if target >= v0 else 4.8
    duration = max(2., 1.875*dv/a_margin, math.sqrt(5.774*dv/6.))
    if duration > 6.:
        target = v0 + np.sign(target-v0)*min(dv, 6*a_margin/1.875)
        duration = 6.
    return dict(initial_speed_mps=float(v0), target_speed_mps=float(target), speed_transition_s=float(duration), speed_t0=.2)


def construct(m, a, e, base):
    ka, ke = match(m, a), match(m, e)
    if ka is None or ke is None: return None, 'no_native_lane_match'
    ja, je = a['lookup'][START], e['lookup'][START]
    xy, exy = a['xy'][ja], e['xy'][je]
    v0, ve = float(a['v'][ja]), float(e['v'][je])
    pa, pe = paths(m, ka), paths(m, ke)
    if not pa or not pe: return None, 'no_directed_route'
    # Choose minimum-turn continuation from current topology, not logged future.
    ra, re = pa[0], pe[0]; route = path_line(m, ra); eroute = path_line(m, re)
    d0, sa = offset(route, xy); se = route.project(Point(exy))
    par = dict(t0=.4, initial_offset_m=d0)
    evidence = dict(map_sources=m.map_sources, source_lane=ka, ego_lane=ke, route_lanes=ra,
                    ego_route_lanes=re, route_choice='native_topology_min_turn_no_future_track')
    family = None; extra = []; window = [1., 8.]
    if ka == ke or ka in re or ke in ra:
        if abs(wrap(a['yaw'][ja]-e['yaw'][je])) > .5: return None, 'same_path_heading_mismatch'
        family = 'C1'; evidence['role_check'] = 'same_directed_corridor'
        if ka in re and ke not in ra:
            ra=re;route=eroute;d0,sa=offset(route,xy);se=route.project(Point(exy))
            par['initial_offset_m']=d0;evidence['route_lanes']=ra
        gap = sa-se
        length = .5*(a['size'][1]+e['size'][1])
        clear = max(0., abs(gap)-length)
        target = ve-clear/4. if gap > 0 else ve+clear/4.
        par.update(speed_parameters(v0, target)); evidence['initial_signed_route_gap_m'] = float(gap)
        evidence['longitudinal_role'] = 'adversary_ahead' if gap > 0 else 'adversary_behind'
    else:
        bd = m.shared(ka, ke)
        heading_diff = abs(wrap(m.tangent(ka,m.line[ka].project(Point(xy))) - m.tangent(ke,m.line[ke].project(Point(xy)))))
        if bd is not None and heading_diff < .35:
            if bd.geom_type == 'MultiLineString': bd = linemerge(bd)
            if bd.geom_type != 'LineString': return None, 'disconnected_shared_boundary'
            dest = np.asarray(m.line[ke].interpolate(m.line[ke].project(Point(xy))).coords[0])
            delta = offset(route, dest)[0]-d0
            if not 1.5 < abs(delta) < 5.5: return None, 'ambiguous_neighbor_offset'
            evidence.update(role_check='same_direction_shared_boundary', same_direction=True,
                            shared_boundary_wkt=bd.wkt, neighbor_lane=ke)
            extra = re
            relative_gap = abs(float(np.dot(xy-exy,[np.cos(e['yaw'][je]),np.sin(e['yaw'][je])])))-.5*(a['size'][1]+e['size'][1])
            if relative_gap < 3.:
                family = 'C3B' if abs(d0) > .4 else 'C3A'
                amp = np.sign(delta)*max(.2, Point(xy).distance(bd)-a['size'][0]/2+.35)
                tin = max(2.6, (60*abs(amp)/6.)**(1/3), math.sqrt(5.774*abs(amp)/3.))
                if tin > 3.5: return None, 'intrusion_duration_exceeds_complete_window'
                par.update(A_m=float(amp),T_in_s=tin,T_hold_s=.5,T_out_s=tin,
                           return_offset_m=0. if family=='C3B' else d0,
                           target_intrusion_m=.35,z_min_m=.2,z_max_m=float(a['size'][0]/2))
                window=[max(1.,.4+.65*tin),.4+tin+.5+.25*tin]
            else:
                family='C2'; duration=max(3.5,(60*abs(delta)/6.)**(1/3),math.sqrt(5.774*abs(delta)/3.))
                par.update(delta_d_m=float(delta),T_lc_s=float(duration))
                window=[.4+.5*duration,min(9.,.4+duration+2.)]
            par.update(speed_parameters(v0,ve))
        else:
            common = set(ra) & set(re)
            if common:
                k = next(k for k in ra if k in common); ia,ie=ra.index(k),re.index(k)
                if ia>0 and ie>0 and m.kind[k]=='lane' and ra[ia-1]!=re[ie-1]:
                    cp=Point(m.line[k].coords[0]); family='C5'
                    evidence.update(role_check='native_distinct_incoming_merge',common_lane=k,
                                    distinct_incoming=[ra[ia-1],re[ie-1]])
            else:
                cp=None
                for ca in ra:
                    for ce in re:
                        if m.kind[ca]!='connector' or m.kind[ce]!='connector':continue
                        if set(m.inc[ca])&set(m.inc[ce]) or set(m.out[ca])&set(m.out[ce]):continue
                        cross=m.line[ca].intersection(m.line[ce])
                        if cross.geom_type!='Point':continue
                        angle=abs(wrap(m.tangent(ca,m.line[ca].project(cross))-m.tangent(ce,m.line[ce].project(cross))))*180/np.pi
                        if 30<=angle<=150:
                            cp=cross;family='C4';evidence.update(role_check='native_distinct_crossing',
                                crossing_connectors=[ca,ce],crossing_point=list(cp.coords[0]),crossing_angle_deg=float(angle));break
                    if family:break
            if family in ('C4','C5'):
                distance=route.project(cp)-sa; edistance=eroute.project(cp)-eroute.project(Point(exy))
                if not 2.<distance<80. or not 2.<edistance<80.:return None,'event_not_ahead_in_local_horizon'
                arrival=float(np.clip(edistance/max(ve,1.),2.,7.))
                desired=2*distance/arrival-v0
                par.update(speed_parameters(v0,desired))
                par.update(arrival_s=arrival,arrival_route_s_m=float(route.project(cp)),centerline_transition_s=3.)
                window=[max(1.,arrival-1.5),min(9.,arrival+1.5)]
    if family is None: return None,'no_supported_five_class_topology'
    corridor_keys=set(ra)|set(extra)
    # Include adjacent predecessor segments so a long rear body at a lane seam
    # is checked against the native road corridor, not an artificial start edge.
    for k in list(corridor_keys): corridor_keys.update(m.inc[k])
    poly=unary_union([m.poly[k] for k in corridor_keys]).buffer(.15)
    body=boxes(xy[None,:],a['yaw'][ja:ja+1],a['size'])[0]
    coverage=body.intersection(poly).area/body.area
    if coverage < .98:return None,'initial_body_corridor_below_98pct'
    ego_body=boxes(exy[None,:],e['yaw'][je:je+1],e['size'])[0]
    if body.intersects(ego_body):return None,'initial_target_overlap'
    polygons=[poly] if poly.geom_type=='Polygon' else list(poly.geoms)
    pieces=[]
    for polygon in polygons:
        parts=[t for t in triangulate(polygon) if polygon.covers(t)] if polygon.interiors else [polygon]
        assert abs(sum(p.area for p in parts)-polygon.area)<1e-5*max(1.,polygon.area), 'Do not fill corridor holes'
        pieces.extend(parts)
    r=copy.deepcopy(base)
    r.update(family=family,parameters=par,geometry_evidence=evidence,route_world_xy=list(route.coords),
             corridor_world_polygons=[list(p.exterior.coords) for p in pieces],interaction_window_s=window,
             route_offset_bounds_m=[-5.,5.] if family in ('C2','C3A','C3B') else [-1.8,1.8],
             initial_feasibility_diagnostic=dict(body_coverage=float(coverage),target_body_distance_m=float(body.distance(ego_body))),
             parameter_source='initial_observed_speed_gap_native_map_geometry_and_unchanged_dynamic_margins',
             route_progress_guidance=family in ('C4','C5'), center_receiving_route=family in ('C4','C5'),
             boundary_relative_template=family in ('C3A','C3B'),phase_specific_intrusion=family in ('C3A','C3B'))
    r['weights']['action']=5. if family=='C1' else 20.
    return r,None


def main():
    assert not DEST.exists(), 'Never overwrite frozen inputs'
    DEST.mkdir()
    cohort=json.loads(Path('/tmp/historical_val_scene_audit_20261021.json').read_text())['historical_union']
    meta=json.loads((SITE/'scene_meta.json').read_text())
    native={x['name']:x for x in json.loads(Path('/path/to/storage/external/data/nuscenes/v1.0-trainval/scene.json').read_text())}
    windows={r['scene']:r for r in csv.DictReader((SITE/'windows.csv').open()) if r['scene'] in cohort and int(r['start_ts'])==START}
    options=collections.defaultdict(list)
    for filename in ('candidates.csv','requests.csv'):
        for r in csv.DictReader((SITE/filename).open()):
            if r['scene'] in cohort and int(r['start_ts'])==START and r.get('adv_id') and r['status'] in ('selected','label_review','semantic_pending','vehicle_only','static_reference_only'):
                options[r['scene']].append(r)
    base=json.loads((REPO/'development_tuning_v11_guided_20261021/C2.json').read_text())
    for key in ('selection_start_revision','configuration_source','parameter_revision','selection_manifest_row'):
        base.pop(key,None)
    base.update(purpose='historical_val50_fixed_start_initial_evaluation',executed_acceleration_guidance=True,
                capability_status='provisional',selection_manifest_row=None)
    base['validation_policy']['formal_evaluation']=True
    maps={};manifest=[];coverage=[]
    for scene in cohort:
        city=meta[scene]['city']
        if city not in maps:maps[city]=NativeMap('nuscenes',city)
        m=maps[city];tr=load_tracks(SITE/'cache'/(scene+'.npz'));e=tr['ego'];reject=[];chosen=None;request=None
        assert all(t in e['lookup'] for t in range(START-30,START+STEPS+1)),scene
        ranked=sorted(options[scene],key=rank,reverse=True);seen=set();fallback=[]
        for candidate in ranked:
            aid=candidate['adv_id']
            if aid in seen:continue
            seen.add(aid);a=tr.get(aid)
            if a is None or any(t not in a['lookup'] for t in range(START-30,START+1)):
                reject.append(dict(actor=aid,reason='incomplete_actual_3_second_history'));continue
            # Keep only actually observed same-time actors. Never synthesize IDs.
            if match(m,a) is not None:fallback.append(candidate)
            if candidate['status']=='static_reference_only' or truth(candidate.get('parked')):
                reject.append(dict(actor=aid,reason='parked_not_an_adversarial_action'));continue
            request,reason=construct(m,a,e,base)
            if request is None:reject.append(dict(actor=aid,reason=reason));continue
            chosen=candidate;break
        row=dict(scene_id=scene,website_fixed31_actor=windows[scene]['recommended_id'],rejections=reject,
                 eligible=request is not None,scene_kept_in_denominator=True)
        if chosen is None:
            # This is an explicit no-adversary sentinel, never a vehicle token.
            chosen={'adv_id':'NO_ADVERSARY'}
            request=dict(version=1,family='NONE',mode='replay_only',no_action_reason='no_eligible_five_class_action',
                         gradient_steps=0,capability_status='not_applicable',geometry_status='not_applicable',
                         corridor_world_polygons=[],validation_policy=base['validation_policy'],engineering_limits=base['engineering_limits'])
        aid=chosen['adv_id'];n=native[scene]
        availability=dict(recorded_reference_frames=sum(t in tr[aid]['lookup'] for t in range(START,START+STEPS)),
                          recorded_final_frame_available=START+STEPS-1 in tr[aid]['lookup']) if aid!='NO_ADVERSARY' else None
        request['reference_availability']=availability
        record=dict(dataset='nuscenes',split='val',scene_id=scene,scene_token=n['token'],log_id=n['log_token'],city=city,
                    adversary_vehicle_id=aid,ego_vehicle_id='ego',vehicle_id_namespace='nuScenes instance_token',
                    start_step_10hz=START,start_time_s=START*.1,selection_source=str(SITE/'windows.csv'),
                    source_window_id=f'{scene}_t031',selection_policy='website_rank_fixed31_with_explicit_runtime_and_native_map_eligibility',
                    website_original_fixed31_actor=windows[scene]['recommended_id'],closed_loop_validated=False,
                    no_action=request['family']=='NONE',reference_availability=availability)
        request.update(scene_id=scene,adversary_vehicle_id=aid,start_step_10hz=START,selection_manifest_row=len(manifest))
        if request['family']!='NONE':
            request.update(allow_selected_stationary_control=True,selected_vehicle_initially_parked=False,
                           stationary_override_source='website_fixed31_candidate_not_parked')
        write(DEST/(scene+'.json'),request)
        row.update(family=request['family'],actor_id=aid,selection_changed=aid!=windows[scene]['recommended_id'],
                   manifest_row=len(manifest),binding_available=True,reference_availability=availability)
        manifest.append(record);coverage.append(row)
        print(scene,request['family'],aid,'changed',row['selection_changed'],flush=True)
    (DEST/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    report=dict(cohort=cohort,total_scenes=50,start_frame=START,steps=STEPS,coverage=coverage,
                family_counts=dict(collections.Counter(r['family'] for r in coverage)),
                selection_changed=sum(r.get('selection_changed',False) for r in coverage),
                candidate_ranking_provenance='website offline rank; its risk proxy can use recorded future; runtime guide does not',
                inputs_sha256={str(SITE/f):hashlib.sha256((SITE/f).read_bytes()).hexdigest() for f in ('windows.csv','candidates.csv','requests.csv')})
    write(DEST/'coverage.json',report)
    print(json.dumps({k:report[k] for k in ('total_scenes','family_counts','selection_changed')}),flush=True)


if __name__=='__main__':main()
