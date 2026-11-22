"""Request-realization metrics for archived CrashDojo development rollouts (no new rollouts).

Frozen definitions (written before any result was inspected):
  times: frame k of data.hdf5 is t = 0.1*k s from rollout start (k = 0 initial state); 10 Hz.
  Frenet (s, d) of the target centroid against request['route_world_xy'] (left positive), same
  nearest-segment rule as the guidance code; d0 = parameters.initial_offset_m.
  C1 : v*(t) speed template; window W = [speed_t0, speed_t0 + speed_transition_s + 1 s].
       realized := mean |v - v*| over the settled second [speed_t0 + T, speed_t0 + T + 1] <= max(0.5 m/s, 0.25 * |target - initial|)
       (the requested speed change was reached, up or down); tracking := mean |v - v*| over W;
       within := fraction of W with |v - v*| <= 0.5 m/s.
  C2 : d*(t) = d0 + delta_d * smooth((t - t0)/T_lc) (effective T_lc); window [t0, t0 + T_lc + 1 s].
       realized := sign(delta_d) * max_W (d - d0) >= 0.5 * |delta_d|;
       tracking := mean |d - d*| over W; within := fraction with |d - d*| <= 0.35 m.
  C3A/C3B: depth(t) = max over the four body corners of intrusion beyond the shared boundary
       (positive = into the neighbor lane; boundary side sign fixed from the initial centroid);
       intrusion realized := max depth over [t0, t0 + T_in + T_hold] >= 0.5 * target_intrusion_m (default 0.35);
       C3B return realized := max depth over t >= t0 + T_in + T_hold + T_out is <= 0.10 m;
       realized := intrusion (C3A) / intrusion and return (C3B); tracking := |max depth - target_intrusion_m|.
  C4/C5: the request's arrival_s is not consistent with its interaction window in several requests, so arrival
       *time* is reported only as a diagnostic. realized := the target reaches the requested arrival station
       (max_t s(t) >= arrival_route_s_m - 2 m) AND the C1 speed-template criterion holds; tracking := speed tracking;
       diagnostic: progress error at arrival_s and the time the station is first reached.
  Frames with unavailable/non-finite target state are excluded; a rollout whose window has < 3 valid frames
  is 'unscored'. Contact is taken from the campaign CSV, not recomputed here.
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
import h5py

DT = 0.1

def smooth(u):
    u = np.clip(u, 0, 1)
    return 10*u**3 - 15*u**4 + 6*u**5

def route_coordinates(xy, route):
    a = route[:-1]; delta = route[1:] - a
    length = np.maximum(np.linalg.norm(delta, axis=-1), 1e-5); unit = delta / length[:, None]
    rel = xy[..., None, :] - a
    along = np.clip((rel * unit).sum(-1), 0, None); along = np.minimum(along, length)
    closest = a + along[..., None] * unit
    dist = ((xy[..., None, :] - closest) ** 2).sum(-1)
    ix = dist.argmin(-1)
    tangent = unit[ix]; nearest = np.take_along_axis(closest, ix[..., None, None], -2).squeeze(-2)
    signed = ((xy - nearest) * np.stack((-tangent[..., 1], tangent[..., 0]), -1)).sum(-1)
    prefix = np.concatenate(([0.], np.cumsum(length)))[:-1]
    s = prefix[ix] + np.take_along_axis(along, ix[..., None], -1).squeeze(-1)
    return s, signed

def speed_reference(ts, p):
    v0 = p['initial_speed_mps']; target = p.get('target_speed_mps', v0)
    t0 = p.get('speed_t0', p.get('t0', 0.)); duration = p['speed_transition_s']
    if 'speed_phases_v10' in p:
        result = np.full_like(ts, v0)
        for ph in p['speed_phases_v10']:
            result = result + (ph['v1'] - ph['v0']) * smooth((ts - ph['start']) / ph['duration'])
        return result
    if p.get('speed_profile') == 'bounded_jerk':
        ramp = p['profile_jerk_ramp_s']; acc = p['profile_acceleration_mps2']
        t = np.clip(ts - t0, 0, duration); sign = 1. if target >= v0 else -1.
        return v0 + sign*acc/(2*ramp)*(t**2 - np.maximum(t-ramp, 0)**2 - np.maximum(t-duration+ramp, 0)**2)
    return v0 + (target - v0) * smooth((ts - t0) / duration)

def load_request(run_dir):
    cands = sorted(run_dir.rglob('effective_request.json')) or sorted(run_dir.rglob('request.json'))
    if not cands:
        return None, None
    return json.loads(cands[0].read_text()), str(cands[0])

def boundary_depth(position, yaw, extent, boundary, sign):
    fwd = np.stack((np.cos(yaw), np.sin(yaw)), -1) * extent[:, 0, None] / 2
    left = np.stack((-np.sin(yaw), np.cos(yaw)), -1) * extent[:, 1, None] / 2
    corners = np.stack((position+fwd+left, position+fwd-left, position-fwd-left, position-fwd+left), 1)
    _, side = route_coordinates(corners, boundary)
    return (-sign * side).max(-1)

def score(run_dir):
    run_dir = Path(run_dir)
    paths = list(run_dir.rglob('data.hdf5'))
    if len(paths) != 1:
        return dict(status='no_hdf5')
    path = paths[0]
    binding = json.loads(path.with_name('selection_binding.json').read_text())
    target = binding['runtime_target_index']
    request, request_path = load_request(run_dir)
    if request is None or target is None:
        return dict(status='no_request')
    with h5py.File(path) as h:
        g = h[next(iter(h))]
        position, yaw, extent, available = [np.asarray(g[k]) for k in ['centroid', 'yaw', 'extent', 'availability']]
    yaw = yaw.reshape(position.shape[:2]); available = available.reshape(position.shape[:2]).astype(bool)
    pos = position[target].astype(float); yw = yaw[target].astype(float); ext = extent[target].astype(float)
    ok = available[target] & np.isfinite(pos).all(-1) & np.isfinite(yw)
    T = len(pos); ts = np.arange(T) * DT
    route = np.asarray(request['route_world_xy'], float)
    s, d = route_coordinates(pos, route)
    vel = np.gradient(pos, DT, axis=0); v = np.linalg.norm(vel, axis=-1)
    p = request['parameters']; fam = request['family']
    scales = request.get('tracking_scales', {}); lat_scale = scales.get('lateral_m', 0.35); spd_scale = scales.get('speed_mps', 0.5)
    out = dict(status='scored', family=fam, request=request_path, frames=T, valid_frames=int(ok.sum()))
    def window(lo, hi):
        w = ok & (ts >= lo) & (ts <= min(hi, ts[-1]))
        return w
    def speed_block():
        t0 = p.get('speed_t0', p.get('t0', 0.)); hi = t0 + p['speed_transition_s'] + 1.0
        w = window(t0, hi)
        if w.sum() < 3: return dict(speed_scored=False)
        vref = speed_reference(ts, p)
        drop_req = p['initial_speed_mps'] - p.get('target_speed_mps', p['initial_speed_mps'])
        v_init = float(v[ok][0]) if ok.any() else float('nan')
        ws = window(t0 + p['speed_transition_s'], t0 + p['speed_transition_s'] + 1.0)
        dv = abs(p.get('target_speed_mps', p['initial_speed_mps']) - p['initial_speed_mps'])
        settled_err = float(np.abs(v - vref)[ws].mean()) if ws.any() else float('nan')
        settled_ok = bool(ws.any() and settled_err <= max(spd_scale, 0.25 * dv))
        return dict(speed_scored=True, settled_error_m_s=settled_err, settled_frames=int(ws.sum()), speed_realized=settled_ok,
                    speed_tracking_m_s=float(np.abs(v - vref)[w].mean()),
                    speed_within=float((np.abs(v - vref)[w] <= spd_scale).mean()),
                    requested_drop_m_s=float(drop_req), executed_drop_m_s=float(v_init - v[w].min()),
                    min_speed_window=float(v[w].min()))
    if fam == 'C1':
        b = speed_block(); out.update(b)
        if b['speed_scored']:
            out['realized'] = b['speed_realized']
            out['tracking'] = b['speed_tracking_m_s']; out['within'] = b['speed_within']
        else: out['status'] = 'unscored'
    elif fam == 'C2':
        t0 = p.get('t0', 0.); d0 = p.get('initial_offset_m', 0.); dd = p['delta_d_m']; Tlc = p['T_lc_s']
        w = window(t0, t0 + Tlc + 1.0)
        if w.sum() < 3: out['status'] = 'unscored'
        else:
            dref = d0 + dd * smooth((ts - t0) / Tlc)
            shift = np.sign(dd) * (d - d0)
            out.update(requested_shift_m=abs(dd), executed_shift_m=float(shift[w].max()),
                       realized=bool(shift[w].max() >= 0.5 * abs(dd)),
                       tracking=float(np.abs(d - dref)[w].mean()), within=float((np.abs(d - dref)[w] <= lat_scale).mean()))
            out.update({k: v_ for k, v_ in speed_block().items()})
    elif fam in ('C3A', 'C3B'):
        from shapely import wkt
        from shapely.ops import linemerge
        bnd = wkt.loads(request['geometry_evidence']['shared_boundary_wkt'])
        if bnd.geom_type == 'MultiLineString': bnd = linemerge(bnd)
        boundary = np.asarray(bnd.coords, float)
        _, d_init = route_coordinates(pos[ok][0], boundary)
        sign = float(np.sign(d_init))
        depth = boundary_depth(pos, yw, ext, boundary, sign)
        t0 = p.get('t0', 0.); tin, th, tout = p['T_in_s'], p['T_hold_s'], p['T_out_s']
        target_depth = p.get('target_intrusion_m', 0.35)
        w_in = window(t0, t0 + tin + th); w_ret = ok & (ts >= t0 + tin + th + tout)
        if w_in.sum() < 3: out['status'] = 'unscored'
        else:
            md = float(depth[w_in].max())
            intr = md >= 0.5 * target_depth
            out.update(target_intrusion_m=target_depth, max_depth_m=md, intrusion_realized=bool(intr),
                       tracking=float(abs(md - target_depth)))
            if fam == 'C3B':
                ret_ok = bool(w_ret.sum() >= 1 and depth[w_ret].max() <= 0.10)
                out.update(return_frames=int(w_ret.sum()), return_max_depth_m=float(depth[w_ret].max()) if w_ret.any() else None,
                           return_realized=ret_ok, realized=bool(intr and ret_ok))
            else:
                out['realized'] = bool(intr)
            out['within'] = float((depth[w_in] >= 0.5 * target_depth).mean())
            out.update({k: v_ for k, v_ in speed_block().items()})
    elif fam in ('C4', 'C5'):
        ta = p['arrival_s']; sa = p['arrival_route_s_m']
        b = speed_block(); out.update(b)
        if not ok.any() or not b['speed_scored']: out['status'] = 'unscored'
        else:
            reached = s[ok] >= sa - 2.0
            t_reach = float(ts[ok][reached][0]) if reached.any() else None
            s_at = float(np.interp(ta, ts[ok], s[ok])) if ts[ok][0] <= ta <= ts[ok][-1] else None
            out.update(arrival_s=ta, arrival_route_s_m=sa, max_progress_m=float(s[ok].max()), station_reached=bool(reached.any()),
                       time_station_reached_s=t_reach, progress_at_arrival_m=s_at,
                       progress_error_m=None if s_at is None else float(s_at - sa),
                       realized=bool(reached.any() and b['speed_realized']), tracking=b['speed_tracking_m_s'], within=b['speed_within'])
    else:
        out['status'] = 'unknown_family'
    return out

def main(csv_paths, out_path, variants=('A0', 'A1', 'A2', 'A3', 'A4'), limit=None):
    rows = []
    for c in csv_paths:
        with open(c) as f:
            for r in csv.DictReader(f):
                if r['variant_id'] in variants and r['status'] == 'completed' and r['hdf5']:
                    rows.append(r)
    if limit: rows = rows[:limit]
    results = []
    for i, r in enumerate(rows):
        run_dir = Path(r['hdf5']).parent.parent
        try:
            res = score(run_dir)
        except Exception as ex:  # keep failures as records, never drop silently
            res = dict(status='error', error=f'{type(ex).__name__}: {ex}')
        res.update(job_id=r['job_id'], scene=r['scene'], seed=int(r['seed']), planner=r['planner'], variant_id=r['variant_id'],
                   CR=r['CR'], target_id=r['target_id'])
        results.append(res)
        if (i + 1) % 100 == 0:
            print(f'{i+1}/{len(rows)}', flush=True)
    Path(out_path).write_text('\n'.join(json.dumps(x) for x in results) + '\n')
    print('written', out_path, len(results))

if __name__ == '__main__':
    args = sys.argv[1:]
    limit = None
    if args and args[0].startswith('--limit='):
        limit = int(args[0].split('=')[1]); args = args[1:]
    main(args[:-1], args[-1], limit=limit)
