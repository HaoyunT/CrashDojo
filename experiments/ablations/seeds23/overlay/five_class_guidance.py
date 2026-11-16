"""Inference-only, fixed-identity five-mechanism endpoint guidance.

No model weights are updated. Requests carry explicit map geometry and clocks.
Natural priors are references, not capability limits. Final rollout validation is
separate from the differentiable losses and from these provisional smoke checks.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from causal_background_occupancy_v1 import enabled as bg_enabled, validate_config as bg_validate, prepare_background, penalty as bg_penalty
from tuning_helpers_v11 import prefix_cost, deviation_cost
from qualified_contact_objective import contact_cost, completion_cost, warm_start
from precise_dynamics import PreciseDynamics, double_context
from ccdiff.evaluation.paper_contract import PaperExecutionMixin, validate_generation
from ablation_contract import generic_contact_only, generic_contact_request


def load_request(path, record):
    raw = Path(path).read_bytes()
    r = json.loads(raw)
    if r.get('family') == 'NONE':
        if r.get('version') != 1 or r.get('mode') != 'replay_only' or not r.get('no_action_reason'):
            raise ValueError('Malformed explicit no-action request')
        if not record.get('no_action') or record.get('adversary_vehicle_id') != 'NO_ADVERSARY':
            raise ValueError('No-action requests require a no-adversary manifest')
        for k in ('scene_id','adversary_vehicle_id','start_step_10hz'):
            if r.get(k) != record[k]:raise ValueError('No-action identity mismatch')
        if r.get('gradient_steps') != 0:raise ValueError('No-action cannot optimize actions')
        r['_sha256']=hashlib.sha256(raw).hexdigest()
        return r
    if r.get('version') != 1 or r.get('family') not in ('C1','C2','C3A','C3B','C4','C5'):
        raise ValueError('Unknown five-class request version/family')
    for k in ('scene_id','adversary_vehicle_id','start_step_10hz'):
        if r.get(k) != record[k]:
            raise ValueError('Request/selection identity mismatch: '+k)
    if r.get('mode') not in ('contact_search','near_conflict'):
        raise ValueError('Explicit pressure mode required')
    if r.get('geometry_status') != 'native_map_verified':
        raise ValueError('Runtime requests require native-map geometry verification')
    route = np.asarray(r.get('route_world_xy'), dtype=float)
    if route.ndim != 2 or route.shape[1] != 2 or len(route)<2 or not np.isfinite(route).all():
        raise ValueError('Finite world-frame route required')
    if not r.get('corridor_world_polygons') or not r.get('geometry_evidence'):
        raise ValueError('Map corridor and provenance required')
    if r.get('capability_status') not in ('provisional','verified'):
        raise ValueError('Explicit capability status required')
    bg_validate(r)
    w=r.get('interaction_window_s', [])
    if len(w)!=2 or not 0<=w[0]<w[1]:raise ValueError('Invalid fixed interaction window')
    if r['mode']=='near_conflict' and not 0<r.get('distance_band_m',[0,0])[0]<=r['distance_band_m'][1]:
        raise ValueError('Positive near-conflict distance band required')
    for name in ('action','pressure','prior','constraint'):
        if not math.isfinite(r['weights'][name]) or r['weights'][name]<0:raise ValueError('Invalid weight')
    solver_budget = 30 if r.get('trajectory_solver','gradient') == 'lbfgs' else 10
    iterations = r.get('gradient_steps',1)
    if type(iterations) is not int or not 0<=iterations<=solver_budget or not 0<=r.get('step_size',.05)<=1:
        raise ValueError('Unbounded guidance optimization requested')
    if not 1<=r.get('last_flow_steps',1)<=20:raise ValueError('Invalid flow-step budget')
    if r.get('trajectory_solver','gradient') not in ('gradient','lbfgs'):
        raise ValueError('Unknown inference trajectory solver')
    evidence=r['geometry_evidence'];family=r['family'];params=r['parameters']
    expected={'C1':'same_directed_corridor','C2':'same_direction_shared_boundary',
        'C3A':'same_direction_shared_boundary','C3B':'same_direction_shared_boundary',
        'C4':'native_distinct_crossing','C5':'native_distinct_incoming_merge'}
    if evidence.get('role_check') != expected[family]:
        raise ValueError('Mechanism topology proof does not match request family')
    for key in ('speed_transition_s',):
        if not math.isfinite(params.get(key,0)) or params.get(key,0)<=0:raise ValueError('Positive template duration required')
    if family=='C2' and params.get('T_lc_s',0)<=0:raise ValueError('Invalid lane change duration')
    if family in ('C3A','C3B') and (params.get('T_in_s',0)<=0 or params.get('T_out_s',0)<=0 or params.get('T_hold_s',-1)<0):
        raise ValueError('Invalid intrusion/return durations')
    if family in ('C2','C3A','C3B') and (not evidence.get('same_direction') or not evidence.get('shared_boundary_wkt')):
        raise ValueError('Same-direction shared boundary required')
    if family=='C4' and not evidence.get('crossing_connectors'):raise ValueError('Native crossing proof required')
    if family=='C5' and not evidence.get('common_lane'):raise ValueError('Common receiving lane required')
    for value in r['engineering_limits'].values():
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('Finite positive engineering bounds required')
    if params.get('speed_profile') == 'bounded_jerk':
        acc=params.get('profile_acceleration_mps2',0);ramp=params.get('profile_jerk_ramp_s',0)
        duration=params['speed_transition_s'];dv=abs(params['target_speed_mps']-params['initial_speed_mps'])
        if not 0<acc<=.9*r['engineering_limits']['max_acceleration_mps2'] or not 0<ramp or acc/ramp>8.:
            raise ValueError('Speed profile exceeds conservative acceleration/jerk margins')
        if duration<2*ramp or abs(duration-(dv/acc+ramp))>1e-6:
            raise ValueError('Inconsistent bounded jerk profile duration')
    for value in r.get('tracking_scales',{}).values():
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('Tracking scales must be finite and positive')
    repair_steps=r.get('prefix_feasibility_repair_steps',0)
    if type(repair_steps) is not int or not 0<=repair_steps<=20:raise ValueError('Unbounded prefix repair budget')
    if not 0<=r.get('contact_margin_m',0.)<=.1:raise ValueError('Excessive contact penetration target')
    if r.get('deviation_guidance'):
        for key in ('ade_weight','fde_weight','soft_ade_budget_m','soft_fde_budget_m'):
            value=r['deviation_guidance'][key]
            if not math.isfinite(value) or not 0<=value<=20:raise ValueError('Invalid deviation regularizer')
    # These are screening values, not automatically certified vehicle capabilities.
    r['_sha256']=hashlib.sha256(raw).hexdigest()
    return r


def release_selected_stationary(mask, target):
    if mask.shape[0] != 1 or not 0 <= target < mask.shape[1]:
        raise ValueError('Stationary exemption requires one valid selected actor')
    result=mask.clone();result[:,target]=False
    return result


def smooth(u):
    u=u.clamp(0,1)
    return 10*u**3-15*u**4+6*u**5


def band(x,lo,hi,scale=1.):
    return (torch.relu(lo-x)**2+torch.relu(x-hi)**2)/(scale*scale)


def route_coordinates(xy, route):
    """Piecewise differentiable nearest segment Frenet coordinates."""
    a=route[:-1];delta=route[1:]-a
    length=delta.norm(dim=-1).clamp_min(1e-5);unit=delta/length[:,None]
    rel=xy[...,None,:]-a
    along=(rel*unit).sum(-1).clamp_min(0)
    along=torch.minimum(along,length)
    closest=a+along[...,None]*unit
    dist=(xy[...,None,:]-closest).square().sum(-1)
    ix=dist.argmin(-1)
    tangent=unit[ix];nearest=closest.gather(-2,ix[...,None,None].expand(*ix.shape,1,2)).squeeze(-2)
    signed=((xy-nearest)*torch.stack((-tangent[...,1],tangent[...,0]),-1)).sum(-1)
    prefix=torch.cat((length.new_zeros(1),length.cumsum(0)))[:-1]
    s=prefix[ix]+along.gather(-1,ix[...,None]).squeeze(-1)
    return s,signed


def route_frame_at_s(s, route):
    delta=route[1:]-route[:-1]
    length=delta.norm(dim=-1).clamp_min(1e-5)
    prefix=torch.cat((length.new_zeros(1),length.cumsum(0)))
    ix=torch.searchsorted(prefix[1:],s.contiguous()).clamp_max(len(length)-1)
    tangent=delta[ix]/length[ix,None]
    along=(s-prefix[ix]).clamp_min(0)
    along=torch.minimum(along,length[ix])
    return route[ix]+along[...,None]*tangent,tangent


def speed_reference(ts, p):
    if 'speed_phases_v10' in p:
        from execution_completion import phased_speed
        return phased_speed(ts, p)
    v0=p['initial_speed_mps'];target=p.get('target_speed_mps',v0)
    t0=p.get('speed_t0',p.get('t0',0.));duration=p['speed_transition_s']
    if p.get('speed_profile') != 'bounded_jerk':
        return v0+(target-v0)*smooth((ts-t0)/duration)
    ramp=p['profile_jerk_ramp_s'];acc=p['profile_acceleration_mps2']
    t=(ts-t0).clamp(0,duration);sign=1. if target>=v0 else -1.
    delta=sign*acc/(2*ramp)*(t.square()-torch.relu(t-ramp).square()-torch.relu(t-duration+ramp).square())
    return v0+delta


def reference_route_progress(ts, p, initial_s):
    if 'speed_phases_v10' in p:
        from execution_completion import phased_progress
        return phased_progress(ts, p, initial_s)
    v0=p['initial_speed_mps'];duration=p['speed_transition_s']
    t0=p.get('speed_t0',p.get('t0',0.))
    if p.get('speed_profile') == 'bounded_jerk':
        ramp=p['profile_jerk_ramp_s'];acc=p['profile_acceleration_mps2']
        target=p['target_speed_mps'];sign=1. if target>=v0 else -1.
        t=(ts-t0).clamp(0,duration)
        extra=sign*acc/(6*ramp)*(t**3-torch.relu(t-ramp)**3-torch.relu(t-duration+ramp)**3)
        extra=extra+(target-v0)*torch.relu(ts-t0-duration)
        return initial_s+v0*ts+extra
    u=((ts-t0)/duration).clamp(0,1)
    integral=duration*(2.5*u**4-3*u**5+u**6)+torch.relu(ts-t0-duration)
    return initial_s+v0*ts+(p.get('target_speed_mps',v0)-v0)*integral


def boundary_intrusion_offset(ts, ctx, p):
    """Solve hold offset against the boundary at each planned route station.

    Uses the fixed speed template and native map, never future observations.
    A constant amplitude derived only at the initial pose misses lane width
    and boundary changes farther along the route.
    """
    station=reference_route_progress(ts,p,ctx['initial_route_s'])
    center,tangent=route_frame_at_s(station,ctx['route'])
    normal=torch.stack((-tangent[...,1],tangent[...,0]),-1)
    bs,bd=route_coordinates(center,ctx['boundary'])
    _,bt=route_frame_at_s(bs,ctx['boundary'])
    bn=torch.stack((-bt[...,1],bt[...,0]),-1)
    sign=ctx['boundary_source_sign']
    derivative=sign*(normal*bn).sum(-1)
    invalid=derivative.abs()<.5
    if bool(invalid.any()):
        # The boundary-relative amplitude cancels exactly outside the intrusion
        # envelope. Do not reject the unused tail of a receding prediction.
        enter=smooth((ts-p.get('t0',0.))/p['T_in_s'])
        leave=smooth((ts-p.get('t0',0.)-p['T_in_s']-p['T_hold_s'])/p['T_out_s'])
        if bool((invalid & (enter != leave)).any()):
            raise ValueError('Intrusion boundary is not locally parallel to route')
        safe_derivative=torch.where(invalid,torch.ones_like(derivative),derivative)
        support=(tangent*bn).sum(-1).abs()*ctx['extent'][0]/2+(normal*bn).sum(-1).abs()*ctx['extent'][1]/2
        offset=(support-p.get('target_intrusion_m',.35)-sign*bd)/safe_derivative
        return torch.where(invalid,torch.full_like(offset,p.get('initial_offset_m',0.)),offset)
    support=(tangent*bn).sum(-1).abs()*ctx['extent'][0]/2+(normal*bn).sum(-1).abs()*ctx['extent'][1]/2
    return (support-p.get('target_intrusion_m',.35)-sign*bd)/derivative


def sat_separation(p,y,e_p,e_y,size,e_size):
    """Positive iff oriented rectangles are separated; <=0 iff contact/overlap."""
    u=torch.stack((y.cos(),y.sin()),-1);v=torch.stack((-y.sin(),y.cos()),-1)
    eu=torch.stack((e_y.cos(),e_y.sin()),-1);ev=torch.stack((-e_y.sin(),e_y.cos()),-1)
    eu=eu.expand_as(u);ev=ev.expand_as(v)
    axes=torch.stack((u,v,eu,ev),-2)
    delta=e_p-p
    sep=(delta[...,None,:]*axes).sum(-1).abs()
    ra=(u[...,None,:]*axes).sum(-1).abs()*size[0]/2+(v[...,None,:]*axes).sum(-1).abs()*size[1]/2
    rb=(eu[...,None,:]*axes).sum(-1).abs()*e_size[0]/2+(ev[...,None,:]*axes).sum(-1).abs()*e_size[1]/2
    return (sep-ra-rb).amax(-1)


def world_pose(traj, transform):
    p=(traj[...,:2].unsqueeze(-2)*transform[:2,:2]).sum(-1)+transform[:2,2]
    y=traj[...,3]+torch.atan2(transform[1,0],transform[0,0])
    return p,y


def body_outside_penalty(pos, yaw, extent, polygons, prefix_weight=0., lookahead_frames=5):
    """Differentiable sampled full-body distance to polygon union.

    This guides corners/edges as well as the centre; independent polygon-area
    validation remains mandatory and is not replaced by this surrogate.
    """
    grid=pos.new_tensor([(x,y) for x in (-.5,-.25,0.,.25,.5) for y in (-.5,0.,.5)])*extent
    u=torch.stack((yaw.cos(),yaw.sin()),-1)
    v=torch.stack((-yaw.sin(),yaw.cos()),-1)
    points=pos[...,None,:]+grid[:,0,None]*u[...,None,:]+grid[:,1,None]*v[...,None,:]
    distances=[]
    for polygon in polygons:
        # A batch groups polygons with identical vertex counts. Edge and polygon
        # reductions remain separate, preserving the original union distance.
        if polygon.ndim==2:polygon=polygon[None]
        a=polygon[:,:-1];b=polygon[:,1:];delta=b-a
        rel=points[...,None,None,:]-a
        q=((rel*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-8)).clamp(0,1)
        d2=(rel-q[...,None]*delta).square().sum(-1).amin(-1)
        x=points[...,0,None,None];y=points[...,1,None,None]
        dy=b[...,1]-a[...,1];safe_dy=torch.where(dy.abs()>1e-8,dy,torch.ones_like(dy))
        crossing=((a[...,1]>y)!=(b[...,1]>y)) & (x < a[...,0]+(y-a[...,1])*(b[...,0]-a[...,0])/safe_dy)
        inside=crossing.sum(-1).remainder(2).bool()
        distances.append(torch.where(inside,torch.zeros_like(d2),d2))
    outside=torch.cat(distances,dim=-1).amin(-1)
    return (outside.mean((-1,-2)) + prefix_weight*outside[:,:lookahead_frames].amax((-1,-2)))/.01


def loss_terms(traj, native, ctx, r):
    """traj [K,T,6] physical local states/actions; one selected actor only."""
    r=generic_contact_request(r)
    T=min(traj.shape[-2],ctx['ego_pos'].shape[-2])
    tr=traj[:,:T];pos,yaw=world_pose(tr,ctx['transform'])
    ts=ctx['elapsed']+torch.arange(1,T+1,device=tr.device,dtype=tr.dtype)*ctx['dt']
    s,d=route_coordinates(pos,ctx['route'])
    p=r['parameters'];f=r['family'];t0=p.get('t0',0.)
    speed=tr[...,2]
    if r.get('ablation_v26', {}).get('arm') == 'U':
        # U retains every common regularizer and feasibility control below.
        # No mechanism tracking or target-contact attraction is evaluated.
        action=tr.new_zeros(tr.shape[0])
        pressure=tr.new_zeros(tr.shape[0])
        dist=sat_separation(pos,yaw,ctx['ego_pos'][:T],ctx['ego_yaw'][:T],ctx['extent'],ctx['ego_extent'])
    elif generic_contact_only(r):
        # Paper A3: no maneuver tracking, phase masks, crossing timing proxy,
        # qualified-contact objective, completion loss, or seeded tracking.
        action=tr.new_zeros(tr.shape[0])
        dist=sat_separation(pos,yaw,ctx['ego_pos'][:T],ctx['ego_yaw'][:T],ctx['extent'],ctx['ego_extent'])
        window=(ts>=r['interaction_window_s'][0])&(ts<=r['interaction_window_s'][1])
        pressure=tr.new_zeros(tr.shape[0])
        if bool(window.any()):
            closest=dist[:,window].amin(-1)
            pressure=torch.relu(closest+float(r.get('contact_margin_m',0.))).square()/4 if r['mode']=='contact_search' else band(closest,*r['distance_band_m'])
    else:
        speed_t0=p.get('speed_t0',t0)
        vref=speed_reference(ts,p)
        dref=torch.full_like(ts,p.get('initial_offset_m',0.))
        if f in ('C4','C5') and r.get('center_receiving_route',False):
            dref=dref*(1-smooth((ts-t0)/p['centerline_transition_s']))
        if f=='C2':
            dref=dref+p['delta_d_m']*smooth((ts-t0)/p['T_lc_s'])
        elif f in ('C3A','C3B'):
            amp=p['A_m'];tin=p['T_in_s'];th=p['T_hold_s'];tout=p['T_out_s']
            if r.get('boundary_relative_template',False):
                amp=boundary_intrusion_offset(ts,ctx,p)-p.get('initial_offset_m',0.)
            ret=0. if f=='C3A' else p['return_offset_m']-p.get('initial_offset_m',0.)
            dref=dref+amp*smooth((ts-t0)/tin)+(ret-amp)*smooth((ts-t0-tin-th)/tout)
        if r.get('execution_completion_v10'):
            from execution_completion import offset_reference
            dref = offset_reference(ts, ctx, r)
        scales=r.get('tracking_scales',{})
        lateral_scale=float(scales.get('lateral_m',1.));speed_scale=float(scales.get('speed_mps',5.))
        tracking=((d-dref)/lateral_scale).square()+((speed-vref)/speed_scale).square()
        action=tracking.mean(-1)
        if r.get('execution_phase_tracking',False):
            # Receding horizon must honor the absolute scene clock in the prefix
            # actually executed, not only promise a later maneuver in the tail.
            action=action+3*tracking[:,:min(5,T)].mean(-1)
            if f=='C1':
                settled=ts>=speed_t0+p['speed_transition_s']
                if bool(settled.any()):action=action+band(speed[:,settled],vref[settled]-.2,vref[settled]+.2,.2).mean(-1)
        if f=='C1' and r.get('qualified_contact_v2'):
            settled=ts>=speed_t0+p['speed_transition_s']
            action=action+completion_cost(speed,vref,settled)
        if f in ('C4','C5'):
            if r.get('route_progress_guidance',False):
                reference_s=reference_route_progress(ts,p,ctx['initial_route_s'])
                action=action+((s-reference_s)/2.).square().mean(-1)
            # Frozen scene-clock arrival target, evaluated when inside this horizon.
            idx=int(round((p['arrival_s']-ctx['elapsed'])/ctx['dt']))-1
            if 0<=idx<T:action=action+((s[:,idx]-p['arrival_route_s_m'])/5.).square()
        if f in ('C2','C3A','C3B') and not (f == 'C3B' and r.get('execution_completion_v10')):
            end=t0+(p['T_lc_s'] if f=='C2' else p['T_in_s']+p['T_hold_s']+p['T_out_s'])
            mask=(ts>=end)&(ts<=end+.5)
            if bool(mask.any()):
                desired=p.get('initial_offset_m',0.)+(p['delta_d_m'] if f=='C2' else (0. if f=='C3A' else p['return_offset_m']-p.get('initial_offset_m',0.)))
                terminal_scale=float(r.get('terminal_tracking_scale',
                    .1 if r.get('boundary_relative_template',False) else 1.))
                action=action+band(d[:,mask],desired-.15,desired+.15,terminal_scale).mean(-1)
        window=(ts>=r['interaction_window_s'][0])&(ts<=r['interaction_window_s'][1])
        if r.get('contact_phase_window',False):
            if f=='C2':window=window & (ts>=t0+.5*p['T_lc_s'])
            if f in ('C3A','C3B'):
                window=window & (ts>=t0+.65*p['T_in_s']) & (ts<=t0+p['T_in_s']+p['T_hold_s']+.25*p['T_out_s'])

        dist=sat_separation(pos,yaw,ctx['ego_pos'][:T],ctx['ego_yaw'][:T],ctx['extent'],ctx['ego_extent'])
        pressure=tr.new_zeros(tr.shape[0])
        if bool(window.any()):
            closest=dist[:,window].amin(-1)
            pressure=torch.relu(closest+float(r.get('contact_margin_m',0.))).square()/4 if r['mode']=='contact_search' else band(closest,*r['distance_band_m'])
            if r.get('qualified_contact_v2') and r['mode']=='contact_search':
                pressure=contact_cost(dist[:,window],ts[window]-ctx['elapsed'],r['qualified_contact_v2'])
        if f in ('C3A','C3B') and ctx.get('boundary') is not None:
            # Full oriented body corners relative to the actual shared boundary.
            u=torch.stack((yaw.cos(),yaw.sin()),-1)*ctx['extent'][0]/2
            v=torch.stack((-yaw.sin(),yaw.cos()),-1)*ctx['extent'][1]/2
            corners=torch.stack((pos+u+v,pos+u-v,pos-u-v,pos-u+v),-2)
            _,side=route_coordinates(corners,ctx['boundary'])
            outward=-ctx['boundary_source_sign']*side
            depth=outward.amax(-1)
            active=(ts>=t0)&(ts<=t0+p['T_in_s']+p['T_hold_s'])
            if bool(active.any()):
                if r.get('phase_specific_intrusion',False):
                    hold=(ts>=t0+p['T_in_s'])&(ts<=t0+p['T_in_s']+p['T_hold_s'])
                    if bool(hold.any()):
                        action=action+band(depth[:,hold],p.get('z_min_m',.2),p.get('z_max_m',float(ctx['extent'][1])/2),.2 if r.get('boundary_relative_template',False) else 1.).mean(-1)
                else:
                    action=action+band(depth[:,active].amax(-1),p.get('z_min_m',.2),p.get('z_max_m',float(ctx['extent'][1])/2))
                # All four corners across the boundary means a full lane entry.
                action=action+torch.relu(outward[:,active].amin(-1)).square().mean(-1)
            returned=ts>=t0+p['T_in_s']+p['T_hold_s']+p['T_out_s']
            if bool(returned.any()):action=action+torch.relu(depth[:,returned]-.03).square().mean(-1)
        if f=='C4' and not r.get('ablation_generic_collision',False) and ctx.get('crossing_point') is not None and bool(window.any()):
            # Body-sized local encounter timing proxy, in addition to exact-pair SAT.
            cp=ctx['crossing_point'];da=(pos-cp).square().sum(-1).sqrt()
            de=(ctx['ego_pos'][:T]-cp).square().sum(-1).sqrt()
            radius=(ctx['extent'].norm()+ctx['ego_extent'].norm())/2
            timing=(torch.relu(da-radius).square()+torch.relu(de-radius).square())[:,window].amin(-1)
            pressure=pressure+.1*timing
    prior=(tr[...,-2:]-native[:,:T,-2:]).square().mean((-1,-2))
    lim=r['engineering_limits'];acc=tr[...,4];omega=tr[...,5];lat=speed*omega
    constraint=band(speed,0,lim['max_speed_mps'],10).mean(-1)+band(acc,-lim['max_braking_mps2'],lim['max_acceleration_mps2'],3).mean(-1)
    constraint=constraint+band(lat,-lim['max_lateral_mps2'],lim['max_lateral_mps2'],3).mean(-1)
    if T>1:constraint=constraint+band(torch.diff(acc,dim=-1)/ctx['dt'],-lim['max_jerk_mps3'],lim['max_jerk_mps3'],5).mean(-1)
    # Include actually observed motion across the replan boundary. Internal
    # action jerk alone misses the first executed step after a fresh FM draw.
    if r.get('executed_jerk_guidance',False) and ctx.get('history_local') is not None:
        hist=ctx['history_local'].expand(tr.shape[0],-1,-1)
        joined=torch.cat((hist,tr[...,:2]),dim=-2)
        velocity=torch.diff(joined,dim=-2)/ctx['dt']
        acceleration=torch.diff(velocity,dim=-2)/ctx['dt']
        jerk=torch.diff(acceleration,dim=-2)/ctx['dt']
        jnorm=jerk.norm(dim=-1)
        # 8 is an optimization margin below the unchanged 10 m/s^3 acceptance.
        margin=min(8.,lim['max_jerk_mps3'])
        penalty=band(jnorm,0,margin,5)
        constraint=constraint+penalty.mean(-1)+penalty.amax(-1)+float(r.get('executed_jerk_prefix_weight',20.))*penalty[:,:5].mean(-1)
        if r.get('executed_acceleration_guidance',False):
            # Apply conservative optimization margins to the same position
            # differences used by the independent execution checker. Native
            # action acceleration is not identical at replan boundaries.
            angle=torch.cat((tr.new_zeros((tr.shape[0],1)),tr[:,:-1,3]),-1)
            unit=torch.stack((angle.cos(),angle.sin()),-1)
            normal=torch.stack((-angle.sin(),angle.cos()),-1)
            longitudinal=(acceleration[:,1:]*unit).sum(-1)
            lateral=(acceleration[:,1:]*normal).sum(-1)
            ap=band(longitudinal,-.9*lim['max_braking_mps2'],
                .9*lim['max_acceleration_mps2'])
            ap=ap+band(lateral,-.9*lim['max_lateral_mps2'],
                .9*lim['max_lateral_mps2'])
            constraint=constraint+ap.mean(-1)+ap.amax(-1)+5*ap[:,:5].mean(-1)
    # Route width is a differentiable proxy only; exact polygon check below.
    constraint=constraint+band(d,r['route_offset_bounds_m'][0],r['route_offset_bounds_m'][1]).mean(-1)
    if r.get('body_corridor_guidance',False):
        constraint=constraint+body_outside_penalty(pos,yaw,ctx['extent'],ctx['corridor_polygons'],float(r.get('corridor_execution_prefix_weight',0.)),int(r.get('corridor_lookahead_frames',5)))
    if r.get('execution_completion_v10') and r.get('ablation_v26', {}).get('arm') != 'U':
        from execution_completion import completion_loss
        action = action + completion_loss(tr, ctx, r)
    terms=dict(action=action,pressure=pressure,prior=prior,constraint=constraint)
    if r.get('conditional_distribution_guidance', {}).get('enabled'):
        from conditional_distribution import penalty as conditional_distribution_penalty
        terms['distribution'] = conditional_distribution_penalty(tr, r)
    total=sum(r['weights'][k]*v for k,v in terms.items())
    if r.get('deviation_guidance') and ctx.get('unguided_anchor') is not None:
        # A frozen predicted clean endpoint captured before the first guide
        # call in this replan. No recorded future or future ego labels enter.
        deviation,_,_=deviation_cost(tr,ctx['unguided_anchor'][:,:T],r['deviation_guidance'])
        total=total+deviation

    if bg_enabled(r):
        background_cost=bg_penalty(pos,yaw,ctx,r)
        total=total+r['non_target_avoidance']['weight']*background_cost
        terms['non_target_avoidance']=background_cost
    return total,terms,dist


class FiveClassGuide(PaperExecutionMixin):
    def __init__(self, request, audit_path):
        request.update(generic_contact_request(request))
        self.request=request;self.audit_path=Path(audit_path);self.context=None;self.calls=0
        self.initialize_contract()

    def prepare(self, obs, ego_action, target, ego, elapsed, dt):
        self.request.update(generic_contact_request(self.request))
        self.begin_replan()
        if target==ego:raise ValueError('Target must not be ego')
        tf=obs['world_from_agent'].double();device=tf.device;dtype=tf.dtype
        ep=torch.as_tensor(ego_action.positions,device=device,dtype=dtype)[0]
        ey=torch.as_tensor(ego_action.yaws,device=device,dtype=dtype)[0].squeeze(-1)
        self.context=dict(transform=tf[target].detach(),ego_pos=((ep.unsqueeze(-2)*tf[ego,:2,:2]).sum(-1)+tf[ego,:2,2]).detach(),
            ego_yaw=(ey+torch.atan2(tf[ego,1,0],tf[ego,0,0])).detach(),
            extent=obs['extent'][target,:2].detach(),ego_extent=obs['extent'][ego,:2].detach(),
            route=torch.as_tensor(self.request['route_world_xy'],device=device,dtype=dtype),elapsed=float(elapsed),dt=float(dt),target=target)

        if bg_enabled(self.request):
            self.context['causal_background']=prepare_background(obs,target,ego,dt)
            self.context['background_uncertainty_cap_m']=self.request['non_target_avoidance'].get('uncertainty_cap_m')
        if not hasattr(self,'_initial_route_s'):
            self._initial_route_s=float(route_coordinates(tf[target,:2,2],self.context['route'])[0])
        self.context['initial_route_s']=self._initial_route_s
        if self.request.get('executed_jerk_guidance',False):
            history=obs['history_positions'][target]
            available=obs['history_availabilities'][target].bool()
            if history.shape[0]<3 or not bool(available[-3:].all()):
                raise ValueError('Three observed positions required for replan jerk')
            if float(history[-1].norm())>.01:
                raise ValueError('Expected chronological local history ending at current origin')
            self.context['history_local']=history[-3:].detach()
        groups={}
        for polygon in self.request['corridor_world_polygons']:
            groups.setdefault(len(polygon),[]).append(polygon)
        self.context['corridor_polygons']=[torch.as_tensor(np.asarray(group[i:i+32]),device=device,dtype=dtype)
            for group in groups.values() for i in range(0,len(group),32)]
        evidence=self.request['geometry_evidence']
        if evidence.get('shared_boundary_wkt'):
            from shapely import wkt
            boundary=wkt.loads(evidence['shared_boundary_wkt'])
            if boundary.geom_type=='MultiLineString':
                from shapely.ops import linemerge
                boundary=linemerge(boundary)
            if boundary.geom_type!='LineString':raise ValueError('Shared boundary must be a line')
            self.context['boundary']=torch.as_tensor(np.asarray(boundary.coords),device=device,dtype=dtype)
            if not hasattr(self,'_boundary_source_sign'):
                _,d=route_coordinates(tf[target,:2,2],self.context['boundary'])
                if abs(float(d))<1e-4:raise ValueError('Initial center lies on ambiguous boundary')
                self._boundary_source_sign=float(torch.sign(d))
            self.context['boundary_source_sign']=self._boundary_source_sign
        if evidence.get('crossing_point'):
            self.context['crossing_point']=torch.as_tensor(evidence['crossing_point'],device=device,dtype=dtype)

        self._request_contract.check(self.request)

    def project(self, model, endpoint, aux_info, num_samp, flow_step):
        r=generic_contact_request(self.request)
        self._request_contract.check(self.request)
        validate_generation(r, model, endpoint, num_samp, model.dt)
        if flow_step<model.n_flow_steps-r.get('last_flow_steps',1):return endpoint
        if self.context is None:raise RuntimeError('No actual ego planner reference supplied')
        if endpoint.ndim!=4 or endpoint.shape[0]!=num_samp:raise ValueError('One scene [K,M,T,2] required')
        self.context=double_context(self.context)
        ctx=self.context;i=ctx['target'];state=aux_info['curr_states'][:,i].double()
        model=PreciseDynamics(model)
        if float(state[...,:2].abs().max())>.01 or float(state[...,3].abs().max())>.01:
            raise ValueError('Five-class hook expects agent-centric current states')
        base=endpoint[:,i].detach().double();x=base.clone();started=time.perf_counter()
        with torch.enable_grad():
            native=model.convert_action_to_state_and_action(base,state,descaled_output=True).detach()
            if r.get('deviation_guidance') and 'unguided_anchor' not in ctx:
                ctx['unguided_anchor']=native.detach().clone()

            def score(z):
                tr=model.convert_action_to_state_and_action(z,state,descaled_output=True)
                total,terms,dist=loss_terms(tr,native,ctx,r)
                return total,terms,dist,tr
            initial,_,_,_=score(x);accepted=0;gradnorm=0.
            solver=r.get('trajectory_solver','gradient')
            solver_evaluations=0
            if solver=='lbfgs' and r.get('gradient_steps',1)>0:
                # Optimize the sampled action tensor only. The model remains
                # frozen. Quasi-Newton scaling helps when boundary jerk and
                # distant lateral tracking have very different curvature.
                trust=r.get('normalized_action_trust',.25)
                seed=warm_start(model,base,state,ctx,r,speed_reference,route_coordinates,route_frame_at_s) if r.get('qualified_contact_v2') and r.get('ablation_v26',{}).get('arm') != 'U' and flow_step==model.n_flow_steps-1 else base
                if r.get('execution_completion_v10') and r.get('ablation_v26',{}).get('arm') != 'U' and flow_step==model.n_flow_steps-1:
                    from execution_completion import tracking_seed
                    seed = tracking_seed(model, base, state, ctx, r)
                    ctx['completion_seed_slot_v10'] = 0
                delta=(seed-base).clamp(-trust,trust).detach().requires_grad_(True)
                opt=torch.optim.LBFGS([delta],lr=1.,max_iter=r['gradient_steps'],
                    max_eval=3*r['gradient_steps'],history_size=10,
                    tolerance_grad=1e-5,tolerance_change=1e-7,
                    line_search_fn='strong_wolfe')
                best=base.clone();bestv=initial.detach().clone()
                def closure():
                    nonlocal best,bestv,accepted,gradnorm,solver_evaluations
                    opt.zero_grad(set_to_none=True)
                    trial=base+delta.clamp(-trust,trust)
                    value,_,_,_=score(trial)
                    grad=torch.autograd.grad(value.sum(),delta)[0]
                    if not torch.isfinite(grad).all():
                        raise FloatingPointError('Nonfinite trajectory solver gradient')
                    delta.grad=grad
                    improve=torch.isfinite(value)&(value.detach()<bestv)
                    best=torch.where(improve[:,None,None],trial.detach(),best)
                    bestv=torch.where(improve,value.detach(),bestv)
                    accepted+=int(improve.sum());solver_evaluations+=1
                    gradnorm=float(grad.norm().detach())
                    return value.sum()
                opt.step(closure)
                x=best
            for _ in range(r.get('gradient_steps',1) if solver=='gradient' else 0):
                x=x.detach().requires_grad_(True);value,_,_,_=score(x)
                grad=torch.autograd.grad(value.sum(),x)[0]
                if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite guide gradient')
                gradnorm=float(grad.norm().detach());grad=grad/grad.square().mean((-1,-2),keepdim=True).sqrt().clamp_min(1.)
                # Per-candidate backtracking; never accept an increasing loss.
                best=x.detach();bestv=value.detach()
                for factor in (1.,.5,.25,.125,.0625,.015625,.001):
                    trial=(x-r.get('step_size',.05)*factor*grad).detach()
                    trust=r.get('normalized_action_trust',.25)
                    trial=base+(trial-base).clamp(-trust,trust)
                    newval,_,_,_=score(trial)
                    improve=torch.isfinite(newval)&(newval<bestv)
                    best=torch.where(improve[:,None,None],trial,best)
                    bestv=torch.where(improve,newval.detach(),bestv)
                    accepted+=int(improve.sum())
                x=best
            repair_audit=None
            if r.get('prefix_feasibility_repair_steps',0)>0 and r.get('gradient_steps',0)>0 and flow_step==model.n_flow_steps-1:
                repair_start=x.detach().clone()
                trust=r.get('normalized_action_trust',.25)
                def repair_score(z):
                    physical=model.convert_action_to_state_and_action(z,state,descaled_output=True)
                    cost=prefix_cost(physical,ctx,r,body_outside_penalty)
                    if r.get('qualified_contact_v2'):
                        cost=cost*float(r['qualified_contact_v2']['prefix_safety_priority'])
                    if bg_enabled(r):
                        pp,yy=world_pose(physical[:,:5],ctx['transform'])
                        cost=cost+r['non_target_avoidance']['weight']*bg_penalty(pp,yy,ctx,r)
                    return cost
                initial_cost=repair_score(repair_start).detach()
                best_repair=repair_start.clone();best_cost=initial_cost.clone()
                repair_calls=0
                if bool((initial_cost>1e-6).any()):
                    delta_repair=torch.zeros_like(x,requires_grad=True)
                    optimizer=torch.optim.LBFGS([delta_repair],lr=1.,max_iter=r['prefix_feasibility_repair_steps'],max_eval=3*r['prefix_feasibility_repair_steps'],history_size=10,line_search_fn='strong_wolfe',tolerance_change=1e-9)
                    def repair_closure():
                        nonlocal best_repair,best_cost,repair_calls
                        optimizer.zero_grad(set_to_none=True)
                        trial=base+(repair_start+delta_repair-base).clamp(-trust,trust)
                        costs=repair_score(trial)
                        value=200*costs+1e-3*(trial-repair_start).square().mean((-1,-2))
                        grad=torch.autograd.grad(value.sum(),delta_repair)[0]
                        if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite feasibility repair gradient')
                        delta_repair.grad=grad
                        improve=torch.isfinite(costs)&(costs.detach()<best_cost)
                        best_repair=torch.where(improve[:,None,None],trial.detach(),best_repair)
                        best_cost=torch.where(improve,costs.detach(),best_cost)
                        repair_calls+=1
                        return value.sum()
                    optimizer.step(repair_closure)
                    x=best_repair
                repair_audit=dict(initial_prefix_cost=initial_cost.cpu().tolist(),final_prefix_cost=best_cost.cpu().tolist(),evaluations=repair_calls,scope='inference_action_optimization_before_execution_not_posthoc_trajectory_smoothing')
            final,terms,dist,tr=score(x)
            if flow_step==model.n_flow_steps-1:
                self.retain_candidates(native, tr)
        if ctx['elapsed']==0 and flow_step==model.n_flow_steps-1:
            torch.save(dict(base=base.detach(),candidate=x.detach(),state=state.detach(),context=ctx,request=r,
                dynamics=dict(acce_bound=model.dyn.acce_bound,vbound=model.dyn.vbound,max_steer=model.dyn.max_steer,max_yawvel=model.dyn.max_yawvel),
                add_coeffs=model.add_coeffs,div_coeffs=model.div_coeffs),self.audit_path.with_name('guide_debug_first.pt'))
        result=endpoint.clone();result[:,i]=x.detach()
        other=[j for j in range(endpoint.shape[1]) if j!=i]
        unchanged=torch.equal(result[:,other],endpoint[:,other])
        if not unchanged:raise AssertionError('Non-target FM rows were modified')
        audit=dict(family=r['family'],request_sha256=r.get('_sha256'),frozen_identity_sha256=self._request_contract.sha256,elapsed_s=ctx['elapsed'],flow_step=flow_step,
            initial_loss=initial.detach().cpu().tolist(),final_loss=final.detach().cpu().tolist(),
            terms={k:v.detach().cpu().tolist() for k,v in terms.items()},gradient_norm=gradnorm,
            accepted_candidate_updates=accepted,target_only=True,model_parameters_updated=False,
            trajectory_solver=solver,trajectory_solver_evaluations=solver_evaluations,guidance_dtype='float64',native_equations_unchanged=True,
            geometry_status=r['geometry_status'],physical_validity=r['capability_status'],
            seconds=time.perf_counter()-started,normalized_action_delta_max=float((x-base).abs().max()),
            obstacle_modes_used=ctx.get('obstacle_modes_used',False),
            surrogate_min_separation=dist.detach().amin(-1).cpu().tolist(),prefix_feasibility_repair=repair_audit,
            deviation_anchor_source='first_unguided_predicted_clean_endpoint_same_replan' if ctx.get('unguided_anchor') is not None else None)
        if r.get('deviation_guidance') and ctx.get('unguided_anchor') is not None:
            _,anchor_ade,anchor_fde=deviation_cost(tr,ctx['unguided_anchor'],r['deviation_guidance'])
            audit.update(anchor_ADE_m=anchor_ade.detach().cpu().tolist(),anchor_FDE_m=anchor_fde.detach().cpu().tolist())
        self.audit_path.parent.mkdir(parents=True,exist_ok=True)
        with self.audit_path.open('a') as f:f.write(json.dumps(audit,allow_nan=False)+'\n')
        self.calls+=1
        return result


def inspect_plan(action, obs, target, ego, request):
    """Independent exact polygon checks on the planned samples (NOT swept CCD)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    def npy(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    tf=npy(obs['world_from_agent']);sizes=npy(obs['extent']);pp=npy(action.positions);yy=npy(action.yaws)[...,0]
    def boxes(i):
        p=pp[i]@tf[i,:2,:2].T+tf[i,:2,2];yaw=yy[i]+np.arctan2(tf[i,1,0],tf[i,0,0]);l,w=sizes[i,:2]
        result=[]
        for xy,a in zip(p,yaw):
            if not np.isfinite(xy).all() or not np.isfinite(a):
                result.append(None);continue
            u=np.array([np.cos(a),np.sin(a)])*l/2;v=np.array([-np.sin(a),np.cos(a)])*w/2
            result.append(Polygon([xy+u+v,xy+u-v,xy-u-v,xy-u+v]))
        return result
    aa,ee=boxes(target),boxes(ego);corr=unary_union([Polygon(p) for p in request['corridor_world_polygons']])
    if any(b is None for b in aa+ee):raise ValueError('Controlled planned pose is absent')
    coverage=[b.intersection(corr).area/max(b.area,1e-8) for b in aa]
    contact=[a.intersects(e) for a,e in zip(aa,ee)]
    distance=[a.distance(e) for a,e in zip(aa,ee)]
    other_contacts=[]
    for j in range(len(pp)):
        if j not in (target,ego):other_contacts.append(any(b is not None and a.intersects(b) for a,b in zip(aa,boxes(j))))
    return dict(planned_sampled_contact=any(contact),planned_min_body_distance_m=min(distance),
        planned_corridor_coverage_min=min(coverage),planned_other_contact=any(other_contacts),
        physical_validity=request['capability_status'],continuous_collision_validated=False,
        mechanism_complete_validated=False,scope='planned_sampled_geometry_not_executed_metrics')
