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


def load_request(path, record):
    raw = Path(path).read_bytes()
    r = json.loads(raw)
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
    w=r.get('interaction_window_s', [])
    if len(w)!=2 or not 0<=w[0]<w[1]:raise ValueError('Invalid fixed interaction window')
    if r['mode']=='near_conflict' and not 0<r.get('distance_band_m',[0,0])[0]<=r['distance_band_m'][1]:
        raise ValueError('Positive near-conflict distance band required')
    for name in ('action','pressure','prior','constraint'):
        if not math.isfinite(r['weights'][name]) or r['weights'][name]<0:raise ValueError('Invalid weight')
    if not 0<=r.get('gradient_steps',1)<=10 or not 0<=r.get('step_size',.05)<=1:
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
    for value in r.get('tracking_scales',{}).values():
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('Tracking scales must be finite and positive')
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


def reference_route_progress(ts, p, initial_s):
    v0=p['initial_speed_mps'];duration=p['speed_transition_s']
    t0=p.get('speed_t0',p.get('t0',0.))
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
    if bool((derivative.abs()<.5).any()):
        raise ValueError('Intrusion boundary is not locally parallel to route')
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


def body_outside_penalty(pos, yaw, extent, polygons):
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
        a=polygon[:-1];b=polygon[1:];delta=b-a
        rel=points[...,None,:]-a
        q=((rel*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-8)).clamp(0,1)
        d2=(rel-q[...,None]*delta).square().sum(-1).amin(-1)
        x=points[...,0,None];y=points[...,1,None]
        dy=b[:,1]-a[:,1];safe_dy=torch.where(dy.abs()>1e-8,dy,torch.ones_like(dy))
        crossing=((a[:,1]>y)!=(b[:,1]>y)) & (x < a[:,0]+(y-a[:,1])*(b[:,0]-a[:,0])/safe_dy)
        inside=crossing.sum(-1).remainder(2).bool()
        distances.append(torch.where(inside,torch.zeros_like(d2),d2))
    outside=torch.stack(distances).amin(0)
    return outside.mean((-1,-2))/.01


def loss_terms(traj, native, ctx, r):
    """traj [K,T,6] physical local states/actions; one selected actor only."""
    T=min(traj.shape[-2],ctx['ego_pos'].shape[-2])
    tr=traj[:,:T];pos,yaw=world_pose(tr,ctx['transform'])
    ts=ctx['elapsed']+torch.arange(1,T+1,device=tr.device,dtype=tr.dtype)*ctx['dt']
    s,d=route_coordinates(pos,ctx['route'])
    p=r['parameters'];f=r['family'];t0=p.get('t0',0.)
    speed=tr[...,2];v0=p['initial_speed_mps']
    speed_t0=p.get('speed_t0',t0)
    vref=v0+(p.get('target_speed_mps',v0)-v0)*smooth((ts-speed_t0)/p.get('speed_transition_s',2.))
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
    if f in ('C4','C5'):
        if r.get('route_progress_guidance',False):
            duration=p['speed_transition_s'];u=((ts-speed_t0)/duration).clamp(0,1)
            integral=duration*(2.5*u**4-3*u**5+u**6)+torch.relu(ts-speed_t0-duration)
            reference_s=ctx['initial_route_s']+v0*ts+(p.get('target_speed_mps',v0)-v0)*integral
            action=action+((s-reference_s)/2.).square().mean(-1)
        # Frozen scene-clock arrival target, evaluated when inside this horizon.
        idx=int(round((p['arrival_s']-ctx['elapsed'])/ctx['dt']))-1
        if 0<=idx<T:action=action+((s[:,idx]-p['arrival_route_s_m'])/5.).square()
    if f in ('C2','C3A','C3B'):
        end=t0+(p['T_lc_s'] if f=='C2' else p['T_in_s']+p['T_hold_s']+p['T_out_s'])
        mask=(ts>=end)&(ts<=end+.5)
        if bool(mask.any()):
            desired=p.get('initial_offset_m',0.)+(p['delta_d_m'] if f=='C2' else (0. if f=='C3A' else p['return_offset_m']-p.get('initial_offset_m',0.)))
            terminal_scale=float(r.get('terminal_tracking_scale',
                .1 if r.get('boundary_relative_template',False) else 1.))
            action=action+band(d[:,mask],desired-.15,desired+.15,terminal_scale).mean(-1)
    window=(ts>=r['interaction_window_s'][0])&(ts<=r['interaction_window_s'][1])
    dist=sat_separation(pos,yaw,ctx['ego_pos'][:T],ctx['ego_yaw'][:T],ctx['extent'],ctx['ego_extent'])
    pressure=tr.new_zeros(tr.shape[0])
    if bool(window.any()):
        closest=dist[:,window].amin(-1)
        pressure=torch.relu(closest).square()/4 if r['mode']=='contact_search' else band(closest,*r['distance_band_m'])
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
    if f=='C4' and ctx.get('crossing_point') is not None and bool(window.any()):
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
        constraint=constraint+penalty.mean(-1)+penalty.amax(-1)+5*penalty[:,:5].mean(-1)
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
        constraint=constraint+body_outside_penalty(pos,yaw,ctx['extent'],ctx['corridor_polygons'])
    terms=dict(action=action,pressure=pressure,prior=prior,constraint=constraint)
    total=sum(r['weights'][k]*v for k,v in terms.items())
    return total,terms,dist


class FiveClassGuide:
    def __init__(self, request, audit_path):
        self.request=request;self.audit_path=Path(audit_path);self.context=None;self.calls=0

    def prepare(self, obs, ego_action, target, ego, elapsed, dt):
        if target==ego:raise ValueError('Target must not be ego')
        tf=obs['world_from_agent'];device=tf.device;dtype=tf.dtype
        ep=torch.as_tensor(ego_action.positions,device=device,dtype=dtype)[0]
        ey=torch.as_tensor(ego_action.yaws,device=device,dtype=dtype)[0].squeeze(-1)
        self.context=dict(transform=tf[target].detach(),ego_pos=((ep.unsqueeze(-2)*tf[ego,:2,:2]).sum(-1)+tf[ego,:2,2]).detach(),
            ego_yaw=(ey+torch.atan2(tf[ego,1,0],tf[ego,0,0])).detach(),
            extent=obs['extent'][target,:2].detach(),ego_extent=obs['extent'][ego,:2].detach(),
            route=torch.as_tensor(self.request['route_world_xy'],device=device,dtype=dtype),elapsed=float(elapsed),dt=float(dt),target=target)

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
        self.context['corridor_polygons']=[torch.as_tensor(p,device=device,dtype=dtype) for p in self.request['corridor_world_polygons']]
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

    def project(self, model, endpoint, aux_info, num_samp, flow_step):
        r=self.request
        if flow_step<model.n_flow_steps-r.get('last_flow_steps',1):return endpoint
        if self.context is None:raise RuntimeError('No actual ego planner reference supplied')
        if endpoint.ndim!=4 or endpoint.shape[0]!=num_samp:raise ValueError('One scene [K,M,T,2] required')
        ctx=self.context;i=ctx['target'];state=aux_info['curr_states'][:,i]
        if float(state[...,:2].abs().max())>.01 or float(state[...,3].abs().max())>.01:
            raise ValueError('Five-class hook expects agent-centric current states')
        base=endpoint[:,i].detach();x=base.clone();started=time.perf_counter()
        with torch.enable_grad():
            native=model.convert_action_to_state_and_action(base,state,descaled_output=True).detach()
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
                delta=torch.zeros_like(base,requires_grad=True)
                trust=r.get('normalized_action_trust',.25)
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
            final,terms,dist,tr=score(x)
        result=endpoint.clone();result[:,i]=x.detach()
        other=[j for j in range(endpoint.shape[1]) if j!=i]
        unchanged=torch.equal(result[:,other],endpoint[:,other])
        if not unchanged:raise AssertionError('Non-target FM rows were modified')
        audit=dict(family=r['family'],request_sha256=r.get('_sha256'),elapsed_s=ctx['elapsed'],flow_step=flow_step,
            initial_loss=initial.detach().cpu().tolist(),final_loss=final.detach().cpu().tolist(),
            terms={k:v.detach().cpu().tolist() for k,v in terms.items()},gradient_norm=gradnorm,
            accepted_candidate_updates=accepted,target_only=True,model_parameters_updated=False,
            trajectory_solver=solver,trajectory_solver_evaluations=solver_evaluations,
            geometry_status=r['geometry_status'],physical_validity=r['capability_status'],
            seconds=time.perf_counter()-started,normalized_action_delta_max=float((x-base).abs().max()),
            surrogate_min_separation=dist.detach().amin(-1).cpu().tolist())
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
            u=np.array([np.cos(a),np.sin(a)])*l/2;v=np.array([-np.sin(a),np.cos(a)])*w/2
            result.append(Polygon([xy+u+v,xy+u-v,xy-u-v,xy-u+v]))
        return result
    aa,ee=boxes(target),boxes(ego);corr=unary_union([Polygon(p) for p in request['corridor_world_polygons']])
    coverage=[b.intersection(corr).area/max(b.area,1e-8) for b in aa]
    contact=[a.intersects(e) for a,e in zip(aa,ee)]
    distance=[a.distance(e) for a,e in zip(aa,ee)]
    other_contacts=[]
    for j in range(len(pp)):
        if j not in (target,ego):other_contacts.append(any(a.intersects(b) for a,b in zip(aa,boxes(j))))
    return dict(planned_sampled_contact=any(contact),planned_min_body_distance_m=min(distance),
        planned_corridor_coverage_min=min(coverage),planned_other_contact=any(other_contacts),
        physical_validity=request['capability_status'],continuous_collision_validated=False,
        mechanism_complete_validated=False,scope='planned_sampled_geometry_not_executed_metrics')
