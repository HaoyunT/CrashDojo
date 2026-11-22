"""Observed-history background occupancy, never logged future trajectories.

The velocity-spread envelope is an uncalibrated planning heuristic. It is not a
guarantee of future occupancy or a replacement for executed-contact validation.
"""
import math
import torch


def enabled(request):
    return bool(request.get('non_target_avoidance', {}).get('enabled', False))


def validate_config(request):
    if not enabled(request):
        return
    cfg = request['non_target_avoidance']
    assert cfg['forecast'] == 'observed_cv_with_velocity_spread_v1'
    for name, bound in [('weight',100000.), ('margin_m',.5), ('prefix_margin_m',.5)]:
        assert math.isfinite(cfg[name]) and 0 <= cfg[name] <= bound
    if 'uncertainty_cap_m' in cfg:
        assert math.isfinite(cfg['uncertainty_cap_m']) and 0<cfg['uncertainty_cap_m']<=2.


def prepare_background(obs, target, ego, dt):
    """Access only transform, history, availability, extent; detached forecasts."""
    tf = obs['world_from_agent'].detach()
    hist = obs['history_positions'].detach()
    available = obs['history_availabilities'].bool()
    size = obs['extent'].detach()
    assert target != ego and dt > 0
    candidates = [j for j in range(len(tf)) if j not in (target,ego)]
    absent_indices = [j for j in candidates if not bool(available[j,-1])]
    insufficient_history_indices = [j for j in candidates
                                    if bool(available[j,-1]) and
                                    (hist.shape[1] < 2 or not bool(available[j,-2:].all()))]
    indices = [j for j in candidates if bool(available[j,-1]) and j not in insufficient_history_indices]
    positions, velocities, spread, headings, sizes = [], [], [], [], []
    for j in indices:
        n = 2
        while n < min(6,hist.shape[1]) and bool(available[j,-n-1]):
            n += 1
        rotation = tf[j,:2,:2]
        current = rotation @ hist[j,-1] + tf[j,:2,2]
        last_velocity = rotation @ ((hist[j,-1]-hist[j,-2])/dt)
        mean_velocity = rotation @ ((hist[j,-1]-hist[j,-n])/((n-1)*dt))
        positions.append(current);velocities.append(last_velocity)
        spread.append((last_velocity-mean_velocity).norm())
        headings.append(torch.atan2(tf[j,1,0],tf[j,0,0]))
        sizes.append(size[j,:2])
    ref = tf
    stack = lambda values, shape: torch.stack(values).detach() if values else ref.new_empty(shape)
    result = dict(indices=indices,absent_observed_indices=absent_indices,
        insufficient_history_indices=insufficient_history_indices,
        position=stack(positions,(0,2)),velocity=stack(velocities,(0,2)),
        spread_mps=stack(spread,(0,)),yaw=stack(headings,(0,)),extent=stack(sizes,(0,2)),dt=float(dt))
    assert all(torch.isfinite(result[k]).all() for k in ['position','velocity','spread_mps','yaw','extent'])
    assert bool((result['extent']>0).all())
    return result


def separations(position, yaw, extent, background, times, uncertainty_cap_m=None):
    """[K,T,J] signed SAT separation minus the observed velocity-spread radius."""
    p = position[:,:,None,:];y = yaw[:,:,None]
    bp = background['position'][None,None] + times[None,:,None,None]*background['velocity'][None,None]
    by = background['yaw'][None,None,:]
    u = torch.stack((y.cos(),y.sin()),-1)
    v = torch.stack((-y.sin(),y.cos()),-1)
    bu = torch.stack((by.cos(),by.sin()),-1).expand(position.shape[0],position.shape[1],-1,2)
    bv = torch.stack((-by.sin(),by.cos()),-1).expand_as(bu)
    u = u.expand_as(bu);v = v.expand_as(bu)
    axes = torch.stack((u,v,bu,bv),-2)
    delta = bp-p
    center = (delta[...,None,:]*axes).sum(-1).abs()
    ra = (u[...,None,:]*axes).sum(-1).abs()*extent[0]/2 + (v[...,None,:]*axes).sum(-1).abs()*extent[1]/2
    sz = background['extent'][None,None,:,None,:]
    rb = (bu[...,None,:]*axes).sum(-1).abs()*sz[...,0]/2 + (bv[...,None,:]*axes).sum(-1).abs()*sz[...,1]/2
    radius=times[None,:,None]*background['spread_mps'][None,None,:]
    if uncertainty_cap_m is not None:
        radius=radius.clamp_max(uncertainty_cap_m)
    return (center-ra-rb).amax(-1)-radius


def penalty(position, yaw, ctx, request):
    bg = ctx.get('causal_background')
    if not enabled(request) or bg is None or not bg['indices']:
        return position.new_zeros(position.shape[0])
    times = torch.arange(1,position.shape[1]+1,device=position.device,dtype=position.dtype)*ctx['dt']
    separation = separations(position,yaw,ctx['extent'],bg,times,ctx.get('background_uncertainty_cap_m'))
    intrusion = torch.relu(request['non_target_avoidance']['margin_m']-separation).square().amax(-1)
    return intrusion.mean(-1)+intrusion.amax(-1)+4*intrusion[:,:min(5,intrusion.shape[1])].mean(-1)


def prefix_clearance(local_positions, local_yaw, ctx):
    """Adaptive linear interpolation check of five proposed execution steps.

    Sampling bounds match the existing geometric audit (5cm, .02rad). This is
    still a forecast check; final executed geometry remains independently judged.
    """
    bg = ctx['causal_background'];tf = ctx['transform'];dt = ctx['dt']
    local_positions=local_positions.to(tf.dtype);local_yaw=local_yaw.to(tf.dtype)
    K = local_positions.shape[0];T = min(5,local_positions.shape[1])
    if not bg['indices']:
        return local_positions.new_full((K,),1e6)
    with torch.no_grad():
        local = torch.cat((local_positions.new_zeros((K,1,2)),local_positions[:,:T]),1)
        angles = torch.cat((local_yaw.new_zeros((K,1)),local_yaw[:,:T]),1)
        pos = local@tf[:2,:2].T+tf[:2,2]
        angles = angles+torch.atan2(tf[1,0],tf[0,0])
        sample_pos=[pos[:,0]];sample_yaw=[angles[:,0]];sample_t=[0.]
        bg_step = float(bg['velocity'].norm(dim=-1).max()*dt)
        for t in range(T):
            turn = torch.atan2(torch.sin(angles[:,t+1]-angles[:,t]),torch.cos(angles[:,t+1]-angles[:,t]))
            movement = float((pos[:,t+1]-pos[:,t]).norm(dim=-1).max())+bg_step
            n=max(1,math.ceil(movement/.05),math.ceil(float(turn.abs().max())/.02))
            if n>200:
                raise ValueError('Unreasonable interpolated background-prefix motion')
            for i in range(1,n+1):
                f=i/n
                sample_pos.append(pos[:,t]+f*(pos[:,t+1]-pos[:,t]))
                sample_yaw.append(angles[:,t]+f*turn);sample_t.append((t+f)*dt)
        values=separations(torch.stack(sample_pos,1),torch.stack(sample_yaw,1),ctx['extent'],bg,
                           pos.new_tensor(sample_t),ctx.get('background_uncertainty_cap_m'))
        return values.amin((-1,-2))


def forecast_audit(ctx):
    bg=ctx.get('causal_background')
    if bg is None:
        return None
    return dict(absent_observed_indices=bg['absent_observed_indices'],
        insufficient_history_indices=bg.get('insufficient_history_indices', []),
        indices=bg['indices'],position=bg['position'].cpu().tolist(),
        velocity=bg['velocity'].cpu().tolist(),spread_mps=bg['spread_mps'].cpu().tolist(),
        forecast='observed_cv_with_velocity_spread_v1',recorded_future_used=False,
        uncertainty_calibrated=False,uncertainty_cap_m=ctx.get('background_uncertainty_cap_m'),
        independent_execution_acceptance_required=True)
