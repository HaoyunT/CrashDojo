"""Causal contact timing and terminal completion costs for frozen FM inference."""
import torch
from causal_background_occupancy_v1 import separations


def contact_cost(separation, relative_times, config):
    margin = float(config['overlap_margin_m'])
    delay = float(config['delay_cost_per_s'])
    if not 0 <= margin <= .1 or not 0 < delay <= 5:
        raise ValueError('Invalid contact objective configuration')
    if separation.ndim != 2 or relative_times.shape != separation.shape[1:]:
        raise ValueError('Expected candidate by time contact geometry')
    if not bool(torch.isfinite(separation).all()) or not bool((relative_times > 0).all()):
        raise ValueError('Contact optimization requires finite causal predictions')
    return (torch.relu(separation + margin).square()/4 + delay*relative_times).amin(-1)


def completion_cost(speed, reference, settled, tolerance=.25):
    if not bool(settled.any()):
        return speed.new_zeros(speed.shape[0])
    # The unchanged evaluator uses the worst error after settling, not its mean.
    violation = torch.relu((speed[:, settled]-reference[settled]).abs()-tolerance).square()
    return (violation.mean(-1)+violation.amax(-1))/(tolerance*tolerance)


def bounded_interval_speed(desired, endpoint_speed, heading, previous_velocity, previous_acceleration, dt, limits):
    unit=torch.stack((heading.cos(),heading.sin()),-1)
    center=previous_velocity+previous_acceleration*dt
    along=(center*unit).sum(-1)
    transverse=(center-along[:,None]*unit).square().sum(-1)
    radius=.8*limits['max_jerk_mps3']*dt*dt
    width=torch.sqrt((radius*radius-transverse).clamp_min(0))
    native_lo=torch.maximum(endpoint_speed-.9*limits['max_braking_mps2']*dt/2,endpoint_speed/2)
    native_hi=endpoint_speed+.9*limits['max_acceleration_mps2']*dt/2
    lower=torch.maximum(along-width,native_lo)
    upper=torch.minimum(along+width,native_hi)
    projected=torch.minimum(torch.maximum(desired,lower),upper)
    fallback=torch.minimum(torch.maximum(along,native_lo),native_hi)
    feasible=(transverse<=radius*radius)&(lower<=upper)
    return torch.where(feasible,projected,fallback),feasible


def obstacle_modes(model, seed, state, context, request, speed_reference, route_coordinates, route_frame_at_s):
    bg=context.get('causal_background')
    if not bg or not bg['indices']:
        return seed
    physical=model.convert_action_to_state_and_action(seed,state,descaled_output=True)
    tf=context['transform'];dt=context['dt']
    position=physical[...,:2]@tf[:2,:2].T+tf[:2,2]
    yaw=physical[...,3]+torch.atan2(tf[1,0],tf[0,0])
    times=torch.arange(1,seed.shape[1]+1,device=seed.device,dtype=seed.dtype)*dt
    # Branch only for an observed obstacle on a near-term proposal.
    check=separations(position[:1],yaw[:1],context['extent'],bg,times,uncertainty_cap_m=.5)
    if not bool((check[:,:min(30,seed.shape[1])]<.5).any()):
        return seed
    if seed.shape[0]!=5:
        raise ValueError('Obstacle modes reuse exactly the frozen five candidates')
    physical=model.descale_traj(seed,[4,5]).detach().clone()
    offsets=seed.new_tensor([0.,.7,-.7,1.4,-1.4])
    xy=state.expand(5,-1)[:,:2].clone();speed=state.expand(5,-1)[:,2].clone();heading=state.expand(5,-1)[:,3].clone()
    history=context['history_local'];v=(history[-1]-history[-2])/dt
    acceleration=(history[-1]-2*history[-2]+history[-3])/(dt*dt)
    previous_velocity=v.expand(5,-1);previous_acceleration=acceleration.expand(5,-1)
    world_heading=torch.atan2(tf[1,0],tf[0,0])
    absolute=context['elapsed']+times
    reference=speed_reference(absolute,request['parameters'])
    desired_offset=float(request['parameters']['initial_offset_m'])+offsets
    limits=request['engineering_limits']
    for i in range(seed.shape[1]):
        world=xy@tf[:2,:2].T+tf[:2,2]
        station,lateral=route_coordinates(world,context['route'])
        _,tangent=route_frame_at_s(station+torch.maximum(speed,torch.ones_like(speed)),context['route'])
        route_heading=torch.atan2(tangent[:,1],tangent[:,0])-world_heading
        correction=torch.atan2(.6*(desired_offset-lateral),speed.abs().clamp_min(.75)).clamp(-.35,.35)
        target=route_heading+correction
        difference=torch.atan2(torch.sin(target-heading),torch.cos(target-heading))
        angular_limit=torch.minimum(.5*speed.abs(),1./speed.abs().clamp_min(.1)).clamp_min(.1)
        omega=(difference/.8).clamp(-angular_limit,angular_limit)
        desired=(speed+reference[i])/2
        interval,feasible=bounded_interval_speed(desired,speed,heading,previous_velocity,previous_acceleration,dt,limits)
        next_speed=2*interval-speed
        acceleration=(next_speed-speed)/dt
        unit=torch.stack((heading.cos(),heading.sin()),-1)
        velocity=interval[:,None]*unit
        next_acceleration=(velocity-previous_velocity)/dt
        xy=xy+velocity*dt
        speed=next_speed;heading=heading+omega*dt
        previous_velocity=velocity;previous_acceleration=next_acceleration
        physical[:,i,0]=acceleration;physical[:,i,1]=omega
    context['obstacle_modes_used']=True
    return model.scale_traj(physical,[4,5]).detach()


def warm_start(model, base, state, context, request, speed_reference, route_coordinates=None, route_frame_at_s=None):
    """Use one of the existing K slots for a causal speed-tracking proposal."""
    if request['family'] != 'C1':
        return base
    physical = model.descale_traj(base, [4, 5]).detach().clone()
    times = context['elapsed'] + torch.arange(1,base.shape[1]+1,device=base.device,dtype=base.dtype)*context['dt']
    target_speed = speed_reference(times, request['parameters'])
    initial = state.expand(base.shape[0], -1)[0, 2]
    accel = torch.diff(torch.cat((initial[None], target_speed)))/context['dt']
    limits = request['engineering_limits']
    physical[0, :, 0] = accel.clamp(-.9*limits['max_braking_mps2'], .9*limits['max_acceleration_mps2'])
    result=model.scale_traj(physical, [4, 5]).detach()
    if request.get('multimodal_obstacle_seed',False):
        result=obstacle_modes(model,result,state,context,request,speed_reference,route_coordinates,route_frame_at_s)
    return result


def test():
    time = torch.tensor([1., 5.])
    distance = torch.tensor([[2., -1.]], requires_grad=True)
    old = torch.relu(distance.amin(-1)).square()/4
    old_grad, = torch.autograd.grad(old.sum(), distance, retain_graph=True)
    assert old_grad.abs().sum() == 0
    new = contact_cost(distance,time,dict(overlap_margin_m=.05,delay_cost_per_s=1.))
    grad, = torch.autograd.grad(new.sum(), distance)
    assert grad[0, 0] > 0 and grad[0, 1] == 0
    speed = torch.tensor([[2., 2., 1.]], requires_grad=True)
    penalty = completion_cost(speed, torch.tensor([2., 2., 2.]), torch.tensor([False, True, True]))
    gradient, = torch.autograd.grad(penalty.sum(), speed)
    assert penalty.item() > 0 and gradient[0, -1] < 0
    assert completion_cost(speed, speed[0].detach(), torch.zeros(3,dtype=torch.bool)).item() == 0
    limits=dict(max_jerk_mps3=10.,max_braking_mps2=6.,max_acceleration_mps2=3.)
    q,valid=bounded_interval_speed(torch.tensor([2.]),torch.tensor([1.]),torch.tensor([0.]),torch.tensor([[1.,0.]]),torch.tensor([[0.,0.]]),.1,limits)
    assert valid.item() and abs(q.item()-1.08)<1e-6
    q,valid=bounded_interval_speed(torch.tensor([2.]),torch.tensor([1.]),torch.tensor([0.]),torch.tensor([[1.,1.]]),torch.tensor([[0.,0.]]),.1,limits)
    assert not valid.item() and torch.isfinite(q).all()
    print('PASS contact plateau, earlier contact gradient, worst-case completion, empty settled horizon')


if __name__ == '__main__':
    test()
