"""Recover signed native endpoint speeds from generated positions and yaw.

The unicycle decoder advances position with interval-average speed along the
previous heading. Consequently v[t] = 2*projected_interval_speed - v[t-1].
Using a displacement norm loses reverse motion and also confuses interval
average speeds with the endpoint speeds used by the model's constraints.
"""
import torch


def signed_candidate_trajectory(positions, yaws, initial_state, dt):
    if positions.ndim!=3 or positions.shape[-1]!=2 or yaws.shape!=positions.shape[:2]:
        raise ValueError('Expected K by T positions and headings')
    if initial_state.ndim!=2 or initial_state.shape[-1]!=4 or initial_state.shape[0] not in (1,len(positions)):
        raise ValueError('Actual native initial states [K,4] are required')
    if not dt>0 or not all(bool(torch.isfinite(x).all()) for x in [positions,yaws,initial_state]):
        raise ValueError('Finite trajectories and positive dt are required')
    initial=initial_state.expand(len(positions),-1)
    displacement=torch.diff(torch.cat((initial[:,None,:2],positions),1),dim=1)/dt
    previous_heading=torch.cat((initial[:,None,3],yaws[:,:-1]),1)
    unit=torch.stack((previous_heading.cos(),previous_heading.sin()),-1)
    normal=torch.stack((-unit[...,1],unit[...,0]),-1)
    transverse=(displacement*normal).sum(-1)
    if float(transverse.abs().max())>1e-3:
        raise ValueError('Generated positions do not follow the native previous-heading integration')
    interval_speed=(displacement*unit).sum(-1)
    speed=[];previous=initial[:,2]
    for i in range(positions.shape[1]):
        previous=2*interval_speed[:,i]-previous
        speed.append(previous)
    speed=torch.stack(speed,1)
    acceleration=torch.diff(torch.cat((initial[:,None,2],speed),1),dim=1)/dt
    difference=yaws-previous_heading
    yaw_rate=torch.atan2(difference.sin(),difference.cos())/dt
    return torch.cat((positions,speed[...,None],yaws[...,None],acceleration[...,None],yaw_rate[...,None]),-1)
