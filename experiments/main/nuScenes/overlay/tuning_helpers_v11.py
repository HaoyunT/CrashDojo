"""Differentiable feasibility and minimum-change terms for inference only."""
import torch


def prefix_cost(tr, ctx, request, body_penalty):
    n=min(5,tr.shape[1]);xy=tr[:,:n,:2];yy=tr[:,:n,3];dt=ctx['dt']
    hist=ctx['history_local'].expand(tr.shape[0],-1,-1)
    vv=torch.diff(torch.cat((hist,xy),1),dim=1)/dt
    aa=torch.diff(vv,dim=1)/dt;jj=torch.diff(aa,dim=1)/dt
    heading=torch.cat((torch.zeros_like(yy[:,:1]),yy[:,:n-1]),1)
    unit=torch.stack((heading.cos(),heading.sin()),-1)
    along=(aa[:,1:]*unit).sum(-1);lateral=(aa[:,1:]*torch.stack((-unit[...,1],unit[...,0]),-1)).sum(-1)
    limits=request['engineering_limits']
    penalties=[torch.relu(jj.norm(dim=-1)-8.).square()/25,
               torch.relu(along-.9*limits['max_acceleration_mps2']).square(),
               torch.relu(-along-.9*limits['max_braking_mps2']).square(),
               torch.relu(lateral.abs()-.9*limits['max_lateral_mps2']).square(),
               torch.relu(vv[:,2:].norm(dim=-1)-limits['max_speed_mps']).square()]
    cost=sum(z.mean(-1)+z.amax(-1) for z in penalties)
    pos=(xy.unsqueeze(-2)*ctx['transform'][:2,:2]).sum(-1)+ctx['transform'][:2,2]
    yaw=yy+torch.atan2(ctx['transform'][1,0],ctx['transform'][0,0])
    return cost+body_penalty(pos,yaw,ctx['extent'],ctx['corridor_polygons'],3.,n)


def deviation_cost(tr, anchor, config):
    n=min(tr.shape[1],anchor.shape[1]);delta=(tr[:,:n,:2]-anchor[:,:n,:2]).norm(dim=-1)
    ade=delta.mean(-1);fde=delta[:,-1]
    penalty=config['ade_weight']*torch.relu(ade-config['soft_ade_budget_m']).square()
    penalty=penalty+config['fde_weight']*torch.relu(fde-config['soft_fde_budget_m']).square()
    return penalty,ade,fde
