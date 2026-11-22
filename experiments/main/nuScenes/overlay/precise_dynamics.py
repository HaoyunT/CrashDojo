"""Float64 inference optimization with the native parallel decoder equations."""
import torch


def native_parallel(dyn, initial, actions, dt):
    acceleration=actions[..., 0].clamp(*dyn.acce_bound)
    speed=torch.cat((initial[..., 2:3],initial[..., 2:3]+(acceleration*dt).cumsum(-1)),-1)
    speed=speed.clamp(*dyn.vbound)
    with torch.no_grad():
        previous=speed[..., :-1].abs()
        bound=torch.minimum(dyn.max_steer*previous,dyn.max_yawvel/previous.clamp_min(.1)).clamp_min(.1)
    omega=torch.minimum(torch.maximum(actions[..., 1],-bound),bound)
    yaw=torch.cat((initial[..., 3:4],initial[..., 3:4]+(omega*dt).cumsum(-1)),-1)
    interval=(speed[..., :-1]+speed[..., 1:])/2
    velocity=interval[..., None]*torch.stack((yaw[..., :-1].cos(),yaw[..., :-1].sin()),-1)
    xy=initial[..., None,:2]+(velocity*dt).cumsum(-2)
    return torch.cat((xy,speed[..., 1:,None],yaw[..., 1:,None]),-1)


class PreciseDynamics:
    def __init__(self, model):
        self.model=model

    def __getattr__(self,name):
        return getattr(self.model,name)

    def convert_action_to_state_and_action(self, actions, initial, scaled_input=True, descaled_output=False):
        physical=self.model.descale_traj(actions,[4,5]) if scaled_input else actions
        trajectory=native_parallel(self.model.dyn,initial,physical,self.model.dt)
        result=torch.cat((trajectory,physical),-1)
        if scaled_input and not descaled_output:
            result=self.model.scale_traj(result,[0,1,2,3,4,5])
        return result


def double_context(value):
    if torch.is_tensor(value):
        return value.double() if value.is_floating_point() else value
    if isinstance(value,dict):return {k:double_context(v) for k,v in value.items()}
    if isinstance(value,list):return [double_context(v) for v in value]
    if isinstance(value,tuple):return tuple(double_context(v) for v in value)
    return value
