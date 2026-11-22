"""Keep a selected FM sample's positions and native yaw together in controls."""
import torch


def first_native_action(generated, info, target, horizon):
    samples=info['action_samples']; positions=samples['positions']; yaws=samples['yaws']
    if positions.ndim != 4 or positions.shape[-1] != 2 or yaws.shape != positions.shape[:-1]+(1,):
        raise ValueError('Expected native agent-centric [M,K,T,2/1] samples')
    if not 0 <= target < positions.shape[0] or not 1 <= horizon <= positions.shape[2]:
        raise ValueError('Invalid actor or native horizon')
    p=positions[target,0,:horizon]; y=yaws[target,0,:horizon]
    if not torch.equal(p,generated.positions[target,:horizon]):
        raise ValueError('Default control positions no longer match native sample zero')
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite native action')
    return p,y
