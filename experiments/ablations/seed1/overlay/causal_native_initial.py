"""Initialize native endpoint speed from causal interval-position observations."""
import torch


def initial_endpoint_speed(history, dt):
    if history.shape != (3,2) or dt<=0 or not bool(torch.isfinite(history).all()):
        raise ValueError('Three finite chronological local positions and positive dt required')
    if float(history[-1].norm())>.01:
        raise ValueError('Current observed pose must be the local origin')
    velocity=torch.diff(history.double(),dim=0)/dt
    # Native positions integrate the average of adjacent endpoint speeds.
    endpoint=velocity[-1]+.5*(velocity[-1]-velocity[-2])
    return endpoint[0].to(history.dtype)
