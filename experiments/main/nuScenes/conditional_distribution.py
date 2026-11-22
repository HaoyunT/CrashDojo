"""Differentiable, soft conditional behavior prior for V26 guidance.

The table is fit on disjoint training logs. It regularizes motion intensity
without clipping valid tails; hard speed, road and executed-prefix checks remain
the independent validity gates.
"""
import math

import torch


def penalty(trajectory, request):
    cfg = request.get('conditional_distribution_guidance')
    if not cfg or not cfg.get('enabled'):
        return trajectory.new_zeros(trajectory.shape[0])
    acceleration = trajectory[..., 4]
    family = request.get('family')
    observed = acceleration.neg().amax(-1) if family == 'C1' else acceleration.abs().amax(-1)
    center = math.exp(float(cfg['intensity_log_median']))
    spread = max(float(cfg.get('intensity_log_mad', .25)), .15)
    # A robust log-scale band keeps the common mass inexpensive and gives a
    # smooth, finite tail cost for rare but physically valid behavior.
    log_observed = torch.log(observed.clamp_min(1e-3))
    z = (log_observed - math.log(center)) / spread
    return torch.relu(z.abs() - 2.).square() + .05 * z.square()
