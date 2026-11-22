"""Frozen inference and transport helpers without the old fixed-actor adapter."""
import hashlib
import numpy as np


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_budget(active, candidates=5):
    old = active.diffuse_args['num_samples']
    active.diffuse_args['num_samples'] = candidates
    config = getattr(active, 'guide_config', None)
    if config is not None and 'params' in config:
        config['params']['num_samples'] = candidates
    return dict(original_candidates=old, common_candidates=candidates,
                sampler_steps_and_losses_unchanged=True)


def legacy_metadata(value):
    if isinstance(value, dict):
        return {key: legacy_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [legacy_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(legacy_metadata(item) for item in value)
    if isinstance(value, (bool, int, float, np.generic)):
        return np.asarray(value)
    return value
