"""Coupled Gaussian sources and explicit removal of conflict costs only."""
import hashlib
import json
from pathlib import Path
from overlay.ablation_contract import generic_contact_request

_context = None


def set_noise_context(binding, step, directory, request):
    global _context
    if _context is not None and _context['calls'] != 1:
        raise RuntimeError('Expected exactly one Gaussian source draw per replan')
    names = [str(n) for n in binding.env.current_agent_names]
    if len(names) != len(set(names)):
        raise RuntimeError('Stable actor names must be unique')
    config = request['ablation_v26']
    _context = dict(names=names, step=int(step), scene=config['scene'],
                    stream=int(config['stream']), directory=Path(directory), calls=0)


def coupled_noise(shape, device):
    import numpy as np
    import torch
    c = _context
    original_shape = tuple(shape)
    if len(shape) == 5:
        if shape[0] != 1: raise RuntimeError('Only one scene per paired rollout')
        shape = shape[1:]
    if c is None or c['calls'] != 0 or len(shape) != 4 or shape[1] != len(c['names']):
        raise RuntimeError('Unexpected sampling layout or unbound source draw')
    c['calls'] += 1
    values = np.empty(tuple(shape), dtype=np.float32)
    records = []
    for sample in range(shape[0]):
        for actor, actor_id in enumerate(c['names']):
            # Arm is deliberately absent. Stable identity, not array position,
            # determines the source for each actor/sample/replan.
            key = ['v26_paired_gaussian_v1', c['scene'], c['stream'], c['step'], sample, actor_id]
            seed = int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:8], 'little')
            draw = np.random.Generator(np.random.PCG64(seed)).standard_normal(tuple(shape[2:])).astype(np.float32)
            values[sample, actor] = draw
            records.append(dict(actor=actor_id, sample=sample, seed=seed,
                                sha256=hashlib.sha256(draw.tobytes()).hexdigest()))
    result = dict(scene=c['scene'], stream=c['stream'], step=c['step'], shape=list(shape),
                  source='independent Gaussian per stable actor and sample; deterministic Euler thereafter',
                  draws=records, bank_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    with (c['directory'] / 'coupled_noise.jsonl').open('a') as f:
        f.write(json.dumps(result) + '\n')
    return torch.from_numpy(values.reshape(original_shape)).to(device=device)


def install_guidance(module):
    original = module.FiveClassGuide
    original_loss = module.loss_terms

    def arm_loss(traj, native, ctx, request):
        if request.get('ablation_v26', {}).get('arm') == 'A3':
            # Keep regularization, road/body/dynamics/jerk, deviation budgets,
            # non-target avoidance, prefix repair, and feasibility selection.
            request = generic_contact_request(request)
        if request.get('ablation_v26', {}).get('arm') == 'A4':
            assert request['weights']['action'] == request['weights']['pressure'] == 0.
        return original_loss(traj, native, ctx, request)

    module.loss_terms = arm_loss

    class AblationGuide(original):
        def prepare(self, obs, ego_action, target, ego, elapsed, dt):
            self.request.update(generic_contact_request(self.request))
            super().prepare(obs, ego_action, target, ego, elapsed, dt)
            arm = self.request['ablation_v26']['arm']
            if arm not in ('A0','A1','A2','A3','A4'):
                raise ValueError('Unimplemented ablation arm: ' + arm)
            proposed = dict(self.request['weights'])
            if arm == 'A3':
                self.request['weights']['action'] = 0.
            if arm == 'A4':
                self.request['weights']['action'] = 0.
                self.request['weights']['pressure'] = 0.
            with self.audit_path.with_name('ablation_steps.jsonl').open('a') as f:
                f.write(json.dumps(dict(arm=arm, elapsed_s=float(elapsed),
                    proposed_weights=proposed, effective_weights=self.request['weights'],
                    conflict_guidance_enabled=arm != 'A4', mechanism_tracking_enabled=arm not in ('A3', 'A4'), feasibility_and_smoothing_preserved=True)) + '\n')

    module.FiveClassGuide = AblationGuide
