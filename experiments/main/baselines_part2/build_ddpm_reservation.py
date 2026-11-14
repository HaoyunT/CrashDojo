"""Freeze the nine val scenes reserved for a separate DDPM host."""

import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch


ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
QUEUE = ROOT / 'eval/baselines_native_v6_remaining225_20261101'
OUT = QUEUE / 'ddpm_friend_reserved_val45_v1.json'
BLOCKS = (42, 43, 44, 48, 50, 194, 197, 204, 224)
PLANNERS = ('idm', 'strive', 'pdm_closed', 'diffusion_planner', 'flow_planner')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert socket.gethostname() == 'di-20261020182309-thst9'
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_path = QUEUE / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    jobs = {job['job_id']: job for job in protocol['jobs']}
    records = []
    for block in BLOCKS:
        for planner in PLANNERS:
            job_id = f'ccdiff_ddpm_remaining225_{block:03d}_{planner}'
            job = jobs[job_id]
            path = QUEUE / 'jobs' / (job_id + '.json')
            status = QUEUE / 'status' / (job_id + '.json')
            output = QUEUE / 'runs' / job_id
            assert job['source_split'] == 'val' and job['initial_state'] == 'queued'
            assert job['steps'] == 70 and not status.exists() and not output.exists()
            assert json.loads(path.read_text()) == job
            reference = Path(job['initial_reference'])
            data = torch.load(reference, map_location='cpu')
            history = np.asarray(data['observed_source']['world_history'])
            assert len(data['names']) <= 30 and np.isfinite(history[:, -3:, :]).all()
            records.append({'job_id': job_id, 'block': block, 'planner': planner,
                            'actors': len(data['names']), 'job_sha256': sha(path),
                            'reference_sha256': sha(reference),
                            'initial_prefix_all_actors_last3_valid': True})
    assert len(records) == 45 and len({row['job_id'] for row in records}) == 45
    manifest = {'version': 'ddpm_friend_reserved_val45_v1',
                'reservation_active': True, 'native_70_frame_completion_unverified': True,
                'host': socket.gethostname(), 'created_unix': time.time(),
                'protocol_sha256': sha(protocol_path),
                'selection_manifest_sha256': protocol['selection_manifest_sha256'],
                'checkpoint_identity': {path: identity for path, identity in
                    protocol['checkpoint_identity'].items() if '/diffusion/' in path},
                'jobs': records}
    temp = OUT.with_suffix('.json.tmp')
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, OUT)
    print(json.dumps({'path': str(OUT), 'sha256': sha(OUT), 'jobs': len(records)}))


if __name__ == '__main__':
    main()
