"""Finite technical checks; never writes to the formal-run directory."""
import argparse
import os
import socket
import subprocess
import time

from common import HERE, QUEUE, PY, environment, load, save

PLANS = {
    2: [('scene-0097_s1_idm_A0', 70), ('scene-0559_s1_idm_A1', 10),
        ('scene-0099_s1_idm_A3', 10), ('scene-0099_s1_idm_A8', 70)],
    3: [('scene-1061_s1_diffusion_planner_A0', 70), ('scene-0798_s1_diffusion_planner_A2', 10),
        ('scene-0099_s1_diffusion_planner_A8', 70)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[2, 3], required=True)
    args = parser.parse_args()
    dest = QUEUE / 'qa' / f'preflight_worker_gpu{args.gpu}.json'
    assert not dest.exists(), 'Do not repeat an existing preflight worker'
    state = dict(status='running', pid=os.getpid(), host=socket.gethostname(), gpu=args.gpu,
        started_unix=time.time(), formal=False, planned=PLANS[args.gpu], completed=[])
    save(dest, state)
    for job_id, steps in PLANS[args.gpu]:
        state['current_job'] = job_id
        state['current_steps'] = steps
        assert not (QUEUE / 'smoke' / job_id).exists()
        log = QUEUE / 'logs' / f'smoke_{job_id}.log'
        with log.open('x') as f:
            process = subprocess.Popen([str(PY), str(HERE / 'run.py'), '--job-id', job_id,
                '--smoke', '--smoke-steps', str(steps)], env=environment(args.gpu), cwd=HERE,
                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            state['child_pid'] = process.pid
            save(dest, state)
            rc = process.wait()
        state['completed'].append(dict(job_id=job_id, steps=steps, returncode=rc))
        if rc:
            state.update(status='failed', finished_unix=time.time())
            save(dest, state)
            return
        save(dest, state)
    state.update(status='completed', finished_unix=time.time())
    save(dest, state)


if __name__ == '__main__':
    main()
