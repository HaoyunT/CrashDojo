"""Persistent 4090D queue with bounded concurrency and an independent final audit."""
import argparse
import collections
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from common import ROOT, HERE, QUEUE, PYTHON, METHODS, load, save, sha, environment, verify_sources, status_counts


def gpu_free(gpu):
    return float(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())


def reference_ready(job):
    reference = Path(job['initial_reference'])
    metadata = reference.with_suffix('.json')
    if not reference.is_file() or not metadata.is_file():
        return False
    try:
        record = load(metadata)
    except (OSError, ValueError):
        return False
    return (record.get('scene_id') == job['scene']
            and record.get('start_frame') == job['start_frame']
            and record.get('observed_past_only') is True)


def drain(active):
    # These Popen handles are owned by this controller, never unrelated GPU PIDs.
    for entry in active.values():
        if entry['process'].poll() is None:
            entry['process'].terminate()
    deadline = time.monotonic() + 35
    for entry in active.values():
        process = entry['process']
        try:
            process.wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        finally:
            entry['stream'].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--slots-per-gpu', type=int, default=1)
    args = parser.parse_args()
    if args.slots_per_gpu < 1 or len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        raise ValueError('GPU identifiers must be unique and slot counts positive')
    if socket.gethostname() != 'gpu-host-6':
        raise ValueError('Baseline dispatch is bound to the inspected 2044 host')
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / 'workers').mkdir(exist_ok=True)
    lock = (QUEUE / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = load(QUEUE / 'protocol.json')
    verify_sources(protocol)
    for path, expected in protocol['dependency_sha256'].items():
        assert sha(path) == expected, 'Frozen dependency changed: ' + path
    gates = load(QUEUE / 'gates.json')
    allowed = [m for m in METHODS if gates.get(m, {}).get('passed')]
    if not allowed:
        raise ValueError('No baseline passed the real GPU smoke gate')
    slots = [(gpu, slot) for gpu in args.gpus for slot in range(args.slots_per_gpu)]
    active = {}
    started = time.time()
    # Alternate methods/domains early so the ETA is not based on a single easy family.
    pending = [j for j in protocol['jobs'] if j['method'] in allowed
               and j['initial_state'] == 'queued']
    pending.sort(key=lambda j: (int(j['source_job']['block'].rsplit('_', 1)[-1]) % 225,
                               j['source_split'], protocol['planners'].index(j['planner']),
                               METHODS.index(j['method'])))
    for job in pending:
        path = QUEUE / 'status' / (job['job_id'] + '.json')
        if path.exists():
            state = load(path)
            if state.get('job') != job or state.get('job_id') != job['job_id']:
                raise ValueError('Existing status belongs to a different frozen job: ' + job['job_id'])
            if state.get('state') == 'running':
                raise RuntimeError('Active/orphan job requires inspection before controller restart: ' + job['job_id'])
    stopping = False

    def interrupted(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            for slot, entry in list(active.items()):
                if entry['process'].poll() is not None:
                    entry['stream'].close()
                    path = QUEUE / 'status' / (entry['job']['job_id'] + '.json')
                    if not path.exists() or load(path)['state'] == 'running':
                        state = load(path) if path.exists() else dict(job=entry['job'],
                            job_id=entry['job']['job_id'], formal=True, training=False)
                        state.update(state='failed', returncode=entry['process'].returncode,
                                     error='worker_exited_without_final_status')
                        save(path, state)
                    del active[slot]
                elif gpu_free(slot[0]) < 4096:
                    entry['process'].terminate()
            running_ids = {entry['job']['job_id'] for entry in active.values()}
            reservation_file = QUEUE / 'gpu_reservations.json'
            reserved_gpus = (load(reservation_file).get('reserved_gpus', [])
                             if reservation_file.exists() else [])
            for slot in slots:
                if stopping or slot in active or slot[0] in reserved_gpus or gpu_free(slot[0]) < 20480:
                    continue
                candidate = None
                for job in pending:
                    if job['job_id'] in running_ids:
                        continue
                    path = QUEUE / 'status' / (job['job_id'] + '.json')
                    if path.exists():
                        old = load(path)
                        if old['state'] == 'running':
                            raise RuntimeError('Unexpected active/orphan job; refusing duplicate dispatch: ' + job['job_id'])
                        continue
                    if not reference_ready(job):
                        continue
                    candidate = job
                    break
                if candidate is None:
                    continue
                stream = (QUEUE / 'workers' / (candidate['job_id'] + '.log')).open('w')
                invocation = [str(PYTHON), str(HERE / 'run_job.py'), '--job-id', candidate['job_id'],
                              '--gpu', str(slot[0])]
                process = subprocess.Popen(invocation, stdout=stream, stderr=subprocess.STDOUT,
                                           env=environment(slot[0]), start_new_session=True)
                active[slot] = dict(process=process, stream=stream, job=candidate)
                running_ids.add(candidate['job_id'])
            counts = status_counts(protocol)
            unstarted = [j for j in pending if not (QUEUE / 'status' / (j['job_id'] + '.json')).exists()
                         and j['job_id'] not in running_ids]
            waiting_preparation = sum(not reference_ready(j) for j in unstarted)
            preparation_path = QUEUE / 'preparation_state.json'
            preparation = load(preparation_path) if preparation_path.exists() else {'state': 'not_started'}
            current = {('%d:%d' % slot): dict(job_id=e['job']['job_id'], pid=e['process'].pid)
                       for slot, e in active.items()}
            scheduler_state = ('stopping' if stopping else 'running' if active else
                               'waiting_preparation' if waiting_preparation else 'waiting_or_finished')
            save(QUEUE / 'scheduler_state.json', dict(state=scheduler_state,
                 counts=counts, current=current, total=protocol['total'], allowed_methods=allowed,
                 waiting_preparation=waiting_preparation, preparation=preparation,
                 host=socket.gethostname(), controller_pid=os.getpid(),
                 slots_per_gpu=args.slots_per_gpu, started_unix=started, updated_unix=time.time()))
            if not active and (stopping or not unstarted):
                break
            time.sleep(5)
    finally:
        drain(active)
    if not stopping:
        subprocess.run([str(PYTHON), str(HERE / 'audit.py')], check=True)


if __name__ == '__main__':
    main()
