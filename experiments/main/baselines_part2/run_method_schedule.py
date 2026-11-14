"""Resume one frozen baseline method on one inspected 4090D card."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from common import HERE, METHODS, PYTHON, QUEUE, environment, load, save, sha, verify_sources


def gpu_free(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())


def emit(path, event):
    with path.open('a') as stream:
        stream.write(json.dumps(dict(event, unix=time.time()), ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'di-20261022123127-4dzkw' or args.gpu not in range(4):
        raise ValueError('Dispatch is bound to inspected host 2044 and GPUs 0-3')
    protocol = load(QUEUE / 'protocol.json')
    verify_sources(protocol)
    for path, digest in protocol['dependency_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen dependency changed: ' + path)
    gate = load(QUEUE / 'gates.json')[args.method]
    assert gate['passed'] and all(sha(item['path']) == item['sha256'] for item in gate['evidence'])
    canary_id = args.method + '_remaining225_000_idm'
    canary = load(QUEUE / 'status' / (canary_id + '.json'))
    assert canary['state'] == 'completed' and canary['returncode'] == 0
    for name, key in [('metrics.json', 'metrics_sha256'),
                      ('native_inference_validation.json', 'validation_sha256')]:
        assert sha(QUEUE / 'runs' / canary_id / name) == canary[key]
    assert load(QUEUE / 'runs' / canary_id / 'native_inference_validation.json')['steps'] == 70

    lock = (QUEUE / ('controller_' + args.method + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = QUEUE / ('scheduler_state_' + args.method + '.json')
    events = QUEUE / ('scheduler_events_' + args.method + '.jsonl')
    workers = QUEUE / 'workers' / args.method
    workers.mkdir(parents=True, exist_ok=True)
    jobs = [job for job in protocol['jobs'] if job['method'] == args.method
            and job['initial_state'] == 'queued']
    assert len(jobs) == 1120
    jobs.sort(key=lambda job: (job['source_split'], job['source_job']['block'],
                               protocol['planners'].index(job['planner'])))
    for job in jobs:
        path = QUEUE / 'status' / (job['job_id'] + '.json')
        if path.exists():
            current = load(path)
            if current.get('job') != job or current.get('job_id') != job['job_id']:
                raise ValueError('Status identity mismatch: ' + job['job_id'])
            if current['state'] == 'running':
                raise ValueError('Inspect active/orphan worker before resuming: ' + job['job_id'])
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    active = None
    started = time.time()
    emit(events, dict(event='controller_start', pid=os.getpid(), gpu=args.gpu,
                      method=args.method, protocol_sha256=sha(QUEUE / 'protocol.json'),
                      scheduler_sha256=sha(__file__)))
    try:
        while True:
            if active is not None:
                process, stream, job = active
                if process.poll() is not None:
                    stream.close()
                    status = load(QUEUE / 'status' / (job['job_id'] + '.json'))
                    emit(events, dict(event='job_end', job_id=job['job_id'],
                                      state=status['state'], returncode=process.returncode))
                    active = None
                elif gpu_free(args.gpu) < 4096:
                    emit(events, dict(event='memory_guard_stop', job_id=job['job_id'],
                                      free_mib=gpu_free(args.gpu)))
                    process.terminate()
                    stopping = True
            unstarted = [job for job in jobs if not (QUEUE / 'status' /
                         (job['job_id'] + '.json')).exists()]
            if active is None and not stopping and unstarted and gpu_free(args.gpu) >= 20480:
                job = next((item for item in unstarted if Path(item['initial_reference']).is_file()
                    and Path(item['initial_reference']).with_suffix('.json').is_file()), None)
                if job is not None:
                    stream = (workers / (job['job_id'] + '.log')).open('w')
                    command = [str(PYTHON), str(HERE / 'run_job.py'),
                               '--job-id', job['job_id'], '--gpu', str(args.gpu)]
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               env=environment(args.gpu), start_new_session=True)
                    active = (process, stream, job)
                    emit(events, dict(event='job_start', job_id=job['job_id'],
                                      pid=process.pid, gpu=args.gpu))
            counts = {state: 0 for state in ('completed', 'failed', 'running', 'queued')}
            for job in jobs:
                path = QUEUE / 'status' / (job['job_id'] + '.json')
                state = load(path)['state'] if path.exists() else 'queued'
                counts[state] = counts.get(state, 0) + 1
            save(state_path, dict(state='stopping' if stopping else 'running' if active else
                'waiting_references' if unstarted else 'finished', method=args.method,
                gpu=args.gpu, pid=os.getpid(), started_unix=started,
                updated_unix=time.time(), counts=counts,
                active_job=active[2]['job_id'] if active else None,
                active_pid=active[0].pid if active else None,
                protocol_sha256=sha(QUEUE / 'protocol.json'), scheduler_sha256=sha(__file__)))
            if active is None and (stopping or not unstarted):
                break
            time.sleep(5)
    finally:
        if active is not None:
            process, stream, _job = active
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            stream.close()
    emit(events, dict(event='controller_end', method=args.method, stopping=stopping))


if __name__ == '__main__':
    main()
