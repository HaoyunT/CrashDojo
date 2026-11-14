"""Resume the isolated remaining-scene v26 queue on one assigned A100."""
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from common import BASE, HERE, PY, QUEUE, REPO, ROOT, load, save, verify


GPU = 1
LOCK = QUEUE / 'scheduler.lock'
STATE = QUEUE / 'scheduler_state.json'
stop = False


def on_signal(*_):
    global stop
    stop = True


def free_mib():
    result = subprocess.run(['nvidia-smi', '-i', str(GPU), '--query-gpu=memory.free',
                             '--format=csv,noheader,nounits'], text=True,
                            capture_output=True, check=True)
    return int(result.stdout.strip().splitlines()[0])


def environment():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(GPU), CCDIFF_PLANNER_GPU='',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_CORETYPE='Haswell', NPY_DISABLE_CPU_FEATURES='AVX512F',
               TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', NVIDIA_TF32_OVERRIDE='0',
               PYTHONUNBUFFERED='1', PYTHONUTF8='1', LC_ALL='C.UTF-8',
               LANG='C.UTF-8', WANDB_MODE='disabled',
               V26_NUPLAN_CACHE=str(HERE / 'nuplan_centered_runtime_cache'),
               CRASHDOJO_CONTEXTUAL_OWNER=str(QUEUE),
               CRASHDOJO_RULE_ADAPTER_DIR=str(HERE / 'rule_adapter'))
    env['PYTHONPATH'] = ':'.join(map(str, [HERE, HERE / 'map_deps', HERE / 'overlay',
        REPO, REPO / 'third_party/tbsim', BASE / 'CCDiff/third_party/trajdata/src',
        BASE / 'planner_sources_20260810/nuplan_source/nuplan-devkit-master']))
    return env


def counts(jobs):
    result = dict(completed=0, failed=0, running=0, queued=0, blocked_short_packet=0)
    for job in jobs:
        if job['packet_duration_shortfall']:
            result['blocked_short_packet'] += 1
            continue
        path = QUEUE / 'runs' / job['job_id'] / 'status.json'
        if not path.exists():
            result['queued'] += 1
        else:
            value = load(path)['status']
            result[value if value in result else 'failed'] += 1
    return result


def update(status, jobs, **extra):
    state = dict(status=status, host=socket.gethostname(), pid=os.getpid(),
                 gpu=GPU, updated_unix=time.time(), counts=counts(jobs), **extra)
    save(STATE, state)
    return state


def run(job, jobs, canary=False):
    path = QUEUE / 'runs' / job['job_id']
    assert not path.exists(), path
    while free_mib() < 40000:
        update('waiting_gpu_memory', jobs, next_job=job['job_id'], free_mib=free_mib())
        if stop:
            return False
        time.sleep(15)
    log = QUEUE / 'logs' / (job['job_id'] + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PY), str(HERE / 'run.py'), '--job-id', job['job_id']]
    with log.open('w') as stream:
        proc = subprocess.Popen(command, cwd=HERE, env=environment(), stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        update('canary_running' if canary else 'running', jobs,
               current_job=job['job_id'], worker_pid=proc.pid,
               free_mib_before=free_mib(), log=str(log))
        try:
            returncode = proc.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            returncode = 124
    status = load(path / 'status.json') if (path / 'status.json').exists() else {}
    validation_path = path / 'runtime_validation.json'
    valid = validation_path.exists() and load(validation_path).get('passed')
    ok = returncode == 0 and status.get('status') == 'completed' and valid
    update('canary_passed' if canary and ok else 'running' if ok else 'job_failed', jobs,
           last_job=job['job_id'], last_returncode=returncode, last_valid=bool(valid))
    return ok


def main():
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = load(QUEUE / 'protocol.json')
        verify(protocol)
        assert protocol['scenes'] == 225 and len(protocol['jobs']) == 1125
        assert protocol['generator_checkpoint']['sha256'] == __import__('hashlib').sha256(
            Path(protocol['generator_checkpoint']['path']).read_bytes()).hexdigest()
        jobs = protocol['jobs']
        assert sum(j['packet_duration_shortfall'] for j in jobs) == 5
        if counts(jobs)['running']:
            update('needs_attention_stale_worker', jobs)
            return 2
        pending = [j for j in jobs if not j['packet_duration_shortfall'] and
                   not (QUEUE / 'runs' / j['job_id']).exists()]
        update('starting', jobs, canary_job='remaining225_000_idm', blocked_index=222)
        if pending and pending[0]['job_id'] == 'remaining225_000_idm':
            if not run(pending.pop(0), jobs, canary=True):
                update('canary_failed', jobs, canary_job='remaining225_000_idm')
                return 1
        for job in pending:
            if stop:
                update('stopped_after_current_job', jobs)
                return 2
            run(job, jobs)
        result = counts(jobs)
        update('finished_with_failures' if result['failed'] else 'finished_1120_jobs', jobs,
               protocol_sha256=__import__('hashlib').sha256((QUEUE / 'protocol.json').read_bytes()).hexdigest())
        return int(bool(result['failed']))


if __name__ == '__main__':
    raise SystemExit(main())
