"""Execute one native baseline without mutating an existing evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import traceback

from common import QUEUE, REPO, PYTHON, HERE, command, environment, load, save, sha, verify_sources


def validate_inference(validation, job, steps):
    for key in ['passed', 'weights_and_buffers_unchanged',
                'supplied_ego_executed_exactly', 'background_exact_replay']:
        assert validation[key] is True, 'Inference validation failed: ' + key
    assert validation['optimizer_steps'] == 0
    assert validation['steps'] == steps
    for key in ['method', 'planner', 'dataset']:
        assert validation[key] == job[key], 'Inference identity differs: ' + key
    assert validation['calls'] == validation['expected_calls'] == steps // 5
    assert validation['common_execution_calls'] == steps // 5
    adapted = (job['method'] == 'safesim'
               and validation.get('selection_adaptation_authorized') is True
               and validation.get('selection_mode') == 'user_authorized_v6_fallback')
    if adapted:
        assert job.get('selection_adaptation_authorized') is True
        assert validation['reused_ours_target'] is True
        assert validation['native_selection'] is False
    else:
        assert not job.get('selection_adaptation_expected', False)
        assert validation['native_selection'] is True
        assert validation['reused_ours_target'] is False
    assert validation['native_guidance'] is True


def frozen_job(protocol, job_id):
    matches = [job for job in protocol['jobs'] if job['job_id'] == job_id]
    if len(matches) != 1:
        raise ValueError('Job must have exactly one frozen protocol entry')
    job = load(QUEUE / 'jobs' / (job_id + '.json'))
    if job != matches[0]:
        raise ValueError('Job file differs from the frozen protocol')
    if job['initial_state'] != 'queued':
        raise ValueError('The frozen protocol does not permit this job to run')
    return job


def validate_reference(job):
    reference = Path(job['initial_reference'])
    metadata = reference.with_suffix('.json')
    record = load(metadata)
    if (record.get('scene_id') != job['scene']
            or record.get('start_frame') != job['start_frame']
            or record.get('observed_past_only') is not True):
        raise ValueError('Prepared reference identity or observed-past contract differs')
    return {'initial_reference_sha256': sha(reference),
            'initial_reference_metadata_sha256': sha(metadata)}


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke-tag')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--timeout', type=int, default=5400)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.job_id):
        raise ValueError('Invalid job identifier')
    if args.timeout <= 0:
        raise ValueError('The job timeout must be positive')
    if args.smoke_tag is not None and (not re.fullmatch(r'[A-Za-z0-9_-]+', args.smoke_tag)
                                      or args.smoke_tag in ['runs', 'status', 'jobs', 'workers', 'inputs']):
        raise ValueError('Smoke output must use a separate namespace')
    protocol = load(QUEUE / 'protocol.json')
    verify_sources(protocol)
    for path, expected in protocol['checkpoint_identity'].items():
        metadata = Path(path).stat()
        assert metadata.st_size == expected['size'] and metadata.st_mtime_ns == expected['mtime_ns']
    job = frozen_job(protocol, args.job_id)
    formal = args.smoke_tag is None
    if formal and args.steps is not None:
        raise ValueError('Formal runs use only the complete frozen frame budget')
    steps = job['steps'] if args.steps is None else args.steps
    if not 0 < steps <= job['steps'] or steps % 5:
        raise ValueError('Invalid five-frame execution budget')
    if formal and not load(QUEUE / 'gates.json').get(job['method'], {}).get('passed'):
        raise ValueError('This method has not passed its real GPU smoke gate')
    output = QUEUE / ('runs' if formal else args.smoke_tag) / args.job_id
    state_path = (QUEUE / 'status' / (args.job_id + '.json') if formal
                  else output.parent / (args.job_id + '.status.json'))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = state_path.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state_path.exists():
        previous = load(state_path)
        if previous.get('state') == 'completed' and previous.get('job') == job:
            assert previous['output'] == str(output)
            for name in ['metrics', 'native_inference_validation']:
                key = 'metrics_sha256' if name == 'metrics' else 'validation_sha256'
                assert sha(output / (name + '.json')) == previous[key]
            return
        raise FileExistsError('Preserve existing job status: ' + str(state_path))
    output.mkdir(parents=True, exist_ok=False)
    state = dict(state='running', job_id=args.job_id, job=job, pid=os.getpid(),
                 host=socket.gethostname(), gpu=args.gpu, started_unix=time.time(),
                 formal=formal, training=False, output=str(output))
    save(state_path, state)
    process = None

    def interrupted(signum, _frame):
        raise InterruptedError('Worker received signal %d' % signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for path, expected in job['input_sha256'].items():
            assert sha(path) == expected, 'Scene input changed: ' + path
        state.update(validate_reference(job))
        save(state_path, state)
        invocation = command(job, output, args.steps)
        save(output.parent / (args.job_id + '.command.json'), dict(command=invocation))
        with (output.parent / (args.job_id + '.log')).open('w') as stream:
            process = subprocess.Popen(invocation, cwd=str(REPO), env=environment(args.gpu),
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            state['child_pid'] = process.pid
            save(state_path, state)
            try:
                returncode = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                stop_process(process)
                raise TimeoutError('Baseline rollout exceeded job timeout')
        if returncode:
            raise RuntimeError('Native adapter exit code %d' % returncode)
        validation = load(output / 'native_inference_validation.json')
        validate_inference(validation, job, steps)
        assert validate_reference(job) == {key: state[key] for key in
            ['initial_reference_sha256', 'initial_reference_metadata_sha256']}
        if 'initial_reference_sha256' in validation:
            assert validation['initial_reference_sha256'] == state['initial_reference_sha256']
        metric_command = [str(PYTHON), str(HERE / 'metrics.py'), '--job-file',
                          str(QUEUE / 'jobs' / (args.job_id + '.json')),
                          '--output', str(output), '--steps', str(validation['steps'])]
        subprocess.run(metric_command, cwd=str(HERE), env=environment(args.gpu),
                       check=True, timeout=600)
        metrics = load(output / 'metrics.json')
        assert metrics['frames'] == validation['steps']
        for key in ['method', 'planner', 'dataset', 'scene']:
            assert metrics[key] == job[key], 'Metric identity differs: ' + key
        assert metrics['ego_collision_10Hz'] is not None
        state.update(state='completed', returncode=0,
                     validation_sha256=sha(output / 'native_inference_validation.json'),
                     metrics_sha256=sha(output / 'metrics.json'))
    except BaseException as exc:
        stop_process(process)
        state.update(state='failed', returncode=1, error=repr(exc), traceback=traceback.format_exc())
    state.update(finished_unix=time.time(), wall_seconds=time.time() - state['started_unix'])
    save(state_path, state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if state['state'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
