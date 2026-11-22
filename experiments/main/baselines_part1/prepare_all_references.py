"""Produce one shared, observed initial reference for each frozen scene."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import time

from common import HERE, QUEUE, PYTHON, load, save, environment, sha


def prepare(job_file):
    job = load(job_file)
    block = job['source_job']['block']
    output = QUEUE / 'inputs/references' / (block + '.pt')
    metadata = output.with_suffix('.json')
    if output.exists() and metadata.exists():
        return block, None
    temporary = output.with_name(block + '.building-%d.pt' % os.getpid())
    log = QUEUE / 'preparation_logs' / (block + '.log')
    entry = (HERE / 'cache_adapter/prepare_reference_isolated.py' if job['source_job']['dataset'] == 'nuplan'
             else HERE / 'safesim_adapter/prepare_reference.py')
    invocation = [str(PYTHON), str(entry),
                  '--job-file', str(job_file), '--output', str(temporary)]
    try:
        with log.open('w') as stream:
            result = subprocess.run(invocation, stdout=stream, stderr=subprocess.STDOUT,
                                    env=environment(''), timeout=600)
        if result.returncode:
            return block, 'CPU reference exit code %d; log=%s' % (result.returncode, log)
        audit = load(temporary.with_suffix('.json'))
        assert audit['scene_id'] == job['source_job']['scene']
        assert audit['start_frame'] == load(job['source_job']['manifest'])['start_step_10hz']
        audit['reference_sha256'] = sha(temporary)
        audit['source_config_sha256'] = sha(job['source_job']['config'])
        temporary.replace(output)
        save(metadata, audit)
        return block, None
    except Exception as exc:
        return block, repr(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    (QUEUE / 'inputs/references').mkdir(parents=True, exist_ok=True)
    (QUEUE / 'preparation_logs').mkdir(parents=True, exist_ok=True)
    jobs = sorted((QUEUE / 'jobs').glob('safesim_*_idm.json'))
    if len(jobs) != 150:
        raise ValueError('Expected exactly 150 CPU scene references')
    completed, errors = 0, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare, path) for path in jobs]
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=5,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                block, error = future.result()
                if error:
                    errors.append(dict(block=block, error=error))
                else:
                    completed += 1
            save(QUEUE / 'preparation_state.json', dict(
                state='running' if pending else ('completed' if not errors else 'failed'),
                total=150, completed=completed, failed=len(errors), errors=errors,
                updated_unix=time.time()))
    print('Prepared', completed, 'failed', len(errors), flush=True)


if __name__ == '__main__':
    main()
