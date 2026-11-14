"""Retry only OOM jobs on A100 using the unchanged frozen contextual runtime."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import ROOT, HERE, QUEUE, SOURCE, PYTHON, PLANNERS, environment, load, save, sha

RECOVERY = SOURCE / 'a100_oom_recovery_v3_20261030'
FROZEN = ROOT / 'launch/causal_guidance_v26_contextual_full_eval_20261101'


def run_one(job_id):
    sys.path[:0] = [str(FROZEN), str(FROZEN / 'map_deps'), str(FROZEN / 'overlay'),
                   str(ROOT / 'code/CCDiff_M2c_planner'),
                   str(ROOT / 'code/CCDiff_M2c_planner/third_party/tbsim'),
                   '/path/to/workspace/home/CCDiff/third_party/trajdata/src']
    sys.modules.pop('common', None)
    specification = importlib.util.spec_from_file_location('unchanged_contextual_run', FROZEN / 'run.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    # Only the artifact destination changes; source hashes and job settings stay frozen.
    module.QUEUE = RECOVERY
    sys.argv = [str(FROZEN / 'run.py'), '--job-id', job_id]
    module.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-job')
    args = parser.parse_args()
    if args.run_job:
        run_one(args.run_job)
        return
    RECOVERY.mkdir(parents=True, exist_ok=True)
    for name in ['protocol.json', 'release.json']:
        destination = RECOVERY / name
        if not destination.exists():
            shutil.copyfile(SOURCE / name, destination)
        assert sha(destination) == sha(SOURCE / name)
    reservation = QUEUE / 'gpu_reservations.json'
    save(reservation, dict(reserved_gpus=[0], reason='unchanged contextual005 OOM recovery',
                           owner_pid=os.getpid(), updated_unix=time.time()))
    records, child = [], None

    def interrupted(_signum, _frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise InterruptedError('Stopped contextual A100 recovery')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for planner in PLANNERS:
            identity = 'eval150_005_' + planner
            old = SOURCE / 'runs' / identity
            original = load(old / 'status.json')
            if original['status'] == 'completed':
                continue
            assert 'out of memory' in original.get('traceback', '').lower()
            original_hash = sha(old / 'status.json')
            log = RECOVERY / (identity + '.log')
            env = environment(0)
            env.update(V26_NUPLAN_CACHE=str(FROZEN / 'nuplan_centered_runtime_cache'),
                       CRASHDOJO_RULE_ADAPTER_DIR=str(FROZEN / 'rule_adapter'),
                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', NVIDIA_TF32_OVERRIDE='0',
                       NPY_DISABLE_CPU_FEATURES='AVX512F')
            base = Path('/path/to/workspace/home')
            env['PYTHONPATH'] = ':'.join(map(str, [FROZEN, FROZEN / 'map_deps',
                FROZEN / 'overlay', ROOT / 'code/CCDiff_M2c_planner',
                ROOT / 'code/CCDiff_M2c_planner/third_party/tbsim',
                base / 'CCDiff/third_party/trajdata/src',
                base / 'planner_sources_20260810/nuplan_source/nuplan-devkit-master']))
            started = time.time()
            with log.open('w') as stream:
                child = subprocess.Popen([str(PYTHON), str(HERE / Path(__file__).name),
                    '--run-job', identity], stdout=stream, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True)
                code = child.wait()
            record = dict(job_id=identity, returncode=code, wall_seconds=time.time()-started,
                          original_failure_sha256=original_hash, original_runtime_modified=False)
            recovered = RECOVERY / 'runs' / identity
            if code == 0:
                state = load(recovered / 'status.json')
                validation = load(recovered / 'runtime_validation.json')
                assert state['status'] == 'completed' and validation['passed']
                assert sha(old / 'status.json') == original_hash
                archive = SOURCE / 'failed_before_a100' / identity
                archive.parent.mkdir(parents=True, exist_ok=True)
                assert not archive.exists()
                old.rename(archive)
                old.symlink_to(recovered, target_is_directory=True)
                record.update(adopted=True, archived_failed_attempt=str(archive),
                              recovered_output=str(recovered))
            records.append(record)
            save(RECOVERY / 'recovery_state.json', dict(state='running', records=records,
                 updated_unix=time.time(), original_protocol_sha256=sha(SOURCE/'protocol.json')))
            if code:
                break
    finally:
        save(reservation, dict(reserved_gpus=[], reason='contextual OOM recovery released GPU0',
                               updated_unix=time.time()))
        save(RECOVERY / 'recovery_state.json', dict(state='finished', records=records,
              updated_unix=time.time(), original_protocol_sha256=sha(SOURCE/'protocol.json')))


if __name__ == '__main__':
    main()
