"""Isolated fixed-scene baseline evaluation paths and artifact helpers."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
HERE = ROOT / 'launch/baselines_native_v6_remaining225_20261101'
QUEUE = ROOT / 'eval/baselines_native_v6_remaining225_20261101'
SOURCE = ROOT / 'eval/causal_guidance_v26_contextual_full_eval_20261101'
REPO = ROOT / 'code/CCDiff_M2c_planner'
BASE = Path('/root/path/to/workspace/home')
PYTHON = BASE / 'miniconda3/envs/ccdiff/bin/python'
PYTHON_LEGACY = BASE / 'miniconda3/envs/safesim/bin/python'
METHODS = ['safesim', 'ccdiff_ddpm', 'ccdiff_fm']
PLANNERS = ['idm', 'strive', 'pdm_closed', 'diffusion_planner', 'flow_planner']


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment(gpu):
    result = os.environ.copy()
    result.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  OPENBLAS_CORETYPE='Haswell', PYTHONUNBUFFERED='1',
                  PYTHONUTF8='1', LC_ALL='C.UTF-8',
                  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', NVIDIA_TF32_OVERRIDE='0',
                  NPY_DISABLE_CPU_FEATURES='AVX512F',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_MODE='disabled')
    return result


def command(job, output, steps=None):
    legacy = job['method'] == 'safesim'
    entry = HERE / ('safesim_adapter/entry_safesim.py' if legacy
                    else 'ccdiff_adapter/entry_ccdiff.py')
    result = [str(PYTHON_LEGACY if legacy else PYTHON), str(entry),
              '--job-file', str(QUEUE / 'jobs' / (job['job_id'] + '.json')),
              '--output', str(output)]
    if steps is not None:
        result += ['--steps', str(steps)]
    return result


def verify_sources(protocol):
    for path, expected in protocol['runtime_sha256'].items():
        if sha(path) != expected:
            raise AssertionError('Frozen baseline runtime changed: ' + path)


def status_counts(protocol):
    counts = {key: 0 for key in ['completed', 'running', 'queued', 'failed',
                                'blocked_selection', 'blocked_input']}
    for job in protocol['jobs']:
        path = QUEUE / 'status' / (job['job_id'] + '.json')
        state = load(path)['state'] if path.exists() else job.get('initial_state', 'queued')
        counts[state if state in counts else 'failed'] += 1
    return counts
