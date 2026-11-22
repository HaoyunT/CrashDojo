"""Install the previously proven eval-only compact attention for a DDPM trial."""

import os
from pathlib import Path
import runpy
import sys


ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO = ROOT / 'code/CCDiff_M2c_planner'
HERE = ROOT / 'launch/baselines_native_v6_remaining225_20261101'
COMPACT = ROOT / 'launch/oom_rerun_20261101'

os.environ.update(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', NVIDIA_TF32_OVERRIDE='0',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.path[:0] = [str(COMPACT), str(REPO), str(REPO / 'third_party/tbsim'),
                '/path/to/workspace/home/CCDiff/third_party/trajdata/src']
from compact_attention import install

output = Path(sys.argv[sys.argv.index('--output') + 1])
install(output / 'compact_attention_audit')
sys.argv[0] = str(HERE / 'ccdiff_adapter/entry_ccdiff.py')
runpy.run_path(sys.argv[0], run_name='__main__')
