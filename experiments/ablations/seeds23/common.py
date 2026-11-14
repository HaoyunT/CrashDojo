"""Isolated ablation paths, physical GPU binding, and frozen-source audit."""
import hashlib, json, os, subprocess
from pathlib import Path
ROOT=Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
HERE=Path(__file__).resolve().parent
QUEUE=ROOT/'eval'/HERE.name
REPO=ROOT/'code/CCDiff_M2c_planner'
OVERLAY=HERE/'overlay'
BASE=Path('/root/path/to/workspace/home')
PY=BASE/'miniconda3/envs/ccdiff/bin/python'
MAP_PY=BASE/'miniconda3/envs/safesim/bin/python'
OLD=ROOT/'launch/causal_guidance_v26_contextual_val150_20261030'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.%s.tmp'%os.getpid());t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def environment(gpu):
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
 uuids={int(x.split(',')[0]):x.split(',')[1].strip() for x in out.splitlines()}
 env=os.environ.copy()
 env.update(CUDA_VISIBLE_DEVICES=str(gpu),CCDIFF_PLANNER_GPU='',CRASHDOJO_EXPECTED_GPU_UUID=uuids[gpu],
 CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 OPENBLAS_CORETYPE='Haswell',NPY_DISABLE_CPU_FEATURES='AVX512F',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',
 NVIDIA_TF32_OVERRIDE='0',PYTHONUNBUFFERED='1',PYTHONUTF8='1',LC_ALL='C.UTF-8',LANG='C.UTF-8',WANDB_MODE='disabled',
 PYTHONPATH=':'.join(map(str,[HERE,OVERLAY,OLD/'map_deps',BASE/'planner_sources_20260810/nuplan_source/nuplan-devkit-master',REPO,REPO/'third_party/tbsim',BASE/'CCDiff/third_party/trajdata/src'])),CRASHDOJO_RULE_ADAPTER_DIR=str(OLD/'rule_adapter'))
 return env
def verify(protocol):
 for group in ['source_sha256','input_sha256']:
  for name,digest in protocol[group].items():
   if sha(name)!=digest:raise RuntimeError('Frozen ablation source/input changed: '+name)
