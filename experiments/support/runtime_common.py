import hashlib
import json
from pathlib import Path
import os
import socket
import time

ROOT=Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO=ROOT/'code/CCDiff_M2c_planner'
BASE=Path('/path/to/workspace/home')
PY=BASE/'miniconda3/envs/ccdiff/bin/python'
HERE=Path(__file__).resolve().parent
NAME='joint_feasibility_v19_20261026'
OVERLAY=REPO/'benchmark50_overlay_20261026_interaction_v11'
PROTOCOL=REPO/'benchmark50_protocol_joint_feasibility_v19_20261026'
QUEUE=ROOT/'eval'/NAME
HOSTS={
'a100-host':dict(hardware='A100',gpus={0:'GPU-REPLACE-WITH-DEVICE-UUID-01',1:'GPU-REPLACE-WITH-DEVICE-UUID-02'}),
'gpu-host-6b':dict(hardware='RTX4090D',gpus={2:'GPU-REPLACE-WITH-DEVICE-UUID-03',3:'GPU-REPLACE-WITH-DEVICE-UUID-04',4:'GPU-REPLACE-WITH-DEVICE-UUID-05',5:'GPU-REPLACE-WITH-DEVICE-UUID-06',6:'GPU-REPLACE-WITH-DEVICE-UUID-07',7:'GPU-REPLACE-WITH-DEVICE-UUID-08'}),
'gpu-host-6':dict(hardware='RTX4090D',gpus={2:'GPU-REPLACE-WITH-DEVICE-UUID-09',3:'GPU-REPLACE-WITH-DEVICE-UUID-10',4:'GPU-REPLACE-WITH-DEVICE-UUID-11',5:'GPU-REPLACE-WITH-DEVICE-UUID-12',6:'GPU-REPLACE-WITH-DEVICE-UUID-13',7:'GPU-REPLACE-WITH-DEVICE-UUID-14'})}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def output(job,smoke=False):
    stage='benchmark50_smoke_'+job['arm']+'_joint_feasibility_v19_20261026' if smoke else job['stage']
    return ROOT/'eval'/stage/job['job_id']
def completed(p):
    s=p/'status.json';v=p/'runtime_validation.json'
    return s.exists() and v.exists() and load(s).get('returncode')==0 and load(v).get('passed') is True

def environment(gpu):
    cfg=HOSTS[socket.gethostname()]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),CCDIFF_PLANNER_GPU='',CRASHDOJO_EXPECTED_GPU_UUID=cfg['gpus'][gpu],
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_CORETYPE='Haswell',NPY_DISABLE_CPU_FEATURES='AVX512F',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',
        NVIDIA_TF32_OVERRIDE='0',PYTHONUNBUFFERED='1',PYTHONUTF8='1',LC_ALL='C.UTF-8',LANG='C.UTF-8',WANDB_MODE='disabled',
        PYTHONPATH=':'.join(map(str,[REPO,REPO/'third_party/tbsim',BASE/'CCDiff/third_party/trajdata/src'])))
    return env
