import hashlib
import json
from pathlib import Path
import os
import socket
import time

ROOT=Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO=ROOT/'code/CCDiff_M2c_planner'
BASE=Path('/root/path/to/workspace/home')
PY=BASE/'miniconda3/envs/ccdiff/bin/python'
HERE=Path(__file__).resolve().parent
NAME='joint_feasibility_v19_20261026'
OVERLAY=REPO/'benchmark50_overlay_20261026_interaction_v11'
PROTOCOL=REPO/'benchmark50_protocol_joint_feasibility_v19_20261026'
QUEUE=ROOT/'eval'/NAME
HOSTS={
'di-20260929051820-z8ljh':dict(hardware='A100',gpus={0:'GPU-ab476251-2fb0-5095-91ff-f9a273f95d13',1:'GPU-0a59120a-ef94-2c1b-f7f8-e7215c4a4c77'}),
'di-20261020200501-6dpnl':dict(hardware='RTX4090D',gpus={2:'GPU-6249c15b-70d0-b8dc-d454-f7add218d0f2',3:'GPU-bff6484b-f8ef-18f2-491a-e0dbd525e5d0',4:'GPU-4f5728fc-ebc5-f756-c567-ee134270634f',5:'GPU-6cea5f47-1c83-8b51-2152-db1d10bb91f7',6:'GPU-199097ad-4c0c-2fdd-581d-0a2c4fb778f3',7:'GPU-9c2764a3-6033-c32a-a729-af2e20b8875d'}),
'di-20261022123127-4dzkw':dict(hardware='RTX4090D',gpus={2:'GPU-7c86d438-ff63-f88e-62a3-6d36f698da9d',3:'GPU-98fadfb4-1dc2-e0fd-b7cb-790afec3fcf7',4:'GPU-ff6612b8-3294-015c-3616-2852e2a1d783',5:'GPU-5c4ab91a-c635-ce08-f7ee-b42c53743c33',6:'GPU-54e1c2fb-6c25-6794-e240-09b7a1179393',7:'GPU-819fa7ae-8f04-0875-35c0-acffb7c3566a'})}

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
