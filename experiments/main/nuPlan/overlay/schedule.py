"""Run the authorized frozen matrix; preserve every failed or no-action cell."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
REPO=ROOT/'code/CCDiff_M2c_planner'
PROTOCOL=REPO/'benchmark50_protocol_20261021_v3/protocol.json'
OVERLAY=REPO/'benchmark50_overlay_20261021_v3'
OUT=ROOT/'eval/benchmark50_run_20261021_v3'
PY='/root/path/to/workspace/home/miniconda3/envs/ccdiff/bin/python'
SA=PY.replace('/ccdiff/','/safesim/')
P=json.loads(PROTOCOL.read_text())
assert P['execution_authorized'] and P['planned_rollouts']==500
OUT.mkdir(exist_ok=True)
lock=(OUT/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'status.json').exists(),'Resume needs explicit reconciliation; never duplicate active jobs'
gate=[j for j in P['jobs'] if j['arm']=='guided' and j['job_id'] in P['startup_gate_job_ids']]
rest=[j for j in P['jobs'] if j not in gate]
# Paired episodes stay near each other. Every planner and all 50 scenes remain.
rest.sort(key=lambda j:(j['scene'],P['planners'].index(j['planner']),j['arm']))
pending=gate+rest;active=[];finished=[];halt=None
for arm in ['guided','noguide']:
    path=ROOT/'eval'/('benchmark50_'+arm+'_20261021_v3');path.mkdir()

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def external_counts():
    counts=[0,0]
    for arm in ['guided','noguide']:
        for path in (ROOT/'eval'/('development_timing_v9_'+arm+'_20261021')).glob('C*/job.json'):
            if not path.with_name('status.json').exists():counts[int(json.loads(path.read_text())['gpu'])]+=1
    for version in ['v1','v2']:
        for arm in ['guided','noguide']:
            for path in (ROOT/'eval'/('benchmark50_'+arm+'_20261021_'+version)).glob('scene-*/job.json'):
                if not path.with_name('status.json').exists():counts[int(json.loads(path.read_text())['gpu'])]+=1
    for path in (ROOT/'eval/benchmark50_future_reference_probe_20261021').glob('scene-*/job.json'):
        if not path.with_name('status.json').exists():counts[int(json.loads(path.read_text())['gpu'])]+=1
    return counts

def start(job,gpu):
    for path,digest in P['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    dest=ROOT/'eval'/job['stage']/job['job_id'];assert not dest.exists()
    command=[PY,str(OVERLAY/'run.py'),'--request',job['request'],'--selection-manifest',job['manifest'],
             '--selection-row',str(job['row']),'--planner',job['planner'],'--gpu',str(gpu),
             '--steps','100','--samples','5','--seed','1','--output',str(dest),
             '--formal','--execute-formal','--frozen-protocol',str(PROTOCOL)]
    log=(dest.parent/(job['job_id']+'_launcher.log')).open('w')
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    active.append(dict(proc=proc,log=log,job=job,gpu=gpu,started=time.time()))
    print('START',job['arm'],job['job_id'],'GPU',gpu,'PID',proc.pid,flush=True)

while pending or active:
    for item in active[:]:
        code=item['proc'].poll()
        if code is None:continue
        job=item['job'];dest=ROOT/'eval'/job['stage']/job['job_id'];dest.mkdir(exist_ok=True)
        item['log'].close()
        if not (dest/'status.json').exists():
            atomic(dest/'status.json',dict(status='failed',returncode=code,formal=True,wall_seconds=time.time()-item['started']))
        checkcode=None
        if code==0:
            with (dest/'runtime_check.log').open('w') as f:
                checkcode=subprocess.run([SA,str(OVERLAY/'validate_runtime.py'),str(dest)],stdout=f,stderr=subprocess.STDOUT).returncode
        result=dict(job,returncode=code,runtime_check_returncode=checkcode,wall_seconds=time.time()-item['started'])
        atomic(dest/'benchmark_job.json',result);finished.append(result);active.remove(item)
        print('DONE',job['arm'],job['job_id'],code,'CHECK',checkcode,flush=True)
        if job in gate and (code!=0 or checkcode!=0):halt='startup_runtime_gate_failed'
        # Do not burn a full matrix on a repeated integration failure.
        if len(finished)>=3 and all(x['returncode']!=0 or x['runtime_check_returncode']!=0 for x in finished[-3:]):
            halt='three_consecutive_runtime_failures'
    counts=external_counts()
    for item in active:counts[item['gpu']]+=1
    gate_done=sum(j['arm']=='guided' and j['job_id'] in P['startup_gate_job_ids'] for j in finished)
    if not halt:
        for gpu in [0,1]:
            while counts[gpu]<2 and pending:
                job=pending[0]
                if job not in gate and gate_done<len(gate):break
                pending.pop(0);start(job,gpu);counts[gpu]+=1
    status=dict(status='draining_after_failure' if halt and active else 'needs_review' if halt else 'running',
                pid=os.getpid(),updated_unix=time.time(),planned=500,completed=len(finished),
                succeeded=sum(x['returncode']==0 and x['runtime_check_returncode']==0 for x in finished),
                failures=[x for x in finished if x['returncode']!=0 or x['runtime_check_returncode']!=0],
                startup_gate_completed=gate_done,startup_gate_planned=len(gate),
                active=[dict(job=x['job'],pid=x['proc'].pid,gpu=x['gpu'],started=x['started']) for x in active],
                pending=len(pending),halt_reason=halt,formal=True,training=False)
    atomic(OUT/'status.json',status)
    atomic(OUT/'completed_jobs.json',finished)
    if halt and not active:raise SystemExit(2)
    time.sleep(5)

post=[]
for arm in ['guided','noguide']:
    name='benchmark50_'+arm+'_20261021_v3';stage=ROOT/'eval'/name
    commands=[[SA,str(OVERLAY/'evaluate_batch.py'),str(stage)],
              [PY,str(OVERLAY/'compute_metrics.py'),'--root',str(ROOT),'--base','/root/path/to/workspace/home',
               '--family',name,'--output',str(stage/'native_and_cfd.json')]]
    codes=[]
    for i,command in enumerate(commands):
        with (stage/('postprocess_'+str(i)+'.log')).open('w') as f:
            code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')).returncode
        codes.append(code)
    post+=codes;atomic(stage/'stage_status.json',dict(status='metrics_ready' if codes==[0,0] else 'needs_review',postprocess_returncodes=codes))
with (OUT/'actor_audit.log').open('w') as f:
    code=subprocess.run([SA,str(OVERLAY/'audit_actor_metrics.py')],stdout=f,stderr=subprocess.STDOUT).returncode
post.append(code)
with (OUT/'summary.log').open('w') as f:
    code=subprocess.run([SA,str(OVERLAY/'summarize.py')],stdout=f,stderr=subprocess.STDOUT).returncode
post.append(code);status.update(status='complete' if all(c==0 for c in post) and not status['failures'] else 'needs_review',
                               postprocess_returncodes=post,active=[],pending=0,updated_unix=time.time())
atomic(OUT/'status.json',status)
print('FINISHED',json.dumps(status),flush=True)
