"""Finite two-arm paired jobs (A0, C), GPU-affine, explicit failures, idle-only admission on cloud hosts."""
import argparse,fcntl,json,os,signal,socket,subprocess,time,traceback
from pathlib import Path
from common import HERE,QUEUE as Q,PY,load,save,sha,environment,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--smoke-pair');a=p.parse_args()
    host=socket.gethostname();smoke=a.smoke_pair is not None
    assert host in ['local-gpu-host','gpu-host-8','a100-host']
    assert a.gpu in range(10 if host=='local-gpu-host' else 8 if host=='gpu-host-8' else 2)
    verify(load(Q/'execution_freeze.json'))
    ident=f'{host}_g{a.gpu}'+('_smoke' if smoke else '')
    sp=Q/'workers'/(ident+'.json');state=dict(pid=os.getpid(),host=host,gpu=a.gpu,status='waiting_for_preflight',started_at=time.time(),completed_groups=[])
    save(sp,state)
    guard=(Q/'locks'/(ident+'.worker.lock')).open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.time()+20*3600
    while not smoke and not (Q/'qa/preflight_passed.json').exists():
        if (Q/'STOP').exists() or time.time()>deadline:return
        time.sleep(10)
    if not smoke:assert load(Q/'qa/preflight_passed.json')['passed']
    groups=load(Q/'groups.json')
    if smoke:groups=[g for g in groups if g['pair_id']==a.smoke_pair];assert len(groups)==1
    else:
        assigned=set(load(Q/'allocation.json')['groups_by_domain']['3090' if host=='local-gpu-host' else 'cloud'])
        groups=[g for g in groups if g['pair_id'] in assigned]
    child=None;failures=[];admitted=False
    folder='smoke' if smoke else 'runs'
    try:
        # Busy project GPUs get a real queued worker, without preempting work.
        consecutive=0
        while not admitted:
            vals=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            free,util=map(int,vals)
            consecutive=consecutive+1 if free>=(10000 if host=='local-gpu-host' else 12000) and util<=(15 if host=='local-gpu-host' else 35) else 0
            state.update(status='waiting_for_gpu',free_mib=free,util=util,updated_at=time.time());save(sp,state)
            if consecutive>=3:admitted=True;break
            if (Q/'STOP').exists() or time.time()>deadline:state['status']='admission_timeout';return
            time.sleep(10)
        phases=['all'] if smoke else ['core']
        for phase in phases:
            for group in groups:
                if (Q/'STOP').exists() or time.time()>deadline:state['status']='stopped_or_deadline';return
                receipt=Q/('smoke_receipts' if smoke else 'group_receipts')/(group['pair_id']+'.json')
                if receipt.exists():continue
                lock=(Q/'locks'/(group['pair_id']+('_smoke' if smoke else '')+'.lock')).open('a')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lock.close();continue
                owner=Q/'owners'/(group['pair_id']+('_smoke' if smoke else '')+'.json')
                if owner.exists() and load(owner)['worker']!=ident:lock.close();continue
                if not owner.exists():save(owner,dict(worker=ident,host=host,gpu=a.gpu,claimed_at=time.time()))
                jobs=group['jobs'] if phase=='all' else [j for j in group['jobs'] if (j.rsplit('_',1)[1] in ['A0', 'A1', 'A2', 'A3', 'A4'])==(phase=='core')]
                group_failed=False
                for job in jobs:
                    d=Q/folder/job
                    if d.exists():
                        if (d/'status.json').exists() and load(d/'status.json')['status']=='completed':continue
                        group_failed=True;break
                    if phase=='secondary' and not (Q/'core_receipts'/(group['pair_id']+'.json')).exists():group_failed=True;break
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                    if free<7000:raise RuntimeError('Insufficient free GPU memory; preserve existing jobs')
                    cmd=[str(PY),str(HERE/'run.py'),'--job-id',job]+(['--smoke','--smoke-steps','10'] if smoke else [])
                    env=environment(a.gpu);env['PYTHONDONTWRITEBYTECODE']='1'
                    if host=='local-gpu-host':env['CCDIFF_PLANNER_LOG_DIR']='/path/to/local/crashdojo_logs'
                    with (Q/'logs'/(job+('_smoke' if smoke else '')+'.log')).open('x') as f:
                        child=subprocess.Popen(cmd,env=env,cwd=HERE,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        state.update(status='running',phase=phase,pair_id=group['pair_id'],job_id=job,child_pid=child.pid,job_started_at=time.time());save(sp,state)
                        while child.poll() is None:
                            if (Q/'STOP').exists() or time.time()-state['job_started_at']>5400:
                                os.killpg(child.pid,signal.SIGTERM);raise RuntimeError('Stop marker or job timeout')
                            state['updated_at']=time.time();save(sp,state);time.sleep(3)
                    if child.returncode!=0:
                        failures.append(job);group_failed=True;child=None;break
                    assert load(d/'status.json')['status']=='completed'
                    if not smoke:
                        from final_metrics import score
                        score(d,load(Q/'metric_protocol.json'))
                    child=None
                if not group_failed:
                    target=receipt
                    recorded=group['jobs']
                    save(target,dict(pair_id=group['pair_id'],host=host,gpu=a.gpu,completed=True,smoke=smoke,phase=phase,finished_at=time.time(),jobs={j:{'status':sha(Q/folder/j/'status.json'),'pair_validation':sha(Q/folder/j/'pair_validation.json')} for j in recorded}))
                    state['completed_groups'].append(group['pair_id']+':'+phase)
                state.update(failures=failures);save(sp,state);lock.close()
        state.update(status='finished_with_failures' if failures else 'finished',finished_at=time.time())
    except BaseException as e:
        state.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        save(sp,state)

if __name__=='__main__':main()
