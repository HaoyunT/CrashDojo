"""Finite scene/planner groups with one shared Full and three controls on one device."""
import argparse,fcntl,json,os,signal,socket,subprocess,time,traceback
from pathlib import Path
from common import HERE,QUEUE as Q,PY,load,save,sha,environment,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--slot',default='0')
    p.add_argument('--smoke-pair');a=p.parse_args()
    host=socket.gethostname()
    assert host in ('di-20261022123127-4dzkw','di-20260929051820-z8ljh','di-20261020182309-thst9','zkyd')
    assert a.gpu in (range(8) if host in ('di-20261020182309-thst9','zkyd') else range(2))
    smoke=a.smoke_pair is not None
    verify(load(Q/'execution_freeze.json'))
    if not smoke:assert load(Q/'qa/preflight_passed.json')['passed']
    ident=f'{host}_g{a.gpu}_s{a.slot}'+('_smoke' if smoke else '')
    state=dict(pid=os.getpid(),host=host,gpu=a.gpu,slot=a.slot,status='starting',started_at=time.time(),completed_groups=[])
    sp=Q/'workers'/(ident+'.json');save(sp,state)
    worker_lock=(Q/'locks'/(ident+'.worker.lock')).open('a');fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    groups=load(Q/'groups.json')
    if smoke:groups=[g for g in groups if g['pair_id']==a.smoke_pair];assert len(groups)==1
    else:
      allocation=load(Q/'allocation.json')
      domain='3090' if host=='zkyd' else 'cloud'
      assigned=set(allocation['groups_by_domain'][domain])
      groups=[g for g in groups if g['pair_id'] in assigned]
      assert len(groups)==len(assigned)
    child=None
    try:
      for group in groups:
        if (Q/'STOP').exists():state['status']='stopped_by_user';return
        receipt=Q/('smoke_receipts' if smoke else 'group_receipts')/(group['pair_id']+'.json')
        if receipt.exists():continue
        lock=(Q/'locks'/(group['pair_id']+('_smoke' if smoke else '')+'.lock')).open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        if receipt.exists():lock.close();continue
        # A failed/partial group is retained for review, never silently retried.
        folder='smoke' if smoke else 'runs'
        if any((Q/folder/j).exists() for j in group['jobs']):lock.close();continue
        state.update(status='running',pair_id=group['pair_id'],updated_at=time.time());save(sp,state)
        for job in group['jobs']:
          free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
          if free<7000:raise RuntimeError('Less than 7 GiB free before group job; preserve pending group')
          cmd=[str(PY),str(HERE/'run.py'),'--job-id',job]
          if smoke:cmd+=['--smoke','--smoke-steps','10']
          env=environment(a.gpu)
          if host=='zkyd':env['CCDIFF_PLANNER_LOG_DIR']='/data2/yanglei/crashdojo_common100_20261110/logs'
          with (Q/'logs'/(job+('_smoke' if smoke else '')+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,env=env,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(job_id=job,child_pid=child.pid,job_started_at=time.time());save(sp,state)
            while child.poll() is None:
              if (Q/'STOP').exists() or time.time()-state['job_started_at']>5400:
                os.killpg(child.pid,signal.SIGTERM);raise RuntimeError('Stop marker or 90-minute job timeout')
              state['updated_at']=time.time();save(sp,state);time.sleep(3)
          assert child.returncode==0,job+' failed; original record preserved'
          d=Q/folder/job;assert load(d/'status.json')['status']=='completed'
          if not smoke:
            from final_metrics import score
            score(d,load(Q/'metric_protocol.json'))
          child=None
        save(receipt,dict(pair_id=group['pair_id'],host=host,gpu=a.gpu,completed=True,smoke=smoke,
          finished_at=time.time(),jobs={j:{'status':sha(Q/folder/j/'status.json'),'pair_validation':sha(Q/folder/j/'pair_validation.json')} for j in group['jobs']}))
        state['completed_groups'].append(group['pair_id']);save(sp,state);lock.close()
      state.update(status='finished',finished_at=time.time())
    except BaseException as e:
      state.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
      if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
      save(sp,state)

if __name__=='__main__':main()
