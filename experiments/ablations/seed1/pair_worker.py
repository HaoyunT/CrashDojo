"""Run each A0/U pair sequentially on the same physical GPU."""
import argparse,fcntl,hashlib,json,os,signal,socket,subprocess,time,traceback
from pathlib import Path
from common import HERE,QUEUE as Q,PY,load,save,environment,verify,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker-id',required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--pair-id');a=ap.parse_args()
    assignment=load(Q/'allocation.json')['workers'][a.worker_id]
    assert assignment['host']==socket.gethostname()
    gpu=assignment['gpu'];suffix='_smoke' if a.smoke else ''
    lock=(Q/'locks'/(a.worker_id+suffix+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(load(Q/'execution_freeze.json'))
    if not a.smoke:
        assert load(Q/'qa/preflight_passed.json')['passed']
    pairs=[a.pair_id] if a.smoke else assignment['pairs']
    state=dict(worker_id=a.worker_id,pid=os.getpid(),host=socket.gethostname(),gpu=gpu,
        status='running',smoke=a.smoke,pairs=pairs,completed_pairs=[],started_at=time.time())
    path=Q/'workers'/(a.worker_id+suffix+'.json');save(path,state)
    current=None
    try:
        for pair in pairs:
            for variant in ['A0','U']:
                if (Q/'STOP').exists():state['status']='stopped_by_request';return
                job=pair+'_'+variant;dest=Q/('smoke' if a.smoke else 'runs')/job
                if (dest/'status.json').exists():
                    assert load(dest/'status.json')['status']=='completed','Inspect prior failure before retry'
                    if not a.smoke:assert (dest/'final_metrics.json').exists()
                    continue
                required=load(Q/'allocation.json')['pair_estimates'][pair]['required_free_mib']
                while True:
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                    if free>=required:break
                    state.update(status='waiting_for_memory',job_id=job,free_mib=free,required_mib=required,updated_at=time.time());save(path,state)
                    if (Q/'STOP').exists():state['status']='stopped_by_request';return
                    time.sleep(10)
                env=environment(gpu)
                if socket.gethostname()=='local-gpu-host':env['CCDIFF_PLANNER_LOG_DIR']='/path/to/local/crashdojo_ablation_20261108/logs'
                cmd=[str(PY),str(HERE/'run.py'),'--job-id',job]
                if a.smoke:cmd+=['--smoke','--smoke-steps','10']
                logfile=Q/'logs'/(job+suffix+'.log')
                with logfile.open('x') as log:
                    current=subprocess.Popen(cmd,env=env,cwd=HERE,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    started=time.time()
                    state.update(status='running',job_id=job,child_pid=current.pid,job_started_at=started,free_mib=free)
                    while current.poll() is None:
                        state['updated_at']=time.time();save(path,state)
                        if time.time()-started>5400:
                            os.killpg(current.pid,signal.SIGTERM)
                            try:current.wait(timeout=15)
                            except subprocess.TimeoutExpired:os.killpg(current.pid,signal.SIGKILL);current.wait()
                            raise RuntimeError('Owned job exceeded 5400-second execution timeout')
                        time.sleep(3)
                if current.returncode or load(dest/'status.json')['status']!='completed':
                    raise RuntimeError('Job failed: '+job+'; inspect '+str(logfile))
                if not a.smoke:
                    from final_metrics import score
                    score(dest,load(Q/'metric_protocol.json'))
                current=None
            if not a.smoke:
                inventory={}
                for v in ['A0','U']:
                    d=Q/'runs'/(pair+'_'+v)
                    inventory[v]={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}
                save(Q/'paired_receipts'/(pair+'.json'),dict(pair_id=pair,completed=True,host=socket.gethostname(),
                    gpu=gpu,files=inventory,completed_at=time.time()))
            state['completed_pairs'].append(pair);state['updated_at']=time.time();save(path,state)
        state.update(status='finished',finished_at=time.time())
    except BaseException as e:
        state.update(status='failed',error=repr(e),traceback=traceback.format_exc(),updated_at=time.time())
        raise
    finally:
        if current is not None and current.poll() is None:
            os.killpg(current.pid,signal.SIGTERM)
        save(path,state)
if __name__=='__main__':main()
