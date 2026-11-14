"""Sequential, memory-gated retries preserving every failed campaign attempt."""
import argparse,fcntl,os,socket,subprocess,time
from common import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--min-free-mib',type=int,default=14000)
    p.add_argument('--plan',required=True)
    a=p.parse_args();plan=load(a.plan);env=environment(a.gpu)
    identity=f'{socket.gethostname()}_gpu{a.gpu}_retry_{os.getpid()}'
    statepath=QUEUE/'workers'/(identity+'.json')
    state=dict(host=socket.gethostname(),pid=os.getpid(),gpu=a.gpu,slot='oom_retry',started_unix=time.time(),status='starting',plan=a.plan,completed_retries=[])
    save(statepath,state)
    for jobid in plan['jobs']:
        dest=QUEUE/'runs'/jobid
        current=load(dest/'status.json')
        if current['status']=='completed':continue
        assert current['status']=='failed',(jobid,current['status'])
        # Validate the original failure including A8's nested generation stage.
        errors='\n'.join(x.read_text() for x in dest.rglob('status.json'))
        assert 'CUDA out of memory' in errors or 'OutOfMemoryError' in errors,jobid
        retry_claim=QUEUE/'retry_claims'/jobid;retry_claim.parent.mkdir(exist_ok=True)
        try:retry_claim.mkdir()
        except FileExistsError:continue
        save(retry_claim/'owner.json',dict(host=socket.gethostname(),pid=os.getpid(),gpu=a.gpu,claimed_unix=time.time()))
        lock=open(QUEUE/'locks'/(jobid+'.lock'),'a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        claim=QUEUE/'claims'/jobid;claim.mkdir(parents=True,exist_ok=True)
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=a.min_free_mib:break
            state.update(status='waiting_for_memory',job_id=jobid,free_mib=free,required_mib=a.min_free_mib);save(statepath,state);time.sleep(20)
        archive=QUEUE/'failed_attempts'/plan['id']/jobid
        archive.mkdir(parents=True,exist_ok=False)
        dest.rename(archive/'run')
        logfile=QUEUE/'logs'/(jobid+'.log')
        if logfile.exists():logfile.rename(archive/'run.log')
        record=dict(job_id=jobid,original_status=current,original_attempt=str(archive/'run'),retry_started_unix=time.time(),host=socket.gethostname(),gpu=a.gpu,free_mib_before_launch=free,configuration_changed=False)
        save(archive/'retry.json',record)
        state.update(status='running',job_id=jobid,archive=str(archive),free_mib_before_launch=free);save(statepath,state)
        with logfile.open('x') as log:
            child=subprocess.Popen([str(PY),str(HERE/'run.py'),'--job-id',jobid],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=child.pid;save(statepath,state)
            try:rc=child.wait(timeout=5400)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,15)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait()
                rc=124
        record.update(retry_returncode=rc,retry_finished_unix=time.time());save(archive/'retry.json',record)
        lock.close()
        if rc:
            state.update(status='stopped_after_retry_failure',last_returncode=rc);save(statepath,state);raise SystemExit(rc)
        state['completed_retries'].append(jobid);state['last_finished_unix']=time.time();save(statepath,state)
    state.update(status='finished',finished_unix=time.time());save(statepath,state)
    # Rejoin the ordinary queue after the bounded recovery batch.
    os.execv(str(PY),[str(PY),str(HERE/'distributed_worker.py'),'--gpu',str(a.gpu),'--slot','after_retry','--min-free-mib',str(a.min_free_mib)])

if __name__=='__main__':main()
