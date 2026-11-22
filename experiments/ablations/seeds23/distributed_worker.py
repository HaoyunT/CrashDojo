"""Host-qualified shared-filesystem queue; campaign-only locks and readiness gates."""
import argparse,fcntl,os,socket,subprocess,time
from common import *
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--slot',default='extra');p.add_argument('--min-free-mib',type=int,default=0);args=p.parse_args()
env=environment(args.gpu);jobs=load(QUEUE/'run_manifest.json')['jobs'];identity=f'{socket.gethostname()}_gpu{args.gpu}_{args.slot}_{os.getpid()}'
statepath=QUEUE/'workers'/(identity+'.json');state=dict(host=socket.gethostname(),pid=os.getpid(),gpu=args.gpu,slot=args.slot,started_unix=time.time(),status='starting');save(statepath,state)
failures=0
while True:
    if args.min_free_mib:
        free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<args.min_free_mib:
            state.update(status='waiting_for_memory',free_mib=free,required_mib=args.min_free_mib);save(statepath,state);time.sleep(20);continue
    ready={'A0','A3','A4'}
    if (QUEUE/'selection_runtime_ready.json').exists():ready.update(['A1','A2'])
    claimed=False;unattempted=0
    # Prioritize newly enabled components; existing workers already cover the other arms.
    order=sorted(jobs,key=lambda j:(0 if j['variant'] in ['A1','A2'] else 1))
    for j in order:
        dest=QUEUE/'runs'/j['job_id']
        if dest.exists():continue
        unattempted+=1
        if j['variant'] not in ready:continue
        if socket.gethostname()=='gpu-host-6' and j['variant'] not in ['A1','A2']:continue
        lock=open(QUEUE/'locks'/(j['job_id']+'.lock'),'a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        if dest.exists():lock.close();continue
        claim=QUEUE/'claims'/j['job_id'];claim.parent.mkdir(exist_ok=True)
        try:claim.mkdir()
        except FileExistsError:lock.close();continue
        save(claim/'owner.json',dict(host=socket.gethostname(),worker_pid=os.getpid(),gpu=args.gpu,claimed_unix=time.time()))
        claimed=True;state.update(status='running',job_id=j['job_id'],ready_variants=sorted(ready));save(statepath,state)
        logfile=QUEUE/'logs'/(j['job_id']+'.log')
        with logfile.open('x') as f:
            child=subprocess.Popen([str(PY),str(HERE/'run.py'),'--job-id',j['job_id']],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=child.pid;save(statepath,state)
            try:rc=child.wait(timeout=5400)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,15)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait()
                rc=124
        state.update(last_returncode=rc,last_finished_unix=time.time());save(statepath,state);lock.close()
        failures=failures+1 if rc else 0
        if failures>=2:
            state.update(status='stopped_after_two_failures');save(statepath,state);raise SystemExit(1)
        break
    if claimed:continue
    if not unattempted:state.update(status='finished',finished_unix=time.time());save(statepath,state);break
    state.update(status='waiting_for_ready_or_claimed',unattempted=unattempted);save(statepath,state);time.sleep(20)
