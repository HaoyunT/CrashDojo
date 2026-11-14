"""Shared-GPU dispatcher, locking only this campaign's jobs."""
import argparse,fcntl,json,os,subprocess,time,socket
from common import *
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-scene');p.add_argument('--variants',nargs='+',default=['A0','A4','A5','A6','A7']);args=p.parse_args()
jobs=load(QUEUE/'run_manifest.json')['jobs'];env=environment(args.gpu)
state_path=QUEUE/'workers'/f'{socket.gethostname()}_gpu{args.gpu}_pid{os.getpid()}_{"smoke" if args.smoke else "runs"}.json'
state=dict(pid=os.getpid(),host=socket.gethostname(),gpu=args.gpu,status='starting',started_unix=time.time(),shared_gpu_authorized=True);save(state_path,state)
if args.smoke:
    first=args.smoke_scene or load(QUEUE/'cohort.json')[0]['scene']
    jobs=[j for j in jobs if j['scene']==first and j['stream']==1]
jobs=[j for j in jobs if j['variant'] in args.variants]
for j in jobs:
    (QUEUE/'locks').mkdir(exist_ok=True);lock=open(QUEUE/'locks'/(('smoke_' if args.smoke else '')+j['job_id']+'.lock'),'w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();continue
    out=QUEUE/('smoke' if args.smoke else 'runs')/j['job_id']
    if out.exists():lock.close();continue
    state.update(status='running',job_id=j['job_id']);save(state_path,state)
    log=QUEUE/'logs'/((('smoke_' if args.smoke else '')+j['job_id'])+'.log');log.parent.mkdir(exist_ok=True)
    cmd=[str(PY),str(HERE/'run.py'),'--job-id',j['job_id']]+(['--smoke'] if args.smoke else [])
    with log.open('x') as f:
        child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(child_pid=child.pid);save(state_path,state)
        try:rc=child.wait(timeout=5400)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,15)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait()
            rc=124
    state.update(last_returncode=rc,last_finished_unix=time.time());save(state_path,state)
    lock.close()
    if args.smoke and rc:state.update(status='smoke_failed');save(state_path,state);break
else:state.update(status='finished',finished_unix=time.time());save(state_path,state)
