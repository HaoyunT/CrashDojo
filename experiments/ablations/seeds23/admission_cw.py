"""Start smoke checks and bounded queued workers for the commit-window common-100 campaign; occupied GPUs queue."""
from pathlib import Path
import json,socket,subprocess,sys,time
R=Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
L=R/'launch/crashdojo_common100_panelA_h7_s23_3090_20261112';Q=R/'eval'/L.name
host=socket.gethostname();assert host in ['gpu-host-8','a100-host']
assert (Q/'execution_freeze.json').exists(),'Inputs not frozen'
sys.path.insert(0,str(L))
from common import load,save,PY,environment
receipt=Q/'qa'/('launch_'+host+'.json');assert not receipt.exists()
allgroups=load(Q/'allocation.json')['groups_by_domain']['cloud']
smoke={'gpu-host-8':[(0,allgroups[0]),(4,allgroups[1])],'a100-host':[(0,allgroups[2]),(1,allgroups[3])]}[host]
workers=[]
for gpu,pair in smoke:
    with (Q/'logs'/f'{host}_g{gpu}_smoke.log').open('x') as f:
        p=subprocess.Popen([str(PY),str(L/'worker_cw.py'),'--gpu',str(gpu),'--smoke-pair',pair],env=environment(gpu),cwd=L,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append(dict(gpu=gpu,pid=p.pid,smoke=True,pair=pair))
for gpu in range(8 if host=='gpu-host-8' else 2):
    with (Q/'logs'/f'{host}_g{gpu}_queued.log').open('x') as f:
        p=subprocess.Popen([str(PY),str(L/'worker_cw.py'),'--gpu',str(gpu)],env=environment(gpu),cwd=L,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append(dict(gpu=gpu,pid=p.pid,smoke=False,admission='wait for technical gate and 3 idle samples (free>=12000 MiB, util<=15%); no preemption'))
save(receipt,dict(at=time.time(),host=host,workers=workers,smoke_pairs=[p for g,p in smoke]))
print(json.dumps(load(receipt)),flush=True)
