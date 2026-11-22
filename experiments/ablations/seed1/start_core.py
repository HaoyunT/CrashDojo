import sys,time,subprocess
from common import *
gpu=int(sys.argv[1]);env=environment(gpu)
rc=subprocess.run([str(PY),str(HERE/'worker.py'),'--gpu',str(gpu),'--smoke','--variants','A1','A2','A3','A4'],env=env).returncode
first=load(QUEUE/'cohort.json')[0]['scene']
for _ in range(180):
 paths=[QUEUE/'smoke'/f'{first}_s1_{p}_{v}'/'status.json' for p in ['idm','diffusion_planner'] for v in ['A0','A1','A2','A3','A4']]
 states=[load(x)['status'] if x.exists() else 'pending' for x in paths]
 if 'failed' in states:raise RuntimeError('Switch validation failed; see retained smoke results')
 if all(s=='completed' for s in states):break
 time.sleep(5)
else:raise RuntimeError('Switch validation timeout')
subprocess.run([str(PY),str(HERE/'worker.py'),'--gpu',str(gpu),'--variants','A0','A1','A2','A3','A4'],env=env,check=True)
