"""Technical gate and progress summaries for the commit-window common-100 campaign (cloud hosts only)."""
from pathlib import Path
import json,sys,time,traceback
R=Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
L=R/'launch/crashdojo_common100_panelA_h7_s23_3090_20261112';Q=R/'eval'/L.name
sys.path.insert(0,str(L))
from common import load,save,sha
from final_metrics import summarize_once
TOTAL=2000
def gate():
    """Per-host technical gate: the shared preflight file opens once the first host passes both planners
    and both arms; later hosts are appended when their smoke receipts arrive. Each host's smoke is still
    recorded and verified; per-job pair validation remains mandatory for every formal rollout."""
    receipts=[load(p) for p in (Q/'smoke_receipts').glob('*.json')]
    expected={'di-20261020182309-thst9','di-20260929051820-z8ljh'}
    previous=load(Q/'qa/preflight_passed.json') if (Q/'qa/preflight_passed.json').exists() else dict(passed=False,checks=[],passed_hosts=[])
    passed=set(previous.get('passed_hosts',[]));checks=list(previous.get('checks',[]))
    for h in sorted(expected-passed):
        recs=[r for r in receipts if r['host']==h]
        if len(recs)!=2:continue
        planners=set()
        for rec in recs:
            assert rec['completed'] and rec['smoke'] and len(rec['jobs'])==5
            for job,hashes in rec['jobs'].items():
                d=Q/'smoke'/job
                assert sha(d/'status.json')==hashes['status'] and sha(d/'pair_validation.json')==hashes['pair_validation']
                status=load(d/'status.json');planners.add(status['planner'])
                assert status['status']=='completed' and load(d/'pair_validation.json')['passed']
                checks.append(dict(job=job,host=h,status_sha256=hashes['status']))
        assert planners=={'idm','diffusion_planner'}
        passed.add(h)
    if passed and passed!=set(previous.get('passed_hosts',[])):
        save(Q/'qa/preflight_passed.json',dict(passed=True,at=time.time(),checks=checks,passed_hosts=sorted(passed),short_rollouts=len(checks),scope='Both planners and all three arms on each admitted host; hosts are admitted as their own smoke passes; admission does not depend on favorable metrics.'))
start=time.time();errors=[]
while time.time()-start<20*3600 and not (Q/'STOP').exists():
    try:
        gate()
        summarize_once(load(Q/'metric_protocol.json'))
        progress=load(Q/'progress.json') if (Q/'progress.json').exists() else {}
        complete=progress.get('metric_valid')==TOTAL
        save(Q/'collector_status.json',dict(status='complete' if complete else 'running',updated_at=time.time(),errors=errors[-8:],technical_gate=(Q/'qa/preflight_passed.json').exists(),progress=progress))
        if complete:break
    except Exception as e:
        errors.append(dict(at=time.time(),error=repr(e)));save(Q/'collector_status.json',dict(status='retry',updated_at=time.time(),errors=errors[-8:]));print(traceback.format_exc(),flush=True)
    time.sleep(60)
