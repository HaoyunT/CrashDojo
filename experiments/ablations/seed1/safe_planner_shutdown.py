"""Lifecycle-only repair; applied after inference and only to the owned child."""
from pathlib import Path
import json,subprocess,time
def install(directory):
    from ccdiff.planners.official_nuplan import OfficialNuPlanPlanner
    def close(self):
        worker=getattr(self,'_worker',None);record=dict(time=time.time(),pid=getattr(worker,'pid',None),forced_after_timeout=False)
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                record['forced_after_timeout']=True;worker.kill();worker.wait(timeout=10)
        record['returncode']=None if worker is None else worker.poll()
        stream=getattr(self,'_worker_log',None)
        if stream is not None and not stream.closed:stream.close()
        self._worker=None
        with (Path(directory)/'planner_shutdown_audit.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    OfficialNuPlanPlanner.close=close
