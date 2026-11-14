"""Query the modern lane implementation in its own dependency environment."""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import numpy as np

BASE = Path('/root/path/to/workspace/home')
ROOT = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')


class CanonicalLaneClient:
    def __init__(self, directory, reference):
        directory = Path(directory)
        spec = directory / 'lane_config.json'
        spec.write_text(json.dumps(dict(cache_location=reference['cache_location'],
                                        params=reference['lane_params'])))
        contextual = ROOT / 'launch/causal_guidance_v26_contextual_val150_20261030'
        repo = ROOT / 'code/CCDiff_M2c_planner'
        paths = [contextual, contextual / 'map_deps', repo, repo / 'third_party/tbsim',
                 BASE / 'CCDiff/third_party/trajdata/src']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=':'.join(map(str, paths)))
        self.log = (directory / 'canonical_lane_worker.log').open('w')
        self.worker = subprocess.Popen([str(BASE / 'miniconda3/envs/ccdiff/bin/python'),
            str(Path(__file__).with_name('lane_worker.py')), str(spec)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        atexit.register(self.close)
        ready = self.receive()
        if not ready.get('ready'):
            raise RuntimeError('Canonical lane worker failed')
        self.provenance = ready['provenance']

    def receive(self):
        if not select.select([self.worker.stdout], [], [], 60)[0]:
            raise TimeoutError('Canonical lane query timed out')
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError('Canonical lane worker exited')
        result = json.loads(line)
        if result.get('ok') is False:
            raise RuntimeError(result['error'])
        return result

    def __call__(self, ego):
        message = {key: np.asarray(ego[key]).tolist() for key in ['world_from_agent', 'agent_from_world']}
        message['agent_hist'] = np.asarray(ego['agent_hist'])[:, -1:].tolist()
        message['map_names'] = list(ego['map_names'])
        self.worker.stdin.write(json.dumps(message, allow_nan=False) + '\n')
        self.worker.stdin.flush()
        return np.asarray(self.receive()['lanes'], dtype=np.float32)

    def close(self):
        if getattr(self, 'worker', None) is not None and self.worker.poll() is None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()
        if getattr(self, 'log', None) is not None and not self.log.closed:
            self.log.close()
