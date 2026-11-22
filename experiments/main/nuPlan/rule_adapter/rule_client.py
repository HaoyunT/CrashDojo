"""One observation/action contract for all four generators and three rule planners."""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np
import torch
from ccdiff.planners.base import EgoPlannerPolicy
from ccdiff.planners.official_nuplan import OfficialNuPlanPlanner
from tbsim.policies.common import Action

R = Path('/path/to/workspace/Friday/safesim_planner_20261012_01')
B = Path('/path/to/workspace/home')


def array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, bytes):
        return x.decode('utf-8')
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        return plain(array(x).tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def normalized_observation(obs, agents=None):
    keys = ['agent_hist', 'neigh_hist', 'num_neigh', 'neigh_types', 'extent',
            'all_other_agents_extents', 'all_other_agents_types', 'neigh_extents',
            'curr_speed', 'world_from_agent', 'agent_from_world', 'centroid', 'yaw',
            'map_names', 'scene_index', 'neighbor_indices', 'track_id']
    out = {k: plain(obs[k]) for k in keys if k in obs}
    if 'agent_hist' not in out:
        n = len(obs['history_positions'])
        out['agent_hist'] = [OfficialNuPlanPlanner._history_from_parsed(obs, i).tolist() for i in range(n)]
        out['neigh_hist'] = [[OfficialNuPlanPlanner._history_from_parsed(obs, i, j).tolist()
                              for j in range(obs['all_other_agents_history_positions'].shape[1])] for i in range(n)]
    for k in ['agent_hist', 'neigh_hist']:
        value = np.asarray(out[k])
        if value.shape[-1] == 9:
            value = value[..., [0, 1, 3, 4, 5, 6, 7, 8]]
        if value.shape[-1] != 8:
            raise ValueError('Unknown history state layout: ' + k + str(value.shape))
        out[k] = value.tolist()
    if 'neigh_types' not in out:
        out['neigh_types'] = out['all_other_agents_types']
    if 'all_other_agents_extents' not in out:
        out['all_other_agents_extents'] = out['neigh_extents']
    if 'closest_lane_point' in obs.get('extras', {}):
        out['extras'] = {'closest_lane_point': plain(obs['extras']['closest_lane_point'])}
    # Legacy simulators collate ego with an empty neighbor list while passing
    # the live simulated traffic separately. Rebuild history in the ego frame.
    if agents is not None:
        if len(out['agent_hist']) != 1:
            raise ValueError('Rule evaluation requires one scene per batch')
        histories, extents, identities = [], [], []
        ego_scene = int(array(obs['scene_index'])[0])
        ego_track = int(array(obs['track_id'])[0])
        inverse = array(obs['agent_from_world'])[0]
        hlen = len(out['agent_hist'][0])
        for j in range(len(agents['centroid'])):
            if int(array(agents['scene_index'])[j]) != ego_scene or int(array(agents['track_id'])[j]) == ego_track:
                continue
            raw = array(agents['agent_hist'])[j] if 'agent_hist' in agents else OfficialNuPlanPlanner._history_from_parsed(agents, j)
            if raw.shape[-1] == 9:
                raw = raw[..., [0, 1, 3, 4, 5, 6, 7, 8]]
            raw = raw.copy()
            transform = inverse @ array(agents['world_from_agent'])[j]
            raw[:, :2] = raw[:, :2] @ transform[:2, :2].T + transform[:2, 2]
            raw[:, 2:4] = raw[:, 2:4] @ transform[:2, :2].T
            raw[:, 4:6] = raw[:, 4:6] @ transform[:2, :2].T
            yaw = np.arctan2(raw[:, 6], raw[:, 7]) + np.arctan2(transform[1, 0], transform[0, 0])
            raw[:, 6], raw[:, 7] = np.sin(yaw), np.cos(yaw)
            if not np.isfinite(raw[-1]).all() or np.linalg.norm(raw[-1, :2]) > 50:
                continue
            if len(raw) < hlen:
                raw = np.concatenate([np.full((hlen-len(raw), 8), np.nan), raw])
            histories.append(raw[-hlen:].tolist())
            extents.append(plain(agents['extent'][j]))
            identities.append(int(array(agents['track_id'])[j]))
        n = len(histories)
        out.update(neigh_hist=[histories or np.zeros((1, hlen, 8)).tolist()], num_neigh=[n],
                   neigh_types=[[3] * n or [0]], all_other_agents_types=[[3] * n or [0]],
                   all_other_agents_extents=[extents or [[0, 0, 0]]], neighbor_indices=[identities])
    return out


class RulePlannerClient(EgoPlannerPolicy):
    supports_candidate_future = False

    def __init__(self, device, kind):
        super().__init__(device)
        self.kind = self.name = kind
        self.calls = 0
        self.out = Path(os.environ['RULE_OUTPUT'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = (self.out / 'rule_worker.log').open('w')
        env = os.environ.copy()
        if kind == 'pdm_closed':
            python = B / 'miniconda3/envs/ctg/bin/python'
            paths = [R / 'code/tuplan_garage_rule_v1', B / 'planner_sources_20260810/nuplan_source/nuplan-devkit-master',
                     B / 'venvs/nuplan_planners_20260810/lib/python3.9/site-packages',
                     B / 'CCDiff/third_party/trajdata/src']
        else:
            python = B / 'miniconda3/envs/ccdiff/bin/python'
            src = R / 'code/CCDiff_M2c_planner'
            paths = [src, src / 'third_party/tbsim', B / 'CCDiff/third_party/trajdata/src']
        env['PYTHONPATH'] = ':'.join(map(str, paths))
        self.worker = subprocess.Popen([str(python), str(Path(__file__).with_name('rule_worker.py')), kind],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                       text=True, bufsize=1, env=env)
        atexit.register(self.close)
        ready = self.receive()
        if not ready.get('ready'):
            raise RuntimeError(ready)
        self.ready = ready

    def receive(self):
        if not select.select([self.worker.stdout], [], [], 300)[0]:
            raise TimeoutError('Rule planner worker timed out; inspect rule_worker.log')
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError('Rule worker exited; inspect rule_worker.log')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    def plan(self, ego_observation, candidate_agents_future=None, strategy_depth=0, agent_observation=None, **kwargs):
        if strategy_depth or candidate_agents_future is not None:
            raise ValueError('Rule planner benchmark uses observed history only, K=0')
        obs = normalized_observation(ego_observation, agent_observation)
        request = {'observation': obs, 'step_index': int(kwargs.get('step_index', self.calls * 5))}
        if self.calls == 0:
            (self.out / 'rule_first_observation.json').write_text(json.dumps(request))
        start = time.monotonic()
        self.worker.stdin.write(json.dumps(request) + '\n')
        self.worker.stdin.flush()
        reply = self.receive()
        p = torch.as_tensor(reply['positions'], dtype=torch.float32, device=self.device)
        y = torch.as_tensor(reply['yaws'], dtype=torch.float32, device=self.device)
        assert p.ndim == 3 and p.shape[1:] == (52, 2) and y.shape == p.shape[:2] + (1,)
        assert torch.isfinite(p).all() and torch.isfinite(y).all(), 'Nonfinite rule planner output'
        self.calls += 1
        attestation = dict(planner=self.kind, calls=self.calls, last_call_seconds=time.monotonic() - start,
                           output_finite=True, positions_shape=list(p.shape), ready=self.ready,
                           worker_info=reply.get('info'), training=False, observed_history_only=True)
        attestation['observed_neighbor_count'] = obs.get('num_neigh')
        tmp = self.out / 'rule_planner_attestation.tmp'
        tmp.write_text(json.dumps(attestation, ensure_ascii=False, indent=2))
        tmp.replace(self.out / 'rule_planner_attestation.json')
        return Action(positions=p, yaws=y), {'candidate_future_consumed': torch.zeros(p.shape[0], dtype=torch.bool, device=self.device)}

    def close(self):
        worker = getattr(self, 'worker', None)
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
        if getattr(self, 'log', None) is not None:
            self.log.close()

