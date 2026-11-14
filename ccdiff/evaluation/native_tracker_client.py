"""Opt-in, common nuPlan execution layer for all five external planners."""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import numpy as np
import torch

from ccdiff.planners.base import EgoPlannerPolicy
from ccdiff.planners.official_nuplan import OfficialNuPlanPlanner
from tbsim.policies.common import Action


class NativeTrackingEgoPolicy(EgoPlannerPolicy):
    supports_candidate_future = False

    def __init__(self, planner, device, directory):
        super().__init__(device)
        self.planner = planner
        self.name = getattr(planner, "name", type(planner).__name__) + "+nuplan_lqr_bicycle"
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        base = Path("/root/path/to/workspace/home")
        paths = [base / "planner_sources_20260810/nuplan_source/nuplan-devkit-master",
                 base / "venvs/nuplan_planners_20260810/lib/python3.9/site-packages"]
        env = dict(os.environ, PYTHONPATH=":".join(map(str, paths)), OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1", LC_ALL="C.UTF-8")
        self.log = (self.directory / "native_tracker_worker.log").open("w")
        self.worker = subprocess.Popen(
            [str(base / "miniconda3/envs/ctg/bin/python"), str(Path(__file__).with_name("native_tracker_worker.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=env)
        atexit.register(self.close)
        if not self._receive().get("ready"):
            raise RuntimeError("Native tracker failed to initialize")

    def _receive(self):
        if not select.select([self.worker.stdout], [], [], 120)[0]:
            raise TimeoutError("Native tracker timeout")
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError("Native tracker exited")
        reply = json.loads(line)
        if reply.get("ok") is False:
            raise RuntimeError(reply["error"])
        return reply

    def plan(self, ego_observation, candidate_agents_future=None, strategy_depth=0,
             agent_observation=None, step_index=0, **kwargs):
        reference, info = self.planner.plan(
            ego_observation, candidate_agents_future=candidate_agents_future,
            strategy_depth=strategy_depth, agent_observation=agent_observation,
            step_index=step_index, **kwargs)
        if reference.positions.shape[0] != 1:
            raise ValueError("Native tracker development protocol requires one ego")
        array = OfficialNuPlanPlanner._numpy
        raw = ego_observation.get("agent_hist")
        history = array(raw)[0] if raw is not None else OfficialNuPlanPlanner._history_from_parsed(ego_observation, 0)
        if history.shape[-1] != 8:
            raise ValueError("Expected explicit x,y,vx,vy,ax,ay,sin,cos state layout")
        request = dict(positions=array(reference.positions)[0].tolist(),
                       yaws=array(reference.yaws)[0].tolist(), history=history.tolist(),
                       extent=array(ego_observation["extent"])[0].tolist(), step_index=int(step_index))
        self.worker.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.worker.stdin.flush()
        reply = self._receive()
        positions = torch.as_tensor(reply["positions"], dtype=reference.positions.dtype, device=self.device)[None]
        yaws = torch.as_tensor(reply["yaws"], dtype=reference.yaws.dtype, device=self.device)[None]
        if not torch.isfinite(positions).all() or not torch.isfinite(yaws).all():
            raise FloatingPointError("Native controller returned nonfinite action")
        record = dict(step_index=int(step_index), planner=self.name, raw_reference_positions=request["positions"],
                      tracked_positions=reply["positions"], tracked_yaws=reply["yaws"],
                      world_from_agent=array(ego_observation["world_from_agent"])[0].tolist(),
                      commands=reply["commands"], provenance=reply["provenance"],
                      raw_planner_reference_executed_directly=False)
        with (self.directory / "native_tracker_execution.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        return Action(positions=positions, yaws=yaws), dict(
            planner=self.name, candidate_future_consumed=False,
            execution_protocol="native_nuplan_lqr_bicycle_v1")

    def eval(self):
        self.planner.eval()
        return self

    def close(self):
        worker = getattr(self, "worker", None)
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        if getattr(self, "log", None) is not None and not self.log.closed:
            self.log.close()
        if hasattr(self.planner, "close"):
            self.planner.close()
