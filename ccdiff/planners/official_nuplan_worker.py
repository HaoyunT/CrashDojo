"""Persistent worker for the official NuPlan-trained planner checkpoints.

This process runs in the planner (PyTorch 2.x) environment. The SafeSim
evaluator sends already-vectorized NumPy-compatible features over JSON lines,
which keeps the PyTorch 1.13 traffic simulator isolated from planner packages.
"""

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


def _strip_module(state):
    return {key.removeprefix("module."): value for key, value in state.items()}


class PlannerRuntime:
    def __init__(self, kind):
        self.kind = kind
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        home = Path(os.environ.get(
            "PLANNER_SOURCES_ROOT",
            "/root/path/to/workspace/home/planner_sources_20260810/source_no_assets",
        ))
        root = home / ("Diffusion-Planner-main" if kind == "diffusion" else "Flow-Planner-main")
        self.config = None
        self.model = None
        with contextlib.redirect_stdout(sys.stderr):
            if kind == "diffusion":
                from diffusion_planner.model.diffusion_planner import Diffusion_Planner
                from diffusion_planner.utils.config import Config

                self.config = Config(str(root / "checkpoints" / "args.json"), None)
                self.config.device = str(self.device)
                self.model = Diffusion_Planner(self.config).to(self.device).eval()
                checkpoint = torch.load(
                    root / "checkpoints" / "model.pth",
                    map_location="cpu",
                    weights_only=False,
                )
                self.model.load_state_dict(_strip_module(checkpoint["ema_state_dict"]), strict=True)
            else:
                from hydra.utils import instantiate
                from omegaconf import OmegaConf

                from flow_planner.data.dataset.nuplan import NuPlanDataSample

                self.NuPlanDataSample = NuPlanDataSample
                self.config = OmegaConf.load(root / "checkpoints" / "model_config.yaml")
                self.config.device = str(self.device)
                self.config.data = {
                    "dataset": {"train": {
                        "future_downsampling_method": "uniform",
                        "predicted_neighbor_num": 0,
                    }}
                }
                self.model = instantiate(self.config.model).to(self.device).eval()
                checkpoint = torch.load(
                    root / "checkpoints" / "model.pth",
                    map_location="cpu",
                    weights_only=False,
                )
                self.model.load_state_dict(_strip_module(checkpoint), strict=True)
        self.root = root

    def infer(self, request):
        torch.manual_seed(int(request["seed"]))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(request["seed"]))
        arrays = {
            name: torch.as_tensor(
                np.asarray(request[name]),
                device=self.device,
                dtype=(torch.bool if name.endswith("has_speed") else torch.float32),
            )
            for name in (
                "neighbors", "static", "lanes", "routes", "lane_speed",
                "lane_has_speed", "route_speed", "route_has_speed",
            )
        }
        if self.kind == "diffusion":
            arrays["ego_current"] = torch.as_tensor(
                np.asarray(request["ego_current_dp"], dtype=np.float32),
                device=self.device,
            )
            raw = {
                "neighbor_agents_past": arrays["neighbors"],
                "ego_current_state": arrays["ego_current"],
                "static_objects": arrays["static"],
                "lanes": arrays["lanes"],
                "lanes_speed_limit": arrays["lane_speed"],
                "lanes_has_speed_limit": arrays["lane_has_speed"].bool(),
                "route_lanes": arrays["routes"],
                "route_lanes_speed_limit": arrays["route_speed"],
                "route_lanes_has_speed_limit": arrays["route_has_speed"].bool(),
            }
            model_input = self.config.observation_normalizer(raw)
            with torch.inference_mode():
                output = self.model(model_input)[1]["prediction"]
            trajectory = output[:, 0]
        else:
            arrays["ego_past"] = torch.as_tensor(
                np.asarray(request["ego_past"], dtype=np.float32),
                device=self.device,
            )
            arrays["ego_current"] = torch.as_tensor(
                np.asarray(request["ego_current_flow"], dtype=np.float32),
                device=self.device,
            )
            data = self.NuPlanDataSample(
                batched=True,
                ego_past=arrays["ego_past"],
                ego_current=arrays["ego_current"],
                ego_future=torch.empty(0, device=self.device),
                neighbor_past=arrays["neighbors"],
                neighbor_future=torch.empty(0, device=self.device),
                neighbor_future_observed=torch.empty(0, device=self.device),
                lanes=arrays["lanes"],
                lanes_speedlimit=arrays["lane_speed"],
                lanes_has_speedlimit=arrays["lane_has_speed"].bool(),
                routes=arrays["routes"],
                routes_speedlimit=arrays["route_speed"],
                routes_has_speedlimit=arrays["route_has_speed"].bool(),
                map_objects=arrays["static"],
            )
            with torch.inference_mode():
                output = self.model(data, mode="inference", use_cfg=False, cfg_weight=1.0)
            if isinstance(output, dict):
                output = output.get("prediction", next(
                    value for value in output.values() if torch.is_tensor(value)
                ))
            trajectory = output[:, 0] if output.ndim == 4 else output
        trajectory = torch.nan_to_num(trajectory).detach().float().cpu()
        if trajectory.ndim != 3 or trajectory.shape[-1] != 4:
            raise ValueError("planner output must be [B,T,4], got {}".format(tuple(trajectory.shape)))
        return trajectory.numpy().tolist()


def main():
    kind = sys.argv[1]
    runtime = PlannerRuntime(kind)
    print(json.dumps({"ready": True, "kind": kind, "device": str(runtime.device)}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            output = runtime.infer(request)
            print(json.dumps({"ok": True, "trajectory": output}), flush=True)
        except Exception as exc:
            print(json.dumps({
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }), flush=True)


if __name__ == "__main__":
    main()
