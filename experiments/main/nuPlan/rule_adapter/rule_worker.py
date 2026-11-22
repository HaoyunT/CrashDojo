"""Isolated canonical rule planner runtime; diagnostics never enter the RPC stream."""
import contextlib
import json
import os
import sys
import traceback


def tensors(value):
    import torch
    if isinstance(value, dict):
        return {k: tensors(v) for k, v in value.items()}
    if isinstance(value, list):
        try:
            return torch.as_tensor(value, device='cpu')
        except (TypeError, ValueError):
            return value
    return value


def main():
    kind = sys.argv[1]
    if os.environ.get("V26_DATASET") == "nuplan" and kind in ("idm", "strive"):
        import ccdiff.planners.safesim as safesim
        original_loader = safesim.load_lane_graphs_trajdata
        def nuplan_loader(dataset, cache_path="~/.unified_data_cache", flip_singapore=False):
            cache = os.environ.get("V26_NUPLAN_CACHE", "/path/to/workspace/Friday/safesim_planner_20261012_01/launch/causal_guidance_v26_contextual_full_eval_20261101/nuplan_centered_runtime_cache")
            return original_loader("nuplan", cache, flip_singapore)
        safesim.load_lane_graphs_trajdata = nuplan_loader
    with contextlib.redirect_stdout(sys.stderr):
        if kind == 'pdm_closed':
            from pdm_nuscenes import NuScenesPDMRuntime
            runtime = NuScenesPDMRuntime()
        else:
            from ccdiff.planners.safesim import IDMPlannerPolicy, STRIVEPlannerPolicy
            runtime = (IDMPlannerPolicy if kind == 'idm' else STRIVEPlannerPolicy)(device='cpu')
            if kind == 'idm':
                from lane_routes import IDMRouteAdapter
                runtime = IDMRouteAdapter(runtime)
    print(json.dumps({'ready': True, 'kind': kind, 'device': 'cpu', 'dt': .1, 'horizon': 52}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if kind == 'pdm_closed':
                    result = runtime.infer(request)
                else:
                    action, info = runtime.plan(tensors(request['observation']), step_index=request['step_index'])
                    result = {'positions': action.positions.detach().cpu().tolist(),
                              'yaws': action.yaws.detach().cpu().tolist(),
                              'info': info if kind == 'idm' else {'implementation': type(runtime).__name__}}
            print(json.dumps(result, allow_nan=False), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error': str(exc)}), flush=True)


if __name__ == '__main__':
    main()
