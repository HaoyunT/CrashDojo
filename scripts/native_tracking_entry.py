"""Explicit experiment entry point; leaves the raw-execution protocol intact."""
import runpy
import sys
from pathlib import Path
import ccdiff.evaluation.five_class_planners as planners
from ccdiff.evaluation.native_tracker_client import NativeTrackingEgoPolicy

repo = Path("/path/to/workspace/Friday/safesim_planner_20261012_01/code/CCDiff_M2c_planner")
out = Path(sys.argv[sys.argv.index("--results_root_dir") + 1])
original = planners.build_selected_ego


def build_tracked(name, device, strive_profile=None):
    return NativeTrackingEgoPolicy(original(name, device, strive_profile), device, out)


planners.build_selected_ego = build_tracked
runpy.run_path(str(repo / "ccdiff/examples/scene_editor.py"), run_name="__main__")
