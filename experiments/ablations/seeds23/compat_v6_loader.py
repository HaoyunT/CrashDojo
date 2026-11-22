"""Use the shared frozen-request dispatcher for this ablation suite."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install(module, output):
    source = HERE.parents[1] / 'support' / 'request_loader_compat.py'
    spec = importlib.util.spec_from_file_location('frozen_request_loader_compat', source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.install(module, output, HERE / 'compatibility_sources.json')
