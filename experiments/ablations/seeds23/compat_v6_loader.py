"""Route frozen v6 random-fallback requests through their original loader.

Keep the current v26 loss, model, identity binding, and runtime audit unchanged.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install(module, output):
    evidence = json.loads((HERE / 'compatibility_sources.json').read_text())
    source = Path(evidence['random_loader_source'])
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == evidence['random_loader_sha256']
    tree = ast.parse(raw)
    function, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_request']
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    compatible = namespace['load_request']
    original = module.load_request

    def load_request(path, record):
        request = json.loads(Path(path).read_text())
        random = request.get('selection_mode') == 'random_fallback'
        result = (compatible if random else original)(path, record)
        audit = dict(
            selected_actor=result['adversary_vehicle_id'],
            scene_id=result['scene_id'],
            original_v6_selected_actor_retained=True,
            request_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            loader='original_v6_random_compatible' if random else 'original_v26_native',
            original_random_loader_sha256=evidence['random_loader_sha256'],
            current_v26_loss_unchanged=True,
            seven_criteria_used_as_dispatch_filter=False,
            geometry_status=result.get('geometry_status'),
            geometry_evidence=result.get('geometry_evidence'),
        )
        (Path(output) / 'request_loader_compatibility.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
        return result

    module.load_request = load_request
