"""Dispatch frozen requests without converting fallback geometry to native geometry.

Native requests keep the suite's own validator. Historical random-fallback
requests require the original compatible loader and its recorded SHA-256.
The compatibility source is loaded only when such a request is encountered.
"""
import ast
import hashlib
import json
from pathlib import Path


def install(module, output, compatibility_config):
    original = module.load_request
    output = Path(output)
    config_path = Path(compatibility_config)
    compatible = None
    compatible_sha256 = None

    def resolve_compatible():
        nonlocal compatible, compatible_sha256
        if compatible is not None:
            return compatible
        if not config_path.is_file():
            raise FileNotFoundError(
                'A frozen random_fallback request requires the original '
                'compatible loader. Configure random_loader_source and '
                'random_loader_sha256 in ' + str(config_path))
        evidence = json.loads(config_path.read_text())
        source = Path(evidence['random_loader_source'])
        if not source.is_absolute():
            source = config_path.parent / source
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != evidence['random_loader_sha256']:
            raise ValueError('Compatible request-loader SHA-256 mismatch')
        functions = [node for node in ast.parse(raw).body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == 'load_request']
        if len(functions) != 1:
            raise ValueError('Compatibility source must define one load_request')
        namespace = dict(vars(module))
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     str(source), 'exec'), namespace)
        compatible = namespace['load_request']
        compatible_sha256 = digest
        return compatible

    def load_request(path, record):
        raw = Path(path).read_bytes()
        frozen = json.loads(raw)
        fallback = frozen.get('selection_mode') == 'random_fallback'
        loader = resolve_compatible() if fallback else original
        result = loader(path, record)
        for key in ('scene_id', 'adversary_vehicle_id', 'start_step_10hz'):
            if frozen.get(key) != record.get(key) or result.get(key) != frozen.get(key):
                raise ValueError('Frozen request/selection identity mismatch: ' + key)
        output.mkdir(parents=True, exist_ok=True)
        audit = dict(
            selected_actor=result['adversary_vehicle_id'],
            scene_id=result['scene_id'],
            original_v6_selected_actor_retained=True,
            request_sha256=hashlib.sha256(raw).hexdigest(),
            loader='original_v6_random_compatible' if fallback else 'original_v26_native',
            original_random_loader_sha256=compatible_sha256 if fallback else None,
            current_v26_loss_unchanged=True,
            seven_criteria_used_as_dispatch_filter=False,
            geometry_status=result.get('geometry_status'),
            geometry_evidence=result.get('geometry_evidence'),
        )
        (output / 'request_loader_compatibility.json').write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
        return result

    module.load_request = load_request
