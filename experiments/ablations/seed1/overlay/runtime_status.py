"""Only intact completed episodes may enter metric numerators."""
import json

def completed_runtime_ok(parent):
    status=parent/'status.json'
    if not status.exists() or json.loads(status.read_text()).get('returncode')!=0:
        return False
    job=json.loads((parent/'job.json').read_text())
    if job.get('formal'):
        validation=parent/'runtime_validation.json'
        return validation.exists() and json.loads(validation.read_text()).get('passed') is True
    return True
