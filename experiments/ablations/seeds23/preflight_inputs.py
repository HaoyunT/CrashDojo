"""Check every frozen native request/context using the actual runtime loaders."""
import collections
import importlib.util
import sys
import time
from pathlib import Path
from common import HERE, QUEUE, load, save, sha

sys.path[:0] = [str(HERE), str(HERE/'overlay')]
from contextual_contract import bind_context
spec = importlib.util.spec_from_file_location('native82_guide_preflight', HERE/'overlay/five_class_guidance.py')
guide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guide)


def main():
    assert load(QUEUE/'selectors_ready.json')['ready']
    jobs = load(QUEUE/'cohort.json')
    records = []
    seen = set()
    for job in jobs:
        choices = [dict(job)]
        for seed in [1]:
            for variant in ['A1','A2','A3']:
                path = QUEUE/'inputs'/job['scene']/f'selection_{variant}_s{seed}.json'
                selected = load(path)
                effective = dict(job, **selected['job_overrides'])
                context = load(effective['context_record'])
                for key, value in [('prior_factor',context['prior']['factor']),('prior_reason',context['prior']['reason']),
                                   ('prior_support_level',context['prior']['support_level'])]:
                    selected['job_overrides'][key] = value
                    effective[key] = value
                save(path, selected)
                assert selected['selected_actor'] in job['candidate_ids']
                assert selected['candidate_ids'] == job['candidate_ids']
                choices.append(effective)
        for effective in choices:
            key = effective['base_request']
            if key in seen:
                continue
            seen.add(key)
            request = guide.load_request(key, load(effective['manifest']))
            assert request['selection_mode'] != 'random_fallback'
            assert request.get('mechanism_type') != 'random_actor_speed_control'
            prior, metadata = bind_context(effective, request)
            records.append(dict(scene=job['scene'],actor=request['adversary_vehicle_id'],family=request['family'],
                request_path=key,request_sha256=sha(key),context_path=effective['context_record'],
                context_sha256=sha(effective['context_record']),prior_factor=prior['factor'],
                start_step=request['start_step_10hz'],runtime_request_loader_passed=True,causal_context_binding_passed=True))
    save(QUEUE/'runtime_input_preflight.json',dict(passed=True,scenes=len(jobs),requests=len(records),
        native_only=True,checked_unix=time.time(),records=records))
    print(dict(stage='runtime_input_preflight_passed',scenes=len(jobs),unique_requests=len(records)),flush=True)


if __name__ == '__main__':
    main()
