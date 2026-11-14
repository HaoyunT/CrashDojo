"""Check actual execution, not just intended switches."""
from pathlib import Path
import hashlib,json
from common import QUEUE,save,load,sha

def jsonlines(p):return [json.loads(s) for s in p.read_text().splitlines()]

def validate(dest,protocol,smoke=False):
    job=load(dest/'job.json');variant=job['variant']
    assert variant in ('A0','U')
    paths=list(dest.rglob('coupled_noise.jsonl'));assert len(paths)==1
    directory=paths[0].parent
    noise=jsonlines(paths[0]);steps=jsonlines(directory/'ablation_steps.jsonl')
    count=2 if smoke else 14
    assert len(noise)==len(steps)==count
    assert [n['step'] for n in noise]==list(range(0,count*5,5))
    assert all(s['arm']==variant for s in steps)
    request=load(dest/'request.json');keys=['prior','constraint']
    for s in steps:
        assert all(s['effective_weights'][k]==request['weights'][k] for k in keys)
        if variant=='U':
            assert s['effective_weights']['action']==s['effective_weights']['pressure']==0.
            assert not s['conflict_guidance_enabled'] and not s['mechanism_tracking_enabled']
    validation=load(directory/'selected_planner_validation.json')
    assert validation['weights_and_buffers_unchanged'] and validation['optimizer_steps']==0
    record=dict(passed=True,job_id=job['job_id'],variant=variant,noise_steps=count,
        noise_banks=[n['bank_sha256'] for n in noise],model_sha256=validation['before_sha256'],
        unchanged_common_weight_keys=keys,source_snapshot_sha256=sha(dest/'source_snapshot.json'))
    if variant=='U':
        other=QUEUE/('smoke' if smoke else 'runs')/job['job_id'].rsplit('_',1)[0].__add__('_A0')
        full=list(other.rglob('coupled_noise.jsonl'));assert len(full)==1
        a=jsonlines(full[0]);assert a==noise,'A0/U actor/sample/time random streams differ'
        model=load(full[0].with_name('selected_planner_validation.json'))
        assert model['before_sha256']==record['model_sha256']
        binding=load(directory/'selection_binding.json');fullbinding=load(full[0].with_name('selection_binding.json'))
        assert binding==fullbinding,'A0/U actor population or initialization differs'
        base=load(other/'request.json');u=json.loads(json.dumps(request))
        for x in (base,u):x.pop('ablation_v26',None)
        for k in ('action','pressure'):base['weights'][k]=u['weights'][k]=0.
        assert base==u,'A0/U undeclared input difference'
        assert load(other/'source_snapshot.json')==load(dest/'source_snapshot.json')
        record.update(paired_full=str(other),paired_random_sources_identical=True,
            model_and_population_identical=True,only_declared_request_differences=True)
    save(dest/'pair_validation.json',record)
