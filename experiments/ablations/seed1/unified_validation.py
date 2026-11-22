"""Validate each ablation condition against one Full, with actor-aligned noise."""
import copy,json,os
from pathlib import Path
from common import QUEUE,load,save,sha
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def validate(dest,protocol,smoke=False):
    job=load(dest/'job.json');arm=job['variant'];assert arm in protocol['variants']
    p,=dest.rglob('coupled_noise.jsonl');noise=lines(p);steps=lines(p.with_name('ablation_steps.jsonl'))
    n=2 if smoke else protocol['steps']//protocol['replan_steps']
    assert len(noise)==len(steps)==n and [x['step'] for x in noise]==list(range(0,n*5,5))
    assert all(x['arm']==arm for x in steps)
    request=load(dest/'request.json')
    assert all(all(x['effective_weights'][k]==request['weights'][k] for k in ['prior','constraint']) for x in steps)
    if arm in ['A3','A4']:assert all(x['effective_weights']['action']==0 and not x['mechanism_tracking_enabled'] for x in steps)
    if arm=='A4':assert all(x['effective_weights']['pressure']==0 and not x['conflict_guidance_enabled'] for x in steps)
    dynamic=lines(p.with_name('dynamic_strength_steps.jsonl'))
    assert len(dynamic)==n
    assert all(x['beta']==protocol['pressure_beta'] for x in dynamic)
    model=load(p.with_name('selected_planner_validation.json'))
    assert model['weights_and_buffers_unchanged'] and model['optimizer_steps']==0
    record=dict(passed=True,job_id=job['job_id'],variant=arm,noise_steps=n,model_sha256=model['before_sha256'],source_snapshot_sha256=sha(dest/'source_snapshot.json'))
    if arm!='A0':
        full=QUEUE/('smoke' if smoke else 'runs')/(job['job_id'].rsplit('_',1)[0]+'_A0')
        fp,=full.rglob('coupled_noise.jsonl')
        for a,b in zip(lines(fp),noise):
            assert a['step']==b['step']
            convert=lambda r:{(x['actor'],x['sample']):(x['seed'],x['sha256']) for x in r['draws']}
            assert convert(a)==convert(b),'Actor-aligned Gaussian inputs differ'
        assert load(fp.with_name('selected_planner_validation.json'))['before_sha256']==model['before_sha256']
        assert load(full/'source_snapshot.json')==load(dest/'source_snapshot.json')
        fs,s=load(full/'status.json'),load(dest/'status.json')
        assert (fs['host'],fs['gpu'])==(s['host'],s['gpu'])
        fj=load(full/'job.json')
        assert all(fj[k]==job[k] for k in ['scene','stream','planner','config','steps'])
        if arm in ['A1','A2']:
            choice=load(dest/'selection.json');assert request['adversary_vehicle_id']==choice['selected_actor']
            assert set(choice['candidate_ids'])==set(fj['candidate_ids']) and len(choice['candidate_ids'])>=2
        else:
            a,b=load(full/'request.json'),copy.deepcopy(request)
            for q in (a,b):q.pop('ablation_v26')
            if arm=='A4':
                for q in (a,b):q['weights']['action']=q['weights']['pressure']=0.
            if arm=='A3':
                for q in (a,b):
                    q['weights']['action']=0.;q['ablation_generic_collision']=True;q['contact_phase_window']=False;q['qualified_contact_v2']=None
                    for k in ['causal_parameter_search_v9','interaction_timing_v11','execution_completion_v10']:q[k]=False
            assert a==b,'Undeclared non-selector request change'
        record.update(paired_full=str(full),stable_actor_noise_identical=True,model_and_source_identical=True,hardware_identical=True,declared_controls_only=True)
    save(dest/'pair_validation.json',record)
