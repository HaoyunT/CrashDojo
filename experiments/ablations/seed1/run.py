"""One separately recorded A0--A8 run; no implicit retry or outcome filtering."""
import argparse,copy,json,os,runpy,sys,time,traceback,socket
from pathlib import Path
from common import ROOT,REPO,HERE,QUEUE,OVERLAY,load,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-steps',type=int,default=10);p.add_argument('--phase',choices=['nominal','fixed']);a=p.parse_args()
    protocol=load(QUEUE/'run_manifest.json')
    assert 0<a.smoke_steps<=protocol['steps'] and a.smoke_steps%5==0
    j=next(x for x in protocol['jobs'] if x['job_id']==a.job_id)
    source=next(x for x in load(QUEUE/'cohort.json') if x['scene']==j['scene'])
    job=dict(source,**j);v=j['variant']
    if v=='A8' and a.phase is None:
        from run_a8 import run
        return run(a,job,protocol)
    dest=QUEUE/('smoke' if a.smoke else 'runs')/j['job_id']
    if a.phase:dest=dest/a.phase
    dest.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',job_id=j['job_id'],variant=v,planner=j['planner'],scene=j['scene'],seed=j['stream'],pid=os.getpid(),host=socket.gethostname(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],started_unix=time.time(),formal=not a.smoke)
    save(dest/'status.json',state)
    try:
        if v in ['A1','A2','A3']:
            selected=load(QUEUE/'inputs'/j['scene']/f'selection_{v}_s{j["stream"]}.json')
            job.update(selected['job_overrides']);q=load(job['base_request']);context=load(job['context_record'])
            save(dest/'selection.json',selected)
        else:q=load(job['base_request']);context=load(job['context_record'])
        prior=context['prior'];effective_prior=1. if v in ['A5','A7'] else prior['factor']
        beta=0. if v in ['A6','A7'] else protocol['pressure_beta']
        q['context_prior_v26']=prior
        q['context_conditioning_v26']=dict(record=job['context_record'],record_sha256=sha(job['context_record']),prior_table_sha256=context['prior_table_sha256'],observation_cutoff_s=context['observation_cutoff_s'],max_input_time_s=context['max_input_time_s'])
        q['dynamic_strength_v26']=dict(mode='calibrated',scale=protocol['pressure_scale'],beta=beta,prior_factor=effective_prior,
            prior_reason=prior['reason'],prior_support_level=prior['support_level'],calibration_state='inherited_frozen_v26_no_retuning')
        q['ablation_v26']=dict(arm=v,scene=j['scene'],stream=j['stream'],protocol=protocol['version'])
        q['ablation_generic_collision']=v=='A4'
        if v=='U':
            q['weights']['action']=0.
            q['weights']['pressure']=0.
        if a.phase:
            assert v=='A8'
            os.environ['ABLATION_A8_PHASE']=a.phase
            q['ablation_fixed_target']=a.phase=='fixed'
        if v=='A4':
            q['weights']['action']=0.
            q['contact_phase_window']=False;q['qualified_contact_v2']=None
            q['causal_parameter_search_v9']=False;q['interaction_timing_v11']=False;q['execution_completion_v10']=False
        save(dest/'request.json',q);save(dest/'job.json',job)
        save(dest/'configuration_delta.json',dict(variant=v,base_request_sha256=sha(job['base_request']),context_multiplier=effective_prior,beta=beta,generic_collision=v=='A4',target=q['adversary_vehicle_id'],family=q['family']))
        (dest/'manifest.jsonl').write_bytes(Path(job['manifest']).read_bytes())
        save(dest/'source_snapshot.json',{str(f):sha(f) for f in HERE.rglob('*.py')})
        os.environ.update(ABLATION_JOB_FILE=str(dest/'job.json'),V26_DATASET='nuscenes',CRASHDOJO_RUN_MODE='smoke' if a.smoke else 'formal',CCDIFF_PLANNER_GPU=os.environ['CUDA_VISIBLE_DEVICES'] if j['planner']=='diffusion_planner' else '')
        os.chdir(REPO);sys.path[:0]=[str(HERE),str(OVERLAY),str(REPO)]
        from numeric_policy import configure
        configure()
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        steps=a.smoke_steps if a.smoke else protocol['steps']
        sys.argv=[str(HERE/'entry.py'),'--config_file',job['config'],'--env','trajdata','--eval_class','CCFlowMatching',
            '--registered_name','trajdata_nusc_ccflow_matching','--results_root_dir',str(dest),'--num_scenes_to_evaluate','1','--num_scenes_per_batch','1',
            '--num_simulation_steps',str(steps),'--n_step_action','5','--seed',str(j['stream']),'--editing_source','none','--disable_guidance',
            '--scene_source','m2_unguided','--interaction_intervention','none','--eval_num_action_samples','5','--prefix','ablation_'+v+'_',
            '--selection_manifest',str(dest/'manifest.jsonl'),'--selection_row','0','--five_class_request',str(dest/'request.json'),'--ego_planner',j['planner']]
        save(dest/'command.json',sys.argv)
        runpy.run_path(str(HERE/'entry.py'),run_name='__main__')
        integrity,=dest.rglob('selected_planner_validation.json');report=load(integrity)
        assert report['weights_and_buffers_unchanged'] and report['optimizer_steps']==0
        updates=[] if a.phase=='fixed' else [json.loads(l) for l in integrity.with_name('dynamic_strength_steps.jsonl').read_text().splitlines()]
        assert len(updates)==(0 if a.phase=='fixed' else steps//5)
        assert all(u['beta']==beta and u['prior_factor']==effective_prior for u in updates)
        logs=[json.loads(l) for l in integrity.with_name('selected_planner_steps.jsonl').read_text().splitlines()]
        assert [r['step_index'] for r in logs]==list(range(0,steps,5))
        assert all(r['ego_action_executed_exactly'] and r['non_target_non_ego_replayed'] for r in logs)
        from metrics import evaluate
        metrics=evaluate(dest,q);assert metrics['frames']==steps
        save(dest/'metrics.json',metrics)
        save(dest/'runtime_validation.json',dict(passed=True,variant=v,guide_calls=report['guide_calls'],pressure_updates=len(updates),effective_prior=effective_prior,beta=beta))
        from unified_validation import validate
        validate(dest,protocol,smoke=a.smoke)
        state.update(status='completed',returncode=0,wall_seconds=time.time()-state['started_unix'])
    except BaseException as exc:
        state.update(status='failed',returncode=1,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.time()-state['started_unix']);save(dest/'status.json',state);raise
    save(dest/'status.json',state)
    print(json.dumps(state),flush=True)
if __name__=='__main__':main()
