"""One isolated frozen-model rollout with audited dynamic pressure activation."""
import argparse
import copy
import json
import os
import runpy
import socket
import sys
import time
import traceback
from pathlib import Path
# Resolve the repository package before runtime-specific path setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ccdiff.evaluation.paper_contract import prepare_paper_request

from common import ROOT, REPO, HERE, QUEUE, OVERLAY, load, save, sha, verify
from contextual_contract import bind_context, audit_updates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--protocol',default='protocol.json')
    parser.add_argument('--smoke-tag',default='smoke')
    args=parser.parse_args()
    protocol=load(QUEUE/args.protocol)
    verify(protocol)
    checkpoint=protocol['generator_checkpoint']
    checkpoint_stat=Path(checkpoint['path']).stat()
    assert checkpoint_stat.st_size==checkpoint['size'] and checkpoint_stat.st_mtime_ns==checkpoint['mtime_ns']
    job=next(j for j in protocol['jobs'] if j['job_id']==args.job_id)
    out=QUEUE/(args.smoke_tag if args.smoke else 'runs')/job['job_id']
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',job=job,pid=os.getpid(),host=socket.gethostname(),
               gpu=os.environ['CUDA_VISIBLE_DEVICES'],started_unix=time.time(),formal=not args.smoke)
    save(out/'status.json',state)
    try:
        if job.get('reuse_no_action_job') and not args.smoke:
            source_job=next(j for j in protocol['jobs'] if j['job_id']==job['reuse_no_action_job'])
            assert job['family']==source_job['family']=='NONE'
            for key in ['base_request','manifest','config']:
                assert sha(job[key])==sha(source_job[key])
            assert all(job[k]==source_job[k] for k in ['scene','planner','steps'])
            source=QUEUE/'runs'/source_job['job_id']
            assert load(source/'status.json')['status']=='completed'
            metrics=load(source/'metrics.json')
            assert metrics['no_action']
            save(out/'metrics.json',metrics)
            save(out/'runtime_validation.json',dict(passed=True,no_action_reference_reuse=True,
                source_job=source_job['job_id'],source_metrics_sha256=sha(source/'metrics.json'),
                reason='identical model, planner, seed, selection, numeric policy; guide not called without a target'))
            state.update(status='completed',returncode=0,reused_from=source_job['job_id'],
                         wall_seconds=time.time()-state['started_unix'])
            save(out/'status.json',state)
            return
        request=load(job['base_request'])
        prior, context_metadata=bind_context(job, request)
        request['context_prior_v26']=prior
        request['context_conditioning_v26']=context_metadata
        if job['dataset']=='nuplan':
            from nuplan_adapter import shift_request
            request=shift_request(request,load(job['manifest'])['city'])
        if job['arm']=='calibrated':
            release=load(QUEUE/'release.json')
            settings=release['datasets'][job['dataset']]
            scale,beta=settings['scale'],settings['beta']
            mode='fixed_reference' if settings['arm']=='fixed_reference' else 'calibrated'
        else:
            scale,beta=job['scale'],job['beta']
            mode=job['arm']
        assert mode != 'fixed_reference', 'Contextual evaluation requires active pressure scheduling'
        request['dynamic_strength_v26']=dict(mode=mode,scale=scale,beta=beta,
            prior_factor=job['prior_factor'],prior_reason=job['prior_reason'],
            prior_support_level=job.get('prior_support_level'),
            prior_events=job.get('prior_events', 0),
            prior_scenes=job.get('prior_scenes', 0),
            prior_logs=job.get('prior_logs', 0),
            prior_fallback_reason=job.get('prior_fallback_reason'),
            calibration_state='inherited_frozen_IDM_scale_beta_no_evaluation_retuning')
        request, contract = prepare_paper_request(request, job['dataset'],
            steps=job['steps'], smoke=args.smoke)
        save(out/'paper_runtime_contract.json', contract)
        save(out/'request.json',request)
        (out/'manifest.jsonl').write_bytes(Path(job['manifest']).read_bytes())
        os.environ['V26_DATASET']=job['dataset']
        os.environ['V26_NUPLAN_SCENE']=job['scene']
        os.environ['V26_COORDINATE_AUDIT']=str(QUEUE/'coordinate_audit')
        os.environ['CRASHDOJO_RUN_MODE']='smoke' if args.smoke else 'formal'
        os.environ['CCDIFF_PLANNER_GPU']=os.environ['CUDA_VISIBLE_DEVICES'] if job['planner'] in ['diffusion_planner','flow_planner'] else ''
        os.chdir(REPO)
        sys.path[:0]=[str(OVERLAY),str(REPO)]
        from numeric_policy import configure
        configure()
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        steps=10 if args.smoke else job['steps']
        cmd=[str(HERE/'entry.py'),'--config_file',job['config'],'--env','trajdata',
            '--eval_class','CCFlowMatching','--registered_name','trajdata_nusc_ccflow_matching',
            '--results_root_dir',str(out),'--num_scenes_to_evaluate','1','--num_scenes_per_batch','1',
            '--num_simulation_steps',str(steps),'--n_step_action','5','--seed','1','--editing_source','none',
            '--disable_guidance','--scene_source','m2_unguided','--interaction_intervention','none',
            '--eval_num_action_samples','5','--prefix','dynamic_strength_v26_',
            '--selection_manifest',str(out/'manifest.jsonl'),'--selection_row','0',
            '--five_class_request',str(out/'request.json'),'--ego_planner',job['planner']]
        save(out/'command.json',dict(command=cmd))
        sys.argv=cmd
        runpy.run_path(str(HERE/'entry.py'),run_name='__main__')
        integrity_files=list(out.rglob('selected_planner_validation.json'))
        assert len(integrity_files)==1
        integrity=load(integrity_files[0])
        assert integrity['weights_and_buffers_unchanged'] and integrity['optimizer_steps']==0
        log=integrity_files[0].with_name('selected_planner_steps.jsonl')
        records=[json.loads(s) for s in log.read_text().splitlines()]
        assert [s['step_index'] for s in records]==list(range(0,steps,5))
        assert all(s['ego_action_executed_exactly'] and s['non_target_non_ego_replayed'] for s in records)
        assert all(s['replay_validity']['valid_replay_positions_exact'] and s['replay_validity']['valid_replay_yaws_exact'] for s in records)
        weights=log.with_name('dynamic_strength_steps.jsonl')
        updates=[]
        if request['family']!='NONE' and mode!='fixed_reference':
            updates=[json.loads(s) for s in weights.read_text().splitlines()]
            assert len(updates)==steps//5 and all(not u['recorded_future_used'] for u in updates)
            audit_updates(updates, prior, context_metadata)
            state['pressure_range']=[min(u['pressure'] for u in updates),max(u['pressure'] for u in updates)]
        from metrics import evaluate
        metrics=evaluate(out,request)
        assert metrics['frames']==steps
        metrics.update(context_prior_factor=prior['factor'],
                       context_prior_level=prior['support_level'],
                       context_prior_reason=prior['reason'],
                       context_conditioning_v26=context_metadata)
        save(out/'metrics.json',metrics)
        save(out/'runtime_validation.json',dict(passed=True,steps=steps,
             weights_and_buffers_unchanged=True,guide_activated=integrity['guide_calls']>0,
             context_prior_applied=True, context_prior_factor=prior['factor'],
             context_record_sha256=context_metadata['record_sha256'],
             context_updates_checked=len(updates), physical_certification=False))
        state.update(status='completed',returncode=0,wall_seconds=time.time()-state['started_unix'])
    except BaseException as exc:
        state.update(status='failed',returncode=1,error=repr(exc),traceback=traceback.format_exc(),
                     wall_seconds=time.time()-state['started_unix'])
        save(out/'status.json',state)
        raise
    save(out/'status.json',state)
    print(json.dumps(dict(job=job['job_id'],status=state['status'],seconds=state['wall_seconds'])),flush=True)


if __name__=='__main__':
    main()
