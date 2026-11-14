"""Dry-run by default. Authorized frozen 50-scene evaluation or bounded smoke."""
import argparse,json,os,subprocess,time,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--selection-manifest',required=True);p.add_argument('--selection-row',type=int,required=True)
 p.add_argument('--planner',choices=['idm','strive','pdm_closed','diffusion_planner','flow_planner'],default='idm');p.add_argument('--gpu',type=int,required=True)
 p.add_argument('--steps',type=int,default=10);p.add_argument('--samples',type=int,default=2);p.add_argument('--output',required=True);p.add_argument('--execute-smoke',action='store_true');p.add_argument('--formal',action='store_true');p.add_argument('--execute-formal',action='store_true');p.add_argument('--frozen-protocol');p.add_argument('--seed',type=int,default=1)
 p.add_argument('--execute-validation',action='store_true',help='Bounded complete-action development check, never formal')
 a=p.parse_args()
 execute=a.execute_smoke or a.execute_formal or a.execute_validation
 if a.execute_validation and (a.execute_smoke or a.formal or a.execute_formal):raise SystemExit("Cannot mix validation with smoke/formal")
 if a.execute_smoke and (a.formal or a.execute_formal):raise SystemExit('Cannot mix smoke and formal execution')
 if a.execute_formal:
  if not a.formal or not a.frozen_protocol:raise SystemExit('Formal execution requires explicit frozen protocol')
  protocol=json.loads(Path(a.frozen_protocol).read_text())
  digest=hashlib.sha256(Path(a.request).read_bytes()).hexdigest()
  manifest_digest=hashlib.sha256(Path(a.selection_manifest).read_bytes()).hexdigest()
  if protocol.get('status')!='frozen' or protocol.get('execution_authorized') is not True:raise SystemExit('Protocol is not frozen/authorized')
  if digest not in protocol.get('request_sha256',[]) or manifest_digest not in protocol.get('selection_manifest_sha256',[]):raise SystemExit('Request/selection not in frozen protocol')
  if a.planner not in protocol.get('planners',[]) or a.seed not in protocol.get('seeds',[]):raise SystemExit('Planner/seed not in frozen protocol')
  if a.steps!=protocol.get('steps') or a.samples!=protocol.get('samples'):raise SystemExit('Budget differs from frozen protocol')
  if not protocol.get('validity_policy') or not protocol.get('evaluator_version'):raise SystemExit('Missing explicit validity/evaluator policy')
 if not 1<=a.steps<=(1000 if a.formal else (100 if a.execute_validation else 30)) or not 1<=a.samples<=(100 if a.formal else 5):raise SystemExit('Invalid episode/candidate budget')
 root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve()
 if out.exists() and execute:raise SystemExit('Refuse overwriting an existing run')
 py='/root/path/to/workspace/home/miniconda3/envs/ccdiff/bin/python'
 cmd=[py,str(Path(__file__).with_name('entry.py')),'--config_file',str(root/'selection_inputs_20261020/nuscenes_selection_config.json'),
 '--env','trajdata','--eval_class','CCFlowMatching','--registered_name','trajdata_nusc_ccflow_matching',
 '--results_root_dir',str(out),'--num_scenes_to_evaluate','1','--num_scenes_per_batch','1','--num_simulation_steps',str(a.steps),
 '--n_step_action','5','--seed',str(a.seed),'--editing_source','none','--disable_guidance','--scene_source','m2_unguided',
 '--interaction_intervention','none','--eval_num_action_samples',str(a.samples),'--prefix','five_class_formal_' if a.formal else ('five_class_validation_' if a.execute_validation else 'five_class_smoke_'),
 '--selection_manifest',a.selection_manifest,'--selection_row',str(a.selection_row),'--five_class_request',str(Path(a.request).resolve()),'--ego_planner',a.planner]
 spec=dict(command=cmd,gpu=a.gpu,formal=a.formal,executed=execute,complete_action_validation=a.execute_validation,steps=a.steps,candidates=a.samples,planner=a.planner)
 if not execute:print(json.dumps(spec,indent=2));return
 out.mkdir(parents=True)
 (out/'request.json').write_bytes(Path(a.request).read_bytes())
 (out/'selection_manifest.jsonl').write_bytes(Path(a.selection_manifest).read_bytes())
 cmd[cmd.index('--five_class_request')+1]=str(out/'request.json')
 cmd[cmd.index('--selection_manifest')+1]=str(out/'selection_manifest.jsonl')
 (out/'job.json').write_text(json.dumps(spec,indent=2))
 env=os.environ.copy();env['CRASHDOJO_RUN_MODE']='formal' if a.formal else ('validation' if a.execute_validation else 'smoke');env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',PYTHONUTF8='1',LC_ALL='C.UTF-8',LANG='C.UTF-8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=':'.join([str(root),str(root/'third_party/tbsim'),'/root/path/to/workspace/home/CCDiff/third_party/trajdata/src']))
 started=time.perf_counter()
 with (out/'run.log').open('w') as f:
  process=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (out/'process.json').write_text(json.dumps(dict(pid=process.pid,process_group=process.pid,started_unix=time.time())))
  timeout=False
  try:returncode=process.wait(timeout=3600)
  except subprocess.TimeoutExpired:
   import signal
   timeout=True
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   returncode=124
 status=dict(returncode=returncode,timeout=timeout,wall_seconds=time.perf_counter()-started,formal=a.formal,status='completed' if returncode==0 else 'failed')
 (out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
 raise SystemExit(returncode)
if __name__=='__main__':main()
