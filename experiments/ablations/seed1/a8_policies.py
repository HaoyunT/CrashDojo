"""A8: generate under a fixed observed-velocity ego, then execute a frozen target."""
import json, os, hashlib, time
from pathlib import Path
import numpy as np
import torch
from tbsim.policies.common import Action, RolloutAction
from tbsim.policies.wrappers import PlannerOverrideRolloutWrapper
from ccdiff.planners.base import EgoPlannerPolicy
from replay_validity_v1 import apply_replay_validity

class NominalEgoPolicy(EgoPlannerPolicy):
    supports_candidate_future=False
    name='fixed_observed_velocity_nominal_ego'
    def __init__(self,device):
        super().__init__(device);self.initial=None
    def eval(self):return self
    def plan(self,ego_observation,step_index=0,**kwargs):
        a=ego_observation;tf=a['world_from_agent'][0];inv=a['agent_from_world'][0]
        if self.initial is None:
            hist=a['history_positions'][0];vel=(hist[-1]-hist[-6])/.5
            self.initial=(tf[:2,2].clone(),tf[:2,:2]@vel,torch.atan2(tf[1,0],tf[0,0]))
        pos,velocity,yaw=self.initial
        t=(torch.arange(1,53,device=self.device,dtype=tf.dtype)+step_index)*.1
        world=pos[None]+t[:,None]*velocity[None]
        local=world@inv[:2,:2].T+inv[:2,2]
        angle=yaw-torch.atan2(tf[1,0],tf[0,0]);angle=torch.atan2(angle.sin(),angle.cos())
        return Action(positions=local[None],yaws=angle.expand(1,52,1)),{'nominal_ego':'initial_observed_velocity_and_heading_frozen'}

class FixedTargetRollout(PlannerOverrideRolloutWrapper):
    def __init__(self,policy,ego_policy,model,binding,request,directory,dt):
        from ccdiff.evaluation.selected_planner_rollout import state_digest
        self.agents_policy=policy;self.ego_policy=ego_policy;self.model=model;self.binding=binding
        self.request=request;self.directory=Path(directory);self.dt=dt;self.steps=[]
        path=Path(os.environ['A8_TARGET_PATH']);self.path=json.loads(path.read_text())
        assert self.path['actor_id']==request['adversary_vehicle_id']
        self.model.eval()
        for p in model.parameters():p.requires_grad_(False)
        self.before=state_digest(model)
    @property
    def device(self):return self.agents_policy.device
    def eval(self):self.model.eval();self.ego_policy.eval();return self
    def get_action(self,obs,step_index):
        from ccdiff.evaluation.five_class_guidance import inspect_plan
        a=obs['agents'];self.binding.bind();target=self.binding.target_index;ego=self.binding.ego_index;n=a['target_positions'].shape[0]
        started=time.perf_counter()
        with torch.no_grad():ea,ei=self.ego_policy.get_action(self._slice_batch(a,[ego],n),step_index=step_index,strategy_depth=0,candidate_agents_future=None,agent_obs=a)
        horizon=min(5,ea.positions.shape[1]);p=a['target_positions'][:,:horizon].clone();y=a['target_yaws'][:,:horizon].clone()
        world=torch.as_tensor(self.path['positions'][step_index+1:step_index+1+horizon],dtype=torch.float64,device=p.device)
        headings=torch.as_tensor(self.path['yaws'][step_index+1:step_index+1+horizon],dtype=y.dtype,device=y.device)
        assert len(world)==horizon
        tf=a['world_from_agent'][target]
        # Reconstruct the rigid transform from the simulator's observed pose.
        # Batched float32 inverse matrices can have a non-unit homogeneous
        # element and millimetre-scale translation errors in global coordinates.
        center=a['centroid'][target].double();angle=a['yaw'][target].reshape(()).double()
        c,s=torch.cos(angle),torch.sin(angle)
        rotation=torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
        p[target]=((world-center)@rotation).to(p.dtype)
        relative=headings.double()-angle;y[target,:,0]=torch.atan2(relative.sin(),relative.cos()).to(y.dtype)
        p[ego]=ea.positions[0,:horizon];y[ego]=ea.yaws[0,:horizon]
        rv=apply_replay_validity(p,y,a,[j for j in range(n) if j not in [ego,target]])
        action=Action(positions=p,yaws=y);audit=inspect_plan(action,a,target,ego,self.request)
        audit.update(step_index=int(step_index),target_index=target,ego_index=ego,target_actor_id=self.request['adversary_vehicle_id'],
            ego_action_executed_exactly=True,non_target_non_ego_replayed=True,replay_validity=rv,guide_calls=0,
            planner_seconds=time.perf_counter()-started,fm_and_guide_seconds=0.,fixed_target_from_nominal=True,
            history_local_last3=a['history_positions'][target,-3:].cpu().tolist(),selected_local_first5=p[target].cpu().tolist(),target_world_from_agent=tf.cpu().tolist())
        with (self.directory/'selected_planner_steps.jsonl').open('a') as f:f.write(json.dumps(audit,allow_nan=False)+'\n')
        self.steps.append(int(step_index))
        return RolloutAction(agents=action,agents_info={'five_class_target_index':torch.full((n,),target,device=self.device),'five_class_ego_index':torch.full((n,),ego,device=self.device)})
    def finalize(self):
        from ccdiff.evaluation.selected_planner_rollout import state_digest
        after=state_digest(self.model);assert self.before==after
        report=dict(weights_and_buffers_unchanged=True,before_sha256=self.before,after_sha256=after,optimizer_steps=0,guide_calls=0,
            fixed_target_execution=True,steps=self.steps,nominal_generation_artifact=self.path['source_hdf5'],target_path_sha256=hashlib.sha256(Path(os.environ['A8_TARGET_PATH']).read_bytes()).hexdigest())
        (self.directory/'selected_planner_validation.json').write_text(json.dumps(report,indent=2))
