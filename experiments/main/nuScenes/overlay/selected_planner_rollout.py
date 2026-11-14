from numeric_policy import configure
from persistent_selected_state import PersistentSelectedState
"""Fixed selected actor FM + external ego + log-replayed other actors."""
import os
import hashlib
import json
import time
from corridor_prefix_gate import make_corridor, prefix_coverages
from pathlib import Path
import torch
from replay_validity_v1 import apply_replay_validity
from causal_background_occupancy_v1 import enabled as bg_enabled, prefix_clearance, forecast_audit
from tbsim.policies.common import Action, RolloutAction
from tbsim.policies.wrappers import PlannerOverrideRolloutWrapper
from ccdiff.evaluation.five_class_guidance import FiveClassGuide, inspect_plan, loss_terms


def state_digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class SelectedPlannerRollout(PlannerOverrideRolloutWrapper):
    def __init__(self, policy, ego_policy, model, binding, request, directory, dt):
        self.agents_policy=policy;self.ego_policy=ego_policy;self.model=model;self.binding=binding
        self.request=request;self.directory=Path(directory);self.dt=dt;self.strategy_depth=0
        self.guide=FiveClassGuide(request,self.directory/'five_class_guidance_audit.jsonl')
        self.model.nets['policy'].five_class_guide=self.guide
        self.model.ema_policy.five_class_guide=self.guide
        self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.before=state_digest(model)
        self._prefix_corridor=make_corridor(request['corridor_world_polygons']) if request['family']!='NONE' else None
        self.no_action_calls=0
        self.native_selected_state=PersistentSelectedState(dt,5)

    @property
    def device(self):return self.agents_policy.device

    def eval(self):
        self.model.eval();self.agents_policy.eval();self.ego_policy.eval();return self

    def get_action(self,obs,step_index):
        numeric_policy=configure()
        a=obs['agents'];n=a['target_positions'].shape[0]
        target=self.binding.target_index;ego=self.binding.ego_index
        if self.request['family']=='NONE':
            self.binding.bind();ego=self.binding.ego_index
            if target is not None:raise ValueError('No-action scene acquired a target')
            ego_obs=self._slice_batch(a,[ego],n)
            started=time.perf_counter()
            with torch.no_grad():
                ego_action,ego_info=self.ego_policy.get_action(ego_obs,step_index=step_index,strategy_depth=0,candidate_agents_future=None,agent_obs=a)
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            horizon=min(a['target_positions'].shape[1],ego_action.positions.shape[1])
            positions=a['target_positions'][:,:horizon].clone();yaws=a['target_yaws'][:,:horizon].clone()
            positions[ego]=ego_action.positions[0,:horizon];yaws[ego]=ego_action.yaws[0,:horizon]
            others=[j for j in range(n) if j!=ego]
            replay_validity=apply_replay_validity(positions,yaws,a,others)
            audit=dict(replay_validity=replay_validity,step_index=int(step_index),target_index=None,ego_index=ego,no_action=True,
                       no_action_reason=self.request['no_action_reason'],non_target_non_ego_replayed=True,
                       planner_seconds=time.perf_counter()-started,fm_and_guide_seconds=0.,guide_calls=0,
                       ego_action_executed_exactly=True)
            with (self.directory/'selected_planner_steps.jsonl').open('a') as f:f.write(json.dumps(audit)+'\n')
            self.no_action_calls+=1
            return RolloutAction(agents=Action(positions=positions,yaws=yaws),agents_info={
                'five_class_target_index':torch.full((n,),-1,device=self.device),
                'five_class_ego_index':torch.full((n,),ego,device=self.device)})
        if target is None or ego is None:raise RuntimeError('Unbound website selection')
        self.binding.bind()
        self.native_selected_state.begin(a,target,int(step_index))
        self.native_selected_state.install(self.model.nets['policy'])
        self.native_selected_state.install(self.model.ema_policy)
        if self.request.get('allow_selected_stationary_control',False):
            if self.request.get('selected_vehicle_initially_parked') is not False:
                raise ValueError('Cannot override stationary filter for unknown/parked vehicle')
            self.model.five_class_stationary_target=target
        else:
            self.model.five_class_stationary_target=None
        ego_obs=self._slice_batch(a,[ego],n)
        # Future ground truth is not input to the guide. External planner owns E.
        started=time.perf_counter()
        with torch.no_grad():
            ego_action,ego_info=self.ego_policy.get_action(ego_obs,step_index=step_index,strategy_depth=0,candidate_agents_future=None,agent_obs=a)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        planner_s=time.perf_counter()-started
        self.guide.prepare(a,ego_action,target,ego,step_index*self.dt,self.dt)
        started=time.perf_counter()
        with torch.no_grad():generated,info=self.agents_policy.get_action(a,step_index=step_index)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        fm_s=time.perf_counter()-started
        positions=a['target_positions'].clone();yaws=a['target_yaws'].clone()
        horizon=min(positions.shape[1],generated.positions.shape[1],ego_action.positions.shape[1])
        if horizon<1:raise RuntimeError('Empty common horizon')
        # Never pad a planner future with a stationary tail for conflict targets.
        positions=positions[:,:horizon];yaws=yaws[:,:horizon]
        selected_sample=0;candidate_scores=None;candidate_violation=None;feasible_candidates=None;candidate_corridor_coverages=None
        background_clearance=None
        if self.request.get('select_guided_candidate',False):
            samples=info['action_samples']
            if samples['positions'].ndim!=4 or samples['positions'].shape[0]!=n:raise ValueError('Expected agent-centric [M,K,T,2] candidates')
            pp=samples['positions'][target,:,:horizon];yy=samples['yaws'][target,:,:horizon,0]
            # Evaluate all K generated candidates with the actual planner reference.
            origin=torch.zeros_like(pp[:,:1]);vel=torch.diff(torch.cat((origin,pp),1),dim=1)/self.dt
            speed=vel.norm(dim=-1);acc=torch.diff(torch.cat((a['curr_speed'][target].expand(pp.shape[0],1),speed),1),dim=1)/self.dt
            omega=torch.diff(torch.cat((torch.zeros_like(yy[:,:1]),yy),1),dim=1)/self.dt
            trajectory=torch.cat((pp,speed[...,None],yy[...,None],acc[...,None],omega[...,None]),-1)
            score,_,_=loss_terms(trajectory,trajectory,self.guide.context,self.request)
            if not torch.isfinite(score).all():raise FloatingPointError('Invalid candidate score')
            hist=a['history_positions'][target,-3:].expand(pp.shape[0],-1,-1)
            joined=torch.cat((hist,pp[:,:5]),1);vv=torch.diff(joined,dim=1)/self.dt;aa=torch.diff(vv,dim=1)/self.dt;jj=torch.diff(aa,dim=1)/self.dt
            heading=torch.cat((torch.zeros_like(yy[:,:1]),yy[:,:4]),1)
            unit=torch.stack((heading.cos(),heading.sin()),-1);along=(aa[:,1:]*unit).sum(-1);lateral=(aa[:,1:]*torch.stack((-unit[...,1],unit[...,0]),-1)).sum(-1)
            limits=self.request['engineering_limits']
            violation=torch.stack((torch.relu(jj.norm(dim=-1).amax(-1)-8.5),torch.relu(along.amax(-1)-limits['max_acceleration_mps2']),torch.relu(-along.amin(-1)-limits['max_braking_mps2']),torch.relu(lateral.abs().amax(-1)-limits['max_lateral_mps2']),torch.relu(vv[:,2:].norm(dim=-1).amax(-1)-limits['max_speed_mps'])),-1).amax(-1)
            if self.request.get('exact_body_prefix_gate',False):
                coverage=prefix_coverages(pp.detach().cpu().numpy(),yy.detach().cpu().numpy(),
                    a['world_from_agent'][target].detach().cpu().numpy(),
                    a['extent'][target].detach().cpu().numpy(),self._prefix_corridor)
                candidate_corridor_coverages=coverage.tolist()
                # An optimization margin, not a change to the 98% acceptance criterion.
                road_violation=torch.relu(pp.new_tensor(.982)-pp.new_tensor(coverage))*100.
                violation=torch.maximum(violation,road_violation)
            if bg_enabled(self.request):
                clearance=prefix_clearance(pp,yy,self.guide.context)
                background_clearance=clearance.cpu().tolist()
                bg_violation=torch.relu(self.request['non_target_avoidance']['prefix_margin_m']-clearance)*100.
                violation=torch.maximum(violation,bg_violation)
            if self.request.get('forward_only_execution_gate_v9'):
                signed_velocity = (vv[:,2:] * unit).sum(-1)
                violation = torch.maximum(violation, torch.relu(-signed_velocity.amin(-1)-.02)*10.)
            feasible=violation<=1e-5
            # Prefer an admissible execution prefix before optimizing conflict loss.
            # If none exists, record this explicitly; no validity claim is made.
            selected_sample=int(torch.where(feasible,score,torch.full_like(score,float('inf'))).argmin()) if bool(feasible.any()) else int(violation.argmin())
            candidate_scores=score.detach().cpu().tolist();candidate_violation=violation.detach().cpu().tolist();feasible_candidates=int(feasible.sum())
            positions[target]=pp[selected_sample];yaws[target]=yy[selected_sample,:,None]
        else:
            from native_candidate_action import first_native_action
            native_positions,native_yaws=first_native_action(generated,info,target,horizon)
            positions[target]=native_positions;yaws[target]=native_yaws
        positions[ego]=ego_action.positions[0,:horizon];yaws[ego]=ego_action.yaws[0,:horizon]
        others=[j for j in range(n) if j not in (target,ego)]
        replay_validity=apply_replay_validity(positions,yaws,a,others)
        self.native_selected_state.commit(a,positions[target],yaws[target])
        action=Action(positions=positions,yaws=yaws)
        audit=inspect_plan(action,a,target,ego,self.request)
        audit['numeric_policy']=numeric_policy
        audit['replay_validity']=replay_validity
        audit.update(causal_background_forecast=forecast_audit(self.guide.context),candidate_background_prefix_clearance_m=background_clearance)
        audit.update(candidate_prefix_corridor_coverage=candidate_corridor_coverages,candidate_execution_violation=candidate_violation,feasible_execution_candidates=feasible_candidates,history_local_last3=a['history_positions'][target,-3:].detach().cpu().tolist(),selected_local_first5=positions[target,:5].detach().cpu().tolist(),target_world_from_agent=a['world_from_agent'][target].detach().cpu().tolist(),selected_sample=selected_sample,candidate_scores=candidate_scores,step_index=int(step_index),target_index=target,ego_index=ego,
            target_actor_id=self.binding.record['adversary_vehicle_id'],planner_seconds=planner_s,fm_and_guide_seconds=fm_s,
            guide_calls=self.guide.calls,selected_stationary_exemption=self.model.five_class_stationary_target is not None,selected_predicted_displacement_max_m=float(generated.positions[target].norm(dim=-1).max()),non_target_non_ego_replayed=True,ego_action_executed_exactly=True,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else 0)
        with (self.directory/'selected_planner_steps.jsonl').open('a') as f:f.write(json.dumps(audit,allow_nan=False)+'\n')
        # Smoke implementation records geometry failures. Formal execution requires
        # a separately frozen validity policy and never implies certification.
        return RolloutAction(agents=action,agents_info={
            'five_class_target_index':torch.full((n,),target,device=self.device),
            'five_class_ego_index':torch.full((n,),ego,device=self.device)})

    def finalize(self):
        (self.directory/'selected_native_state_audit.json').write_text(json.dumps(self.native_selected_state.report(),indent=2))
        active=self.model.ema_policy if self.model.use_ema else self.model.nets['policy']
        audits=getattr(active,'_selected_kinematics_audits',[])
        (self.directory/'selected_kinematic_state_audit.json').write_text(json.dumps(audits,indent=2))
        after=state_digest(self.model)
        result={'weights_and_buffers_unchanged':after==self.before,'before_sha256':self.before,'after_sha256':after,
            'guide_calls':self.guide.calls,'optimizer_steps':0,'formal_evaluation':os.environ.get('CRASHDOJO_RUN_MODE')=='formal',
            'no_action_calls':self.no_action_calls,'no_action':self.request['family']=='NONE',
            'validation_scope':'frozen_inference_runtime_integrity_not_physical_or_mechanism_certification'}
        (self.directory/'selected_planner_validation.json').write_text(json.dumps(result,indent=2))
        if after!=self.before:raise AssertionError('Frozen model changed')
        if self.guide.calls==0 and not (self.request['family']=='NONE' and self.no_action_calls>0):raise AssertionError('Guide never executed')
