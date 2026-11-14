"""Opt-in signed initial unicycle speed from observed travel direction.

Keep the original speed magnitude and all observation/conditioning tensors.
Only the selected actor's dynamics initialization changes when its last
observed displacement is backward relative to its current body heading.
This is not a relaxation of motion, route or action acceptance criteria.
"""
import torch


def correct_selected_direction(aux,batch,target):
    if target is None:return aux,None
    states=aux['curr_states']
    history=batch['history_positions'];yaw=batch['history_yaws'];available=batch['history_availabilities']
    if states.ndim!=3 or states.shape[0]!=1 or states.shape[-1]!=4:
        raise ValueError('One unified scene with native unicycle states required')
    if type(target) is not int or not 0<=target<states.shape[1]:raise ValueError('Invalid selected actor')
    if history.shape[:2]!=states.shape[:2] or history.shape[-2]<2 or not bool(available[:,target,-2:].all()):
        raise ValueError('Two observed selected-actor positions required')
    delta=history[:,target,-1]-history[:,target,-2]
    heading=yaw[:,target,-1,0]
    along=(delta*torch.stack((heading.cos(),heading.sin()),dim=-1)).sum(-1)
    old=states[:,target,2]
    if not torch.isfinite(along).all() or not torch.isfinite(old).all():raise ValueError('Nonfinite selected observed motion')
    corrected=torch.where(along<0,-old.abs(),old)
    changed=not torch.equal(old,corrected)
    result=aux
    if changed:
        result=dict(aux)
        result['curr_states']=states.clone()
        result['curr_states'][:,target,2]=corrected
    audit=dict(target_index=target,changed=changed,original_speed_mps=old.detach().cpu().tolist(),
               dynamics_speed_mps=corrected.detach().cpu().tolist(),
               observed_longitudinal_displacement_m=along.detach().cpu().tolist(),
               magnitude_preserved=True,observation_and_neural_features_unchanged=True,
               source='last_two_observed_positions_and_current_heading',acceptance_bounds_changed=False)
    return result,audit


def install(policy,target):
    policy.selected_kinematic_target=target
    if getattr(policy,'_signed_kinematics_installed',False):return
    original=policy.get_aux_info
    policy._selected_kinematics_audits=[]
    def get_aux_info(batch,*args,**kwargs):
        aux=original(batch,*args,**kwargs)
        result,audit=correct_selected_direction(aux,batch,policy.selected_kinematic_target)
        if audit is not None:policy._selected_kinematics_audits.append(audit)
        return result
    policy.get_aux_info=get_aux_info
    policy._signed_kinematics_installed=True
