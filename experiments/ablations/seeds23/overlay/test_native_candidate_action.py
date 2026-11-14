from types import SimpleNamespace
import unittest
import torch
from ccdiff.models.diffuser_helpers import unicyle_forward_dynamics
from ccdiff.algos.algo_utils import yaw_from_pos
from native_candidate_action import first_native_action
from signed_candidate_kinematics import signed_candidate_trajectory


def fixture():
    initial=torch.tensor([[0.,0.,4.,0.]])
    dyn=SimpleNamespace(acce_bound=(-10.,8.),vbound=(-10.,30.),max_steer=.5,max_yawvel=2*torch.pi)
    actions=torch.zeros(1,5,2);actions[...,1]=.1
    native=unicyle_forward_dynamics(dyn,initial,actions,.1,mode='parallel')
    p=native[:,:,:2];y=native[:,:,3:4]
    samples=dict(positions=p[:,None].repeat(1,5,1,1),yaws=y[:,None].repeat(1,5,1,1))
    rewritten=yaw_from_pos(torch.cat([torch.zeros_like(p[:,:1]),p],dim=1),dt=.1,yaw_correction_speed=.1)
    return initial,p,y,samples,SimpleNamespace(positions=p.clone(),yaws=rewritten)


class Tests(unittest.TestCase):
    def test_native_pair_restores_actual_decoder_identity(self):
        initial,p,y,samples,generated=fixture()
        with self.assertRaisesRegex(ValueError,'previous-heading'):
            signed_candidate_trajectory(p,generated.yaws[...,0],initial,.1)
        fixed_p,fixed_y=first_native_action(generated,dict(action_samples=samples),0,5)
        reconstructed=signed_candidate_trajectory(fixed_p[None],fixed_y[...,0][None],initial,.1)
        self.assertTrue(torch.allclose(reconstructed[...,2],torch.full((1,5),4.),atol=1e-5))
        self.assertTrue(torch.equal(fixed_p,p[0]));self.assertTrue(torch.equal(fixed_y,y[0]))

    def test_position_selection_cannot_silently_change(self):
        _,_,_,samples,generated=fixture();generated.positions=generated.positions+.01
        with self.assertRaisesRegex(ValueError,'sample zero'):
            first_native_action(generated,dict(action_samples=samples),0,5)

    def test_does_not_modify_model_samples_or_generated_wrapper(self):
        _,_,_,samples,generated=fixture();before={k:v.clone() for k,v in samples.items()};yaw=generated.yaws.clone()
        first_native_action(generated,dict(action_samples=samples),0,3)
        self.assertTrue(torch.equal(yaw,generated.yaws))
        for k,v in samples.items():self.assertTrue(torch.equal(before[k],v))


if __name__=='__main__':unittest.main()
