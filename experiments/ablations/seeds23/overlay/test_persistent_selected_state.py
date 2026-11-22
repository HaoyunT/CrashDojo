import copy
from types import SimpleNamespace
import unittest

import torch

from ccdiff.models.diffuser_helpers import unicyle_forward_dynamics
from persistent_selected_state import PersistentSelectedState


def fixture(speed=2.):
    state=torch.zeros(1,2,4);state[0,:,2]=torch.tensor([4.,speed])
    hist=torch.zeros(1,2,3,2);hist[0,1,:,0]=torch.tensor([-.4,-.2,0.])
    batch=dict(history_positions=hist,history_yaws=torch.zeros(1,2,3,1),
               history_availabilities=torch.ones(1,2,3,dtype=torch.bool))
    obs=dict(centroid=torch.zeros(2,2),yaw=torch.zeros(2),world_from_agent=torch.eye(3).repeat(2,1,1))
    aux=dict(curr_states=state,neural=torch.arange(12).reshape(1,2,6).float())
    class Policy:
        def get_aux_info(self,batch):return aux
    return Policy(),aux,batch,obs


def native(initial,steps=5):
    dyn=SimpleNamespace(acce_bound=(-10.,8.),vbound=(-10.,30.),max_steer=.5,max_yawvel=2*torch.pi)
    u=torch.zeros(1,steps,2);u[...,0]=2.5
    return unicyle_forward_dynamics(dyn,initial[None],u,.1,mode='parallel')[0]


class Tests(unittest.TestCase):
    def test_native_continuity_removes_reset_induced_jerk_without_changing_acceleration(self):
        policy,aux,batch,obs=fixture();tracker=PersistentSelectedState();tracker.install(policy)
        tracker.begin(obs,1,0);first_state=policy.get_aux_info(batch)['curr_states'][0,1]
        first=native(first_state);tracker.commit(obs,first[:,:2],first[:,3])
        self.assertAlmostEqual(tracker.pending['endpoint_speed'],3.25,places=4)
        end=first[-1,:2];obs['centroid'][1]=end;obs['world_from_agent'][1,:2,2]=end
        aux['curr_states'][0,1,2]=3.125  # Cache supplies the last interval average.
        tracker.begin(obs,1,5);second_state=policy.get_aux_info(batch)['curr_states'][0,1]
        self.assertAlmostEqual(float(second_state[2]),3.25,places=4)
        second=native(second_state);wrong=native(aux['curr_states'][0,1])
        path=torch.cat([torch.zeros(1,2),first[:,:2],second[:,:2]+end])
        bad=torch.cat([torch.zeros(1,2),first[:,:2],wrong[:,:2]+end])
        self.assertLess(float(torch.diff(path,n=3,dim=0).norm(dim=-1).max()/.1**3),.01)
        self.assertGreater(float(torch.diff(bad,n=3,dim=0).norm(dim=-1).max()/.1**3),10.)
        self.assertTrue(tracker.records[-1]['previous_prefix_execution_verified'])

    def test_auxiliary_history_and_other_actors_are_not_mutated(self):
        policy,aux,batch,obs=fixture();before=copy.deepcopy(batch);features=aux['neural'].clone();others=aux['curr_states'][0,0].clone()
        tracker=PersistentSelectedState();tracker.install(policy);tracker.begin(obs,1,0)
        state=policy.get_aux_info(batch)['curr_states'][0,1];plan=native(state);tracker.commit(obs,plan[:,:2],plan[:,3])
        obs['centroid'][1]=plan[-1,:2];obs['world_from_agent'][1,:2,2]=plan[-1,:2]
        aux['curr_states'][0,1,2]=3.125;tracker.begin(obs,1,5);result=policy.get_aux_info(batch)
        self.assertEqual(float(aux['curr_states'][0,1,2]),3.125)
        self.assertTrue(torch.equal(result['curr_states'][0,0],others))
        self.assertTrue(torch.equal(result['neural'],features))
        for key in batch:self.assertTrue(torch.equal(before[key],batch[key]))

    def test_pose_mismatch_does_not_reuse_unexecuted_state(self):
        policy,aux,batch,obs=fixture();tracker=PersistentSelectedState();tracker.install(policy);tracker.begin(obs,1,0)
        plan=native(policy.get_aux_info(batch)['curr_states'][0,1]);tracker.commit(obs,plan[:,:2],plan[:,3])
        obs['centroid'][1]=plan[-1,:2]+.01
        with self.assertRaisesRegex(ValueError,'Observed pose'):tracker.begin(obs,1,5)

    def test_timing_and_duplicate_commit_are_rejected(self):
        policy,aux,batch,obs=fixture();tracker=PersistentSelectedState();tracker.install(policy);tracker.begin(obs,1,0)
        plan=native(policy.get_aux_info(batch)['curr_states'][0,1]);tracker.commit(obs,plan[:,:2],plan[:,3])
        with self.assertRaisesRegex(ValueError,'committed twice'):tracker.commit(obs,plan[:,:2],plan[:,3])
        with self.assertRaisesRegex(ValueError,'number of executed'):tracker.begin(obs,1,10)

    def test_initial_reverse_sign_follows_observation_without_changing_magnitude(self):
        policy,aux,batch,obs=fixture(.2);batch['history_positions'][0,1,-2,0]=.02
        tracker=PersistentSelectedState();tracker.install(policy);tracker.begin(obs,1,0)
        result=policy.get_aux_info(batch)
        self.assertAlmostEqual(float(result['curr_states'][0,1,2]),-.2)
        self.assertAlmostEqual(float(aux['curr_states'][0,1,2]),.2)


if __name__=='__main__':unittest.main()
