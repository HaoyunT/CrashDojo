import unittest
from types import SimpleNamespace
import torch
from selected_kinematic_state import correct_selected_direction,install


def fixture():
    states=torch.zeros(1,3,4);states[0,:,2]=torch.tensor([2.,.35,4.])
    history=torch.zeros(1,3,3,2);history[0,1,-2,0]=.035
    batch=dict(history_positions=history,history_yaws=torch.zeros(1,3,3,1),history_availabilities=torch.ones(1,3,3,dtype=torch.bool))
    return dict(curr_states=states,neural_feature=torch.randn(1,3,8)),batch


class TestSignedState(unittest.TestCase):
    def test_backward_selected_only_and_no_mutation(self):
        aux,batch=fixture();before=aux['curr_states'].clone();hist=batch['history_positions'].clone()
        result,audit=correct_selected_direction(aux,batch,1)
        expected=before.clone();expected[0,1,2]*=-1
        self.assertTrue(torch.equal(result['curr_states'],expected))
        self.assertTrue(torch.equal(aux['curr_states'],before));self.assertTrue(torch.equal(batch['history_positions'],hist))
        self.assertIs(result['neural_feature'],aux['neural_feature']);self.assertTrue(audit['changed'])

    def test_forward_is_bitwise_unchanged(self):
        aux,batch=fixture();batch['history_positions'][0,1,-2,0]*=-1
        result,audit=correct_selected_direction(aux,batch,1)
        self.assertIs(result,aux);self.assertFalse(audit['changed'])

    def test_direction_uses_current_heading(self):
        aux,batch=fixture();batch['history_yaws'][0,1,-1,0]=torch.pi
        result,audit=correct_selected_direction(aux,batch,1)
        self.assertIs(result,aux);self.assertFalse(audit['changed'])

    def test_existing_signed_speed_is_not_flipped_twice(self):
        aux,batch=fixture();aux['curr_states'][0,1,2]*=-1
        result,audit=correct_selected_direction(aux,batch,1)
        self.assertIs(result,aux);self.assertFalse(audit['changed'])

    def test_no_action_and_missing_history(self):
        aux,batch=fixture();self.assertEqual(correct_selected_direction(aux,batch,None),(aux,None))
        batch['history_availabilities'][0,1,-2]=False
        with self.assertRaises(ValueError):correct_selected_direction(aux,batch,1)

    def test_install_is_idempotent_and_target_can_be_cleared(self):
        aux,batch=fixture();policy=SimpleNamespace(get_aux_info=lambda *a,**k:aux)
        install(policy,1);original_wrapper=policy.get_aux_info;install(policy,1)
        self.assertIs(policy.get_aux_info,original_wrapper)
        self.assertLess(float(policy.get_aux_info(batch)['curr_states'][0,1,2]),0.)
        install(policy,None);self.assertIs(policy.get_aux_info(batch),aux)
        self.assertEqual(len(policy._selected_kinematics_audits),1)


if __name__=='__main__':unittest.main()
