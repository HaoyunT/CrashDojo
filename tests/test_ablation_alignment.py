"""Paper actor-selection and generic-contact contracts, without datasets/GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SUITES = [ROOT/'experiments/ablations'/name for name in ('seed1', 'seeds23')]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guide_module(suite):
    with patch.object(sys, 'path', [str(suite/'overlay'), str(suite)] + sys.path):
        return load_module('alignment_guide_' + suite.name,
                           suite/'overlay/five_class_guidance.py')


def candidate(actor, speed, score=0):
    return dict(actor=actor, family='C1', source='native_search',
                judgement=dict(n_fail=score, n_insufficient=score,
                               crash_level='type_statistics_and_similar_cases',
                               strength_tier='natural_L1'),
                evidence=dict(contact=score == 0,
                              minimum_body_gap_in_effective_window_m=score,
                              peak_braking_mps2=score, peak_acceleration_mps2=score),
                request=dict(family='C1', parameters=dict(initial_speed_mps=4.,
                             target_speed_mps=speed, speed_transition_s=1.),
                             geometry_evidence={'kind': 'native'},
                             interaction_window_s=[0., 2.]))


class SelectionAlignmentTests(unittest.TestCase):
    def test_judgements_and_outcomes_do_not_choose_within_actor(self):
        for suite in SUITES:
            with self.subTest(suite=suite.name):
                rules = load_module('rules_' + suite.name, suite/'selector_rules.py')
                pool = [candidate('a', 8., 0), candidate('a', 2., 100), candidate('b', 5., 3)]
                selected = rules.canonical_candidates(pool)
                self.assertEqual(selected['a']['request']['parameters']['target_speed_mps'], 2.)
                class Unreadable(dict):
                    def __getitem__(self, key):
                        raise AssertionError('An evidence ranking was consulted')
                    def get(self, key, default=None):
                        raise AssertionError('An evidence ranking was consulted')
                for item in pool:
                    item['judgement'] = Unreadable()
                    item['evidence'] = Unreadable()
                self.assertEqual(rules.canonical_candidates(list(reversed(pool))), selected)

    def test_same_actor_cannot_restore_fulls_different_parameters(self):
        for suite in SUITES:
            rules = load_module('rules_' + suite.name, suite/'selector_rules.py')
            chosen = candidate('a', 2.)
            full = dict(candidate('a', 8.)['request'], adversary_vehicle_id='a')
            self.assertFalse(rules.matches_full_request(chosen, full))
            full['parameters'] = copy.deepcopy(chosen['request']['parameters'])
            self.assertTrue(rules.matches_full_request(chosen, full))

    def test_old_selection_cache_is_rejected(self):
        for suite in SUITES:
            rules = load_module('rules_' + suite.name, suite/'selector_rules.py')
            for arm in ('A1', 'A2'):
                with self.assertRaisesRegex(ValueError, 'Regenerate'):
                    rules.validate_selection_revision({}, arm)
                rules.validate_selection_revision({'selector_revision': rules.SELECTOR_REVISION}, arm)

    def test_actual_selector_generates_declared_seeds_and_neutral_requests(self):
        for suite in SUITES:
            with self.subTest(suite=suite.name), tempfile.TemporaryDirectory() as temp:
                queue = Path(temp)
                def save(path, data):
                    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data))
                def read(path):
                    return json.loads(Path(path).read_text())
                full = dict(candidate('a', 8.)['request'], adversary_vehicle_id='a',
                            scene_id='test-scene', start_step_10hz=5,
                            weights=dict(action=5., pressure=2., prior=1., constraint=1.))
                save(queue/'full.json', full)
                save(queue/'manifest.json', {'start_step_10hz': 5, 'city': 'test-city'})
                save(queue/'prior.json', {})
                save(queue/'context.json', {'prior_table': str(queue/'prior.json')})
                job = dict(scene='test-scene', candidate_ids=['a','b'], cache='unused',
                           family='C1', base_request=str(queue/'full.json'),
                           manifest=str(queue/'manifest.json'), context_record=str(queue/'context.json'))
                save(queue/'cohort.json', [job])
                save(queue/'run_manifest.json', {'seeds': [1,2,3]})
                pool = [candidate('a', 8., 0), candidate('a', 2., 10), candidate('b', 2., 1)]
                save(queue/'candidate_capture/00/candidate_pools/test-scene.json', {'pool': pool})
                tracks = {actor: dict(lookup={i:i for i in range(6)},
                                      xy=np.zeros((6,2))+offset, size=np.array([4.,2.]))
                          for actor,offset in [('ego',0.),('a',10.),('b',20.)]}
                common = ModuleType('common')
                common.ROOT=queue; common.HERE=suite; common.QUEUE=queue
                common.load=read; common.save=save
                common.sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                maps = ModuleType('q16_map_events_v2_20261020')
                maps.load_tracks=lambda p:tracks
                rules = load_module('rules_' + suite.name, suite/'selector_rules.py')
                builder = SimpleNamespace(
                    SOURCES={'nuscenes':queue}, read=lambda p:{'test-scene':{}},
                    NuScenesSource=lambda records:SimpleNamespace(scene=lambda scene:tracks),
                    event_snapshot=lambda:(None,None), Predictor=lambda **kw:None,
                    RoadMap=lambda *args:None, derive=lambda *args:{'prior':{'factor':1.}},
                    alter_future=lambda *args,**kw:tracks,
                    canonical=lambda value:json.dumps(value,sort_keys=True))
                with patch.dict(sys.modules, {'common':common,'selector_rules':rules,
                                             'q16_map_events_v2_20261020':maps}), \
                     patch.object(sys, 'path', list(sys.path)):
                    selector = load_module('selector_' + suite.name, suite/'prepare_selectors.py')
                    fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None))
                    with patch.object(selector.importlib.util,'spec_from_file_location',return_value=fake_spec), \
                         patch.object(selector.importlib.util,'module_from_spec',return_value=builder):
                        selector.main()
                self.assertEqual(read(queue/'selectors_ready.json')['seeds'], [1,2,3])
                for seed in (1,2,3):
                    for arm in ('A1','A2'):
                        selection=read(queue/f'inputs/test-scene/selection_{arm}_s{seed}.json')
                        self.assertEqual(selection['selector_revision'],rules.SELECTOR_REVISION)
                        effective=read(selection['job_overrides']['base_request'])
                        self.assertEqual(effective['parameters']['target_speed_mps'],2.)
                        if arm=='A2':
                            self.assertEqual(selection['selected_actor'],'a')
                            self.assertNotEqual(selection['job_overrides']['base_request'],job['base_request'])


def contact_case():
    count=10
    trajectory=torch.zeros((2,count,6),dtype=torch.float64)
    trajectory[:,:,0]=torch.arange(1,count+1,dtype=torch.float64)*.2
    trajectory[:,:,2]=2.
    context=dict(transform=torch.eye(3,dtype=torch.float64),
                 ego_pos=torch.tensor([[8.,0.]]*count,dtype=torch.float64),
                 ego_yaw=torch.zeros(count,dtype=torch.float64),
                 extent=torch.tensor([4.,2.],dtype=torch.float64),
                 ego_extent=torch.tensor([4.,2.],dtype=torch.float64),
                 route=torch.tensor([[0.,0.],[100.,0.]],dtype=torch.float64),
                 initial_route_s=0.,elapsed=0.,dt=.1,target=0)
    request=dict(family='C4',parameters={},ablation_v26={'arm':'A4'},
                 weights=dict(action=20.,pressure=2.,prior=1.,constraint=1.),
                 mode='contact_search',interaction_window_s=[0.,2.],
                 contact_margin_m=0.,route_offset_bounds_m=[-4.,4.],
                 engineering_limits=dict(max_speed_mps=30.,max_acceleration_mps2=5.,
                     max_braking_mps2=8.,max_lateral_mps2=5.,max_jerk_mps3=10.),
                 route_world_xy=[[0.,0.],[100.,0.]],
                 corridor_world_polygons=[[[0.,-5.],[100.,-5.],[100.,5.],[0.,5.],[0.,-5.]]],
                 geometry_evidence={},geometry_status='native_map_verified',capability_status='test',
                 contact_phase_window=True,qualified_contact_v2={'invalid':'must not be read'},
                 causal_parameter_search_v9=True,interaction_timing_v11=True,
                 execution_completion_v10=True)
    return trajectory,context,request


class GenericContactAlignmentTests(unittest.TestCase):
    def test_generic_loss_is_invariant_to_maneuver_family_and_parameters(self):
        for suite in SUITES:
            guide=guide_module(suite)
            trajectory,context,request=contact_case()
            reference=None
            with patch.object(guide,'speed_reference',side_effect=AssertionError('tracking called')), \
                 patch.object(guide,'contact_cost',side_effect=AssertionError('qualified contact called')):
                for family in ('C1','C2','C3A','C3B','C4','C5'):
                    request['family']=family
                    context['crossing_point']=torch.tensor([1000.,1000.])
                    total,terms,_=guide.loss_terms(trajectory,trajectory,context,request)
                    self.assertTrue(torch.equal(terms['action'],torch.zeros(2)))
                    self.assertTrue(bool((terms['pressure']>0).all()))
                    if reference is None:reference=total
                    torch.testing.assert_close(total,reference)
            # Direct loss calls do not rewrite the caller's request.
            self.assertEqual(request['weights']['action'],20.)
            self.assertTrue(request['contact_phase_window'])

    def test_fixed_window_keeps_early_contact_and_empty_window_is_zero(self):
        for suite in SUITES:
            guide=guide_module(suite)
            trajectory,context,request=contact_case()
            request['family']='C2'
            request['parameters']={'t0':0.,'T_lc_s':2.}
            context['ego_pos'][0]=trajectory[0,0,:2]
            _,terms,_=guide.loss_terms(trajectory,trajectory,context,request)
            torch.testing.assert_close(terms['pressure'],torch.zeros(2,dtype=torch.float64))
            request['interaction_window_s']=[3.,4.]
            _,terms,_=guide.loss_terms(trajectory,trajectory,context,request)
            torch.testing.assert_close(terms['pressure'],torch.zeros(2,dtype=torch.float64))

    def test_common_penalties_and_contact_gradients_remain_active(self):
        for suite in SUITES:
            guide=guide_module(suite)
            trajectory,context,request=contact_case()
            native=trajectory.clone()
            trajectory[:,:,4]=2.
            trajectory.requires_grad_()
            total,terms,_=guide.loss_terms(trajectory,native,context,request)
            self.assertTrue(bool((terms['prior']>0).all()))
            self.assertTrue(bool((terms['pressure']>0).all()))
            gradient=torch.autograd.grad(total.sum(),trajectory)[0]
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient[:,:,:2].abs().sum()),0.)
            self.assertGreater(float(gradient[:,:,4].abs().sum()),0.)

    def test_direct_guide_cannot_invoke_tracking_or_parameter_rewriters(self):
        # Import optimizer internals before patch.dict restores sys.modules;
        # otherwise PyTorch 2.x can try to register its lazy operators twice.
        torch.optim.LBFGS([torch.zeros(1,requires_grad=True)])
        for suite in SUITES:
            with self.subTest(suite=suite.name),tempfile.TemporaryDirectory() as temp:
                module=guide_module(suite)
                _,_,request=contact_case()
                request.update(trajectory_solver='lbfgs',gradient_steps=1,
                               last_flow_steps=1,normalized_action_trust=.1)
                forbidden=SimpleNamespace(
                    choose_request=lambda *a:(_ for _ in ()).throw(AssertionError('parameter rewrite')),
                    screen_request=lambda *a:(_ for _ in ()).throw(AssertionError('timing rewrite')),
                    prepare_request=lambda *a:(_ for _ in ()).throw(AssertionError('completion rewrite')),
                    tracking_seed=lambda *a:(_ for _ in ()).throw(AssertionError('tracking seed')))
                with patch.dict(sys.modules,{name:forbidden for name in
                         ('conflict_parameter_search','interaction_timing','execution_completion')}), \
                     patch.object(module,'warm_start',side_effect=AssertionError('warm start')):
                    guide=module.FiveClassGuide(request,Path(temp)/'guide.jsonl')
                    transforms=torch.eye(3,dtype=torch.float64).repeat(2,1,1)
                    transforms[1,0,2]=8.
                    obs={'world_from_agent':transforms,
                         'extent':torch.tensor([[4.,2.,1.],[4.,2.,1.]],dtype=torch.float64)}
                    action=SimpleNamespace(positions=torch.zeros(1,10,2,dtype=torch.float64),
                                           yaws=torch.zeros(1,10,1,dtype=torch.float64))
                    guide.prepare(obs,action,0,1,0.,.1)
                    model=SimpleNamespace(n_flow_steps=20,dt=.1,
                        dyn=SimpleNamespace(acce_bound=(-8.,5.),vbound=(0.,30.),max_steer=.5,max_yawvel=1.),
                        descale_traj=lambda x,idx:x,scale_traj=lambda x,idx:x,
                        add_coeffs=torch.zeros(6),div_coeffs=torch.ones(6))
                    endpoint=torch.zeros(2,2,10,2,dtype=torch.float64)
                    states=torch.tensor([[[0.,0.,2.,0.],[0.,0.,0.,0.]]],dtype=torch.float64).repeat(2,1,1)
                    result=guide.project(model,endpoint,{'curr_states':states},2,19)
                    self.assertTrue(bool(torch.isfinite(result).all()))
                    self.assertTrue(torch.equal(result[:,1],endpoint[:,1]))
                    self.assertEqual(request['weights']['action'],0.)
                    self.assertFalse(request['execution_completion_v10'])
                    self.assertTrue((Path(temp)/'guide.jsonl').exists())


if __name__=='__main__':
    unittest.main()
