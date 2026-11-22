import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
module_path=Path(__file__).with_name('selection_manifest.py')
if not module_path.exists():
    module_path=Path(__file__).resolve().parents[2]/'ccdiff/evaluation/selection_manifest.py'
spec=importlib.util.spec_from_file_location('selection_manifest',module_path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.record=dict(dataset='nuscenes',scene_id='scene-0015',log_id='log',scene_token='token',adversary_vehicle_id='car-a',ego_vehicle_id='ego',vehicle_id_namespace='nuScenes instance_token',start_time_s=10.1,start_step_10hz=101)
    def test_time_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'selection.jsonl';p.write_text(json.dumps(self.record))
            self.assertEqual(m.load_selection(p,0)['start_step_10hz'],101)
            self.record['start_time_s']=10.2;p.write_text(json.dumps(self.record))
            with self.assertRaises(ValueError):m.load_selection(p,0)
    def test_token_not_row_index(self):
        self.assertEqual(m.resolve_agents(['other','ego','car-a'],self.record),(2,1))
        with self.assertRaises(ValueError):m.resolve_agents(['ego','other'],self.record)
        with self.assertRaises(ValueError):m.resolve_agents(['ego','car-a','car-a'],self.record)
    def test_scene_exact_name_dt_and_duplicates(self):
        scene=NS(name='scene-0015',env_metadata=NS(name='nusc_trainval'),dt=.1,length_timesteps=200)
        ds=NS(num_scenes=lambda:2,get_scene=lambda i: NS(name='other') if i==0 else scene)
        self.assertEqual(m.resolve_scene(ds,self.record),1)
        scene.dt=.5
        with self.assertRaises(ValueError):m.resolve_scene(ds,self.record)
    def test_reset_and_order_guard(self):
        env=NS(current_agent_names=['other','ego','car-a'],current_scene_names=['scene-0015'],reset=lambda **kw:[True],get_observation=lambda:{})
        control=NS(set_controllable_set=lambda x:setattr(control,'indices',x))
        model=NS(set_guidance_dim=lambda x,t:setattr(model,'indices',x))
        with tempfile.TemporaryDirectory() as d:
            binding=m.SelectionBinding(env,self.record,5,control,model,5,Path(d)/'audit.json')
            env.reset(scene_indices=[5],start_frame_index=[101])
            self.assertEqual(control.indices,[2]);self.assertEqual(model.indices,[2])
            self.assertEqual(binding.ego_index,1)
            with self.assertRaises(ValueError):env.reset(scene_indices=[5],start_frame_index=[31])
            env.current_agent_names=['ego','other','car-a']
            with self.assertRaises(ValueError):env.get_observation()
    def test_insufficient_history_rejected_before_action(self):
        class History(list):
            def __getitem__(self, key):
                value=super().__getitem__(key)
                return History(value) if isinstance(key,slice) else value
            def all(self):return all(self)
        history=History([False]+[True]*30)
        env=NS(current_agent_names=['ego','car-a'],current_scene_names=['scene-0015'],reset=lambda **kw:[True],get_observation=lambda:{'agents':{'history_availabilities':[History([True]*31),history]}})
        control=NS(set_controllable_set=lambda x:None)
        model=NS(set_guidance_dim=lambda x,t:None)
        with tempfile.TemporaryDirectory() as d:
            m.SelectionBinding(env,self.record,0,control,model,5,Path(d)/'audit.json',history_num_frames=30)
            env.reset(scene_indices=[0],start_frame_index=[101])
            with self.assertRaises(ValueError):env.get_observation()
            history[0]=True
            env.get_observation()

    def test_nuplan_log_identity(self):
        rec=dict(self.record,dataset='nuplan',scene_id='log=token')
        scene=NS(name='log=token',env_metadata=NS(name='nuplan_mini'),dt=.1,length_timesteps=200)
        ds=NS(num_scenes=lambda:1,get_scene=lambda i:scene)
        self.assertEqual(m.resolve_scene(ds,rec),0)
        rec['log_id']='wrong'
        with self.assertRaises(ValueError):m.resolve_scene(ds,rec)

if __name__=='__main__':unittest.main()
