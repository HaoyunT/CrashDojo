"""Check argument forwarding without importing or launching the GPU runtime."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('release_train', Path(__file__).resolve().parents[1]/'scripts/train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

class TrainingCommandTests(unittest.TestCase):
    def args(self, **overrides):
        fields=dict(python='python3',gpus=1,config='trajdata_nusc_ccflow_matching',data='/tmp/data with spaces',output='/tmp/runs',name='example',steps=100,smoke_steps=None,resume=None)
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_single_gpu_paths_are_single_arguments(self):
        cmd=train.command(self.args())
        self.assertEqual(cmd[cmd.index('--dataset_path')+1],str(Path('/tmp/data with spaces').resolve()))
        self.assertNotIn('torch.distributed.run',cmd)
        self.assertNotIn('--remove_exp_dir',cmd)

    def test_ddp_and_resume_are_forwarded(self):
        cmd=train.command(self.args(gpus=4,resume='/tmp/last.ckpt'))
        self.assertIn('--nproc_per_node=4',cmd)
        self.assertEqual(cmd[cmd.index('--parallel_strategy')+1],'ddp')
        self.assertEqual(cmd[cmd.index('--resume_ckpt')+1],str(Path('/tmp/last.ckpt').resolve()))

if __name__=='__main__':
    unittest.main()
