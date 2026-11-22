#!/usr/bin/env python3
"""Portable argument adapter for the actual Lightning training entrypoint."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def command(args):
    cmd = [args.python]
    if args.gpus > 1:
        cmd += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                '--nproc_per_node=' + str(args.gpus)]
    cmd += [str(ROOT / 'ccdiff/examples/train.py'), '--config_name', args.config,
            '--dataset_path', str(Path(args.data).expanduser().resolve()),
            '--output_dir', str(Path(args.output).expanduser().resolve()),
            '--name', args.name, '--num_gpus', str(args.gpus)]
    if args.gpus > 1:
        cmd += ['--parallel_strategy', 'ddp']
    if args.steps is not None:
        cmd += ['--num_steps', str(args.steps)]
    if args.smoke_steps is not None:
        cmd += ['--smoke_steps', str(args.smoke_steps)]
    if args.resume:
        cmd += ['--resume_ckpt', str(Path(args.resume).expanduser().resolve())]
    return cmd

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='trajdata_nusc_ccflow_matching',
                   choices=['trajdata_nusc_ccflow_matching', 'trajdata_nusc_ccdiff'])
    p.add_argument('--data', required=True, help='nuScenes root, with the required prepared cache')
    p.add_argument('--output', default='runs/train')
    p.add_argument('--name', default='crashdojo_flow')
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--steps', type=int)
    p.add_argument('--smoke-steps', type=int)
    p.add_argument('--resume')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dry-run', action='store_true', help='Print command without importing ML dependencies or starting training')
    args = p.parse_args(argv)
    if args.gpus < 1 or any(x is not None and x < 1 for x in (args.steps, args.smoke_steps)):
        p.error('GPU and step counts must be positive')
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    if not Path(args.data).expanduser().is_dir():
        p.error('Dataset directory does not exist')
    if args.resume and not Path(args.resume).expanduser().is_file():
        p.error('Resume checkpoint does not exist')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT/'third_party/tbsim'), env.get('PYTHONPATH','')])
    env.setdefault('WANDB_MODE', 'disabled')
    return subprocess.call(cmd, cwd=ROOT, env=env)

if __name__ == '__main__':
    sys.exit(main())
