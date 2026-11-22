#!/usr/bin/env python3
"""Locate and launch a preserved evaluation/ablation runner after runtime setup."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main(argv=None):
    catalog = json.loads((ROOT/'configs/experiments.json').read_text())
    p = argparse.ArgumentParser(description=__doc__, epilog='The snapshots retain frozen server paths. Read docs/EVALUATION.md before executing. --dry-run needs no ML dependencies.')
    p.add_argument('--suite', choices=sorted(catalog))
    p.add_argument('--list', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('runner_args', nargs=argparse.REMAINDER, help='Arguments after -- are passed unchanged to the historical runner')
    args = p.parse_args(argv)
    if args.list:
        for name, suite in catalog.items():
            print(name + ': ' + suite['path'] + '/' + suite['entrypoint'] + ' [' + suite['status'] + ']')
        return 0
    if not args.suite:
        p.error('--suite or --list is required')
    suite = catalog[args.suite]
    extra = args.runner_args[1:] if args.runner_args[:1] == ['--'] else args.runner_args
    cmd = [args.python, str(ROOT/suite['path']/suite['entrypoint'])] + extra
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(ROOT/'third_party/tbsim'),env.get('PYTHONPATH','')])
    return subprocess.call(cmd, cwd=ROOT/suite['path'], env=env)

if __name__ == '__main__':
    sys.exit(main())
