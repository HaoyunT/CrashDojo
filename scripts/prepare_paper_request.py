#!/usr/bin/env python3
"""Prepare an explicit manuscript configuration without running an experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccdiff.evaluation.paper_contract import prepare_paper_request, read_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-profile', action='store_true')
    parser.add_argument('--request', type=Path)
    parser.add_argument('--dataset', choices=('nuscenes', 'nuplan'))
    parser.add_argument('--arm', choices=('A0', 'A1', 'A2', 'A4', 'U'), default='A0',
                        help='Internal label; paper A3=A4 and paper A4=U')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.show_profile:
        print(json.dumps(read_profile(), indent=2))
        return
    if not args.request or not args.dataset or not args.output:
        parser.error('--request, --dataset and --output are required')
    source = json.loads(args.request.read_text())
    request, audit = prepare_paper_request(source, args.dataset, args.arm, args.steps)
    # Never overwrite a frozen input or an earlier prepared request.
    audit_path = args.output.with_suffix('.audit.json')
    if args.output == audit_path or args.output.exists() or audit_path.exists():
        raise FileExistsError('Use fresh request and audit output paths')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(request, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with audit_path.open('x') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
