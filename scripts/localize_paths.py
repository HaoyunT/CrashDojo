#!/usr/bin/env python
# Copyright 2026 The CrashDojo authors
# SPDX-License-Identifier: Apache-2.0
"""Replace the placeholder paths of the preserved runners with local paths.

The runners under ``experiments/`` were archived from the machines that
produced the reported numbers, with machine paths replaced by placeholders such
as ``/path/to/workspace``. This script maps each placeholder prefix to an
environment variable and rewrites it in place.

It only previews the changes unless ``--apply`` is given. Placeholders whose
variable is unset are left unchanged and listed in the report. Markdown files
are skipped, so the documentation keeps the placeholders.

Example::

    export CRASHDOJO_WORKSPACE=/data/crashdojo_ws
    export CRASHDOJO_STORAGE=/data/crashdojo_storage
    export NUSCENES_ROOT=/data/nuscenes
    python scripts/localize_paths.py            # preview
    python scripts/localize_paths.py --apply    # rewrite
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Longest prefixes first, so interpreter paths are not split by the
# workspace prefix.
MAPPING = [
    ("/path/to/workspace/home/miniconda3/envs/ccdiff/bin/python", "CRASHDOJO_PYTHON"),
    ("/path/to/workspace/home/miniconda3/envs/ctg/bin/python", "CRASHDOJO_SAFESIM_PYTHON"),
    ("/path/to/workspace", "CRASHDOJO_WORKSPACE"),
    ("/path/to/storage", "CRASHDOJO_STORAGE"),
    ("/path/to/local", "CRASHDOJO_LOCAL"),
    ("/path/to/nuscenes", "NUSCENES_ROOT"),
    ("/path/to/nuScenes", "NUSCENES_ROOT"),
    ("/path/to/results", "CRASHDOJO_RESULTS_DIR"),
]

SCAN_DIRS = ["analysis", "baselines", "ccdiff", "configs", "experiments",
             "planners", "scripts", "third_party/tbsim"]
SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".txt", ".cfg", ".ini"}
SKIP_PARTS = {".git", "__pycache__"}

# A placeholder ends at a path separator, a quote, whitespace or end of text,
# so "/path/to/local" never matches inside "/path/to/localhost".
_BOUNDARY = r"(?=[/'\"\s,;:)\]}]|$)"


def _pattern():
    alternatives = "|".join(re.escape(p) for p, _ in MAPPING)
    return re.compile(f"({alternatives}){_BOUNDARY}", re.MULTILINE)


def iter_files(root):
    for name in SCAN_DIRS:
        base = root / name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if (path.is_file() and path.suffix in SUFFIXES
                    and not SKIP_PARTS.intersection(path.parts)
                    and path.resolve() != Path(__file__).resolve()):
                yield path


def localize(text, values, pattern, seen, missing):
    def repl(match):
        placeholder = match.group(1)
        var = dict(MAPPING)[placeholder]
        seen[placeholder] += 1
        value = values.get(var)
        if not value:
            missing[var] += 1
            return placeholder
        return value.rstrip("/")
    return pattern.sub(repl, text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true",
                        help="rewrite files (default: preview only)")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root (default: this checkout)")
    args = parser.parse_args(argv)

    values = {var: os.environ.get(var, "") for _, var in MAPPING}
    if not values["CRASHDOJO_PYTHON"]:
        values["CRASHDOJO_PYTHON"] = sys.executable

    pattern = _pattern()
    seen, missing = Counter(), Counter()
    changed = []
    for path in iter_files(args.root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "/path/to/" not in text:
            continue
        new = localize(text, values, pattern, seen, missing)
        if new != text:
            changed.append(path.relative_to(args.root))
            if args.apply:
                path.write_text(new, encoding="utf-8")

    print("placeholder occurrences:")
    for placeholder, var in MAPPING:
        if seen[placeholder]:
            target = values.get(var) or "(unset, left unchanged)"
            print(f"  {seen[placeholder]:4d}  {placeholder}  ->  ${var} = {target}")
    verb = "rewrote" if args.apply else "would rewrite"
    print(f"{verb} {len(changed)} file(s)")
    for path in changed:
        print(f"  {path}")
    if missing:
        print("unset variables: " + ", ".join(sorted(missing)))
    if not args.apply and changed:
        print("preview only; rerun with --apply to rewrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
