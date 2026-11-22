#!/usr/bin/env bash
# Copyright 2026 The CrashDojo authors
# SPDX-License-Identifier: Apache-2.0
#
# Launch a preserved evaluation or ablation runner through
# scripts/experiment.py. Run scripts/localize_paths.py --apply first so the
# runners point at your datasets, checkpoints and output directories.
# Arguments are passed unchanged to the runner, e.g.
#   CRASHDOJO_SUITE=main_nuscenes bash scripts/run_eval.sh --job-id JOB_ID

set -euo pipefail

CRASHDOJO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${CRASHDOJO_SUITE:?Set CRASHDOJO_SUITE (see: python scripts/experiment.py --list)}"

extra=()
if [[ "${CRASHDOJO_DRY_RUN:-0}" == "1" ]]; then
  extra+=(--dry-run)
fi

python "${CRASHDOJO_ROOT}/scripts/experiment.py" \
  --suite "${CRASHDOJO_SUITE}" \
  --python "${CRASHDOJO_PYTHON:-python}" \
  ${extra[@]+"${extra[@]}"} \
  -- "$@"
