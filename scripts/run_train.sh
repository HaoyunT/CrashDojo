#!/usr/bin/env bash
# Copyright 2026 The CrashDojo authors
# SPDX-License-Identifier: Apache-2.0
#
# Train the CrashDojo world model through scripts/train.py.
# Extra arguments are passed through, e.g. --dry-run or --smoke-steps 5.

set -euo pipefail

CRASHDOJO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes dataset directory}"

args=(
  --config "${CRASHDOJO_CONFIG:-trajdata_nusc_ccflow_matching}"
  --data "${NUSCENES_ROOT}"
  --output "${CRASHDOJO_OUTPUT_DIR:-runs/train}"
  --name "${CRASHDOJO_RUN_NAME:-crashdojo_flow}"
  --gpus "${CRASHDOJO_GPUS:-1}"
)
if [[ -n "${CRASHDOJO_STEPS:-}" ]]; then
  args+=(--steps "${CRASHDOJO_STEPS}")
fi
if [[ -n "${CRASHDOJO_RESUME:-}" ]]; then
  args+=(--resume "${CRASHDOJO_RESUME}")
fi

python "${CRASHDOJO_ROOT}/scripts/train.py" "${args[@]}" "$@"
