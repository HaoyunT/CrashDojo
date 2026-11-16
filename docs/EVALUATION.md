# Evaluation

## Source map

| Suite | Directory | Entrypoint |
|---|---|---|
| CrashDojo nuScenes | `experiments/main/nuScenes` | `run.py` |
| CrashDojo nuPlan | `experiments/main/nuPlan` | `run.py` |
| Baselines, part 1 | `experiments/main/baselines_part1` | `run_job.py` |
| Baselines, part 2 | `experiments/main/baselines_part2` | `run_job.py` |

## Runtime setup required

Absolute paths in the runners use the placeholders `/path/to/workspace` (project
root), `/path/to/storage` (shared storage) and `/path/to/local`. Replace them with
your own locations. Before executing:

1. Install the generator environment, datasets, maps and trajdata cache. Install
   each external ego planner and obtain its checkpoint separately.
2. Adapt `common.py` (`ROOT`, `HERE`, `QUEUE`, `REPO`, `BASE`, Python interpreters)
   and the adapters' dataset/cache/planner paths to the new machine. The new core
   is the repository root, and each suite's `HERE` is its directory above.
3. Main CrashDojo `common.py` imports
   `runtime_common.py`, included as `experiments/support/runtime_common.py`. Point
   the import to this copy and configure its host/GPU mapping for your machine.
   Referenced `map_deps` remain external. Do not treat an import failure as an
   experimental failure.
4. Build cohort, request, job and protocol manifests for the intended run, and
   generate their digests.
5. Run a bounded smoke evaluation, inspect actor identity/metric alignment and
   environment versions, then freeze the full protocol and start the new campaign.

Datasets, checkpoints, request files and results are not part of this repository.

Host names such as `a100-host`, `gpu-host-8` and `local-gpu-host`, and device IDs
starting with `GPU-REPLACE-`, are deployment placeholders. Replace them
consistently in worker admission rules, allocation manifests and
`experiments/support/runtime_common.py` before running a campaign. Local log
paths use `/path/to/local`; no original machine identity is required.

`configs/legacy/metric_contract.json` and its parser in
`ccdiff/evaluation/contract.py` preserve an earlier development contract. This
contract is not the paper's main-table metric definition and is not loaded by
the main evaluation entrypoints. Use the metric definitions and runtime
protocol for the intended suite; do not substitute this archived contract.

## Entrypoint examples

```bash
python scripts/experiment.py --suite main_nuscenes --dry-run -- --job-id JOB_ID
python scripts/experiment.py --suite main_nuplan --dry-run -- --job-id JOB_ID --smoke
python scripts/experiment.py --suite baselines_part1 --dry-run -- --job-id JOB_ID --gpu 0
```

Remove `--dry-run` after the setup above.

## Baselines and planner packages

`baselines/safe_sim` contains its own `tbsim` package and environment definition.
Use a separate environment to avoid import collisions.
`planners/tuplan_garage` supplies the preserved rule-planner integration; external
nuPlan/planner packages and weights remain separate dependencies.

## Frozen request geometry

The main entrypoints preserve each request's declared geometry source. Native
requests use the existing native-map validator. Requests marked
`selection_mode: random_fallback` use the original compatible loader, supplied
through `compatibility_sources.json` in the selected suite directory:

```json
{
  "random_loader_source": "/path/to/original/five_class_guidance.py",
  "random_loader_sha256": "<SHA-256 of the original source file>"
}
```

The dispatcher checks the source digest before extracting its `load_request`
function, preserves scene/actor/start-time identity, and records the selected
loader and request digest in `request_loader_compatibility.json`. Missing or
mismatched compatibility inputs fail explicitly. Native-only runs do not need
this configuration. This dispatch does not supply the historical fallback
loader or request manifests and does not reinterpret fallback geometry as native.

The paper's main cohort sizes and horizons belong to the frozen campaign
manifests. Other preparation utilities retain their development defaults; use
the supplied protocol rather than treating every script's default as the main
evaluation configuration.

## Executable paper settings

The two main runners now apply `configs/paper_runtime.json` before saving the
effective request. This checks domain calibration and episode length, supplies
the manuscript's solver and loss settings, and disables runtime request rewriting.
The active guide also checks the 20-step flow, five-candidate, 52-frame generation
budget. Changed controls and input/effective request hashes are recorded in
`paper_runtime_contract.json` for each run. Source/input freeze checks remain in
force: create a new campaign with digests for this revision.

Inspect the profile without datasets or ML packages:

```bash
python scripts/prepare_paper_request.py --show-profile
```

Prepare an existing request with explicit calibration and per-request settings:

```bash
python scripts/prepare_paper_request.py --request INPUT.json --dataset nuscenes \
  --steps 100 --output NEW_REQUEST.json
```

The command writes a request and adjacent audit JSON, refuses to overwrite either,
and does not launch inference. Ablation inputs carry `ablation_v26` and use a
70-frame horizon; pass the internal arm mapping documented in `ABLATIONS.md`.
Missing geometry, original per-request weights/margins, historical fallback
loaders or candidate-evidence sources are not supplied by this profile. The file
is a manuscript-derived implementation specification, not proof of a historical
run's configuration or results.
