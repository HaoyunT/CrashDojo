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
