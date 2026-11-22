# Contributing to CrashDojo

Thank you for improving CrashDojo. Please open an issue before a large change
so its scope and compatibility requirements can be discussed.

## Development checks

Before submitting a pull request, run:

```bash
python -m unittest discover -s tests -v
bash -n scripts/*.sh
python scripts/experiment.py --list
```

Keep CrashDojo-specific logic inside `ccdiff/` (actor selection, mechanism
guidance and planner interfaces) and `analysis/` where possible, and avoid
duplicating code already provided by the vendored tbsim runtime. The preserved
runners under `experiments/` reproduce the reported numbers; change them only
to fix a bug, and say so in the pull request. New behavior should include a
focused test and documentation for any new configuration or output field.

## Release boundaries

Do not contribute:

- nuScenes or nuPlan data, model checkpoints without redistribution
  permission, or raw evaluation outputs;
- credentials, machine-specific absolute paths, or personal information.

Changes under `baselines/safe_sim/` and `ccdiff/planners/vendor/` remain under
CC BY-NC 4.0, and `planners/tuplan_garage/` under its own license. Other
contributions are submitted under the repository's Apache-2.0 license unless a
file states otherwise. Contributors must have the right to submit their work
under the applicable license.
