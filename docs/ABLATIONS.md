# Ablations

Source for the component ablations. Results are not included.

- `experiments/ablations/seed1`: development source with `run.py`, `entry.py`,
  `metrics.py`, `final_metrics.py`, hooks, model override and its overlay.
- `experiments/ablations/seeds23`: the seeds-2/3 runner, kept separate from seed 1
  because its runtime version differs.
- `analysis/realization/realization_metrics.py`: request-realization scorer.

## Ablation arms

| Arm | Meaning | Internal code arm |
|---|---|---|
| A0 | Full CrashDojo | A0 |
| A1 | Random actor | A1 |
| A2 | TTC actor | A2 |
| A3 | Generic contact pressure | **A4** |
| A4 | No adversarial guidance | **U** |

Internal A3 (geometry-only selection), A5 (no prior), A6 (no online), A7 (neither)
and A8 (fixed-target) are additional development variants, not the five arms above.
Do not relabel the source constants in place: use an explicit reporting map.

## Running

The described five-arm study uses 100 nuScenes development scenes, IDM and
Diffusion Planner, a seven-second horizon, and seeds 1/2/3. These protocol values
are not result records. Use fresh run directories and preserve failed or missing
jobs in the new manifest. Do not infer per-seed outcomes from summary means.

Follow the external dependency and path setup in [EVALUATION.md](EVALUATION.md).
The ablation runtime additionally needs the map/rule adapter environment of the
main runner. Its `compat_v6_loader.py` requires a locally supplied
`compatibility_sources.json` with `random_loader_source` and
`random_loader_sha256`, pointing to the original compatible `load_request`
implementation. Supply the compatible source,
calculate its SHA-256, and create a fresh two-field configuration; do not bypass
the loader hash check.

```bash
python scripts/experiment.py --suite ablation_seed1 --dry-run -- --job-id JOB_ID
python scripts/experiment.py --suite ablation_seeds23 --dry-run -- --job-id JOB_ID
```

Data, checkpoints, requests, cohort and job manifests must exist before
executing. Save results under an ignored runtime directory; `.gitignore` excludes
weights, trajectories, run folders and JSONL logs.
