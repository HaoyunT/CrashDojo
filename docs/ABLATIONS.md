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

## Implementation alignment

The manuscript's ablation definitions are the implementation contract: A1/A2
remove Eq. (2)'s evidence ordering for actor and within-actor request choice;
paper A3 replaces mechanism-specific guidance with generic contact pressure.
The revised supplementary material specifies the deterministic within-actor
tie rule below. This source archive implements those definitions.

A1 remains uniform over eligible actor IDs. A2 uses the earliest constant-velocity
TTC with actor-ID ties. Both now use an evidence-independent canonical
family/parameter/geometry identity to choose a request within the selected actor;
neither uses the Eq. (2) verdict, crash-support, conflict, strength, gap, or effort
ordering. The deterministic within-actor tie rule is documented in the revised
supplementary material. Selecting the same actor as Full does not restore Full's differently
ranked mechanism/parameters.

`prepare_selectors.py` and `preflight_inputs.py` obtain their seeds from the run
manifest, covering seeds 1/2/3 when declared. A1/A2 cached selections must carry
`selector_revision: paper_actor_selection_v2`; regenerate older selections before
running. Use fresh campaign directories for this implementation revision.

Paper A3 (internal A4) keeps generic pair-contact pressure over the fixed request
window and the common prior, dynamics/road, deviation, background, and feasibility
controls. It evaluates no action-tracking loss, mechanism-phase mask, crossing
timing proxy, qualified-contact cost, parameter/timing/completion rewrite, or
tracking proposal. The formal runner already disabled the corresponding switches;
the same contract is now enforced at direct guide and loss entrypoints as well.
Contextual pressure scheduling remains unchanged.

These are source changes and regression checks, not rerun benchmark results.
The archive does not establish that the existing manuscript's Table 4 values were
produced by the corrected selector. Establishing that correspondence requires
running the affected experiment with this revision; no result values are changed
or synthesized here.

For CPU regression tests (PyTorch, NumPy, and Shapely required):

```bash
python -m unittest discover -s tests -p 'test_ablation_alignment.py' -v
```

## Running

The described five-arm study uses 100 nuScenes development scenes, IDM and
Diffusion Planner, a seven-second horizon, and seeds 1/2/3. These protocol values
are not result records. Use fresh run directories and preserve failed or missing
jobs in the new manifest. Do not infer per-seed outcomes from summary means.

Follow the external dependency and path setup in [EVALUATION.md](EVALUATION.md).
The ablation runtime additionally needs the map/rule adapter environment of the
main runner. Its `compat_v6_loader.py` uses the shared frozen-request dispatcher. Native-only
runs use the existing native validator and need no legacy loader configuration.
Only requests explicitly marked `selection_mode: random_fallback` require
`compatibility_sources.json` with `random_loader_source` and
`random_loader_sha256`. Supply the original compatible source and its digest;
the dispatcher verifies it before use and records which loader handled the
request. It never relabels fallback geometry as native geometry.

```bash
python scripts/experiment.py --suite ablation_seed1 --dry-run -- --job-id JOB_ID
python scripts/experiment.py --suite ablation_seeds23 --dry-run -- --job-id JOB_ID
```

Data, checkpoints, requests, cohort and job manifests must exist before
executing. Save results under an ignored runtime directory; `.gitignore` excludes
weights, trajectories, run folders and JSONL logs.
