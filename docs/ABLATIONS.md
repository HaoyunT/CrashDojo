# Ablations

Source for the component ablations. Results are not included.

- `experiments/ablations/seed1`: development source with `run.py`, `entry.py`,
  `metrics.py`, `final_metrics.py`, hooks, model override and its overlay.
- `experiments/ablations/seeds23`: the seeds-2/3 runner, kept separate from seed 1
  because its runtime version differs.
- `analysis/realization/realization_metrics.py`: request-realization scorer.

## Ablation arms

| Arm | Meaning |
|---|---|
| A0 | Full CrashDojo |
| A1 | Random actor, canonical request |
| A2 | TTC actor, canonical request |
| A3 | Generic contact pressure |
| A4 | No adversarial guidance |

The code uses the same labels as the paper.

## Implementation alignment

The manuscript's ablation definitions are the implementation contract: A1/A2
remove Eq. (2)'s evidence ordering for actor and within-actor request choice;
paper A3 replaces mechanism-specific guidance with generic contact pressure.
Supplementary Appendix A.1 (accident-grounded actor--mechanism selection)
specifies the deterministic within-actor tie rule below. This source archive implements those definitions.

A1 remains uniform over eligible actor IDs. A2 uses the earliest constant-velocity
TTC with actor-ID ties. Both now use an evidence-independent canonical
family/parameter/geometry identity to choose a request within the selected actor;
neither uses the Eq. (2) verdict, crash-support, conflict, strength, gap, or effort
ordering. The deterministic within-actor tie rule is documented in Supplementary
Appendix A.1. Selecting the same actor as Full does not restore Full's differently
ranked mechanism/parameters.

`prepare_selectors.py` and `preflight_inputs.py` obtain their seeds from the run
manifest, covering seeds 1/2/3 when declared. A1/A2 cached selections must carry
`selector_revision: paper_actor_selection_v2`; regenerate older selections before
running. Use fresh campaign directories for this implementation revision.

A3 keeps generic pair-contact pressure over the fixed request
window and the common prior, dynamics/road, deviation, background, and feasibility
controls. It evaluates no action-tracking loss, mechanism-phase mask, crossing
timing proxy, qualified-contact cost, parameter/timing/completion rewrite, or
tracking proposal. The formal runner already disabled the corresponding switches;
the same contract is now enforced at direct guide and loss entrypoints as well.
Contextual pressure scheduling remains unchanged.

The five reported arms also use `configs/paper_runtime.json` through the shared
paper request contract. All freeze their maneuver parameters and window before
rollout; initialization-time parameter/timing/completion rewriting is disabled.
Execution ranking includes the motion prior against the final solver's native
proposal, with candidate identities checked before selection. The corresponding
regressions are in `tests/test_paper_runtime_contract.py`.

The A1/A2 rows of the manuscript's Table 4 were produced with the canonical-request
selector specified in Supplementary Appendix A.1, which is the rule implemented
here; the regression tests check that rule but do not re-run the benchmark.
The runtime contract and candidate scoring described above are likewise the
versions used for the reported results; the regression tests check them but do
not re-run the benchmark, and no result values are changed or synthesized here.

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
