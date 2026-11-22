# Paper implementation map

The manuscript definitions govern the following code paths. This map separates
implemented behavior from the external inputs required to reproduce a campaign.

| Paper component | Implementation |
|---|---|
| Evidence-ordered actor/mechanism ranking | `experiments/ablations/seed1/prepare_selectors.py:full_key`; the original candidate enumeration and judging pipeline is supplied externally |
| Persistent request, absolute maneuver clock, geometry tracking | `experiments/main/{nuScenes,nuPlan}/overlay/five_class_guidance.py` and `persistent_selected_state.py` |
| Context-conditioned reference and online pressure | `hierarchical_prior.py`, `contextual_contract.py`, `dynamic_strength.py` in each main suite |
| Frozen generation and executed-prefix candidate checks | `overlay/selected_planner_rollout.py`, `overlay/five_class_guidance.py`, and the main `run.py` integrity checks |
| Manuscript runtime settings and immutable requests | `configs/paper_runtime.json` and `ccdiff/evaluation/paper_contract.py`; applied by both main runners and the five paper ablation arms |
| Native/fallback request loading | `experiments/support/request_loader_compat.py`, installed by each main and ablation entrypoint |
| A1 random actor and A2 TTC actor | `selector_rules.py` and `prepare_selectors.py` in each ablation suite |
| A3 generic contact pressure | `overlay/ablation_contract.py` and `overlay/five_class_guidance.py` in each ablation suite |
| A4 no adversarial guidance | `run.py`, `ablation_hooks.py`, and the A4 branch of `loss_terms` |
| Main metric definitions | `experiments/main/*/metrics.py` and baseline-suite metric code; the archived legacy metric contract is separate |
| VLM-assisted case inspection | `docs/VLM_ASSESSMENT.md`; the original inference runner and complete prompt are external inputs |

## Local checks

```bash
python -m unittest discover -s tests -v
python scripts/experiment.py --list
python scripts/experiment.py --suite main_nuscenes --dry-run -- --job-id JOB_ID
```

The unit tests exercise actor/request selection independently of evidence scores,
seed coverage and stale-cache rejection, generic-contact isolation and gradients,
frozen request dispatch, identity preservation and source-digest validation, and
training argument forwarding. They do not launch a CUDA benchmark.

Datasets, model/planner weights, frozen cohort/request/protocol manifests,
accident/context source tables, and the historical fallback loader remain
external. The A1/A2 selector (Supplementary Appendix A.1), the request/runtime
contract, and the candidate scoring implemented in this archive are the versions
used for the reported results; the regression tests check them but do not re-run
the benchmark. Numerical results are not modified by these code changes.

## Request and execution contracts

`configs/paper_runtime.json` records the manuscript's solver, tracking, contact,
motion-limit, sample-budget, and domain-calibration settings for new runs. The
main runners and the A0–A4 ablation runners apply them before writing
the effective request. `paper_runtime_contract.json` records the input/effective
digests and changed controls. Per-request deviation weights, background margins,
and the prefix-repair priority still require explicit inputs; they are not inferred
from missing historical configurations. No geometry or maneuver parameters are
invented by this preparation step.

The guide freezes actor, family, parameters, intervention window, motion limits,
geometry and contextual inputs at construction. Runtime parameter search, timing
rewrite and completion rewrite are disabled in the paper profile and rejected by
direct guide calls. Live pressure updates remain enabled. The absolute maneuver
clock advances through each replan without changing the frozen request.

Execution ranking uses the final flow step's decoded physical candidates and
their corresponding pre-optimization model proposals. It includes the motion
prior from the same complete objective used by endpoint optimization, instead of
passing a candidate as its own reference. Candidate value/order checks reject
misalignment, and each replan clears the stored references. Feasibility-first
selection and the least-violating fallback retain their existing policy.

These implementation contracts are the ones used for the reported results.
The supplied profile restates their manuscript-level settings for new runs; it is
not a recovered campaign manifest, and per-request inputs from the original runs
are not included. The external candidate
enumerator, seven-check implementation, fallback loader and original VLM inputs
remain required for their respective reproduction paths.
