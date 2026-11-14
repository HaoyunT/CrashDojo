# Legacy Baselines On The Fixed Mixed Cohort

`entry_safesim.py --job-file JOB.json --output RUN_DIR [--steps 10]` runs
the original SafeSim diffusion generator. The job has `method`,
`planner`, and `source_job`; source_job contains the contextual cohort's
`dataset`, `config`, `manifest`, `steps`, and planner identity. An optional outer
`initial_reference` points to the shared initial-reference PT file.

The adapter preserves native generator weights, sampler steps, loss terms,
candidate ranking, and guidance. It standardizes K=5, the same actual planner
plus `NativeTrackingEgoPolicy`, available logged background replay, the initial
actor pool, 0.1-second frames, and five-frame replanning. The manifest's native
scene ID and `start_step_10hz` control scene/start; nuPlan has 70 steps and
nuScenes has 100 steps in full runs.

SafeSim uses its official predefined actor assignment when it binds to the live
population and protected ego; otherwise it runs on the request's target actor
(see `entry_safesim.py`). Its original guidance remains active in both cases.
`native_assignment_source.json` records each run's actual assignment.

`prepare_reference.py --job-file JOB.json --output REFERENCE.pt` uses the modern
checkpoint's config and EnvUnifiedBuilder on CPU, without loading the model.
It freezes the common filtered actor IDs, real initial 31-frame observed
history, map cache and lane parameters. Legacy inputs retain the original
11-frame 9-state layout, while the external planner receives 31 observed
8-state frames and no recorded actor future. Subsequent planner histories are
assembled from actual executed observations.

An optional `source_job.nuplan_cache` selects the new private cache and is
checked against the frozen reference. The legacy nuPlan dataset adapter reads
only that prepared scene and city. It never rebuilds raw data/maps or applies
the centering transform again, avoiding unavailable raw-map dependencies in the
legacy Python environment. SafeSim's original cause-collision ego-plan
assignment uses an explicit, contiguous broadcast over the five samples to
avoid an old CUDA advanced-indexing error; this repeats exactly the same plan
and preserves the loss formula and other actors.

The copied v5 availability adapter retains absent actors as NaN in logs and
occupancy. Only the native raster center-point OffRoadRate collectors remain;
the simulator's native ADE/FDE are still computed by get_metrics. The common
eight-metric postprocessor owns collision recomputation and qualification.
Execution records include selected actor IDs each block, actual tracked ego
and exact background replay evidence, plus selected actors' three observed
local positions and five executed local positions for dynamics verification.

`native_inference_validation.json` reports frozen model hashes, zero optimizer
steps, actual/expected replan counts, selection provenance, and execution
invariants. A successful short smoke is not a completed full-cohort evaluation.
