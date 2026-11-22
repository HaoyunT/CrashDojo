# Native CCDiff DDPM/FM adapter for the mixed 150-scene cohort

Entry: `entry_ccdiff.py --job-file JOB.json --output OUTPUT [--steps 10]`.
`JOB.json` contains `method`, `planner`, and a `source_job` copied from the
contextual protocol. A flat job with those fields also works. Source keys used:
`scene`, `dataset`, `config`, `manifest`, `steps`; optional `nuplan_cache`.
The source manifest resolves only the scene/start identity. Its selected target
is not passed to the generator, ranker, guidance, or execution wrapper.

The adapter preserves the native TTC transform and released CIG clique ranker,
Top-5 budget, seeded fill, partial guidance, and model sample ranking. CIG is
recomputed once from the admitted initial observed history for both domains;
it does not blindly apply old nuScenes actor indices to a new population.
`native_cig_initial_inputs.npz` and ranking JSON retain provenance for audits.

The external planner uses exactly the contextual runtime's source
`NativeTrackingEgoPolicy`: raw plan, LQR bicycle tracking, then exact execution.
Nonselected/non-ego traffic is logged replay with explicit absent slots. Each
block records names/track IDs, executed world poses, selected IDs, observed
last-three positions and next-five local executed positions.

This adapter deliberately imports the contextual runtime's read-only numeric,
availability, nuPlan centering, and rule-planner adapters. The protocol builder
must hash these transitive dependencies and the shared source tracker. It must
also provide a writable cache location outside frozen inputs if rebuilding is
needed. No contextual guide or dynamic-strength installer is imported.

`native_inference_validation.json` contains the agreed validation schema.
Native stats/HDF5 remain under `native_results/`. Unified eight-metric
aggregation is the outer runner's responsibility; the v5 vector-corner ORR
postprocessor is intentionally not used.

Validation completed locally: Python syntax compilation. GPU smoke on each
domain/method, exact execution audit, native guide-update audit, FM transport
attestation, candidate/actor admission parity, and source/checkpoint integrity
must pass before the formal queue is marked ready. The original seeded fill
can produce duplicate indices; the v5 transport fails closed in that case
rather than silently changing the selector.
