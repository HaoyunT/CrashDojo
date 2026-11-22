# Planner reference provenance

`safesim_hardcoded.py` is copied from SafeSim commit
`27c96a84e7bf5fbca4b47f6edde386811d76c6e7`.

`third_party/tbsim/tbsim/policies/strive_planner_trajdata.py` is copied from
CCFM commit `fdb43d54e79488acc86106f0fab1498c328a8964`. It is equivalent to the
SafeSim STRIVE core except for CCFM's interpolation-range clipping fix.

The wrappers in `ccdiff/planners/` provide the CCDiff-facing interface and do
not change the planner objectives. SafeSim's `final_tuned_val_1` limits
(`smax=20 m/s`, `accmax=4 m/s^2`) are ego-planner parameters, not dataset-wide
physical-validity thresholds.
