# Third-party notices

CrashDojo builds on the CCDiff traffic-model codebase. The internal `ccdiff` import
name is retained for checkpoint and import compatibility; it does not imply that
the upstream authors developed or endorse CrashDojo's additions.

| Component | Upstream | License preserved |
|---|---|---|
| CCDiff and its inherited tbsim fork | https://github.com/cruise-automation/CCDiff | Apache-2.0; preserve file headers |
| SAFE-SIM snapshot | https://github.com/jxmmy7777/safe-sim | CC BY-NC 4.0 |
| CCFM (STRIVE planner fix vendored in `ccdiff/planners/vendor`) | https://github.com/KELISBU/CCFM | CC BY-NC 4.0 |
| tuplan_garage | https://github.com/autonomousvision/tuplan_garage | Original license in `LICENSES/tuplan_garage.txt` |
| Historical TBSim ancestry | https://github.com/NVlabs/traffic-behavior-simulation | NVIDIA Source Code License-NC, retained for provenance; inherited files also carry Cruise notices |

Local code snapshots contain project modifications and are not claimed to equal
those public revisions. Dependencies such as trajdata, nuPlan, datasets and
pretrained planners retain their own terms and must be obtained separately.
This repository does not redistribute their model weights or datasets.
