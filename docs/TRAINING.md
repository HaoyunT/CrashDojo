# Training

The actual entrypoint is `ccdiff/examples/train.py`, which constructs the data
module, model and PyTorch Lightning trainer and calls `trainer.fit`. The convenient
launcher in `scripts/train.py` passes paths and distributed-training options to it.

## Environment and data

The inherited environment targets Python 3.8, torch 1.13.1, torchvision 0.14.1,
Lightning 1.8.3.post0, numpy 1.23.4 and einops 0.6.0. Older Lightning dependency
metadata may require pip older than 24.1. These are the declared versions; record the
exact environment you use.

The official CCDiff installation additionally uses these repositories:

```bash
git clone https://github.com/AIasd/trajdata.git third_party/trajdata
python -m pip install -r third_party/trajdata/trajdata_requirements.txt
python -m pip install -e third_party/trajdata
git clone https://github.com/NVlabs/spline-planner.git third_party/Pplan
python -m pip install -e third_party/Pplan
```

Record and test the revisions you install. Read their licenses before reuse.
Obtain nuScenes under its dataset terms and prepare maps and trajdata caches.
Inspect `third_party/tbsim/tbsim/configs/registry.py` and the registered
`trajdata_nusc_ccflow_matching` configuration for dataset/cache paths and training
settings. The launcher's `--data` override updates the nuScenes dataset root; it
does not build missing caches or download datasets.

## Commands

```bash
# Inspect a single-GPU flow training command.
python scripts/train.py --data /data/nuScenes --name flow_run --dry-run

# A bounded run, after environment and cache preparation.
python scripts/train.py --data /data/nuScenes --name flow_smoke --smoke-steps 5

# Four-GPU distributed training. Choose the stop step for your protocol.
python scripts/train.py --data /data/nuScenes --name flow_run --gpus 4 --steps 100000

# Resume with optimizer/model/EMA/global-step state.
python scripts/train.py --data /data/nuScenes --name flow_resume --resume /models/last.ckpt
```

For other original options, use `python ccdiff/examples/train.py --help` in the
installed environment. For a full JSON config, use its `--config_file` option.
The launcher does not delete an existing experiment directory. Weights are not
included.
