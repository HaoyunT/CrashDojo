#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import argparse
import json
import sys
import os
import socket
import time

import wandb
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tbsim.utils.log_utils import PrintLogger
import tbsim.utils.train_utils as TrainUtils
from tbsim.utils.env_utils import RolloutCallback
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from ccdiff.algos.factory import algo_factory

os.environ["WANDB_DISABLE_CODE"] = "True"


def is_global_zero_env():
    return int(os.environ.get("RANK", "0")) == 0


def rank_zero_print(*args, **kwargs):
    if is_global_zero_env():
        print(*args, **kwargs)


class TrainingTimeCallback(pl.Callback):
    """Persist the exact wall-clock time and stopping step for paired runs."""

    def __init__(self, summary_path):
        super().__init__()
        self.summary_path = summary_path
        self.train_start_time = None

    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.perf_counter()
        self.initial_global_step = int(trainer.global_step)

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero or self.train_start_time is None:
            return
        elapsed_seconds = time.perf_counter() - self.train_start_time
        global_step = int(trainer.global_step)
        steps_this_run = global_step - self.initial_global_step
        summary = {
            "global_step": global_step,
            "initial_global_step": self.initial_global_step,
            "steps_this_run": steps_this_run,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hours": elapsed_seconds / 3600.0,
            "seconds_per_step": (
                elapsed_seconds / steps_this_run if steps_this_run > 0 else None
            ),
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("TRAINING_SUMMARY {}".format(json.dumps(summary, sort_keys=True)))


def main(
    cfg,
    auto_remove_exp_dir=False,
    debug=False,
    max_hours=None,
    resume_ckpt=None,
):
    pl.seed_everything(cfg.seed)
    
    if cfg.env.name == "l5kit":
        set_global_batch_type("l5kit")
    elif cfg.env.name in ["nusc", "trajdata"]:
        set_global_batch_type("trajdata")
        if cfg.env.name == "nusc":
            set_global_trajdata_batch_env("nusc_trainval")
        elif cfg.env.name == "trajdata":
            # assumes all used trajdata datasets use share same map layers
            set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])
        set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer) # determines cfg for rasterizing agents
    else:
        raise NotImplementedError("Env {} is not supported".format(cfg.env.name))

    rank_zero_print("\n============= New Training Run with Config =============")
    # print(cfg)
    # print("")
    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.name,
        output_dir=cfg.root_dir,
        save_checkpoints=cfg.train.save.enabled,
        auto_remove_exp_dir=auto_remove_exp_dir
    )

    # Save experiment config to the training dir
    if is_global_zero_env():
        cfg.dump(os.path.join(root_dir, version_key, "config.json"))

    global_train_batch_size = int(cfg.train.training.batch_size)
    configured_validation_batch_size = int(cfg.train.validation.batch_size)

    if (
        cfg.train.logging.terminal_output_to_txt
        and not debug
        and int(os.environ.get("LOCAL_RANK", "0")) == 0
    ):
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    train_callbacks = [
        TrainingTimeCallback(
            summary_path=os.path.join(root_dir, version_key, "run_summary.json")
        )
    ]

    # Training Parallelism
    assert cfg.train.parallel_strategy in [
        "dp",
        "ddp",
        "ddp_spawn",
        None,
    ]  # TODO: look into other strategies
    if not cfg.devices.num_gpus > 1:
        # Override strategy when training on a single GPU
        with cfg.train.unlocked():
            cfg.train.parallel_strategy = None
    if cfg.train.parallel_strategy in ["ddp", "ddp_spawn"]:
        if global_train_batch_size % cfg.devices.num_gpus != 0:
            raise ValueError(
                "Global training batch size {} must be divisible by {} GPUs".format(
                    global_train_batch_size, cfg.devices.num_gpus
                )
            )
        with cfg.train.training.unlocked():
            cfg.train.training.batch_size = (
                global_train_batch_size // cfg.devices.num_gpus
            )
        with cfg.train.validation.unlocked():
            # The official validation batch is 1. Dividing it by four GPUs
            # produces an invalid DataLoader batch size of zero, so validation
            # keeps at least one scene per rank.
            cfg.train.validation.batch_size = max(
                1,
                (configured_validation_batch_size + cfg.devices.num_gpus - 1)
                // cfg.devices.num_gpus,
            )

    if cfg.devices.num_gpus > torch.cuda.device_count():
        raise RuntimeError(
            "Requested {} GPUs, but only {} CUDA devices are visible".format(
                cfg.devices.num_gpus, torch.cuda.device_count()
            )
        )

    run_contract = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "devices_num_gpus": int(cfg.devices.num_gpus),
        "parallel_strategy": cfg.train.parallel_strategy,
        "global_training_batch_size": global_train_batch_size,
        "per_rank_training_batch_size": int(cfg.train.training.batch_size),
        "configured_validation_batch_size": configured_validation_batch_size,
        "per_rank_validation_batch_size": int(cfg.train.validation.batch_size),
        "training_num_steps": int(cfg.train.training.num_steps),
        "max_hours": max_hours,
        "resume_ckpt": resume_ckpt,
        "n_diffusion_steps": int(cfg.algo.n_diffusion_steps),
        "n_flow_steps": (
            int(cfg.algo.n_flow_steps)
            if "n_flow_steps" in cfg.algo
            else None
        ),
        "transport": (
            "flow_matching"
            if cfg.algo.name == "ccflow_matching"
            else "ddpm"
        ),
        "predict_epsilon": bool(cfg.algo.predict_epsilon),
        "diffuser_input_mode": cfg.algo.diffuser_input_mode,
        "loss_type": cfg.algo.loss_type,
        "conditioning_drop_graph_p": float(cfg.algo.conditioning_drop_graph_p),
        "horizon": int(cfg.algo.horizon),
        "history_num_frames": int(cfg.algo.history_num_frames),
        "seed": int(cfg.seed),
        "trajdata_data_dirs": dict(cfg.train.trajdata_data_dirs),
        "output_root": root_dir,
        "run_version": version_key,
    }
    if is_global_zero_env():
        with open(
            os.path.join(root_dir, version_key, "run_contract.json"), "w"
        ) as f:
            json.dump(run_contract, f, indent=2, sort_keys=True)
        print("RUN_CONTRACT {}".format(json.dumps(run_contract, sort_keys=True)))

    # Dataset
    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )
    datamodule.setup()

    # Environment for close-loop evaluation
    if cfg.train.rollout.enabled:
        # Run rollout at regular intervals
        rollout_callback = RolloutCallback(
            exp_config=cfg,
            every_n_steps=cfg.train.rollout.every_n_steps,
            warm_start_n_steps=cfg.train.rollout.warm_start_n_steps,
            verbose=True,
            save_video=cfg.train.rollout.save_video,
            video_dir=video_dir
        )
        train_callbacks.append(rollout_callback)

    # Model
    model = algo_factory(
        config=cfg,
        modality_shapes=datamodule.modality_shapes
    )

    # Checkpointing
    if cfg.train.validation.enabled and cfg.train.save.save_best_validation:
        assert (
            cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps
        ), "checkpointing frequency ("+str(cfg.train.save.every_n_steps)+") needs to be greater than validation frequency ("+str(cfg.train.validation.every_n_steps)+")"
        for metric_name, metric_key in model.checkpoint_monitor_keys.items():
            print(
                "Monitoring metrics {} under alias {}".format(metric_key, metric_name)
            )
            ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="iter{step}_ep{epoch}_%s{%s:.2f}" % (metric_name, metric_key),
                # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,  # save the best k models
                monitor=metric_key,
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                verbose=True,
            )
            train_callbacks.append(ckpt_valid_callback)

    if cfg.train.rollout.enabled and cfg.train.save.save_best_rollout:
        assert (
            cfg.train.save.every_n_steps > cfg.train.rollout.every_n_steps
        ), "checkpointing frequency needs to be greater than rollout frequency"
        ckpt_rollout_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="iter{step}_ep{epoch}_simADE{rollout/metrics_ego_ADE:.2f}",
            # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
            auto_insert_metric_name=False,
            save_top_k=cfg.train.save.best_k,  # save the best k models
            monitor="rollout/metrics_ego_ADE",
            mode="min",
            every_n_train_steps=cfg.train.save.every_n_steps,
            verbose=True,
        )
        train_callbacks.append(ckpt_rollout_callback)

    # a ckpt monitor to save at fixed interval
    ckpt_fixed_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="iter{step}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        monitor=None,
        every_n_train_steps=10000,
        save_last=True,
        verbose=True,
    )
    train_callbacks.append(ckpt_fixed_callback)

    # # NOTE: this is annoying to be able to load in EMA weights at test time,
    # #           implemented in the lightning module class instead.
    # if 'use_ema' in cfg.algo.__dict__ and cfg.algo.use_ema:
    #     from tbsim.utils.ema import EMA
    #     ema_callback = EMA(decay=cfg.algo.ema_decay, every_n_train_steps=cfg.algo.ema_step)
    #     train_callbacks.append(ema_callback)

    # Logging
    assert not (cfg.train.logging.log_tb and cfg.train.logging.log_wandb)
    logger = None
    if debug:
        print("Debugging mode enabled.")
    if cfg.train.logging.log_tb:
        logger = TensorBoardLogger(
            save_dir=root_dir, version=version_key, name=None, sub_dir="logs/"
        )
        print("Tensorboard event will be saved at {}".format(logger.log_dir))
    elif cfg.train.logging.log_wandb:
        assert (
            "WANDB_APIKEY" in os.environ
        ), "Set api key by `export WANDB_APIKEY=<your-apikey>`"
        apikey = os.environ["WANDB_APIKEY"]
        wandb.login(key=apikey)
        logger = WandbLogger(
            name=cfg.name, project=cfg.train.logging.wandb_project_name
        )
        # record the entire config on wandb
        logger.experiment.config.update(cfg.to_dict())
        logger.watch(model=model)
    else:
        print("WARNING: not logging training stats")

    # Train
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        # checkpointing
        enable_checkpointing=cfg.train.save.enabled,
        # logging
        logger=logger,
        # flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
        log_every_n_steps=cfg.train.logging.log_every_n_steps,
        # training
        max_steps=cfg.train.training.num_steps,
        max_time={"hours": max_hours} if max_hours is not None else None,
        # validation
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch,
        # all callbacks
        callbacks=train_callbacks,
        # device & distributed training setup
        gpus=cfg.devices.num_gpus,
        strategy=cfg.train.parallel_strategy,
        # setting for overfit debugging
        # limit_val_batches=0,
        # overfit_batches=2
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="(optional) create experiment config from a preregistered name (see configs/registry.py)",
    )
    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="(optional) if provided, override the wandb project name defined in the config",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset root path",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )

    parser.add_argument(
        "--remove_exp_dir",
        action="store_true",
        help="Whether to automatically remove existing experiment directory of the same name (remember to set this to "
        "True to avoid unexpected stall when launching cloud experiments).",
    )

    parser.add_argument(
        "--on_ngc",
        action="store_true",
        help="whether running the script on ngc (this will change some behaviors like avoid writing into dataset)"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Debug mode, suppress wandb logging, etc."
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Override cfg.devices.num_gpus after applying CUDA_VISIBLE_DEVICES.",
    )

    parser.add_argument(
        "--parallel_strategy",
        choices=("dp", "ddp", "ddp_spawn"),
        default=None,
        help="Override the configured PyTorch Lightning parallel strategy.",
    )

    parser.add_argument(
        "--smoke_steps",
        type=int,
        default=None,
        help="Run a bounded training smoke test without changing the default formal run.",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Override the formal training stop step (used to pair M2 with M1).",
    )

    parser.add_argument(
        "--max_hours",
        type=float,
        default=None,
        help="Optional wall-clock training limit in hours.",
    )

    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Resume optimizer, model, EMA, and global step from a Lightning checkpoint.",
    )

    args = parser.parse_args()

    if args.config_name is not None:
        default_config = get_registered_experiment_config(args.config_name)
        rank_zero_print('args.config_name', args.config_name)
        rank_zero_print('default_config', default_config)
    elif args.config_file is not None:
        # Update default config with external json file
        default_config = get_experiment_config_from_file(args.config_file, locked=False)
    else:
        raise Exception(
            "Need either a config name or a json file to create experiment config"
        )

    if args.name is not None:
        default_config.name = args.name

    if args.dataset_path is not None:
        default_config.train.dataset_path = args.dataset_path
        if hasattr(default_config.train, "trajdata_data_dirs"):
            for dataset_key in default_config.train.trajdata_data_dirs:
                if dataset_key.startswith("nusc"):
                    default_config.train.trajdata_data_dirs[dataset_key] = args.dataset_path

    if args.output_dir is not None:
        default_config.root_dir = os.path.abspath(args.output_dir)

    if args.wandb_project_name is not None:
        default_config.train.logging.wandb_project_name = args.wandb_project_name

    if args.on_ngc:
        ngc_job_id = socket.gethostname()
        default_config.name = default_config.name + "_" + ngc_job_id

    default_config.train.on_ngc = args.on_ngc

    if args.num_gpus is not None:
        if args.num_gpus < 0:
            raise ValueError("--num_gpus must be non-negative")
        default_config.devices.num_gpus = args.num_gpus

    if args.parallel_strategy is not None:
        default_config.train.parallel_strategy = args.parallel_strategy

    if args.resume_ckpt is not None:
        args.resume_ckpt = os.path.abspath(args.resume_ckpt)
        if not os.path.isfile(args.resume_ckpt):
            raise FileNotFoundError("Resume checkpoint not found: {}".format(args.resume_ckpt))

    if args.debug:
        # Keep debug runs bounded and avoid expensive closed-loop rollouts.
        default_config.train.validation.every_n_steps = 5
        default_config.train.save.every_n_steps = 10
        default_config.train.rollout.enabled = False

    if args.smoke_steps is not None:
        if args.smoke_steps <= 0:
            raise ValueError("--smoke_steps must be positive")
        default_config.train.training.num_steps = args.smoke_steps
        default_config.train.validation.every_n_steps = min(100, args.smoke_steps)
        default_config.train.validation.num_steps_per_epoch = 2
        default_config.train.save.every_n_steps = args.smoke_steps
        default_config.train.rollout.enabled = False

    if args.num_steps is not None:
        if args.num_steps <= 0:
            raise ValueError("--num_steps must be positive")
        if args.smoke_steps is not None:
            raise ValueError("Use only one of --num_steps and --smoke_steps")
        default_config.train.training.num_steps = args.num_steps

    # make rollout evaluation config consistent with the rest of the config
    if default_config.train.rollout.enabled:
        default_config.eval.env = default_config.env.name
        assert default_config.algo.eval_class is not None, \
            "Please set an eval_class for {}".format(default_config.algo.name)
        default_config.eval.eval_class = default_config.algo.eval_class
        default_config.eval.dataset_path = default_config.train.dataset_path
        for k in default_config.eval[default_config.eval.env]:  # copy env-specific config to the global-level
            default_config.eval[k] = default_config.eval[default_config.eval.env][k]
        default_config.eval.pop("nusc")
        default_config.eval.pop("l5kit")
        # default_config.eval.pop("trajdata")

    default_config.lock()  # Make config read-only
    main(
        default_config,
        auto_remove_exp_dir=args.remove_exp_dir,
        debug=args.debug,
        max_hours=args.max_hours,
        resume_ckpt=args.resume_ckpt,
    )
