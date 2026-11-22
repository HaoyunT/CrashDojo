#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

"""
This file contains several utility functions used to define the main training loop. It
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import socket
import shutil
import time


def infinite_iter(data_loader):
    """
    Get an infinite generator
    Args:
        data_loader (DataLoader): data loader to iterate through

    """
    c_iter = iter(data_loader)
    while True:
        try:
            data = next(c_iter)
        except StopIteration:
            c_iter = iter(data_loader)
            data = next(c_iter)
        yield data


def get_exp_dir(exp_name, output_dir, save_checkpoints=True, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        exp_name (str): name of the experiment
        output_dir (str): output directory of the experiment
        save_checkpoints (bool): if save checkpoints
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.

    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = output_dir
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.abspath(base_output_dir)
    base_output_dir = os.path.join(base_output_dir, exp_name)
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    sync_file = os.environ.get("CCDIFF_RUN_SYNC_FILE")
    if world_size > 1 and not sync_file:
        raise RuntimeError(
            "CCDIFF_RUN_SYNC_FILE must be set for a multi-process torchrun launch"
        )

    if global_rank == 0:
        if os.path.exists(base_output_dir):
            if not auto_remove_exp_dir:
                ans = input(
                    "WARNING: model directory ({}) already exists! "
                    "\noverwrite? (y/n)\n".format(base_output_dir)
                )
            else:
                ans = "y"
            if ans == "y":
                print("REMOVING")
                shutil.rmtree(base_output_dir)
        os.makedirs(base_output_dir, exist_ok=True)

        # Version selection and directory creation are rank-zero-only. This
        # avoids races when torchrun starts all ranks at the same time.
        existing_runs = [
            a
            for a in os.listdir(base_output_dir)
            if os.path.isdir(os.path.join(base_output_dir, a))
        ]
        run_counts = [-1]
        for ep in existing_runs:
            m = ep.split("run")
            if len(m) == 2 and m[0] == "" and m[1].isnumeric():
                run_counts.append(int(m[1]))
        version_str = "run{}".format(max(run_counts) + 1)

        if save_checkpoints:
            os.makedirs(
                os.path.join(base_output_dir, version_str, "checkpoints"),
                exist_ok=True,
            )
        os.makedirs(
            os.path.join(base_output_dir, version_str, "logs"), exist_ok=True
        )
        os.makedirs(
            os.path.join(base_output_dir, version_str, "videos"), exist_ok=True
        )

        if world_size > 1:
            sync_parent = os.path.dirname(sync_file)
            if sync_parent:
                os.makedirs(sync_parent, exist_ok=True)
            temporary_sync_file = "{}.{}.tmp".format(sync_file, os.getpid())
            with open(temporary_sync_file, "w") as f:
                f.write(version_str)
            os.replace(temporary_sync_file, sync_file)
    else:
        deadline = time.time() + 120.0
        while not os.path.exists(sync_file):
            if time.time() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for rank zero to create {}".format(sync_file)
                )
            time.sleep(0.1)
        with open(sync_file, "r") as f:
            version_str = f.read().strip()
        if not version_str.startswith("run"):
            raise RuntimeError("Invalid run version in {}".format(sync_file))

    # only make model directory if model saving is enabled
    ckpt_dir = None
    if save_checkpoints:
        ckpt_dir = os.path.join(base_output_dir, version_str, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, version_str, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # video directory
    video_dir = os.path.join(base_output_dir, version_str, "videos")
    os.makedirs(video_dir, exist_ok=True)
    return base_output_dir, log_dir, ckpt_dir, video_dir, version_str
