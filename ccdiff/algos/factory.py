#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

"""Factory methods for creating models"""
from pytorch_lightning import LightningDataModule
from tbsim.configs.base import ExperimentConfig

from ccdiff.algos.algos import (
    CCDiffTrafficModel
)
from ccdiff.models.ccdiff import CCFlowMatching


def algo_factory(config: ExperimentConfig, modality_shapes: dict):
    """
    A factory for creating training algos

    Args:
        config (ExperimentConfig): an ExperimentConfig object,
        modality_shapes (dict): a dictionary that maps observation modality names to shapes

    Returns:
        algo: pl.LightningModule
    """
    algo_config = config.algo
    algo_name = algo_config.name
    if algo_name == "ccdiff":
        algo = CCDiffTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes, registered_name=config.registered_name)
    elif algo_name == "ccflow_matching":
        algo = CCDiffTrafficModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            registered_name=config.registered_name,
            policy_model_cls=CCFlowMatching,
            policy_model_kwargs={"n_flow_steps": algo_config.n_flow_steps},
        )
    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo
