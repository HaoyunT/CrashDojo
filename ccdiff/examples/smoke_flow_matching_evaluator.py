#!/usr/bin/env python3
"""CPU-only proof that the M2 evaluator reconstructs the FM transport."""

import argparse
import json
from pathlib import Path

import torch

from ccdiff.evaluation.flow_matching_policy_composer import (
    CCFlowMatchingPolicyComposer,
)
from ccdiff.models.ccdiff import CCFlowMatching
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-key", default="last.ckpt")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        raise RuntimeError(
            "Evaluator type smoke must run with CUDA_VISIBLE_DEVICES empty"
        )

    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env("nusc_trainval")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SceneEditingConfig(
        registered_name="trajdata_nusc_ccflow_matching"
    )
    cfg.eval_class = "CCFlowMatching"
    cfg.results_dir = str(output_dir)
    cfg.ckpt.policy.ckpt_dir = args.checkpoint_dir
    cfg.ckpt.policy.ckpt_key = args.checkpoint_key

    composer = CCFlowMatchingPolicyComposer(cfg, torch.device("cpu"))
    wrapped, experiment = composer.get_policy()
    model = wrapped.model
    if type(model.nets["policy"]) is not CCFlowMatching:
        raise TypeError("non-EMA policy was reconstructed with the wrong type")
    if type(model.ema_policy) is not CCFlowMatching:
        raise TypeError("EMA policy was reconstructed with the wrong type")

    attestation_path = output_dir / "fm_evaluator_attestation.json"
    with open(attestation_path, "r", encoding="utf-8") as stream:
        attestation = json.load(stream)
    if attestation["transport"] != "flow_matching":
        raise AssertionError("transport attestation is not flow_matching")
    if attestation["sample_calls"] != 0:
        raise AssertionError("type smoke unexpectedly sampled a trajectory")

    report = {
        "algo": experiment.algo.name,
        "checkpoint_sha256": attestation["checkpoint_sha256"],
        "ema_model_type": type(model.ema_policy).__name__,
        "non_ema_model_type": type(model.nets["policy"]).__name__,
        "sample_calls": attestation["sample_calls"],
        "status": "pass",
    }
    report_path = output_dir / "type_smoke_report.json"
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("FM_EVALUATOR_TYPE_SMOKE {}".format(json.dumps(report, sort_keys=True)))


if __name__ == "__main__":
    main()
