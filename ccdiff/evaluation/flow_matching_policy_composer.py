"""Checkpoint composer that cannot silently reconstruct M2 as DDPM."""

import hashlib
import inspect
import json
import os
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import torch

from ccdiff.algos.algos import CCDiffTrafficModel
from ccdiff.models.ccdiff import CCFlowMatching as CCFlowMatchingModel
from tbsim.evaluation.policy_composers import PolicyComposer
from tbsim.policies.wrappers import (
    AgentCentricToSceneCentricWrapper,
    NaiveAgentCentricToSceneCentricWrapper,
    PolicyWrapper,
)
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.utils.experiment_utils import get_checkpoint


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _synchronize_if_cuda(model):
    parameter = next(model.parameters())
    if parameter.is_cuda:
        torch.cuda.synchronize(parameter.device)


def _instrument_sampler(model, label, state, attestation_path):
    if type(model) is not CCFlowMatchingModel:
        raise TypeError(
            "Expected exact CCFlowMatching policy, got {}".format(
                type(model).__qualname__
            )
        )
    if type(model).p_sample_loop is not CCFlowMatchingModel.p_sample_loop:
        raise TypeError("Flow-matching p_sample_loop has been overridden")
    if model.n_flow_steps != 20:
        raise ValueError(
            "M2 evaluation is frozen at 20 Euler steps, got {}".format(
                model.n_flow_steps
            )
        )

    original = model.p_sample_loop
    signature = inspect.signature(original)

    def audited_p_sample_loop(self, *args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        cfg_weight = float(bound.arguments.get("class_free_guide_w", 0.0))
        per_step_evaluations = 2 if cfg_weight != 0.0 else 1

        _synchronize_if_cuda(self)
        start_time = time.perf_counter()
        result = original(*args, **kwargs)
        _synchronize_if_cuda(self)
        elapsed = time.perf_counter() - start_time

        state["sample_calls"] += 1
        state["total_model_evaluations"] += (
            self.n_flow_steps * per_step_evaluations
        )
        state["sampler_only_seconds_total"] += elapsed
        correlated_prior = self.marginal_preserving_correlated_flow_prior
        pcn_beta = self.pcn_joint_flow_prior_beta
        persistent_prior = self.persistent_copula_flow_prior_attestation()
        if persistent_prior is not None:
            initial_prior = "persistent_stationary_gaussian_copula"
        elif correlated_prior is not None:
            initial_prior = "marginal_preserving_correlated_gaussian"
        elif pcn_beta is not None:
            initial_prior = "pcn_standard_gaussian_proposal_bank"
        else:
            initial_prior = "standard_gaussian"
        state["last_call"] = {
            "apply_guidance": bool(
                bound.arguments.get("apply_guidance", True)
            ),
            "class_free_guide_w": cfg_weight,
            "elapsed_seconds": elapsed,
            "label": label,
            "model_evaluations": self.n_flow_steps * per_step_evaluations,
            "num_samples": int(bound.arguments.get("num_samp", 1)),
            "initial_prior": initial_prior,
            "correlated_flow_prior": correlated_prior,
            "pcn_joint_flow_prior_beta": pcn_beta,
            "persistent_copula_flow_prior": persistent_prior,
        }
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(attestation_path, state)
        return result

    model.p_sample_loop = types.MethodType(audited_p_sample_loop, model)


class CCFlowMatchingPolicyComposer(PolicyComposer):
    """Load M2 with an exact transport assertion and runtime sampler audit."""

    def get_policy(self, policy=None):
        if policy is not None:
            raise ValueError(
                "The audited M2 evaluator requires an on-disk checkpoint"
            )

        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            ckpt_dir=self.eval_config.ckpt.policy.ckpt_dir,
            ckpt_root_dir=self.ckpt_root_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        if policy_cfg.algo.name != "ccflow_matching":
            raise ValueError(
                "M2 evaluator requires algo.name=ccflow_matching, got {}".format(
                    policy_cfg.algo.name
                )
            )
        if int(policy_cfg.algo.n_flow_steps) != 20:
            raise ValueError("M2 evaluator requires n_flow_steps=20")
        if int(policy_cfg.algo.n_diffusion_steps) != 100:
            raise ValueError(
                "The frozen time-embedding scale requires n_diffusion_steps=100"
            )

        policy = CCDiffTrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.get_modality_shapes(policy_cfg),
            registered_name=policy_cfg["registered_name"],
            policy_model_cls=CCFlowMatchingModel,
            policy_model_kwargs={
                "n_flow_steps": int(policy_cfg.algo.n_flow_steps)
            },
        ).to(self.device).eval()

        if type(policy.nets["policy"]) is not CCFlowMatchingModel:
            raise TypeError("Non-EMA policy was not reconstructed as M2")
        if type(policy.ema_policy) is not CCFlowMatchingModel:
            raise TypeError("EMA policy was not reconstructed as M2")

        selected_use_ema = bool(self.eval_config.policy.use_ema)
        policy.use_ema = selected_use_ema

        attestation_path = Path(self.eval_config.results_dir) / (
            "fm_evaluator_attestation.json"
        )
        state = {
            "attestation_version": 1,
            "checkpoint_path": str(Path(policy_ckpt_path).resolve()),
            "checkpoint_sha256": _sha256(policy_ckpt_path),
            "configured_n_flow_steps": 20,
            "configured_time_embedding_max": 99,
            "deterministic_euler_after_initial_noise": True,
            "ema_model_type": type(policy.ema_policy).__qualname__,
            "evaluator": type(self).__qualname__,
            "non_ema_model_type": type(policy.nets["policy"]).__qualname__,
            "per_step_ddpm_noise": False,
            "selected_weight_mode": "ema" if selected_use_ema else "raw",
            "sample_calls": 0,
            "sampler_method": (
                CCFlowMatchingModel.p_sample_loop.__module__
                + "."
                + CCFlowMatchingModel.p_sample_loop.__qualname__
            ),
            "sampler_only_seconds_total": 0.0,
            "total_model_evaluations": 0,
            "transport": "flow_matching",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        policy._fm_evaluator_audit_state = state
        policy._fm_evaluator_attestation_path = str(attestation_path)
        _instrument_sampler(
            policy.nets["policy"], "non_ema", state, attestation_path
        )
        _instrument_sampler(
            policy.ema_policy, "ema", state, attestation_path
        )
        _write_json_atomic(attestation_path, state)

        policy = PolicyWrapper.wrap_controller(
            policy,
            num_action_samples=self.eval_config.policy.num_action_samples,
            class_free_guide_w=self.eval_config.policy.class_free_guide_w,
            guide_as_filter_only=self.eval_config.policy.guide_as_filter_only,
            guide_with_gt=self.eval_config.policy.guide_with_gt,
            guide_clean=self.eval_config.policy.guide_clean,
        )
        policy_cfg = policy_cfg.clone()
        if policy_cfg.algo.coordinate == "agent_centric":
            policy = NaiveAgentCentricToSceneCentricWrapper(policy)
        elif policy_cfg.algo.coordinate == "scene_centric":
            policy = AgentCentricToSceneCentricWrapper(policy)
        else:
            raise NotImplementedError(
                "Unsupported coordinate system: {}".format(
                    policy_cfg.algo.coordinate
                )
            )
        return policy, policy_cfg
