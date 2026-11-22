import unittest

import torch

from ccdiff.algos.algos import CCDiffTrafficModel
from ccdiff.utils.guidance_loss import PerturbationGuidance
from tbsim.utils.guidance_loss import DiffuserGuidance, GuidanceConfig


class _QuadraticEndpointGuidance:
    def compute_guidance_loss(self, trajectory, data_batch):
        loss = (trajectory[..., 0] - 10.0).square().mean()
        return loss, {"quadratic": loss.detach()}


class _PartitionedB4Guidance:
    def compute_b4_energy_partition(self, trajectory, data_batch):
        del data_batch
        attack = (trajectory[..., 0] - 1.0).square().mean()
        feasibility = torch.relu(trajectory.abs() - 2.0).square().mean()
        return attack, feasibility, {
            "attack": attack.detach(),
            "feasibility": feasibility.detach(),
        }


class _ConstantPartitionLoss:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, trajectory, data_batch, agt_mask=None):
        del data_batch
        selected = trajectory if agt_mask is None else trajectory[agt_mask]
        return selected[..., 0, 0] * 0.0 + self.value


def _action_to_state(action, data_batch, params, bsize, num_samp):
    del data_batch, params, bsize, num_samp
    zeros = torch.zeros(
        (*action.shape[:-1], 4), device=action.device, dtype=action.dtype
    )
    return torch.cat((action, zeros), dim=-1)


class FlowVelocityGuidanceTest(unittest.TestCase):
    def test_diffuser_guidance_explicit_b4_partition(self):
        guidance = DiffuserGuidance.__new__(DiffuserGuidance)
        guidance.num_scenes = 1
        attack = GuidanceConfig(
            "target_speed", 2.0, {}, None, partition="attack"
        )
        attack.func = _ConstantPartitionLoss(3.0)
        feasible = GuidanceConfig(
            "map_collision", 4.0, {}, None, partition="feasibility"
        )
        feasible.func = _ConstantPartitionLoss(5.0)
        guidance.guide_configs = [[attack, feasible]]
        trajectory = torch.zeros((2, 1, 4, 6))
        data_batch = {"scene_index": torch.zeros(2, dtype=torch.long)}
        attack_value, feasible_value, audit = (
            guidance.compute_b4_energy_partition(trajectory, data_batch)
        )
        self.assertEqual(float(attack_value), 6.0)
        self.assertEqual(float(feasible_value), 20.0)
        self.assertEqual(
            audit["partition_counts"], {"attack": 1, "feasibility": 1}
        )

    def test_diffuser_guidance_b4_partition_fails_closed(self):
        guidance = DiffuserGuidance.__new__(DiffuserGuidance)
        guidance.num_scenes = 1
        legacy = GuidanceConfig("target_speed", 1.0, {}, None)
        legacy.func = _ConstantPartitionLoss(1.0)
        guidance.guide_configs = [[legacy]]
        trajectory = torch.zeros((1, 1, 4, 6))
        data_batch = {"scene_index": torch.zeros(1, dtype=torch.long)}
        with self.assertRaisesRegex(ValueError, "explicit attack/feasibility"):
            guidance.compute_b4_energy_partition(trajectory, data_batch)

    def test_b4_monotone_complete_joint_runtime_seam(self):
        guidance = PerturbationGuidance(
            transform=_action_to_state,
            transform_params={},
        )
        guidance.current_guidance = _PartitionedB4Guidance()
        native = torch.zeros((2, 3, 5, 2), dtype=torch.float32)
        action_flow = torch.zeros_like(native)
        guided, losses, audit = guidance.perturb_flow_velocity(
            native,
            action_flow,
            remaining=0.5,
            data_batch={},
            opt_params={"optimizer": "sgd", "lr": 1.0, "grad_steps": 1},
            num_samp=2,
            monotone_config={
                "initial_step_rms": 0.1,
                "trust_max_rms": 0.1,
                "trust_max_linf": 0.2,
            },
        )
        self.assertTrue(audit["accepted"])
        self.assertLess(
            audit["final_attack_energy"], audit["initial_attack_energy"]
        )
        self.assertLessEqual(
            audit["final_feasibility_energy"],
            audit["initial_feasibility_energy"],
        )
        self.assertFalse(audit["direct_trajectory_edit"])
        self.assertFalse(audit["row_replacement_or_mixing"])
        self.assertEqual(audit["optimized_variable"], "complete_joint_flow_velocity")
        self.assertIn("attack", losses)
        self.assertEqual(guided.shape, native.shape)

    def test_b4_monotone_refuses_unpartitioned_guidance(self):
        guidance = PerturbationGuidance(
            transform=_action_to_state,
            transform_params={},
        )
        guidance.current_guidance = _QuadraticEndpointGuidance()
        native = torch.zeros((1, 2, 5, 2), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "explicit attack/feasibility"):
            guidance.perturb_flow_velocity(
                native,
                torch.zeros_like(native),
                remaining=0.5,
                data_batch={},
                opt_params={"optimizer": "sgd", "lr": 1.0, "grad_steps": 1},
                num_samp=1,
                monotone_config={
                    "initial_step_rms": 0.1,
                    "trust_max_rms": 0.1,
                    "trust_max_linf": 0.2,
                },
            )

    def test_algo_wrapper_delegates_to_policy_and_ema(self):
        class _Inner:
            def __init__(self):
                self.configs = []

            def set_flow_velocity_guidance(self, config=None):
                self.configs.append(config)

        policy = _Inner()
        ema_policy = _Inner()
        outer = type("_Outer", (), {})()
        outer.nets = {"policy": policy}
        outer.use_ema = True
        outer.ema_policy = ema_policy

        config = {"trust_max_rms": 0.05, "trust_max_linf": 0.20}
        CCDiffTrafficModel.set_flow_velocity_guidance(outer, config)
        CCDiffTrafficModel.set_flow_velocity_guidance(outer, None)

        self.assertEqual(policy.configs, [config, None])
        self.assertEqual(ema_policy.configs, [config, None])

    def test_updates_velocity_only_and_obeys_trust_region(self):
        guidance = PerturbationGuidance(
            transform=_action_to_state,
            transform_params={},
        )
        guidance.current_guidance = _QuadraticEndpointGuidance()
        native_velocity = torch.zeros((1, 2, 5, 2))
        action_flow = torch.zeros_like(native_velocity)
        guided, losses, audit = guidance.perturb_flow_velocity(
            native_velocity,
            action_flow,
            remaining=0.5,
            data_batch={},
            opt_params={
                "optimizer": "sgd",
                "lr": 10.0,
                "grad_steps": 2,
            },
            num_samp=1,
            trust_max_rms=0.05,
            trust_max_linf=0.08,
        )
        delta = guided - native_velocity
        rms = torch.sqrt(delta.square().mean())
        self.assertGreater(float(delta.abs().max()), 0.0)
        self.assertLessEqual(float(rms), 0.050001)
        self.assertLessEqual(float(delta.abs().max()), 0.080001)
        self.assertTrue(torch.equal(action_flow, torch.zeros_like(action_flow)))
        self.assertEqual(audit["optimized_variable"], "flow_velocity")
        self.assertFalse(audit["direct_trajectory_edit"])
        self.assertIn("quadratic", losses)

    def test_multi_agent_sample_batch_uses_flattened_agent_batch(self):
        transform_calls = []

        def _spy_transform(action, data_batch, params, bsize, num_samp):
            del data_batch, params
            transform_calls.append(
                {
                    "shape": tuple(action.shape),
                    "bsize": bsize,
                    "num_samp": num_samp,
                }
            )
            zeros = torch.zeros(
                (*action.shape[:-1], 4),
                device=action.device,
                dtype=action.dtype,
            )
            return torch.cat((action, zeros), dim=-1)

        guidance = PerturbationGuidance(
            transform=_spy_transform,
            transform_params={},
        )
        guidance.current_guidance = _QuadraticEndpointGuidance()
        native_velocity = torch.zeros((20, 15, 5, 2))
        action_flow = torch.zeros_like(native_velocity)

        guided, _, audit = guidance.perturb_flow_velocity(
            native_velocity,
            action_flow,
            remaining=0.5,
            data_batch={},
            opt_params={
                "optimizer": "sgd",
                "lr": 1.0,
                "grad_steps": 1,
            },
            num_samp=20,
            trust_max_rms=0.05,
            trust_max_linf=0.08,
        )

        self.assertEqual(tuple(guided.shape), (20, 15, 5, 2))
        self.assertEqual(
            transform_calls,
            [{"shape": (300, 5, 2), "bsize": 15, "num_samp": 20}],
        )
        self.assertLessEqual(audit["max_velocity_delta_rms"], 0.050001)
        self.assertLessEqual(audit["max_velocity_delta_linf"], 0.080001)


if __name__ == "__main__":
    unittest.main()
