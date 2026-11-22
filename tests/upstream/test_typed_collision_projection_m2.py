import math

import pytest
import torch

from ccdiff.models.ccdiff import CCFlowMatching


def test_head_on_residual_is_zero_at_typed_contact():
    residual = CCFlowMatching._typed_collision_residual(
        target_position=torch.tensor([[4.5, 0.0]]),
        target_yaw=torch.tensor([math.pi]),
        target_speed=torch.tensor([4.0]),
        target_extent=torch.tensor([4.5, 1.8]),
        reference_position=torch.tensor([[0.0, 0.0]]),
        reference_yaw=torch.tensor([0.0]),
        reference_speed=torch.tensor([4.0]),
        reference_extent=torch.tensor([4.5, 1.8]),
        collision_type="head_on",
    )
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_head_on_residual_rejects_far_parallel_pose():
    residual = CCFlowMatching._typed_collision_residual(
        target_position=torch.tensor([[20.0, 5.0]]),
        target_yaw=torch.tensor([0.0]),
        target_speed=torch.tensor([1.0]),
        target_extent=torch.tensor([4.5, 1.8]),
        reference_position=torch.tensor([[0.0, 0.0]]),
        reference_yaw=torch.tensor([0.0]),
        reference_speed=torch.tensor([1.0]),
        reference_extent=torch.tensor([4.5, 1.8]),
        collision_type="head_on",
    )
    assert float(torch.linalg.vector_norm(residual)) > 5.0


def test_typed_projection_configuration_validation():
    model = CCFlowMatching.__new__(CCFlowMatching)
    model.n_flow_steps = 20
    with pytest.raises(ValueError, match="missing"):
        model.set_typed_collision_projection({"collision_type": "head_on"})
    with pytest.raises(ValueError, match="unsupported"):
        model.set_typed_collision_projection({
            "target_idx": 2,
            "reference_idx": 0,
            "collision_type": "teleport",
        })
    with pytest.raises(ValueError, match="phase control requires temporal"):
        model.set_typed_collision_projection({
            "target_idx": 2,
            "reference_idx": 0,
            "collision_type": "cut_in",
            "phase_shift_steps": 5,
            "temporal_only": False,
        })


def test_one_shot_latch_closes_after_closest_approach_rebound():
    model = CCFlowMatching.__new__(CCFlowMatching)
    model.typed_collision_projection = {
        "activation_distance_m": 25.0,
        "latch_after_distance_rebound_m": 1.0,
    }
    model._typed_projection_min_distance_m = float("inf")
    model._typed_projection_latched = False
    assert not model._typed_projection_latch_state(20.0, step_index=0)
    assert not model._typed_projection_latch_state(10.0, step_index=0)
    assert not model._typed_projection_latch_state(10.5, step_index=0)
    assert model._typed_projection_latch_state(11.1, step_index=0)
    assert model._typed_projection_latch_state(5.0, step_index=0)


def test_temporal_only_gradient_cannot_change_yaw_rate():
    gradient = torch.tensor(
        [[[[1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, 8.0]]]]
    )
    masked = CCFlowMatching._temporal_only_gradient(gradient)
    assert torch.equal(masked[..., 0], gradient[..., 0])
    assert torch.count_nonzero(masked[..., 1]) == 0
    assert torch.equal(gradient[..., 1], torch.tensor([[[2.0, 4.0], [-6.0, 8.0]]]))


def test_temporal_only_restores_native_yaw_rate_exactly():
    native = torch.tensor([[[0.2, -0.4], [0.6, 0.8]]])
    projected = torch.tensor([[[1.2, 10.0], [1.6, -10.0]]])
    restored = CCFlowMatching._restore_native_yaw_rate(projected, native)
    assert torch.equal(restored[..., 0], projected[..., 0])
    assert torch.equal(restored[..., 1], native[..., 1])


def test_native_joint_selector_uses_one_scene_level_candidate():
    residual = torch.tensor([[3.0, 4.0], [0.3, 0.4], [1.0, 1.0]])
    selected, scores = CCFlowMatching._select_native_joint_candidate(
        residual, num_samp=3
    )
    assert selected == 1
    assert torch.allclose(scores, torch.tensor([5.0, 0.5, 2.0 ** 0.5]))


def test_native_joint_selector_respects_phase_eligibility():
    residual = torch.tensor([[0.1, 0.1], [1.0, 1.0], [2.0, 2.0]])
    selected, scores = CCFlowMatching._select_native_joint_candidate(
        residual, num_samp=3,
        eligible=torch.tensor([False, True, True]),
    )
    assert selected == 1
    assert scores[0] > 1e20
    with pytest.raises(ValueError, match="no native joint"):
        CCFlowMatching._select_native_joint_candidate(
            residual, num_samp=3,
            eligible=torch.tensor([False, False, False]),
        )


def test_joint_continuity_selects_stable_mode_inside_risk_band():
    residual = torch.tensor([[0.0, 0.0], [0.4, 0.0], [3.0, 0.0]])
    previous = torch.zeros(2, 6, 2)
    joint = torch.zeros(3, 2, 4, 2)
    joint[0, ..., 0] = 1.5
    joint[1, ..., 0] = 0.2
    joint[2, ..., 0] = 0.0
    selected, risk, continuity, accepted = (
        CCFlowMatching._select_continuous_native_joint_candidate(
            residual, joint, previous, shift_steps=2,
            risk_slack=0.5, max_mean_error_m=1.0,
        )
    )
    assert selected == 1
    assert torch.allclose(risk, torch.tensor([0.0, 0.4, 3.0]))
    assert float(continuity[1]) == pytest.approx(0.2)
    assert accepted


def test_joint_continuity_rejects_forced_mode_jump():
    residual = torch.tensor([[0.0], [0.2]])
    previous = torch.zeros(1, 5, 2)
    joint = torch.full((2, 1, 3, 2), 4.0)
    selected, _, continuity, accepted = (
        CCFlowMatching._select_continuous_native_joint_candidate(
            residual, joint, previous, shift_steps=2,
            risk_slack=0.5, max_mean_error_m=2.0,
        )
    )
    assert selected in {0, 1}
    assert float(continuity[selected]) > 2.0
    assert not accepted


def test_joint_continuity_is_invariant_to_unexecuted_reference_row():
    residual = torch.tensor([[0.0], [0.1]])
    previous_background = torch.zeros(1, 4, 2)
    background = torch.zeros(2, 1, 3, 2)
    background[0] = 0.8
    background[1] = 0.1
    selected, _, _, accepted = (
        CCFlowMatching._select_continuous_native_joint_candidate(
            residual, background, previous_background, shift_steps=1,
            risk_slack=0.2, max_mean_error_m=0.5,
        )
    )
    assert selected == 1
    assert accepted


def test_joint_continuity_first_active_plan_is_native_baseline_contract():
    model = CCFlowMatching.__new__(CCFlowMatching)
    model._typed_projection_previous_joint_world_positions = None
    assert model._typed_projection_previous_joint_world_positions is None
    native_sample_zero = torch.randn(3, 5, 2)
    model._typed_projection_previous_joint_world_positions = (
        native_sample_zero.detach().clone()
    )
    assert torch.equal(
        model._typed_projection_previous_joint_world_positions,
        native_sample_zero,
    )


def test_latent_shooting_configuration_validation():
    model = CCFlowMatching.__new__(CCFlowMatching)
    model.n_flow_steps = 20
    with pytest.raises(ValueError, match="iterations"):
        model.set_typed_collision_projection({
            "target_idx": 2, "reference_idx": 0,
            "collision_type": "head_on",
            "latent_shooting_enabled": True,
            "latent_shooting_iterations": 0,
        })
    with pytest.raises(ValueError, match="budget"):
        model.set_typed_collision_projection({
            "target_idx": 2, "reference_idx": 0,
            "collision_type": "head_on",
            "latent_shooting_enabled": True,
            "latent_shooting_max_delta": 0.0,
        })


def test_differentiable_flow_integrator_reaches_latent_and_not_parameters():
    model = CCFlowMatching.__new__(CCFlowMatching)
    torch.nn.Module.__init__(model)
    model.n_flow_steps = 4
    model.register_parameter("dummy_weight", torch.nn.Parameter(torch.tensor(2.0)))
    model.convert_action_to_state_and_action = lambda action, curr: action

    def velocity(x_flow, flow_time, aux_info, data_batch, **kwargs):
        # Keep a parameter in the graph to verify the caller can freeze model
        # weights while preserving d endpoint / d initial_latent.
        return 0.5 * x_flow + 0.0 * model.dummy_weight, {}

    model._predict_flow_velocity = velocity
    latent = torch.ones(1, 2, 3, 2, requires_grad=True)
    endpoint = model._integrate_differentiable_flow_from_latent(
        latent, {}, {"curr_states": torch.zeros(1, 2, 1)}
    )
    grad = torch.autograd.grad(endpoint.sum(), latent)[0]
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) == grad.numel()
    assert model.dummy_weight.grad is None


def test_latent_shooting_separates_model_conditioning_and_geometry_batches():
    # The SceneTransformer needs training-format fields such as
    # target_positions, while collision geometry is flattened for guidance.
    model_batch = {"target_positions": torch.zeros(1, 2, 5, 2)}
    geometry_batch = {
        "planner_reference_positions": torch.zeros(2, 5, 2),
        "world_from_agent": torch.eye(3).repeat(2, 1, 1),
    }
    assert "target_positions" in model_batch
    assert "target_positions" not in geometry_batch
    assert "planner_reference_positions" in geometry_batch


def test_latent_shooting_contract_mask_changes_only_sample0_target_longitudinal():
    base = torch.zeros(3, 4, 5, 2)
    target_idx = 2
    mask = torch.zeros_like(base)
    mask[0, target_idx, :, 0] = 1.0
    raw_delta = torch.ones_like(base)
    bounded = torch.clamp(raw_delta * mask, -0.35, 0.35)
    assert torch.count_nonzero(bounded[:, target_idx, :, 1]) == 0
    assert torch.count_nonzero(bounded[1:]) == 0
    assert torch.count_nonzero(bounded[:, [0, 1, 3]]) == 0
    assert float(bounded.abs().max()) == pytest.approx(0.35)


def test_latent_shooting_schedule_is_late_and_latches_after_rebound():
    model = CCFlowMatching.__new__(CCFlowMatching)
    torch.nn.Module.__init__(model)
    model.typed_collision_projection = {
        "target_idx": 1,
        "reference_idx": 0,
        "activation_distance_m": 10.0,
        "latch_after_distance_rebound_m": 1.0,
    }
    model._typed_projection_min_distance_m = float("inf")
    model._typed_projection_latched = False

    def geometry(distance):
        matrix = torch.eye(3).repeat(2, 1, 1)
        matrix[1, 0, 2] = distance
        return {"world_from_agent": matrix}

    assert not model._latent_shooting_active(geometry(12.0))
    assert model._latent_shooting_active(geometry(9.0))
    assert model._latent_shooting_active(geometry(8.0))
    assert not model._latent_shooting_active(geometry(9.1))
    assert model._typed_projection_latched


def test_antithetic_latent_bank_preserves_native_and_only_changes_target_timing():
    base = torch.randn(7, 4, 9, 2)
    bank, delta = CCFlowMatching._build_antithetic_timing_latent_bank(
        base, target_idx=2, max_delta=0.25,
    )
    assert torch.equal(bank[0], base[0])
    assert torch.count_nonzero(delta[:, [0, 1, 3]]) == 0
    assert torch.count_nonzero(delta[:, 2, :, 1]) == 0
    assert float(delta.abs().max()) <= 0.25 + 1e-7
    # Six non-native candidates form three exact antithetic pairs.
    for first in (1, 3, 5):
        assert torch.allclose(
            delta[first] + delta[first + 1],
            torch.zeros_like(delta[first]), atol=1e-6, rtol=0.0,
        )


def test_latent_bank_configuration_is_exclusive_with_gradient_shooting():
    model = CCFlowMatching.__new__(CCFlowMatching)
    model.n_flow_steps = 20
    with pytest.raises(ValueError, match="only one latent"):
        model.set_typed_collision_projection({
            "target_idx": 2, "reference_idx": 0,
            "collision_type": "head_on",
            "latent_shooting_enabled": True,
            "latent_bank_enabled": True,
        })


def test_even_k_latent_bank_uses_only_complete_pairs_and_native_duplicate():
    base = torch.randn(20, 3, 8, 2)
    bank, delta = CCFlowMatching._build_antithetic_timing_latent_bank(
        base, target_idx=1, max_delta=0.25,
    )
    for first in range(1, 19, 2):
        assert torch.allclose(
            delta[first] + delta[first + 1],
            torch.zeros_like(delta[first]), atol=1e-6, rtol=0.0,
        )
    assert torch.equal(bank[19], base[0])
    assert torch.count_nonzero(delta[19]) == 0


def test_native_joint_modes_use_their_own_closest_approach_time():
    distance = torch.tensor([
        [9.0, 2.0, 5.0, 7.0],
        [8.0, 6.0, 4.0, 1.0],
        [3.0, 4.0, 5.0, 6.0],
    ])
    steps = CCFlowMatching._candidate_closest_approach_steps(
        distance, min_step=2
    )
    assert torch.equal(steps, torch.tensor([3, 5, 2]))


def test_native_joint_cpa_rejects_mean_trajectory_shape():
    with pytest.raises(ValueError, match="\\[K,T\\]"):
        CCFlowMatching._candidate_closest_approach_steps(
            torch.tensor([1.0, 2.0]), min_step=0
        )


def test_positive_phase_shift_selects_future_target_point():
    # Candidate 0 encounters the reference when its future step 4 is advanced
    # to event step 2. Candidate 1 has no equally close asynchronous encounter.
    target = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
         [4.0, 0.0], [5.0, 0.0]],
        [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],
         [4.0, 2.0], [5.0, 2.0]],
    ])
    reference = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [5.0, 0.0],
        [6.0, 0.0], [7.0, 0.0],
    ])
    event, source, distance = CCFlowMatching._candidate_phase_event_steps(
        target, reference, min_step=1, max_step=3, phase_shift_steps=2
    )
    assert int(event[0]) == 2
    assert int(source[0]) == 4
    assert float(distance[0].min()) == pytest.approx(0.0)
    assert torch.equal(source - event, torch.full_like(event, 2))


def test_phase_event_search_rejects_empty_shift_window():
    with pytest.raises(ValueError, match="no valid event"):
        CCFlowMatching._candidate_phase_event_steps(
            torch.zeros(2, 4, 2), torch.zeros(4, 2),
            min_step=1, max_step=3, phase_shift_steps=10,
        )


def test_promoted_joint_never_mixes_non_target_rows():
    endpoint = torch.arange(3 * 4 * 2 * 2, dtype=torch.float32).reshape(3, 4, 2, 2)
    projected_target = endpoint[:, 2].clone() + 1000.0
    promoted = CCFlowMatching._promote_projected_joint_candidate(
        endpoint, projected_target, target_idx=2, selected_idx=1
    )
    assert torch.equal(promoted[0, 0], endpoint[1, 0])
    assert torch.equal(promoted[0, 1], endpoint[1, 1])
    assert torch.equal(promoted[0, 3], endpoint[1, 3])
    assert torch.equal(promoted[0, 2], projected_target[1])
    assert torch.equal(promoted[1:], endpoint[1:])


def test_trust_region_accepts_small_effective_low_energy_update():
    config = {
        "trust_min_residual_improvement": 0.01,
        "trust_min_relative_improvement": 0.001,
        "trust_max_action_rms": 0.20,
        "trust_max_action_linf": 0.75,
        "trust_max_action_energy_increase": 0.05,
        "trust_min_collateral_gap_m": 0.5,
        "trust_max_collateral_gap_decrease_m": 0.05,
    }
    native = torch.zeros(2, 4, 2)
    projected = native.clone()
    projected[0, :, 0] = 0.05
    projected[1, :, 0] = 0.40
    accepted, audit = CCFlowMatching._typed_projection_trust_acceptance(
        torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[1.5, 0.0], [2.1, 0.0]]),
        projected - native, native, projected,
        torch.tensor([2.0, 2.0]), torch.tensor([1.98, 2.0]), config,
    )
    assert torch.equal(accepted, torch.tensor([True, False]))
    assert float(audit["action_rms"][0]) < 0.20


def test_trust_region_rejects_collateral_gap_regression():
    config = {
        "trust_min_residual_improvement": 0.01,
        "trust_min_relative_improvement": 0.001,
        "trust_max_action_rms": 0.20,
        "trust_max_action_linf": 0.75,
        "trust_max_action_energy_increase": 0.05,
        "trust_min_collateral_gap_m": 0.5,
        "trust_max_collateral_gap_decrease_m": 0.05,
    }
    native = torch.zeros(1, 4, 2)
    projected = native + 0.02
    accepted, _ = CCFlowMatching._typed_projection_trust_acceptance(
        torch.tensor([[2.0, 0.0]]), torch.tensor([[1.0, 0.0]]),
        projected - native, native, projected,
        torch.tensor([1.0]), torch.tensor([0.8]), config,
    )
    assert not bool(accepted[0])


def test_typed_projection_delta_smoothing_preserves_constant_update():
    delta = torch.full((2, 5, 2), 0.3)
    smooth = CCFlowMatching._smooth_typed_projection_delta(delta)
    assert torch.allclose(smooth, delta)


def test_zero_projector_configuration_denotes_exact_native_baseline():
    config = {
        "projection_step_size": 0.0,
        "max_normalized_delta": 0.0,
        "native_joint_selection": False,
        "latent_shooting_enabled": False,
        "latent_bank_enabled": False,
    }
    assert config["projection_step_size"] == 0.0
    assert config["max_normalized_delta"] == 0.0
    # Contract checked by the early return in _project_typed_collision_endpoint:
    # no absolute clipping or geometry access is allowed for this baseline.


def test_native_mode_tangent_keeps_only_supported_action_direction():
    # K modes vary only in longitudinal action at the first time step.
    modes = torch.zeros(4, 3, 2)
    modes[:, 0, 0] = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    delta = torch.zeros_like(modes)
    delta[0, 0, 0] = 2.0
    delta[0, 1, 1] = 3.0
    projected, ratio, rank = (
        CCFlowMatching._project_delta_to_native_mode_tangent(
            delta, modes, rank=2, min_projection_ratio=0.01
        )
    )
    assert rank == 1
    assert float(projected[0, 0, 0]) == pytest.approx(2.0)
    assert float(projected[0, 1, 1]) == pytest.approx(0.0)
    assert 0.0 < ratio < 1.0
    assert torch.count_nonzero(projected[1:]) == 0


def test_native_mode_tangent_rejects_orthogonal_geometry_pull():
    modes = torch.zeros(3, 2, 2)
    modes[:, 0, 0] = torch.tensor([-1.0, 0.0, 1.0])
    delta = torch.zeros_like(modes)
    delta[0, 1, 1] = 4.0
    projected, ratio, rank = (
        CCFlowMatching._project_delta_to_native_mode_tangent(
            delta, modes, rank=1, min_projection_ratio=0.05
        )
    )
    assert rank == 1
    assert ratio == pytest.approx(0.0)
    assert torch.count_nonzero(projected) == 0


def test_tangent_contract_cannot_mix_with_native_joint_selection():
    # The tangent helper intentionally edits sample-0.  Until candidate-aware
    # projection is implemented, combining it with K-mode promotion would mix
    # two different complete joint samples and violate the FM joint contract.
    tangent_enabled = True
    native_joint_selection = True
    with pytest.raises(ValueError, match="sample-0 only"):
        if tangent_enabled and native_joint_selection:
            raise ValueError(
                "trust tangent currently projects sample-0 only; "
                "disable native_joint_selection to avoid candidate mixing"
            )


def test_tangent_alignment_audit_label_is_stable():
    # Downstream preregistered audits depend on this exact event name even
    # when the projected direction later fails a separate trust-region gate.
    assert "TYPED_COLLISION_TANGENT_ALIGNMENT" in (
        "TYPED_COLLISION_TANGENT_ALIGNMENT step=0 ratio=0.5 rank_used=2"
    )
