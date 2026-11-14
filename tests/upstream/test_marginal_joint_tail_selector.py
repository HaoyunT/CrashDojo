import torch

from tbsim.policies.wrappers import (
    select_marginal_joint_tail_action,
    select_multi_agent_marginal_joint_tail_action,
    select_sparse_marginal_joint_tail_action,
)


def identity_transforms(num_agents):
    return torch.eye(3).repeat(1, num_agents, 1, 1)


def strict_speed_naturalness_profile():
    return {
        "speed_bin_upper_bounds_mps": [2.0],
        "q99": {
            "speed_mps": [1.5, 100.0],
            "abs_acc_mps2": [100.0, 100.0],
            "abs_jerk_mps3": [100.0, 100.0],
            "abs_yaw_rate_rps": [100.0, 100.0],
            "abs_lateral_acc_mps2": [100.0, 100.0],
        },
    }


def test_selects_untouched_lower_clearance_sample():
    samples = torch.zeros(1, 2, 3, 3, 2)
    samples[0, 0, 1, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    samples[0, 0, 1, :, 1] = 5.0
    samples[0, 1, 1, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    samples[0, 1, 1, :, 1] = 1.2
    samples[:, :, 2, :, 1] = 10.0
    sample_yaws = torch.zeros(1, 2, 3, 3, 1)
    baseline = samples[:, 0].clone()
    baseline_yaws = sample_yaws[:, 0].clone()
    ego = torch.zeros(1, 3, 2)
    ego[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    extents = torch.ones(1, 3, 3)

    positions, yaws, audit = select_marginal_joint_tail_action(
        samples,
        sample_yaws,
        baseline,
        baseline_yaws,
        ego,
        torch.zeros(1, 3, 1),
        identity_transforms(3),
        extents,
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.0,
        min_risk_improvement_m=0.05,
    )

    assert audit["selected_sample_index"] == 1
    assert audit["trajectory_modified"] is False
    torch.testing.assert_close(positions[0, 1], samples[0, 1, 1])
    torch.testing.assert_close(positions[0, 0], baseline[0, 0])
    torch.testing.assert_close(positions[0, 2], baseline[0, 2])
    torch.testing.assert_close(yaws[0, 1], sample_yaws[0, 1, 1])


def test_rejects_candidate_with_collateral_contact():
    samples = torch.zeros(1, 2, 3, 3, 2)
    samples[0, 0, 1, :, 1] = 5.0
    samples[0, 1, 1, :, 1] = 1.2
    samples[:, :, 2, :, 1] = 1.2
    sample_yaws = torch.zeros(1, 2, 3, 3, 1)
    baseline = samples[:, 0].clone()
    # The third actor's factual trajectory occupies the lower-clearance
    # candidate path.  The selector must reject that candidate as collateral.
    baseline[0, 2, :, 1] = 1.2
    extents = torch.ones(1, 3, 3)

    positions, _, audit = select_marginal_joint_tail_action(
        samples,
        sample_yaws,
        baseline,
        sample_yaws[:, 0],
        torch.zeros(1, 3, 2),
        torch.zeros(1, 3, 1),
        identity_transforms(3),
        extents,
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.0,
        min_risk_improvement_m=0.05,
    )

    assert audit["selected_sample_index"] == 0
    torch.testing.assert_close(positions, baseline)


def test_rejects_mode_below_reference_planner_safety_margin():
    samples = torch.zeros(1, 2, 3, 3, 2)
    samples[0, 0, 1, :, 1] = 5.0
    samples[0, 1, 1, :, 1] = 1.2
    samples[:, :, 2, :, 1] = 10.0
    yaws = torch.zeros(1, 2, 3, 3, 1)

    positions, _, audit = select_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=torch.zeros(1, 3, 2),
        ego_yaws=torch.zeros(1, 3, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.0,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )

    assert audit["selected_sample_index"] == 0
    torch.testing.assert_close(positions, samples[:, 0])


def test_accepts_agent_centric_wrapper_layout_and_history_extents():
    samples = torch.zeros(3, 2, 3, 2)
    samples[1, 0, :, 1] = 5.0
    samples[1, 1, :, 1] = 1.2
    samples[2, :, :, 1] = 10.0
    sample_yaws = torch.zeros(3, 2, 3, 1)
    baseline = samples[:, 0].clone()
    history_extents = torch.ones(3, 4, 3)

    positions, yaws, audit = select_marginal_joint_tail_action(
        samples,
        sample_yaws,
        baseline,
        sample_yaws[:, 0],
        torch.zeros(1, 3, 2),
        torch.zeros(1, 3, 1),
        torch.eye(3).repeat(3, 1, 1),
        history_extents,
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.0,
        min_risk_improvement_m=0.05,
    )

    assert positions.shape == baseline.shape
    assert yaws.shape == sample_yaws[:, 0].shape
    assert audit["selected_sample_index"] == 1
    torch.testing.assert_close(positions[1], samples[1, 1])


def test_multi_agent_composition_keeps_each_receiver_on_a_native_mode():
    samples = torch.zeros(1, 2, 3, 3, 2)
    samples[0, 0, 1, :, 1] = 5.0
    samples[0, 1, 1, :, 1] = 1.2
    samples[0, 0, 2, :, 1] = -5.0
    samples[0, 1, 2, :, 1] = -1.2
    yaws = torch.zeros(1, 2, 3, 3, 1)

    positions, _, audits = select_multi_agent_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=torch.zeros(1, 3, 2),
        ego_yaws=torch.zeros(1, 3, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_indices=[1, 2],
        collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )

    assert [row["selected_sample_index"] for row in audits] == [1, 1]
    torch.testing.assert_close(positions[0, 1], samples[0, 1, 1])
    torch.testing.assert_close(positions[0, 2], samples[0, 1, 2])


def test_rejects_unphysical_native_mode_before_risk_selection():
    samples = torch.zeros(1, 2, 3, 5, 2)
    samples[0, 0, 1, :, 1] = 5.0
    # This mode would pass closest to ego but makes a 10 m jump in 0.1 s.
    samples[0, 1, 1, :, 1] = torch.tensor([5.0, 5.0, 0.5, 0.5, 0.5])
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    baseline = samples[:, 0].clone()

    positions, _, audit = select_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=baseline,
        baseline_yaws=yaws[:, 0],
        ego_positions=torch.zeros(1, 5, 2),
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.0,
        min_risk_improvement_m=0.05,
        dt=0.1,
        max_speed_mps=20.0,
    )

    assert audit["selected_sample_index"] == 0
    assert audit["kinematic_valid_sample_count"] == 1
    torch.testing.assert_close(positions, baseline)


def test_rejects_mode_with_cross_replan_velocity_discontinuity():
    samples = torch.zeros(1, 2, 3, 5, 2)
    samples[0, 0, 1, :, 0] = torch.arange(1.0, 6.0)
    samples[0, 0, 1, :, 1] = 5.0
    # Internally smooth, but it abruptly stops from the 10 m/s history.
    samples[0, 1, 1, :, 1] = 1.2
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    history = torch.zeros(3, 3, 2)
    history[1, :, 0] = torch.tensor([-2.0, -1.0, 0.0])
    history[1, :, 1] = 5.0

    positions, _, audit = select_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=torch.zeros(1, 5, 2),
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.0,
        min_risk_improvement_m=0.05,
        history_positions=history,
        history_yaws=torch.zeros(3, 3, 1),
        history_availabilities=torch.ones(3, 3),
    )

    assert audit["boundary_kinematics_used"] is True
    assert audit["selected_sample_index"] == 0
    assert audit["kinematic_valid_sample_count"] == 1
    torch.testing.assert_close(positions, samples[:, 0])


def test_replaces_invalid_baseline_with_valid_native_mode():
    samples = torch.zeros(1, 2, 3, 5, 2)
    samples[0, 0, 1, :, 1] = 5.0
    samples[0, 1, 1, :, 0] = torch.arange(1.0, 6.0)
    samples[0, 1, 1, :, 1] = 5.0
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    history = torch.zeros(3, 3, 2)
    history[1, :, 0] = torch.tensor([-2.0, -1.0, 0.0])
    history[1, :, 1] = 5.0

    positions, _, audit = select_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=torch.zeros(1, 5, 2),
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.0,
        min_risk_improvement_m=0.05,
        history_positions=history,
        history_yaws=torch.zeros(3, 3, 1),
        history_availabilities=torch.ones(3, 3),
    )

    assert audit["baseline_hard_valid"] is False
    assert audit["hard_gate_fallback"] is False
    assert audit["selected_sample_index"] == 1
    torch.testing.assert_close(positions[0, 1], samples[0, 1, 1])


def test_dual_planner_selects_target_failure_while_reference_stays_safe():
    samples = torch.zeros(1, 2, 3, 5, 2)
    # The factual receiver is four metres from the target planner.  Native
    # mode one overlaps the target planner at y=5, while remaining four metres
    # clear of the executed reference planner at y=0.
    samples[0, 0, 1, :, 1] = 9.0
    samples[0, 1, 1, :, 1] = 5.0
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    reference_ego = torch.zeros(1, 5, 2)
    target_ego = torch.zeros(1, 5, 2)
    target_ego[0, :, 1] = 5.0

    positions, _, audit = select_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=reference_ego,
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.25,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
        max_target_clearance_m=2.0,
        target_ego_positions=target_ego,
        target_ego_yaws=torch.zeros(1, 5, 1),
        minimize_causal_change=True,
    )

    assert audit["dual_planner_objective"] is True
    assert audit["selected_sample_index"] == 1
    assert audit["selected_target_min_clearance_m"] <= 0.0
    assert audit["selected_reference_min_clearance_m"] > 0.25
    torch.testing.assert_close(positions[0, 1], samples[0, 1, 1])


def test_sparse_auto_search_selects_minimum_causal_change_native_receiver():
    samples = torch.zeros(1, 2, 4, 5, 2)
    samples[0, 0, 1, :, 1] = 5.0
    samples[0, 1, 1, :, 1] = 2.0
    samples[0, 0, 2, :, 1] = 6.0
    samples[0, 1, 2, :, 1] = 1.2
    samples[:, :, 3, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 4, 5, 1)
    reference_ego = torch.zeros(1, 5, 2)
    reference_ego[0, :, 1] = -5.0
    target_ego = torch.zeros(1, 5, 2)

    positions, _, audit = select_sparse_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=reference_ego,
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(4),
        extents=torch.ones(1, 4, 3),
        ego_index=0,
        receiver_indices=[1, 2, 3],
        collateral_margin_m=0.25,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
        target_ego_positions=target_ego,
        target_ego_yaws=torch.zeros(1, 5, 1),
        minimize_causal_change=True,
    )

    assert audit["auto_receiver_search"] is True
    # Both modes meet the target-severity gate, but receiver 1 needs a 3 m
    # marginal change while receiver 2 needs 4.8 m.  The constrained
    # objective must select the smaller causal intervention.
    assert audit["selected_receiver_index"] == 1
    assert audit["selected_sample_index"] == 1
    assert audit["constrained_objective"] == "min_mean_displacement_from_factual"
    assert audit["selectable_receiver_count"] == 2
    assert audit["sparse_noop"] is False
    torch.testing.assert_close(positions[0, 1], samples[0, 1, 1])
    torch.testing.assert_close(positions[0, 2], samples[0, 0, 2])


def test_single_receiver_minimizes_change_subject_to_target_failure():
    samples = torch.zeros(1, 3, 3, 5, 2)
    samples[0, 0, 1, :, 1] = 6.0
    # Both non-factual modes pass a 2 m target-clearance gate.  Mode 1 is a
    # mild 3 m shift; mode 2 is a deeper but unnecessary 4.8 m shift.
    samples[0, 1, 1, :, 1] = 3.0
    samples[0, 2, 1, :, 1] = 1.2
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 3, 3, 5, 1)
    reference_ego = torch.zeros(1, 5, 2)
    reference_ego[0, :, 1] = -5.0

    positions, _, audit = select_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=reference_ego,
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.25,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
        max_target_clearance_m=2.0,
        target_ego_positions=torch.zeros(1, 5, 2),
        target_ego_yaws=torch.zeros(1, 5, 1),
        minimize_causal_change=True,
    )

    assert audit["selected_sample_index"] == 1
    assert audit["selected_mean_displacement_from_factual_m"] == 3.0
    torch.testing.assert_close(positions[0, 1], samples[0, 1, 1])


def test_sparse_auto_search_respects_locked_receiver_across_replans():
    samples = torch.zeros(1, 2, 4, 5, 2)
    samples[0, 0, 1, :, 1] = 5.0
    samples[0, 1, 1, :, 1] = 2.0
    samples[0, 0, 2, :, 1] = 6.0
    samples[0, 1, 2, :, 1] = 1.2
    samples[:, :, 3, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 4, 5, 1)
    reference_ego = torch.zeros(1, 5, 2)
    reference_ego[0, :, 1] = -5.0

    positions, _, audit = select_sparse_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=reference_ego,
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(4),
        extents=torch.ones(1, 4, 3),
        ego_index=0,
        receiver_indices=[1, 2, 3],
        locked_receiver_index=1,
        collateral_margin_m=0.25,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
        target_ego_positions=torch.zeros(1, 5, 2),
        target_ego_yaws=torch.zeros(1, 5, 1),
    )

    assert audit["receiver_lock_active"] is True
    assert audit["locked_receiver_index"] == 1
    assert audit["evaluated_receiver_count"] == 1
    assert audit["selected_receiver_index"] == 1
    assert audit["selected_sample_index"] == 1
    torch.testing.assert_close(positions[0, 1], samples[0, 1, 1])
    torch.testing.assert_close(positions[0, 2], samples[0, 0, 2])


def test_sparse_auto_search_keeps_factual_when_no_mode_improves():
    samples = torch.zeros(1, 2, 3, 5, 2)
    samples[:, :, 1, :, 1] = 5.0
    samples[:, :, 2, :, 1] = 10.0
    yaws = torch.zeros(1, 2, 3, 5, 1)

    positions, _, audit = select_sparse_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=torch.zeros(1, 5, 2),
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_indices=[1, 2],
        collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )

    assert audit["selected_receiver_index"] == -1
    assert audit["selected_sample_index"] == 0
    assert audit["sparse_noop"] is True
    torch.testing.assert_close(positions, samples[:, 0])


def test_sparse_auto_search_does_not_repair_irrelevant_invalid_baseline():
    samples = torch.zeros(1, 2, 3, 5, 2)
    # Factual mode is close to the target ego but discontinuous with history.
    samples[0, 0, 1, :, 1] = 1.2
    # Native mode one is physically continuous but much less adversarial.
    samples[0, 1, 1, :, 0] = torch.arange(1.0, 6.0)
    samples[0, 1, 1, :, 1] = 5.0
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    history = torch.zeros(3, 3, 2)
    history[1, :, 0] = torch.tensor([-2.0, -1.0, 0.0])
    history[1, :, 1] = 5.0

    positions, _, audit = select_sparse_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=torch.zeros(1, 5, 2),
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_indices=[1],
        collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
        history_positions=history,
        history_yaws=torch.zeros(3, 3, 1),
        history_availabilities=torch.ones(3, 3),
    )

    assert audit["fallback_receiver_count"] == 0
    assert audit["selectable_receiver_count"] == 0
    assert audit["sparse_noop"] is True
    torch.testing.assert_close(positions, samples[:, 0])


def test_logged_q99_gate_rejects_new_marginal_tail_violation():
    samples = torch.zeros(1, 2, 3, 5, 2)
    samples[0, 0, 1, :, 0] = torch.arange(0.1, 0.6, 0.1)
    samples[0, 0, 1, :, 1] = 5.0
    # This mode is more adversarial but drives at 1.8 m/s, above the
    # independent logged-data q99 of 1.5 m/s for the synthetic low-speed bin.
    samples[0, 1, 1, :, 0] = torch.arange(0.18, 0.91, 0.18)
    samples[0, 1, 1, :, 1] = 1.2
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    reference_ego = torch.zeros(1, 5, 2)
    reference_ego[:, :, 1] = -5.0

    positions, _, audit = select_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=reference_ego,
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.25,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
        naturalness_profile=strict_speed_naturalness_profile(),
        target_ego_positions=torch.zeros(1, 5, 2),
        target_ego_yaws=torch.zeros(1, 5, 1),
    )

    assert audit["naturalness_profile_enabled"] is True
    assert audit["naturalness_valid_sample_count"] == 1
    assert audit["sample_naturalness_q99_violation_counts"]["speed_mps"] == [0, 4]
    assert audit["selected_sample_index"] == 0
    torch.testing.assert_close(positions, samples[:, 0])


def test_logged_q99_gate_keeps_native_adversarial_mode_when_nonworsening():
    samples = torch.zeros(1, 2, 3, 5, 2)
    samples[0, 0, 1, :, 0] = torch.arange(0.1, 0.6, 0.1)
    samples[0, 0, 1, :, 1] = 5.0
    samples[0, 1, 1, :, 0] = torch.arange(0.14, 0.71, 0.14)
    samples[0, 1, 1, :, 1] = 1.2
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    reference_ego = torch.zeros(1, 5, 2)
    reference_ego[:, :, 1] = -5.0

    positions, _, audit = select_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=reference_ego,
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_index=1,
        collateral_margin_m=0.25,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
        naturalness_profile=strict_speed_naturalness_profile(),
        target_ego_positions=torch.zeros(1, 5, 2),
        target_ego_yaws=torch.zeros(1, 5, 1),
    )

    assert audit["naturalness_valid_sample_count"] == 2
    assert audit["sample_naturalness_q99_violation_counts"]["speed_mps"] == [0, 0]
    assert audit["selected_sample_index"] == 1
    torch.testing.assert_close(positions[0, 1], samples[0, 1, 1])


def test_target_severity_gate_prefers_noop_to_irrelevant_far_receiver():
    samples = torch.zeros(1, 2, 3, 5, 2)
    samples[0, 0, 1, :, 1] = 80.0
    samples[0, 1, 1, :, 1] = 70.0
    samples[:, :, 2, :, 1] = 100.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    reference_ego = torch.zeros(1, 5, 2)
    reference_ego[:, :, 1] = -5.0

    positions, _, audit = select_sparse_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=reference_ego,
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_indices=[1],
        collateral_margin_m=0.25,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
        max_target_clearance_m=2.0,
        target_ego_positions=torch.zeros(1, 5, 2),
        target_ego_yaws=torch.zeros(1, 5, 1),
    )

    assert audit["selectable_receiver_count"] == 0
    assert audit["sparse_noop"] is True
    torch.testing.assert_close(positions, samples[:, 0])


def test_baseline_target_safety_gate_rejects_already_dangerous_factual_mode():
    samples = torch.zeros(1, 2, 3, 5, 2)
    # With unit extents, mode 0 already has only 0.5 m target clearance.
    samples[0, 0, 1, :, 1] = 1.5
    samples[0, 1, 1, :, 1] = 1.2
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    reference_ego = torch.zeros(1, 5, 2)
    reference_ego[:, :, 1] = -5.0

    positions, _, audit = select_sparse_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=reference_ego,
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_indices=[1],
        collateral_margin_m=0.25,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
        max_target_clearance_m=2.0,
        min_baseline_target_clearance_m=2.5,
        target_ego_positions=torch.zeros(1, 5, 2),
        target_ego_yaws=torch.zeros(1, 5, 1),
        minimize_causal_change=True,
    )

    assert audit["sparse_noop"] is True
    assert audit["selectable_receiver_count"] == 0
    assert audit["baseline_target_unsafe_receiver_count"] == 1
    torch.testing.assert_close(positions, samples[:, 0])


def test_baseline_target_safety_gate_accepts_safe_to_dangerous_transition():
    samples = torch.zeros(1, 2, 3, 5, 2)
    # Factual mode 0 is 4 m clear; native mode 1 reduces clearance to 0.2 m.
    samples[0, 0, 1, :, 1] = 5.0
    samples[0, 1, 1, :, 1] = 1.2
    samples[:, :, 2, :, 1] = 20.0
    yaws = torch.zeros(1, 2, 3, 5, 1)
    reference_ego = torch.zeros(1, 5, 2)
    reference_ego[:, :, 1] = -5.0

    positions, _, audit = select_sparse_marginal_joint_tail_action(
        sample_positions=samples,
        sample_yaws=yaws,
        baseline_positions=samples[:, 0],
        baseline_yaws=yaws[:, 0],
        ego_positions=reference_ego,
        ego_yaws=torch.zeros(1, 5, 1),
        world_from_agent=identity_transforms(3),
        extents=torch.ones(1, 3, 3),
        ego_index=0,
        receiver_indices=[1],
        collateral_margin_m=0.25,
        reference_safety_margin_m=0.25,
        min_risk_improvement_m=0.05,
        max_target_clearance_m=2.0,
        min_baseline_target_clearance_m=2.5,
        target_ego_positions=torch.zeros(1, 5, 2),
        target_ego_yaws=torch.zeros(1, 5, 1),
        minimize_causal_change=True,
    )

    assert audit["sparse_noop"] is False
    assert audit["selected_receiver_index"] == 1
    assert audit["selected_sample_index"] == 1
    assert audit["baseline_target_safe_gate_pass"] is True
    torch.testing.assert_close(positions[0, 1], samples[0, 1, 1])
