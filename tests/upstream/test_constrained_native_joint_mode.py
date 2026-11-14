import numpy as np
import torch

from tbsim.policies.wrappers import (
    select_constrained_native_joint_mode_action,
)


def fixture():
    # Agent-centric [M,K,T,D]. Agent 0 is ego and must never be replaced.
    samples = torch.zeros(3, 3, 4, 2)
    samples[1, 0, :, 0] = 3.0
    samples[1, 1, :, 0] = 0.8  # target-planner collision, reference stays safe
    samples[1, 2, :, 0] = 9.5  # collides with agent 2 -> collateral rejection
    samples[2, :, :, 0] = 10.0
    yaws = torch.zeros(3, 3, 4, 1)
    baseline = samples[:, 0].clone()
    baseline_yaw = yaws[:, 0].clone()
    target_ego = torch.zeros(1, 4, 2)
    target_ego_yaw = torch.zeros(1, 4, 1)
    reference_ego = torch.zeros(1, 4, 2)
    reference_ego[:, :, 1] = 10.0
    reference_ego_yaw = torch.zeros(1, 4, 1)
    transforms = torch.eye(3).repeat(3, 1, 1)
    extents = torch.ones(3, 1, 3)
    return (
        samples, yaws, baseline, baseline_yaw,
        reference_ego, reference_ego_yaw,
        target_ego, target_ego_yaw, transforms, extents,
    )


def select(**kwargs):
    values = fixture()
    return select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05, **kwargs
    )


def test_selects_one_complete_joint_mode_without_row_mixing():
    positions, yaws, audit = select()
    samples = fixture()[0]
    torch.testing.assert_close(positions[0], fixture()[2][0])
    torch.testing.assert_close(positions[1:], samples[1:, 1])
    torch.testing.assert_close(yaws[1:], fixture()[1][1:, 1])
    assert audit["selected_sample_index"] == 1
    assert audit["all_non_ego_same_joint_sample"]
    assert not audit["trajectory_modified"]
    assert not audit["projection"]
    assert not audit["row_mixing"]
    assert not audit["hard_gate_fallback"]


def test_reference_safety_is_a_hard_gate_and_falls_back_to_mode_zero():
    values = list(fixture())
    # Reference follows the same colliding trajectory as target.
    values[4] = values[6].clone()
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    torch.testing.assert_close(positions, values[2])
    assert audit["selected_sample_index"] == 0
    assert audit["hard_gate_fallback"]
    assert audit["valid_joint_mode_count"] == 0


def test_naturalness_tail_gate_rejects_an_individually_abnormal_mode():
    values = list(fixture())
    # Make the only target-critical mode contain an implausible speed jump.
    values[0][1, 1, 1:, 0] = torch.tensor([0.8, 5.8, 10.8])
    profile = {
        "speed_bin_upper_bounds_mps": [2.0, 5.0, 10.0, 15.0, 25.0],
        "q99": {
            "speed_mps": [20.0] * 6,
            "abs_acc_mps2": [2.0] * 6,
            "abs_jerk_mps3": [5.0] * 6,
            "abs_yaw_rate_rps": [1.0] * 6,
            "abs_lateral_acc_mps2": [4.0] * 6,
        },
    }
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05, naturalness_profile=profile,
        max_agent_tail_fraction=0.01, max_speed_mps=100.0,
        max_acc_mps2=1000.0, max_jerk_mps3=10000.0,
    )
    torch.testing.assert_close(positions, values[2])
    assert audit["hard_gate_fallback"]
    assert audit["proposal_sample0_naturalness_gate"]
    assert audit["proposal_sample0_physical_gate"]
    assert audit["proposal_sample0_max_agent_q99_tail_fraction"] == 0.0
    assert set(audit["proposal_sample0_q99_tail_fraction_by_agent"]) == {1, 2}


def test_reports_ineligible_sample_zero_without_changing_fallback_semantics():
    values = list(fixture())
    # Make proposal 0 locally abnormal while leaving policy baseline and the
    # target-critical proposal untouched. The new attestation must expose the
    # invalid pCN centre but remain diagnostic-only.
    values[0][1, 0, 1:, 0] = torch.tensor([3.0, 8.0, 13.0])
    profile = {
        "speed_bin_upper_bounds_mps": [2.0, 5.0, 10.0, 15.0, 25.0],
        "q99": {
            "speed_mps": [100.0] * 6,
            "abs_acc_mps2": [2.0] * 6,
            "abs_jerk_mps3": [5.0] * 6,
            "abs_yaw_rate_rps": [1.0] * 6,
            "abs_lateral_acc_mps2": [4.0] * 6,
        },
    }
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05, naturalness_profile=profile,
        max_agent_tail_fraction=0.01, max_speed_mps=100.0,
        max_acc_mps2=1000.0, max_jerk_mps3=10000.0,
    )
    assert not audit["proposal_sample0_naturalness_gate"]
    assert audit["proposal_sample0_max_agent_q99_tail_fraction"] > 0.01
    assert audit["selected_sample_index"] == 1
    assert not audit["hard_gate_fallback"]
    torch.testing.assert_close(positions[1:], values[0][1:, 1])


def test_raw_jerk_limit_is_diagnostic_but_empirical_jerk_q99_remains_hard():
    values = list(fixture())
    # Mode 1 contains a high raw third difference. A permissive empirical q99
    # permits it despite the fixed raw jerk constant; a strict empirical q99
    # must still reject it. This prevents duplicate finite-difference vetoes
    # while keeping data-grounded jerk naturalness as a hard condition.
    values[0][1, 1, :, 0] = torch.tensor([0.8, 0.85, 0.8, 0.8])
    permissive = {
        "speed_bin_upper_bounds_mps": [2.0, 5.0, 10.0, 15.0, 25.0],
        "q99": {
            "speed_mps": [100.0] * 6,
            "abs_acc_mps2": [100.0] * 6,
            "abs_jerk_mps3": [1000.0] * 6,
            "abs_yaw_rate_rps": [10.0] * 6,
            "abs_lateral_acc_mps2": [100.0] * 6,
        },
    }
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05, naturalness_profile=permissive,
        max_agent_tail_fraction=0.01, max_jerk_mps3=0.01,
        max_acc_mps2=100.0,
    )
    assert not audit["hard_gate_fallback"]
    assert audit["physical_gate_counts_by_metric"]["jerk"] < audit["num_samples"]
    assert audit["raw_jerk_regulatory_limit_diagnostic_only"]
    torch.testing.assert_close(positions[1:], values[0][1:, 1])

    strict = dict(permissive)
    strict["q99"] = dict(permissive["q99"])
    strict["q99"]["abs_jerk_mps3"] = [0.01] * 6
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05, naturalness_profile=strict,
        max_agent_tail_fraction=0.01, max_jerk_mps3=1000.0,
        max_acc_mps2=100.0,
    )
    assert audit["hard_gate_fallback"]
    torch.testing.assert_close(positions, values[2])


def test_existing_mode0_collateral_overlap_is_not_misclassified_as_new():
    values = list(fixture())
    # Add a second background agent that overlaps agent 1 identically in both
    # native modes. This is pre-existing in mode-0, not newly introduced by
    # selecting the target-critical complete joint sample.
    values[0] = torch.cat((values[0], values[0][1:2]), dim=0)
    values[1] = torch.cat((values[1], values[1][1:2]), dim=0)
    values[2] = torch.cat((values[2], values[2][1:2]), dim=0)
    values[3] = torch.cat((values[3], values[3][1:2]), dim=0)
    values[8] = torch.cat((values[8], values[8][1:2]), dim=0)
    values[9] = torch.cat((values[9], values[9][1:2]), dim=0)
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    assert audit["selected_sample_index"] == 1
    assert audit["selected_collateral_nonworsening_vs_mode0"]
    torch.testing.assert_close(positions[1], values[0][1, 1])


def test_rejects_single_frame_risk_when_persistent_risk_is_required():
    values = list(fixture())
    values[0][1, 1, :, 0] = torch.tensor([3.0, 0.8, 3.0, 3.0])
    values[0][1, 2, :, 0] = torch.tensor([3.0, 0.8, 0.8, 3.0])
    values[0][2, 2, :, 0] = 10.0
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        target_min_persistence_steps=2,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05, max_speed_mps=100.0,
        max_acc_mps2=1000.0, max_jerk_mps3=10000.0,
    )
    assert audit["selected_sample_index"] == 2
    assert audit["selected_target_max_consecutive_risk_steps"] >= 2
    torch.testing.assert_close(positions[1:], values[0][1:, 2])


def test_rejects_when_factual_mode_is_already_target_critical():
    values = list(fixture())
    values[0][1, 0, :, 0] = 0.8
    values[2] = values[0][:, 0].clone()
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        baseline_min_target_clearance_m=2.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    torch.testing.assert_close(positions, values[2])
    assert not audit["baseline_target_safe_gate"]
    assert audit["hard_gate_fallback"]


def test_risk_improvement_is_a_gate_before_minimum_displacement():
    values = list(fixture())
    # Mode 0 is already within the target-risk band and has zero displacement.
    # Mode 1 is strictly riskier but still passes every other hard gate.
    values[0][1, 0, :, 0] = 1.5
    values[0][1, 1, :, 0] = 0.8
    values[2] = values[0][:, 0].clone()
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        baseline_min_target_clearance_m=None,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    assert audit["selected_sample_index"] == 1
    assert audit["selected_target_risk_improvement_m"] >= 0.05
    assert not audit["hard_gate_fallback"]
    torch.testing.assert_close(positions[1:], values[0][1:, 1])


def test_progressive_barrier_accumulates_safe_native_clearance_progress():
    values = list(fixture())
    # Mode 1 improves the target-planner clearance versus mode 0, but remains
    # outside the terminal <=1 m risk band. This is a valid receding-horizon
    # progress step only when the progressive barrier is explicitly enabled.
    values[0][1, 1, :, 0] = 2.5
    values[2] = values[0][:, 0].clone()

    default_positions, _, default_audit = (
        select_constrained_native_joint_mode_action(
            *values, ego_indices=[0], target_max_clearance_m=1.0,
            target_min_persistence_steps=3,
            reference_safety_margin_m=0.25, collateral_margin_m=0.25,
            min_risk_improvement_m=0.05,
        )
    )
    torch.testing.assert_close(default_positions, values[2])
    assert default_audit["hard_gate_fallback"]
    assert default_audit["target_gate_mode"] == "terminal_persistent_severity"

    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        target_min_persistence_steps=3, progressive_target_barrier=True,
        progressive_activation_max_clearance_m=8.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    assert audit["selected_sample_index"] == 1
    assert not audit["hard_gate_fallback"]
    assert audit["target_gate_mode"] == "progressive_clearance_barrier"
    assert audit["terminal_target_gate_count"] == 0
    assert audit["selected_target_risk_improvement_m"] >= 0.05
    torch.testing.assert_close(positions[1:], values[0][1:, 1])


def test_progressive_barrier_is_inactive_outside_factual_risk_domain():
    values = list(fixture())
    values[0][1, 0, :, 0] = 20.0
    values[0][1, 1, :, 0] = 15.0
    values[2] = values[0][:, 0].clone()
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        target_min_persistence_steps=3, progressive_target_barrier=True,
        progressive_activation_max_clearance_m=8.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    torch.testing.assert_close(positions, values[2])
    assert not audit["progressive_activation_gate"]
    assert audit["hard_gate_fallback"]


def test_progressive_risk_first_prefers_terminal_mode_before_parsimony():
    values = list(fixture())
    # Mode 1 is a small but non-terminal change; mode 2 is a larger complete
    # joint change that reaches persistent terminal severity. The legacy
    # progressive objective chooses mode 1 by minimum displacement, whereas
    # continuation must first reach terminal severity and only then optimize
    # causal parsimony.
    values[0][1, 0, :, 0] = 3.0
    values[0][1, 1, :, 0] = 2.5
    values[0][1, 2, :, 0] = 1.5
    values[0][2, :, :, 0] = 10.0
    values[2] = values[0][:, 0].clone()

    legacy, _, legacy_audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        target_min_persistence_steps=3, progressive_target_barrier=True,
        progressive_activation_max_clearance_m=8.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    assert legacy_audit["selected_sample_index"] == 1
    torch.testing.assert_close(legacy[1:], values[0][1:, 1])

    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        target_min_persistence_steps=3, progressive_target_barrier=True,
        progressive_activation_max_clearance_m=8.0,
        progressive_risk_first=True,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    assert audit["selected_sample_index"] == 2
    assert audit["selection_stage"] == (
        "terminal_severity_then_minimum_displacement"
    )
    torch.testing.assert_close(positions[1:], values[0][1:, 2])


def test_progressive_risk_first_maximizes_safe_progress_before_terminal():
    values = list(fixture())
    values[0][1, 0, :, 0] = 3.0
    values[0][1, 1, :, 0] = 2.7  # smaller displacement, weaker progress
    values[0][1, 2, :, 0] = 2.3  # larger displacement, stronger progress
    values[0][2, :, :, 0] = 10.0
    values[2] = values[0][:, 0].clone()
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=0.5,
        target_min_persistence_steps=3, progressive_target_barrier=True,
        progressive_activation_max_clearance_m=8.0,
        progressive_risk_first=True,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    assert audit["selected_sample_index"] == 2
    assert audit["selection_stage"] == "maximum_safe_risk_progress"
    torch.testing.assert_close(positions[1:], values[0][1:, 2])


def test_progressive_persistence_first_prefers_sustained_near_miss():
    values = list(fixture())
    # Baseline signed gap is 3m. Mode 1 has a stronger one-frame 0.6m spike;
    # mode 2 sustains a 1.2m near-miss over the complete proposal horizon.
    # Neither reaches the 0.5m terminal band.
    values[0][1, 0, :, 0] = 4.0
    values[0][1, 1, :, 0] = torch.tensor([1.6, 4.0, 4.0, 4.0])
    values[0][1, 2, :, 0] = 2.2
    values[0][2, :, :, 0] = 10.0
    values[2] = values[0][:, 0].clone()

    risk_positions, _, risk_audit = (
        select_constrained_native_joint_mode_action(
            *values, ego_indices=[0], target_max_clearance_m=0.5,
            target_min_persistence_steps=3, progressive_target_barrier=True,
            progressive_activation_max_clearance_m=8.0,
            progressive_risk_first=True,
            reference_safety_margin_m=0.25, collateral_margin_m=0.25,
            min_risk_improvement_m=0.05, max_speed_mps=100.0,
            max_acc_mps2=1000.0,
        )
    )
    assert risk_audit["selected_sample_index"] == 1
    torch.testing.assert_close(risk_positions[1:], values[0][1:, 1])

    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=0.5,
        target_min_persistence_steps=3, progressive_target_barrier=True,
        progressive_activation_max_clearance_m=8.0,
        progressive_risk_first=True, progressive_persistence_first=True,
        progressive_exposure_clearance_m=2.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05, max_speed_mps=100.0,
        max_acc_mps2=1000.0,
    )
    assert audit["selected_sample_index"] == 2
    assert audit["selection_stage"] == "maximum_safe_risk_persistence"
    assert audit["selected_target_max_consecutive_exposure_steps"] == 4
    torch.testing.assert_close(positions[1:], values[0][1:, 2])


def test_progressive_persistence_first_rejects_invalid_exposure_band():
    values = list(fixture())
    try:
        select_constrained_native_joint_mode_action(
            *values, ego_indices=[0], target_max_clearance_m=1.0,
            progressive_target_barrier=True,
            progressive_persistence_first=True,
            progressive_exposure_clearance_m=0.5,
        )
    except ValueError as error:
        assert "must be at least" in str(error)
    else:
        raise AssertionError("invalid exposure band was accepted")


def test_progressive_dose_first_balances_exposure_depth_and_duration():
    values = list(fixture())
    # Baseline gap is 3m. Mode1 has four shallow 1.9m exposure frames
    # (dose=0.4 at band2); mode2 has three deeper 1.7m frames (dose=0.9).
    # Neither enters the terminal 1m band.
    values[0][1, 0, :, 0] = 4.0
    values[0][1, 1, :, 0] = 2.9
    values[0][1, 2, :, 0] = torch.tensor([2.7, 2.7, 2.7, 4.0])
    values[0][2, :, :, 0] = 10.0
    values[2] = values[0][:, 0].clone()
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        target_min_persistence_steps=3, progressive_target_barrier=True,
        progressive_activation_max_clearance_m=8.0,
        progressive_dose_first=True, progressive_exposure_clearance_m=2.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.0, max_speed_mps=100.0,
        max_acc_mps2=1000.0,
    )
    assert audit["selected_sample_index"] == 2
    assert audit["selection_stage"] == "maximum_safe_target_risk_dose"
    assert audit["selected_target_risk_dose_m_steps"] > 0.8
    torch.testing.assert_close(positions[1:], values[0][1:, 2])


def test_fallback_preserves_policy_baseline_not_correlated_sample_zero():
    values = list(fixture())
    values[2] = values[2].clone()
    values[2][1, :, 0] = 7.0  # policy baseline differs from sample mode0 at 3m
    values[4] = values[6].clone()  # force reference hard-gate failure
    positions, yaws, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_max_clearance_m=1.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    torch.testing.assert_close(positions, values[2])
    torch.testing.assert_close(yaws, values[3])
    assert audit["hard_gate_fallback"]
    assert audit["fallback_preserved_policy_baseline"]
    assert not audit["proposal_sample_executed"]
    assert audit["selected_mean_displacement_from_policy_baseline_m"] == 0.0
    assert audit["selected_max_displacement_from_policy_baseline_m"] == 0.0
    assert audit["selected_collateral_nonworsening_vs_policy_baseline"]
    assert audit["proposal_sample_max_agent_q99_tail_fraction"] is not None
    assert audit["executed_sample_max_agent_q99_tail_fraction"] is None
    assert audit["selected_max_agent_q99_tail_fraction"] is None


def test_locked_target_cannot_be_satisfied_by_unrelated_close_background():
    values = list(fixture())
    # Add an unrelated background row copied from the risky target, then keep
    # the preregistered target row far from the target-planner ego in all modes.
    for index in (0, 1, 2, 3, 8, 9):
        values[index] = torch.cat((values[index], values[index][1:2]), dim=0)
    values[0][1, :, :, 0] = 8.0
    values[2][1, :, 0] = 8.0
    positions, _, audit = select_constrained_native_joint_mode_action(
        *values, ego_indices=[0], target_agent_indices=[1],
        target_max_clearance_m=1.0,
        reference_safety_margin_m=0.25, collateral_margin_m=0.25,
        min_risk_improvement_m=0.05,
    )
    torch.testing.assert_close(positions, values[2])
    assert audit["hard_gate_fallback"]
    assert audit["target_agent_indices"] == [1]
    assert audit["target_attribution_locked"]
