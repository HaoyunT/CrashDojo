import numpy as np
import pytest
import torch

from ccdiff.models.ccdiff import CCFlowMatching


def test_correlated_prior_preserves_standard_normal_marginals_and_sets_copula():
    torch.manual_seed(7)
    rho = -0.7
    noise = CCFlowMatching.sample_marginal_preserving_correlated_prior(
        (4000, 5, 4, 6, 2), [(2, 1)], rho, device="cpu"
    )
    # sample 0 is a reserved independent factual baseline; copula statistics
    # are measured only on proposal modes 1..K-1.
    flattened = noise[:, 1:].reshape(-1, 4, 6 * 2)
    for agent in range(4):
        values = flattened[:, agent].reshape(-1).numpy()
        assert abs(float(values.mean())) < 0.02
        assert abs(float(values.std()) - 1.0) < 0.02
    receiver = flattened[:, 2].reshape(-1).numpy()
    anchor = flattened[:, 1].reshape(-1).numpy()
    unrelated = flattened[:, 3].reshape(-1).numpy()
    assert abs(float(np.corrcoef(receiver, anchor)[0, 1]) - rho) < 0.02
    assert abs(float(np.corrcoef(receiver, unrelated)[0, 1])) < 0.02


def test_shared_anchor_preserves_marginals_and_creates_common_factor():
    torch.manual_seed(9)
    rho = 0.6
    noise = CCFlowMatching.sample_marginal_preserving_correlated_prior(
        (8000, 2, 4, 6, 2), [(2, 0), (3, 0)], rho, device="cpu"
    )
    proposals = noise[:, 1:]
    for agent in range(4):
        values = proposals[:, :, agent].reshape(-1).numpy()
        assert abs(float(values.mean())) < 0.02
        assert abs(float(values.std()) - 1.0) < 0.02
    anchor = proposals[:, :, 0].reshape(-1).numpy()
    left = proposals[:, :, 2].reshape(-1).numpy()
    right = proposals[:, :, 3].reshape(-1).numpy()
    assert abs(float(np.corrcoef(left, anchor)[0, 1]) - rho) < 0.02
    assert abs(float(np.corrcoef(right, anchor)[0, 1]) - rho) < 0.02
    assert abs(float(np.corrcoef(left, right)[0, 1]) - rho * rho) < 0.02


def test_correlated_prior_is_deterministic_for_fixed_seed():
    shape = (8, 3, 4, 6, 2)
    torch.manual_seed(11)
    first = CCFlowMatching.sample_marginal_preserving_correlated_prior(
        shape, [(2, 1)], 0.5, device="cpu"
    )
    torch.manual_seed(11)
    second = CCFlowMatching.sample_marginal_preserving_correlated_prior(
        shape, [(2, 1)], 0.5, device="cpu"
    )
    assert torch.equal(first, second)


def test_sample_zero_is_exact_independent_policy_baseline_for_fixed_seed():
    shape = (8, 20, 4, 6, 2)
    torch.manual_seed(13)
    independent = torch.randn(shape)
    torch.manual_seed(13)
    correlated = CCFlowMatching.sample_marginal_preserving_correlated_prior(
        shape, [(2, 1)], -0.7, action_dims=[0], device="cpu"
    )
    assert torch.equal(correlated[:, 0], independent[:, 0])
    assert not torch.equal(correlated[:, 1:], independent[:, 1:])


@pytest.mark.parametrize("lag_steps", [3, 7])
def test_lagged_correlated_prior_preserves_white_marginals_and_sets_lag(lag_steps):
    torch.manual_seed(17)
    rho = 0.5
    noise = CCFlowMatching.sample_marginal_preserving_correlated_prior(
        (8000, 2, 3, 12, 2), [(2, 0)], rho,
        lag_steps=lag_steps, device="cpu"
    )
    receiver = noise[:, 1:, 2].numpy()
    anchor = noise[:, 1:, 0].numpy()
    values = receiver.reshape(-1)
    assert abs(float(values.mean())) < 0.02
    assert abs(float(values.std()) - 1.0) < 0.02
    # Every receiver coordinate remains mutually independent despite the lag.
    flat = receiver.reshape(-1, receiver.shape[-2] * receiver.shape[-1])
    corr = np.corrcoef(flat, rowvar=False)
    off_diagonal = corr - np.eye(corr.shape[0])
    assert float(np.max(np.abs(off_diagonal))) < 0.04
    lagged_receiver = receiver[:, :, lag_steps:].reshape(-1)
    lagged_anchor = anchor[:, :, :-lag_steps].reshape(-1)
    synchronous_anchor = anchor[:, :, lag_steps:].reshape(-1)
    assert abs(float(np.corrcoef(lagged_receiver, lagged_anchor)[0, 1]) - rho) < 0.02
    assert abs(float(np.corrcoef(lagged_receiver, synchronous_anchor)[0, 1])) < 0.02


def test_dimension_selective_prior_correlates_only_requested_action_dim():
    torch.manual_seed(29)
    rho = -0.5
    noise = CCFlowMatching.sample_marginal_preserving_correlated_prior(
        (10000, 2, 3, 8, 2), [(2, 0)], rho,
        action_dims=[1], device="cpu"
    )
    receiver = noise[:, 1:, 2].numpy()
    anchor = noise[:, 1:, 0].numpy()
    acc_corr = np.corrcoef(
        receiver[..., 0].reshape(-1), anchor[..., 0].reshape(-1)
    )[0, 1]
    yaw_corr = np.corrcoef(
        receiver[..., 1].reshape(-1), anchor[..., 1].reshape(-1)
    )[0, 1]
    assert abs(float(acc_corr)) < 0.02
    assert abs(float(yaw_corr) - rho) < 0.02
    assert abs(float(receiver.mean())) < 0.02
    assert abs(float(receiver.std()) - 1.0) < 0.02


@pytest.mark.parametrize(
    "pairs,rho,exception",
    [
        ([(1, 1)], 0.5, ValueError),
        ([(1, 2), (2, 3)], 0.5, ValueError),
        ([(1, 2), (1, 3)], 0.5, ValueError),
        ([(1, 2)], 1.0, ValueError),
        ([(1, 5)], 0.5, IndexError),
    ],
)
def test_correlated_prior_rejects_invalid_contracts(pairs, rho, exception):
    with pytest.raises(exception):
        CCFlowMatching.sample_marginal_preserving_correlated_prior(
            (4, 2, 4, 3, 2), pairs, rho, device="cpu"
        )


@pytest.mark.parametrize("lag_steps", [-1, 3])
def test_correlated_prior_rejects_invalid_lags(lag_steps):
    with pytest.raises(ValueError):
        CCFlowMatching.sample_marginal_preserving_correlated_prior(
            (4, 2, 4, 3, 2), [(1, 2)], 0.5,
            lag_steps=lag_steps, device="cpu"
        )


@pytest.mark.parametrize("action_dims", [[], [-1], [2]])
def test_correlated_prior_rejects_invalid_action_dims(action_dims):
    with pytest.raises(ValueError):
        CCFlowMatching.sample_marginal_preserving_correlated_prior(
            (4, 2, 4, 3, 2), [(1, 2)], 0.5,
            action_dims=action_dims, device="cpu"
        )


@pytest.mark.parametrize("beta", [0.25, 0.5, 1.0])
def test_pcn_joint_prior_preserves_exact_original_joint_gaussian(beta):
    torch.manual_seed(41)
    noise = CCFlowMatching.sample_pcn_joint_prior(
        (6000, 4, 3, 6, 2), beta, device="cpu"
    )
    proposals = noise[:, 1:].numpy()
    assert abs(float(proposals.mean())) < 0.02
    assert abs(float(proposals.std()) - 1.0) < 0.02
    # Within each complete proposal, different agents remain independent.
    left = proposals[:, :, 0].reshape(-1)
    right = proposals[:, :, 2].reshape(-1)
    assert abs(float(np.corrcoef(left, right)[0, 1])) < 0.02
    # The only intended change is cross-mode proximity to factual sample 0.
    factual = np.broadcast_to(
        noise[:, :1].numpy(), noise[:, 1:].shape
    ).reshape(-1)
    proposal = proposals.reshape(-1)
    expected = np.sqrt(1.0 - beta * beta)
    assert abs(float(np.corrcoef(factual, proposal)[0, 1]) - expected) < 0.02


def test_pcn_sample_zero_is_exact_independent_policy_baseline_for_fixed_seed():
    shape = (8, 20, 4, 6, 2)
    torch.manual_seed(43)
    independent = torch.randn(shape)
    torch.manual_seed(43)
    pcn = CCFlowMatching.sample_pcn_joint_prior(
        shape, 0.35, device="cpu"
    )
    assert torch.equal(pcn[:, 0], independent[:, 0])
    assert not torch.equal(pcn[:, 1:], independent[:, 1:])


def test_pcn_beta_one_matches_independent_sampler_exactly_for_fixed_seed():
    shape = (5, 7, 4, 3, 2)
    torch.manual_seed(47)
    independent = torch.randn(shape)
    torch.manual_seed(47)
    pcn = CCFlowMatching.sample_pcn_joint_prior(shape, 1.0, device="cpu")
    assert torch.equal(pcn, independent)


@pytest.mark.parametrize("beta", [-0.1, 0.0, 1.01])
def test_pcn_joint_prior_rejects_invalid_beta(beta):
    with pytest.raises(ValueError):
        CCFlowMatching.sample_pcn_joint_prior(
            (4, 2, 3, 5, 2), beta, device="cpu"
        )
