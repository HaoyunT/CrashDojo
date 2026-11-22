import math

import torch

from ccdiff.models.persistent_copula_prior import (
    PersistentCopulaConfig,
    PersistentCopulaSource,
)


def correlation(first, second):
    first = first.flatten().double()
    second = second.flatten().double()
    first = first - first.mean()
    second = second - second.mean()
    return float(
        (first * second).mean()
        / (first.std(unbiased=False) * second.std(unbiased=False))
    )


def test_first_call_sample0_matches_native_draw():
    shape = (2, 5, 4, 9, 2)
    config = PersistentCopulaConfig(0.7, 3, ((2, 0),), 0.6)
    state = PersistentCopulaSource(config)
    generator = torch.Generator().manual_seed(17)
    actual = state.sample(shape, global_replan_step=0, generator=generator)
    expected = torch.randn(shape, generator=torch.Generator().manual_seed(17))
    assert torch.equal(actual[:, 0], expected[:, 0])


def test_temporal_persistence_and_cross_actor_correlation():
    shape = (24, 12, 4, 25, 2)
    alpha = 0.72
    rho = -0.55
    state = PersistentCopulaSource(
        PersistentCopulaConfig(alpha, 5, ((2, 0), (3, 0)), rho)
    )
    state.sample(
        shape,
        global_replan_step=0,
        generator=torch.Generator().manual_seed(1),
    )
    previous_independent = state._previous_independent.clone()
    second = state.sample(
        shape,
        global_replan_step=1,
        generator=torch.Generator().manual_seed(2),
    )
    current_independent = state._previous_independent

    assert abs(
        correlation(
            previous_independent[:, 1:, :, 5:],
            current_independent[:, 1:, :, :20],
        )
        - alpha
    ) < 0.03
    assert abs(correlation(second[:, 1:, 2], second[:, 1:, 0]) - rho) < 0.03
    assert abs(correlation(second[:, 1:, 3], second[:, 1:, 0]) - rho) < 0.03
    assert abs(
        correlation(second[:, 1:, 2], second[:, 1:, 3]) - rho * rho
    ) < 0.04
    for actor in range(shape[2]):
        values = second[:, 1:, actor].flatten()
        assert abs(float(values.mean())) < 0.03
        assert abs(float(values.std(unbiased=False)) - 1.0) < 0.03


def test_zero_parameters_are_native_and_reset_works():
    shape = (2, 4, 3, 8, 2)
    state = PersistentCopulaSource(PersistentCopulaConfig(0.0, 2))
    actual = state.sample(
        shape,
        global_replan_step=4,
        generator=torch.Generator().manual_seed(9),
    )
    expected = torch.randn(shape, generator=torch.Generator().manual_seed(9))
    assert torch.equal(actual, expected)
    state.reset()
    replay = state.sample(
        shape,
        global_replan_step=0,
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.equal(replay, expected)


def test_within_horizon_temporal_copula_preserves_marginals_and_sample0():
    shape = (256, 24, 5, 25, 2)
    temporal_rho = 0.62
    state = PersistentCopulaSource(
        PersistentCopulaConfig(
            0.0,
            5,
            temporal_rho=temporal_rho,
            temporal_action_dims=(0,),
        )
    )
    actual = state.sample(
        shape,
        global_replan_step=0,
        generator=torch.Generator().manual_seed(31),
    )
    expected = torch.randn(shape, generator=torch.Generator().manual_seed(31))

    assert torch.equal(actual[:, 0], expected[:, 0])
    assert torch.equal(actual[:, 1:, :, :, 1], expected[:, 1:, :, :, 1])
    modified = actual[:, 1:, :, :, 0]
    assert abs(float(modified.mean())) < 0.015
    assert abs(float(modified.std(unbiased=False)) - 1.0) < 0.015
    assert abs(
        correlation(modified[..., :-1], modified[..., 1:]) - temporal_rho
    ) < 0.02


def test_temporal_and_actor_copulas_compose_without_recursive_receivers():
    shape = (192, 20, 4, 25, 2)
    temporal_rho = 0.48
    actor_rho = -0.35
    state = PersistentCopulaSource(
        PersistentCopulaConfig(
            0.0,
            5,
            ((2, 0), (3, 0)),
            actor_rho,
            temporal_rho=temporal_rho,
        )
    )
    actual = state.sample(
        shape,
        global_replan_step=0,
        generator=torch.Generator().manual_seed(47),
    )
    proposals = actual[:, 1:]
    assert abs(
        correlation(proposals[:, :, 2], proposals[:, :, 0]) - actor_rho
    ) < 0.025
    assert abs(
        correlation(proposals[:, :, 3], proposals[:, :, 0]) - actor_rho
    ) < 0.025
    assert abs(
        correlation(proposals[:, :, 2], proposals[:, :, 3]) - actor_rho**2
    ) < 0.03
    for actor in range(shape[2]):
        values = proposals[:, :, actor]
        assert abs(float(values.mean())) < 0.02
        assert abs(float(values.std(unbiased=False)) - 1.0) < 0.02
        assert abs(
            correlation(values[:, :, :-1, :], values[:, :, 1:, :])
            - temporal_rho
        ) < 0.025


def test_kl_contract_is_finite_and_exact_per_coordinate():
    config = PersistentCopulaConfig(
        0.5,
        5,
        ((2, 0), (3, 0)),
        0.25,
        temporal_rho=0.4,
        temporal_action_dims=(0,),
    )
    state = PersistentCopulaSource(config)
    audit = state.normalized_kl_contract((1, 20, 4, 25, 2), replan_calls=16)
    assert math.isclose(
        audit["persistence_kl_per_coordinate"], -0.5 * math.log(0.75)
    )
    assert math.isclose(
        audit["copula_kl_per_receiver_coordinate"], -0.5 * math.log(0.9375)
    )
    assert audit["persistence_total_kl"] > 0
    assert audit["copula_total_kl"] > 0
    assert math.isclose(
        audit["temporal_kl_per_transition_coordinate"],
        -0.5 * math.log(0.84),
    )
    assert audit["temporal_transition_coordinates"] == 1 * 19 * 4 * 24 * 1 * 16
    assert audit["temporal_total_kl"] > 0


def test_invalid_contracts_fail_closed():
    state = PersistentCopulaSource(PersistentCopulaConfig(1.0, 2))
    try:
        state.sample((1, 2, 3, 8, 2), global_replan_step=0)
    except ValueError:
        pass
    else:
        raise AssertionError("singular persistence must fail")

    state = PersistentCopulaSource(
        PersistentCopulaConfig(0.5, 2, ((1, 0),), 0.2)
    )
    state.sample((1, 2, 3, 8, 2), global_replan_step=1)
    try:
        state.sample((1, 2, 3, 8, 2), global_replan_step=1)
    except ValueError:
        pass
    else:
        raise AssertionError("non-increasing replan step must fail")

    state = PersistentCopulaSource(
        PersistentCopulaConfig(0.0, 2, temporal_rho=1.0)
    )
    try:
        state.sample((1, 2, 3, 8, 2), global_replan_step=0)
    except ValueError:
        pass
    else:
        raise AssertionError("singular temporal correlation must fail")
