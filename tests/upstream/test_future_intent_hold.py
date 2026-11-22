import pytest
import torch

from ccdiff.models.scenetemporal import apply_directed_future_intent_hold


def _features(batch=1, agents=3, hist=4, fut=5, width=2):
    history = torch.arange(
        batch * agents * agents * hist * width, dtype=torch.float32
    ).reshape(batch, agents, agents, hist, width)
    future = 1000.0 + torch.arange(
        batch * agents * agents * fut * width, dtype=torch.float32
    ).reshape(batch, agents, agents, fut, width)
    return history, future


def test_future_hold_changes_only_requested_directed_future_edge():
    history, future = _features()
    mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    mask[:, 2, 0] = True

    result = apply_directed_future_intent_hold(future, history, mask)

    expected = history[0, 2, 0, -1].expand_as(result[0, 2, 0])
    assert torch.equal(result[0, 2, 0], expected)
    assert torch.equal(result[0, 0, 2], future[0, 0, 2])
    assert torch.equal(result[0, 1, 1], future[0, 1, 1])
    # The helper is functional: factual history and candidate future inputs are
    # not edited in place.
    _, future_again = _features()
    assert torch.equal(future, future_again)


def test_empty_mask_is_value_identical_and_none_is_same_object():
    history, future = _features()
    empty = torch.zeros((1, 3, 3), dtype=torch.bool)
    assert torch.equal(
        apply_directed_future_intent_hold(future, history, empty), future
    )
    assert apply_directed_future_intent_hold(future, history, None) is future


@pytest.mark.parametrize(
    "future_shape,history_shape,mask_shape",
    [
        ((1, 3, 3, 5), (1, 3, 3, 4, 2), (1, 3, 3)),
        ((1, 3, 3, 5, 2), (1, 3, 2, 4, 2), (1, 3, 3)),
        ((1, 3, 3, 5, 2), (1, 3, 3, 4, 2), (1, 3, 2)),
    ],
)
def test_future_hold_rejects_shape_contract_violations(
    future_shape, history_shape, mask_shape
):
    with pytest.raises(ValueError):
        apply_directed_future_intent_hold(
            torch.zeros(future_shape),
            torch.zeros(history_shape),
            torch.zeros(mask_shape, dtype=torch.bool),
        )
