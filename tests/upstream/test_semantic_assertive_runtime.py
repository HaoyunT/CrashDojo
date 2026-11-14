import torch

from ccdiff.models.scenetemporal import apply_semantic_assertive_token


def test_assertive_token_is_exact_identity_without_receiver_mask():
    embeddings = torch.randn(5, 2, 4, 32)
    assert apply_semantic_assertive_token(embeddings, None) is embeddings


def test_assertive_token_changes_only_selected_receiver():
    embeddings = torch.zeros(5, 2, 4, 32)
    receiver_mask = torch.zeros(2, 4, dtype=torch.bool)
    receiver_mask[0, 2] = True
    conditioned = apply_semantic_assertive_token(
        embeddings, receiver_mask, scale=1.0
    )
    delta = conditioned - embeddings
    assert torch.isfinite(conditioned).all()
    assert torch.count_nonzero(delta[:, 0, 2]).item() > 0
    assert torch.count_nonzero(delta[:, 0, :2]).item() == 0
    assert torch.count_nonzero(delta[:, 0, 3:]).item() == 0
    assert torch.count_nonzero(delta[:, 1]).item() == 0


def test_assertive_token_scale_is_linear():
    embeddings = torch.zeros(3, 1, 2, 16)
    receiver_mask = torch.tensor([[False, True]])
    unit = apply_semantic_assertive_token(embeddings, receiver_mask, scale=1.0)
    half = apply_semantic_assertive_token(embeddings, receiver_mask, scale=0.5)
    torch.testing.assert_close(half, unit * 0.5, rtol=0.0, atol=0.0)
