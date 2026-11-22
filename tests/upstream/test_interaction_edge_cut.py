import torch

from ccdiff.models.scenetemporal import apply_directed_interaction_edge_cut


def test_multiple_directed_edges_do_not_mask_reverse_edges():
    neighbor_mask = torch.zeros((1, 5, 5, 4), dtype=torch.bool)
    edge_cut = torch.zeros((1, 5, 5), dtype=torch.bool)
    edge_cut[:, 3, 0] = True
    edge_cut[:, 4, 0] = True

    result = apply_directed_interaction_edge_cut(neighbor_mask, edge_cut)

    assert bool(result[0, 3, 0].all())
    assert bool(result[0, 4, 0].all())
    assert not bool(result[0, 0, 3].any())
    assert not bool(result[0, 0, 4].any())
    assert int(result.sum()) == 8
