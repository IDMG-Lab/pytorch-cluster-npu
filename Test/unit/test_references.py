import torch
from fps_grid import fps_reference, grid_reference
from knn import reference as knn_reference
from nearest import reference as nearest_reference
from graclus import verify as verify_graclus
from radius import verify as verify_radius
import pytest


def test_fps_greedy_order_and_segment_offsets():
    pos = torch.tensor([[0.], [1.], [3.], [7.]])
    assert fps_reference(pos, torch.tensor([0, 4]), .75).tolist() == [0, 3, 2]
    assert fps_reference(pos, torch.tensor([0, 2, 4]), .5).tolist() == [0, 2]


def test_grid_truncation_for_negative_coordinates():
    pos = torch.tensor([[-.5, 1.9], [-1.5, -2.1]], dtype=torch.float64)
    result = grid_reference(pos, torch.ones(2), torch.zeros(2), torch.full((2,), 3.))
    assert result.tolist() == [4, -9]


def test_grid_wide_labels_keep_all_integer_bits():
    result = grid_reference(torch.ones(1, 64), torch.full((64,), .5),
                            torch.zeros(64), torch.ones(64))
    expected = (3**64-1) % 2**64
    if expected >= 2**63:
        expected -= 2**64
    assert result.tolist() == [expected]


def test_knn_graph_direction_and_self_exclusion():
    pos = torch.tensor([[0.], [1.], [5.]])
    assert knn_reference(pos, pos, None, None, 1, False).tolist() == [[0, 1, 2], [0, 1, 2]]
    assert knn_reference(pos, pos, None, None, 1, True).tolist() == [[1, 0, 1], [0, 1, 2]]


def test_nearest_exact_tie_uses_lowest_center_index():
    result = nearest_reference(torch.tensor([[0.]]), torch.tensor([[-1.], [1.]]), None, None)
    assert result.tolist() == [0]


def test_weighted_matching_rejects_ignoring_all_better_edges():
    row = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    col = torch.tensor([1, 2, 0, 3, 0, 3, 1, 2])
    weight = torch.tensor([1., 3., 1., 3., 3., 1., 3., 1.])
    verify_graclus(row, col, weight, torch.tensor([0, 1, 0, 1]), 4)
    with pytest.raises(AssertionError, match='greedy'):
        verify_graclus(row, col, weight, torch.tensor([0, 0, 2, 2]), 4)


def test_radius_allows_any_capped_subset_but_rejects_wrong_or_missing_edges():
    x = torch.tensor([[0.], [.1], [.2], [9.]])
    y = x[:1]
    verify_radius(x, y, torch.tensor([[0, 0], [1, 2]]), 1., 2, False)
    with pytest.raises(AssertionError, match='Nonexistent'):
        verify_radius(x, y, torch.tensor([[0], [3]]), 1., 2, False)
    with pytest.raises(AssertionError, match='Missing'):
        verify_radius(x, y, torch.tensor([[0], [1]]), 1., 2, False)
