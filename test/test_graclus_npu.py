import pytest
import torch
import torch_npu  # noqa: F401

import torch_cluster  # noqa: F401; registers low-level operators


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize("weighted", [False, True])
def test_graclus_matches_seeded_cpu_permutation(weighted):
    rowptr = torch.tensor([0, 2, 4, 7, 9, 10], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 2, 0, 1, 3, 2, 4, 3], dtype=torch.long)
    weight = (torch.tensor([2, 1, 2, 3, 1, 3, 4, 4, 5, 5], dtype=torch.float)
              if weighted else None)
    torch.manual_seed(2026)
    cpu = torch.ops.torch_cluster.graclus(rowptr, col, weight)
    torch.manual_seed(2026)
    npu = torch.ops.torch_cluster.graclus(
        rowptr.npu(), col.npu(), weight.npu() if weighted else None).cpu()
    assert torch.equal(npu, cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize("num_nodes", [0, 1, 100])
def test_graclus_isolated_nodes(num_nodes):
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    col = torch.empty(0, dtype=torch.long)
    torch.manual_seed(33)
    cpu = torch.ops.torch_cluster.graclus(rowptr, col, None)
    torch.manual_seed(33)
    npu = torch.ops.torch_cluster.graclus(rowptr.npu(), col.npu(), None).cpu()
    assert torch.equal(npu, cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize('values', [
    [1.] * 10,
    [-1., 0., 0., -2., 0., 0., -1., 0., 0., -3.],
    [float('nan'), 1., float('inf'), 2., 1., float('nan'),
     float('inf'), float('inf'), 0., 0.],
])
def test_weight_pruning_preserves_ties_and_nonfinite_values(values):
    rowptr = torch.tensor([0, 3, 5, 7, 9, 10], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 0, 2, 0, 1, 2, 4, 3], dtype=torch.long)
    weight = torch.tensor(values, dtype=torch.float32)
    for seed in (23, 67):
        torch.manual_seed(seed)
        expected = torch.ops.torch_cluster.graclus(rowptr, col, weight)
        torch.manual_seed(seed)
        actual = torch.ops.torch_cluster.graclus(
            rowptr.npu(), col.npu(), weight.npu()).cpu()
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_weight_tiles_preserve_order_and_tail():
    n = 1053
    rowptr = torch.arange(n + 1, dtype=torch.long) * (n - 1)
    col = torch.cat([torch.cat((torch.arange(u), torch.arange(u + 1, n)))
                     for u in range(n)])
    weight = torch.ones(col.numel(), dtype=torch.float32)
    weight[::101] = float('nan')
    weight[::137] = -1.
    weight[::211] = float('inf')
    torch.manual_seed(67)
    expected = torch.ops.torch_cluster.graclus(rowptr, col, weight)
    torch.manual_seed(67)
    actual = torch.ops.torch_cluster.graclus(
        rowptr.npu(), col.npu(), weight.npu()).cpu()
    assert torch.equal(actual, expected)
