import pytest
import torch
import torch_npu  # noqa: F401

from torch_cluster import random_walk


def test_zero_step_cpu_and_empty_edges():
    row = torch.empty(0, dtype=torch.long)
    col = torch.empty(0, dtype=torch.long)
    start = torch.tensor([0, 2], dtype=torch.long)
    nodes, edges = random_walk(row, col, start, walk_length=0,
                               return_edge_indices=True)
    assert nodes.tolist() == [[0], [2]]
    assert tuple(edges.shape) == (2, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize("length,p,q", [(0, 1.0, 1.0), (4, 1.0, 1.0),
                                         (4, 2.0, 0.5)])
def test_degree_one_and_isolated_match_cpu(length, p, q):
    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1, 0], dtype=torch.long)
    start = torch.tensor([0, 1, 2], dtype=torch.long)
    cpu = random_walk(row, col, start, length, p=p, q=q,
                      num_nodes=3, return_edge_indices=True)
    npu = random_walk(row.npu(), col.npu(), start.npu(), length,
                      p=p, q=q, num_nodes=3, return_edge_indices=True)
    assert all(torch.equal(a, b.cpu()) for a, b in zip(cpu, npu))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize("p,q", [(1.0, 1.0), (2.0, 0.5)])
def test_random_paths_use_real_edges(p, q):
    row = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)
    start = torch.randint(3, (64,), generator=torch.Generator().manual_seed(7))
    nodes, edges = random_walk(row.npu(), col.npu(), start.npu(), 8,
                               p=p, q=q, num_nodes=3,
                               return_edge_indices=True)
    nodes, edges = nodes.cpu(), edges.cpu()
    assert tuple(nodes.shape) == (64, 9)
    assert tuple(edges.shape) == (64, 8)
    assert torch.equal(nodes[:, 0], start)
    for walk in range(64):
        for step in range(8):
            eid = int(edges[walk, step])
            assert 0 <= eid < col.numel()
            assert int(row[eid]) == int(nodes[walk, step])
            assert int(col[eid]) == int(nodes[walk, step + 1])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize('p,q', [(1., 1.), (2., .5)])
@pytest.mark.parametrize('length', [2, 8])
def test_node2vec_transition_distribution_and_seed(p, q, length):
    # Given first transition 0 -> 1, the second-step candidates 0, 2, 3
    # have return/near/far weights 1/p, 1, 1/q respectively.
    rowptr = torch.tensor([0, 2, 5, 7, 8], dtype=torch.long).npu()
    col = torch.tensor([1, 2, 0, 2, 3, 0, 1, 1], dtype=torch.long).npu()
    start = torch.zeros(32768, dtype=torch.long).npu()
    torch.manual_seed(12345)
    result = torch.ops.torch_cluster.random_walk(rowptr, col, start, length, p, q)
    torch.manual_seed(12345)
    repeated = torch.ops.torch_cluster.random_walk(rowptr, col, start, length, p, q)
    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(result, repeated))
    nodes = result[0].cpu()
    next_node = nodes[nodes[:, 1] == 1, 2]
    observed = torch.tensor([(next_node == value).float().mean()
                             for value in (0, 2, 3)])
    expected = torch.tensor([1 / p, 1., 1 / q])
    expected /= expected.sum()
    assert torch.all((observed - expected).abs() < .02)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize('length', [8, 1024, 1025])
def test_buffered_output_boundary_and_core_tail(length):
    rowptr = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    col = torch.tensor([1, 0], dtype=torch.long)
    start = torch.arange(41, dtype=torch.long) % 3
    expected = torch.ops.torch_cluster.random_walk(
        rowptr, col, start, length, 2., .5)
    actual = torch.ops.torch_cluster.random_walk(
        rowptr.npu(), col.npu(), start.npu(), length, 2., .5)
    assert all(torch.equal(a, b.cpu()) for a, b in zip(expected, actual))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_degree_one_validation_and_changed_csr_fallback():
    n = 129
    generator = torch.Generator().manual_seed(2026)
    col = torch.randint(n, (n,), generator=generator)
    start = torch.arange(4097, dtype=torch.long) % n
    nrowptr = torch.empty(n + 1, dtype=torch.long).npu()
    ncol, nstart = col.npu(), start.npu()
    for degree_one in (True, False, True):
        rowptr = torch.arange(n + 1, dtype=torch.long)
        if not degree_one:
            # Same E==N shape: row 0 has degree 2, row 1 has degree 0.
            rowptr[1] = 2
        nrowptr.copy_(rowptr)
        torch.manual_seed(84)
        actual = torch.ops.torch_cluster.random_walk(
            nrowptr, ncol, nstart, 8, 2., .5)
        nodes, edges = (t.cpu() for t in actual)
        if degree_one:
            expected = torch.ops.torch_cluster.random_walk(
                rowptr, col, start, 8, 2., .5)
            assert all(torch.equal(a, b) for a, b in zip(expected, (nodes, edges)))
        else:
            current, following = nodes[:, :-1], nodes[:, 1:]
            valid = edges >= 0
            assert torch.all(edges[valid] >= rowptr[current[valid]])
            assert torch.all(edges[valid] < rowptr[current[valid] + 1])
            assert torch.equal(col[edges[valid]], following[valid])
            assert torch.equal(current[~valid], following[~valid])
            assert torch.all(rowptr[current[~valid]] == rowptr[current[~valid] + 1])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize('degree', [128, 129])
@pytest.mark.parametrize('length', [2, 8])
def test_biased_cdf_against_captured_draws(degree, length):
    counts = torch.tensor([1, degree] + [1] * (degree - 1))
    rowptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    candidates = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(2, degree + 1)))
    col = torch.cat((torch.ones(1, dtype=torch.long), candidates,
                     torch.ones(degree - 1, dtype=torch.long)))
    start = torch.zeros(2048, dtype=torch.long).npu()
    nrowptr, ncol = rowptr.npu(), col.npu()
    torch.manual_seed(1026)
    draws = torch.rand((2048, length), device=start.device).cpu()
    torch.manual_seed(1026)
    nodes, edges = torch.ops.torch_cluster.random_walk(
        nrowptr, ncol, start, length, 2., .5)
    nodes, edges = nodes.cpu(), edges.cpu()
    weights = torch.full((degree,), 2.)
    weights[0] = .5
    offsets = torch.searchsorted(weights.cumsum(0), draws[:, 1] * weights.sum(), right=True)
    offsets.clamp_(max=degree - 1)
    assert torch.all(nodes[:, 0] == 0)
    assert torch.all(nodes[:, 1] == 1)
    assert torch.all(edges[:, 0] == 0)
    assert torch.equal(nodes[:, 2], candidates[offsets])
    assert torch.equal(edges[:, 1], offsets + 1)
