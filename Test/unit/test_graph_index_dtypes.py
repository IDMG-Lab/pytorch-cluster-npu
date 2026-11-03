"""Real int32/int64 paths, cache-line ownership, and overflow-safe COO order."""
import pytest
import torch
import torch_npu  # noqa: F401
import torch_cluster


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('length', [0, 8, 1025])
def test_walk_dtype_and_output_tail(dtype, length):
    rp = torch.tensor([0, 1, 2, 2], dtype=dtype)
    col = torch.tensor([1, 0], dtype=dtype)
    start = (torch.arange(41) % 3).to(dtype)
    cpu = torch.ops.torch_cluster.random_walk(rp, col, start, length, 1., 1.)
    npu = torch.ops.torch_cluster.random_walk(
        rp.npu(), col.npu(), start.npu(), length, 1., 1.)
    assert all(t.dtype == dtype for t in npu)
    assert all(torch.equal(a, b.cpu()) for a, b in zip(cpu, npu))


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('p,q', [(1., 1.), (2., .5)])
def test_walk_sampling_and_repeatability(dtype, p, q):
    rp = torch.tensor([0, 2, 5, 7, 8], dtype=dtype).npu()
    col = torch.tensor([1, 2, 0, 2, 3, 0, 1, 1], dtype=dtype).npu()
    start = torch.zeros(32768, dtype=dtype).npu()
    torch.manual_seed(12345)
    nodes, edges = torch.ops.torch_cluster.random_walk(rp, col, start, 2, p, q)
    torch.manual_seed(12345)
    repeated = torch.ops.torch_cluster.random_walk(rp, col, start, 2, p, q)
    assert all(torch.equal(a, b) for a, b in zip((nodes, edges), repeated))
    nodes, edges = nodes.cpu().long(), edges.cpu().long()
    assert torch.equal(col.cpu().long()[edges], nodes[:, 1:])
    selected = nodes[nodes[:, 1] == 1, 2]
    observed = torch.tensor([(selected == i).float().mean() for i in (0, 2, 3)])
    expected = torch.tensor([1 / p, 1., 1 / q])
    expected /= expected.sum()
    assert torch.all((observed - expected).abs() < .02)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('shape', [(3, 7, 2), (2048, 66003, 1031),
                                   (65539, 1114163, 1031), (3, 0, 2)])
def test_subgraph_dtype_parallel_and_empty(dtype, shape):
    n, e, q = shape
    gen = torch.Generator().manual_seed(419)
    rp = (torch.arange(n + 1) * e // n).to(dtype)
    col = torch.randint(n, (e,), generator=gen).to(dtype)
    nodes = torch.randperm(n, generator=gen)[:q].to(dtype)
    expected = torch_cluster.subgraph(rp, col, nodes)
    actual = torch_cluster.subgraph(rp.npu(), col.npu(), nodes.npu())
    assert all(t.dtype == dtype for t in actual)
    assert all(torch.equal(a, b.cpu()) for a, b in zip(expected, actual))
    without_ids = torch_cluster.subgraph(
        rp.npu(), col.npu(), nodes.npu(), return_edge_id=False)
    assert len(without_ids) == 2
    assert all(torch.equal(a, b.cpu()) for a, b in zip(expected, without_ids))


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_subgraph_duplicates_and_node_order(dtype):
    rp = torch.tensor([0, 3, 3, 5], dtype=dtype)
    col = torch.tensor([0, 0, 2, 0, 2], dtype=dtype)
    nodes = torch.tensor([2, 0], dtype=dtype)
    expected = ([0, 2, 5], [1, 0, 1, 1, 0], [3, 4, 0, 1, 2])
    for device in ('cpu', 'npu'):
        result = torch_cluster.subgraph(rp.to(device), col.to(device), nodes.to(device))
        assert tuple(t.cpu().tolist() for t in result) == expected


@pytest.mark.parametrize('device', ['cpu', 'npu'])
def test_int32_coo_sort_does_not_overflow(device):
    row = torch.tensor([65537, 0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([1, 65537, 0], dtype=torch.int32, device=device)
    start = torch.tensor([0, 1, 65537], dtype=torch.int32, device=device)
    result = torch_cluster.random_walk(row, col, start, 2, num_nodes=70000)
    assert result.cpu().tolist() == [[0, 65537, 1], [1, 0, 65537], [65537, 1, 0]]
    assert result.dtype == torch.int32


@pytest.mark.parametrize('device', ['cpu', 'npu'])
def test_mixed_graph_index_dtypes_rejected(device):
    rp = torch.tensor([0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0], dtype=torch.int64, device=device)
    nodes = torch.tensor([0], dtype=torch.int32, device=device)
    with pytest.raises(TypeError, match='same int32 or int64'):
        torch_cluster.subgraph(rp, col, nodes)
    with pytest.raises(RuntimeError, match='same int32 or int64'):
        torch.ops.torch_cluster.random_walk(rp, col, nodes, 1, 1., 1.)
