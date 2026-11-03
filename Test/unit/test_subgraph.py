import pytest
import torch
import torch_npu  # noqa: F401; registers the PrivateUse1 NPU backend

from torch_cluster import subgraph


def reference(rowptr, col, nodes):
    """Independent edge-list reference for the CSR subset contract."""
    rows = torch.repeat_interleave(
        torch.arange(rowptr.numel() - 1), rowptr[1:] - rowptr[:-1])
    mapping = {int(node): i for i, node in enumerate(nodes.tolist())}
    kept = [(mapping[int(src)], mapping[int(dst)], eid)
            for eid, (src, dst) in enumerate(zip(rows.tolist(), col.tolist()))
            if int(src) in mapping and int(dst) in mapping]
    out_rowptr = [0]
    out_col = []
    out_eid = []
    for src in range(nodes.numel()):
        for new_src, new_dst, eid in kept:
            if new_src == src:
                out_col.append(new_dst)
                out_eid.append(eid)
        out_rowptr.append(len(out_col))
    return out_rowptr, out_col, out_eid


@pytest.mark.parametrize("rowptr,col,nodes", [
    ([0, 2, 4, 6], [1, 2, 0, 2, 0, 1], [2, 0]),
    ([0, 3, 3, 5], [0, 0, 2, 0, 2], [2, 0]),
    ([0, 0, 0, 0], [], [2, 0]),
    ([0], [], []),
    ([0, 67, 67], [0] * 67, [0]),
])
def test_subgraph_cpu(rowptr, col, nodes):
    rp = torch.tensor(rowptr, dtype=torch.long)
    ci = torch.tensor(col, dtype=torch.long)
    nd = torch.tensor(nodes, dtype=torch.long)
    result = subgraph(rp, ci, nd)
    expected = reference(rp, ci, nd)
    assert tuple(t.tolist() for t in result) == expected
    assert len(subgraph(rp, ci, nd, return_edge_id=False)) == 2


def test_subgraph_invalid_cpu():
    rowptr = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([0], dtype=torch.long)
    with pytest.raises(RuntimeError, match="unique"):
        subgraph(rowptr, col, torch.tensor([0, 0]))
    with pytest.raises(RuntimeError, match="invalid CSR"):
        subgraph(torch.tensor([1, 1]), torch.empty(0, dtype=torch.long),
                 torch.tensor([0]))


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(),
                    reason="NPU unavailable")
def test_subgraph_npu():
    rowptr = torch.tensor([0, 3, 5, 7, 9], dtype=torch.long)
    col = torch.tensor([1, 2, 2, 0, 3, 0, 2, 1, 3], dtype=torch.long)
    nodes = torch.tensor([3, 1, 2], dtype=torch.long)
    expected = reference(rowptr, col, nodes)
    result = subgraph(rowptr.npu(), col.npu(), nodes.npu())
    assert tuple(t.cpu().tolist() for t in result) == expected


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize('skewed', [False, True])
def test_parallel_subgraph_cache_boundaries_and_reuse(skewed):
    from torch_cluster import subgraph
    generator = torch.Generator().manual_seed(419)
    n, e, q = 2048, 66003, 1031
    if skewed:
        # A single row crosses many output-core boundaries; remaining rows
        # are empty. Count and output sizes are not cache-line multiples.
        rowptr = torch.cat((torch.zeros(1, dtype=torch.long),
                            torch.full((n,), e, dtype=torch.long)))
    else:
        rowptr = torch.arange(n + 1) * e // n
    nodes = torch.cat((torch.zeros(1, dtype=torch.long),
                       torch.randperm(n - 1, generator=generator)[:q - 1] + 1))
    nodes = nodes[torch.randperm(q, generator=generator)]
    nrowptr, nnodes = rowptr.npu(), nodes.npu()
    ncol = torch.empty(e, dtype=torch.long).npu()
    for _ in range(3):
        col = torch.randint(n, (e,), generator=generator)
        ncol.copy_(col)
        expected = subgraph(rowptr, col, nodes)
        actual = subgraph(nrowptr, ncol, nnodes)
        assert all(torch.equal(a, b.cpu()) for a, b in zip(expected, actual))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_parallel_mapping_owner_boundaries_and_reuse():
    generator = torch.Generator().manual_seed(20261001)
    n, q = 65539, 1031
    e = n * 17
    rowptr = torch.arange(n + 1, dtype=torch.long) * 17
    protected = torch.tensor([0, 15, 16, n - 1], dtype=torch.long)
    extra = torch.randperm(n, generator=generator)
    extra = extra[~torch.isin(extra, protected)]
    nodes = torch.cat((protected, extra[:q - protected.numel()]))
    nrowptr = rowptr.npu()
    ncol = torch.empty(e, dtype=torch.long).npu()
    nnodes = torch.empty(q, dtype=torch.long).npu()
    for iteration in range(3):
        col = torch.randint(n, (e,), generator=generator)
        col[:17] = protected[torch.arange(17) % protected.numel()]
        current_nodes = nodes.roll(iteration * 31)
        ncol.copy_(col)
        nnodes.copy_(current_nodes)
        expected = subgraph(rowptr, col, current_nodes)
        actual = subgraph(nrowptr, ncol, nnodes)
        assert all(torch.equal(a, b.cpu()) for a, b in zip(expected, actual))
