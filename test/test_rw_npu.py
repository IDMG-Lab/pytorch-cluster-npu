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
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_regular_degree_validation_and_captured_draws(dtype):
    # Keep E/N unchanged while breaking regularity in the middle of the CSR.
    # Reusing the same storage must never reuse an earlier validation result.
    n, degree, count, length = 129, 8, 4097, 65
    gen = torch.Generator().manual_seed(128)
    col = torch.randint(n, (n * degree,), generator=gen, dtype=dtype)
    start = (torch.arange(count) % n).to(dtype)
    nrowptr = torch.empty(n + 1, dtype=dtype, device='npu')
    ncol, nstart = col.npu(), start.npu()
    for regular in (True, False, True):
        rowptr = torch.arange(n + 1, dtype=dtype) * degree
        if not regular:
            rowptr[64] += degree
        nrowptr.copy_(rowptr)
        torch.manual_seed(128)
        draws = torch.rand((count, length), device='npu').cpu()
        torch.manual_seed(128)
        nodes, edges = torch.ops.torch_cluster.random_walk(
            nrowptr, ncol, nstart, length, 1., 1.)
        nodes, edges = nodes.cpu(), edges.cpu()
        current = start.long()
        assert torch.equal(nodes[:, 0], start)
        for step in range(length):
            begin = rowptr[current].long()
            degrees = (rowptr[current + 1] - rowptr[current]).long()
            selected = begin + (draws[:, step] * degrees).long()
            selected = torch.where(degrees > 0, selected, -1)
            following = torch.where(degrees > 0, col[selected.clamp_min(0)].long(), current)
            assert torch.equal(edges[:, step].long(), selected)
            assert torch.equal(nodes[:, step + 1].long(), following)
            current = following


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_regular_degree_bounds_validation_not_cached():
    n, degree, count, length = 129, 8, 4097, 65
    rp = (torch.arange(n + 1, dtype=torch.int32) * degree).npu()
    gen = torch.Generator().manual_seed(319)
    good_col = torch.randint(n, (n * degree,), generator=gen, dtype=torch.int32)
    nc = good_col.npu()
    ns = (torch.arange(count, dtype=torch.int32) % n).npu()
    for invalid in (False, True, False):
        col = good_col.clone()
        start = torch.arange(count, dtype=torch.int32) % n
        if invalid:
            col[5] = n + 7
            start[:2] = torch.tensor([-1, n], dtype=torch.int32)
        nc.copy_(col)
        ns.copy_(start)
        torch.manual_seed(319)
        draws = torch.rand((count, length), device='npu').cpu()
        torch.manual_seed(319)
        nodes, edges = torch.ops.torch_cluster.random_walk(rp, nc, ns, length, 1., 1.)
        nodes, edges = nodes.cpu(), edges.cpu()
        current = start.long()
        for step in range(length):
            active = (current >= 0) & (current < n)
            selected = torch.where(active, current * degree + (draws[:, step] * degree).long(), -1)
            following = torch.where(active, col[selected.clamp_min(0)].long(), current)
            assert torch.equal(edges[:, step].long(), selected)
            assert torch.equal(nodes[:, step + 1].long(), following)
            current = following


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize('degree,length', [(8, 17), (16, 65)])
def test_large_regular_degree_tiles_against_captured_draws(degree, length):
    n, count = 8193, 16401
    gen = torch.Generator().manual_seed(711)
    col = torch.randint(n, (n * degree,), generator=gen, dtype=torch.int32)
    start = torch.arange(count, dtype=torch.int32) % n
    rp = (torch.arange(n + 1, dtype=torch.int32) * degree).npu()
    torch.manual_seed(711)
    draws = torch.rand((count, length), device='npu').cpu()
    torch.manual_seed(711)
    nodes, edges = torch.ops.torch_cluster.random_walk(rp, col.npu(), start.npu(), length, 1., 1.)
    nodes, edges = nodes.cpu(), edges.cpu()
    assert torch.equal(nodes[:, 0], start)
    current = start.long()
    for step in range(length):
        selected = current * degree + (draws[:, step] * degree).long()
        current = col[selected].long()
        assert torch.equal(edges[:, step].long(), selected)
        assert torch.equal(nodes[:, step + 1].long(), current)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize('concentrated,length,n,count', [
    (False, 17, 8192, 16401), (True, 65, 8192, 16401),
    (False, 33, 8192, 20480), (False, 65, 524288, 16401),
    (False, 17, 131072, 16401), (True, 17, 524288, 262144),
    ('coarse', 17, 524288, 262144), (False, 17, 524288, 262144)])
def test_sharded_regular_captured_draws_and_complete_queue(concentrated, length, n, count):
    degree = 8
    gen = torch.Generator().manual_seed(281)
    limit = 65536 if concentrated == 'coarse' else 4096 if concentrated else n
    col = torch.randint(limit, (n * degree,),
                        generator=gen, dtype=torch.int32)
    start = torch.zeros(count, dtype=torch.int32) if concentrated else torch.arange(count, dtype=torch.int32) % n
    rp = (torch.arange(n + 1, dtype=torch.int32) * degree).npu()
    torch.manual_seed(281)
    draws = torch.rand((count, length), device='npu').cpu()
    torch.manual_seed(281)
    nodes, edges = torch.ops.torch_cluster.random_walk(rp, col.npu(), start.npu(), length, 1., 1.)
    nodes, edges = nodes.cpu(), edges.cpu()
    assert torch.equal(nodes[:, 0], start)
    current = start.long()
    for step in range(length):
        selected = current * degree + (draws[:, step] * degree).long()
        current = col[selected].long()
        assert torch.equal(edges[:, step].long(), selected)
        assert torch.equal(nodes[:, step + 1].long(), current)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_sharded_layout_mutation_validation_and_safe_fallback():
    n, count, degree, length = 8192, 8193, 8, 17
    gen = torch.Generator().manual_seed(903)
    good_col = torch.randint(n, (n * degree,), generator=gen, dtype=torch.int32)
    nrp = torch.empty(n + 1, dtype=torch.int32, device='npu')
    nc = good_col.npu()
    ns = torch.empty(count, dtype=torch.int32, device='npu')
    for mode in ('valid', 'irregular', 'invalid_indices', 'valid'):
        rowptr = torch.arange(n + 1, dtype=torch.int32) * degree
        col = good_col.clone()
        start = torch.arange(count, dtype=torch.int32) % n
        if mode == 'irregular':
            rowptr[64] += degree
        if mode == 'invalid_indices':
            col[5] = n + 7
            start[:2] = torch.tensor([-1, n], dtype=torch.int32)
        nrp.copy_(rowptr); nc.copy_(col); ns.copy_(start)
        torch.manual_seed(903)
        draws = torch.rand((count, length), device='npu').cpu()
        torch.manual_seed(903)
        nodes, edges = torch.ops.torch_cluster.random_walk(nrp, nc, ns, length, 1., 1.)
        nodes, edges = nodes.cpu(), edges.cpu()
        current = start.long()
        assert torch.equal(nodes[:, 0], start)
        for step in range(length):
            active = (current >= 0) & (current < n)
            safe = current.clamp(0, n - 1)
            begin = rowptr[safe].long()
            degrees = (rowptr[safe + 1] - rowptr[safe]).long()
            selected = torch.where(active & (degrees > 0),
                                   begin + (draws[:, step] * degrees).long(), -1)
            current = torch.where(selected >= 0, col[selected.clamp_min(0)].long(), current)
            assert torch.equal(edges[:, step].long(), selected)
            assert torch.equal(nodes[:, step + 1].long(), current)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize('seed,offset', [(281, 36), ((1 << 64) - 1, (1 << 63) + 64)])
def test_step_major_rng_exact_state_and_changed_inputs(seed, offset):
    n, count, degree, length = 8192, 16401, 8, 64
    gen = torch.Generator().manual_seed(281)
    good_col = torch.randint(n, (n * degree,), generator=gen, dtype=torch.int32)
    nrp = torch.empty(n + 1, dtype=torch.int32, device='npu')
    nc = good_col.npu()
    ns = torch.empty(count, dtype=torch.int32, device='npu')
    npu_gen = torch.Generator(device='npu').manual_seed(seed)
    npu_gen.set_offset(offset)
    torch.npu.set_rng_state(npu_gen.get_state())
    for mode in ('valid', 'irregular', 'invalid_indices'):
        rowptr = torch.arange(n + 1, dtype=torch.int32) * degree
        col = good_col.clone()
        start = torch.arange(count, dtype=torch.int32) % n
        if mode == 'irregular':
            rowptr[64] += degree
        if mode == 'invalid_indices':
            col[5] = n + 7
            start[:2] = torch.tensor([-1, n], dtype=torch.int32)
        nrp.copy_(rowptr); nc.copy_(col); ns.copy_(start)
        before = torch.npu.get_rng_state().clone()
        draws = torch.rand((count, length), device='npu').cpu()
        expected_state = torch.npu.get_rng_state().clone()
        expected_next = torch.rand(19, device='npu').cpu()
        torch.npu.set_rng_state(before)
        nodes, edges = torch.ops.torch_cluster.random_walk(nrp, nc, ns, length, 1., 1.)
        assert torch.equal(torch.npu.get_rng_state(), expected_state)
        assert torch.equal(torch.rand(19, device='npu').cpu(), expected_next)
        nodes, edges = nodes.cpu(), edges.cpu()
        assert torch.equal(nodes[:, 0], start)
        current = start.long()
        for step in range(length):
            active = (current >= 0) & (current < n)
            safe = current.clamp(0, n - 1)
            begin = rowptr[safe].long()
            degrees = (rowptr[safe + 1] - rowptr[safe]).long()
            selected = torch.where(active & (degrees > 0),
                                   begin + (draws[:, step] * degrees).long(), -1)
            current = torch.where(selected >= 0, col[selected.clamp_min(0)].long(), current)
            assert torch.equal(edges[:, step].long(), selected)
            assert torch.equal(nodes[:, step + 1].long(), current)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_step_major_rng_graph_capture_uses_framework_state():
    n, count, degree, length = 8192, 8193, 8, 4
    gen = torch.Generator().manual_seed(137)
    col = torch.randint(n, (n * degree,), generator=gen, dtype=torch.int32)
    start = torch.arange(count, dtype=torch.int32) % n
    nrp = (torch.arange(n + 1, dtype=torch.int32) * degree).npu()
    nc, ns = col.npu(), start.npu()
    # Warm both native eager RNG and the framework capture fallback.
    for _ in range(2):
        torch.ops.torch_cluster.random_walk(nrp, nc, ns, length, 1., 1.)
        torch.rand((count, length), device='npu')
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        nodes, edges = torch.ops.torch_cluster.random_walk(nrp, nc, ns, length, 1., 1.)
    for _ in range(2):
        before = torch.npu.get_rng_state().clone()
        draws = torch.rand((count, length), device='npu').cpu()
        expected_state = torch.npu.get_rng_state().clone()
        torch.npu.set_rng_state(before)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(torch.npu.get_rng_state(), expected_state)
        walks, eids = nodes.cpu(), edges.cpu()
        current = start.long()
        for step in range(length):
            selected = current * degree + (draws[:, step] * degree).long()
            current = col[selected].long()
            assert torch.equal(eids[:, step].long(), selected)
            assert torch.equal(walks[:, step + 1].long(), current)


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
