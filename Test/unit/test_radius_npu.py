"""NPU query, compaction, batch and output-tail correctness."""
import pytest
import torch
import torch_npu  # noqa: F401
from torch_cluster import radius

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason='NPU unavailable')


def check_edges(x, y, r, cap, edges, bx=None, by=None, ignore=False):
    assert edges.device.type == 'npu'
    assert edges.dtype == torch.int64 and edges.is_contiguous()
    edges = edges.cpu()
    distance = ((y.double()[:, None, :] - x.double()[None, :, :]) ** 2).sum(-1)
    valid = distance <= r * r
    if bx is not None:
        valid &= by[:, None] == bx[None, :]
    if ignore:
        valid &= torch.arange(y.size(0))[:, None] != torch.arange(x.size(0))[None, :]
    counts = torch.bincount(edges[0], minlength=y.size(0))
    assert torch.equal(counts, valid.sum(1).clamp(max=cap))
    assert torch.all(valid[edges[0], edges[1]])
    assert torch.unique(edges[0] * x.size(0) + edges[1]).numel() == edges.size(1)


@pytest.mark.parametrize('dims', [1, 3, 6])
@pytest.mark.parametrize('cap', [1, 7, 32])
def test_independent_queries_and_compaction_tail(dims, cap):
    gen = torch.Generator().manual_seed(8192 + dims)
    x = torch.randn(257, dims, generator=gen)
    y = torch.randn(131, dims, generator=gen)
    r = .71
    result = radius(x.npu(), y.npu(), r, max_num_neighbors=cap)
    check_edges(x, y, r, cap, result)


def test_batches_with_empty_segments():
    gen = torch.Generator().manual_seed(98)
    x = torch.randn(101, 3, generator=gen)
    y = torch.randn(53, 3, generator=gen)
    bx = torch.cat((torch.zeros(51, dtype=torch.long), torch.full((50,), 2)))
    by = torch.cat((torch.zeros(17, dtype=torch.long), torch.ones(19, dtype=torch.long),
                    torch.full((17,), 2)))
    result = radius(x.npu(), y.npu(), .71, bx.npu(), by.npu(), max_num_neighbors=7, batch_size=3)
    check_edges(x, y, .71, 7, result, bx, by)


@pytest.mark.parametrize('ignore', [False, True])
def test_dense_same_input_and_self_filter(ignore):
    x = torch.zeros(1025, 3)
    nx = x.npu()
    result = radius(nx, nx, 1., max_num_neighbors=32, ignore_same_index=ignore)
    check_edges(x, x, 1., 32, result, ignore=ignore)


def test_inclusive_distance_boundary_and_empty_output():
    x = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    y = torch.tensor([[0., 0., 0.], [10., 0., 0.]])
    result = radius(x.npu(), y.npu(), 1.)
    check_edges(x, y, 1., 32, result)
    result = radius(x.npu(), y[1:].npu(), 1.)
    assert result.shape == (2, 0)


def test_equal_storage_validation_and_changed_queries():
    gen = torch.Generator().manual_seed(913)
    x = .02 * torch.randn(257, 3, generator=gen)
    nx, ny = x.npu(), x.npu()
    for equal in (True, False, True):
        y = x.clone()
        if not equal:
            y[73] += 10
        ny.copy_(y)
        result = radius(nx, ny, .71, max_num_neighbors=7)
        check_edges(x, y, .71, 7, result)


@pytest.mark.parametrize('dtype', [torch.float16, torch.int32])
def test_other_native_dtypes(dtype):
    x = (torch.arange(257) % 5).to(dtype).view(-1, 1)
    y = (torch.arange(131) % 5).to(dtype).view(-1, 1)
    result = radius(x.npu(), y.npu(), 1.01, max_num_neighbors=7)
    check_edges(x, y, 1.01, 7, result)
