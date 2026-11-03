import pytest
import torch
import torch_npu  # noqa: F401

from torch_cluster import knn


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize("k", [1, 2, 5])
def test_knn_matches_cpu_for_random_points(k):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(4, 3, generator=generator)
    y = torch.randn(6, 3, generator=generator)
    cpu = knn(x, y, k)
    npu = knn(x.npu(), y.npu(), k).cpu()
    assert torch.equal(npu, cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_knn_batch_and_empty_reference_batch():
    x = torch.tensor([[0.0, 0.0], [2.0, 0.0], [8.0, 0.0]])
    y = torch.tensor([[0.2, 0.0], [4.0, 0.0], [8.1, 0.0]])
    batch_x = torch.tensor([0, 0, 2])
    batch_y = torch.tensor([0, 1, 2])
    cpu = knn(x, y, 2, batch_x, batch_y, batch_size=3)
    npu = knn(x.npu(), y.npu(), 2, batch_x.npu(), batch_y.npu(),
              batch_size=3).cpu()
    assert torch.equal(npu, cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_knn_cosine():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    y = torch.tensor([[0.9, 0.1], [-0.9, 0.1]])
    result = knn(x.npu(), y.npu(), 1, cosine=True).cpu()
    assert result.tolist() == [[0, 1], [0, 2]]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_knn_maximum_k_and_short_reference():
    x = torch.arange(8, dtype=torch.float32).view(-1, 1)
    y = torch.tensor([[.25], [6.25]])
    assert torch.equal(knn(x, y, 100), knn(x.npu(), y.npu(), 100).cpu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize('kind', ['random', 'translated', 'mixed_scale', 'ties'])
def test_large_knn_distance_and_tie_safety(kind):
    generator = torch.Generator().manual_seed(311)
    x = torch.randn(512, 32, generator=generator)
    y = torch.randn(64, 32, generator=generator)
    if kind == 'translated':
        x, y = x + 100000, y + 100000
    elif kind == 'mixed_scale':
        x[0] = 100000
        y = x[1:65] + 0.001
    elif kind == 'ties':
        x.zero_()
        y.zero_()
    got = knn(x.npu(), y.npu(), 4).cpu()
    if kind == 'ties':
        # The scalar NPU contract breaks exact ties by reference index.
        assert torch.equal(got[1].view(-1, 4), torch.arange(4).expand(64, -1))
    else:
        oracle = ((y.double().unsqueeze(1) - x.double())**2).sum(2)
        expected_distance = oracle.sort(dim=1).values[:, :4]
        actual_distance = oracle.gather(1, got[1].view(-1, 4))
        assert torch.allclose(expected_distance, actual_distance,
                              rtol=1e-5, atol=1e-5)
