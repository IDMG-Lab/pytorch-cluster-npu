import pytest
import torch
import torch_npu
from torch_cluster import subgraph, subgraph_950, subgraph_910b


def test_backend_operator_names_coexist():
    assert torch.ops.torch_cluster.subgraph_950.default._schema.name == 'torch_cluster::subgraph_950'
    assert torch.ops.torch_cluster.subgraph_910b.default._schema.name == 'torch_cluster::subgraph_910b'


@pytest.mark.skipif(not torch.npu.is_available(), reason='NPU unavailable')
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('edge_id', [True, False])
def test_explicit_950_and_automatic_backend(dtype, edge_id):
    if '950' not in torch.npu.get_device_name():
        pytest.skip('950 hardware required')
    cpu = tuple(torch.tensor(x, dtype=dtype) for x in
                ([0, 3, 3, 5], [0, 0, 2, 0, 2], [2, 0]))
    inputs = tuple(x.npu() for x in cpu)
    expected = subgraph(*cpu, return_edge_id=edge_id)
    for op in (subgraph_950, subgraph):
        result = op(*inputs, return_edge_id=edge_id)
        assert len(result) == len(expected)
        assert all(torch.equal(a.cpu(), b) for a, b in zip(result, expected))
    with pytest.raises(RuntimeError, match='requires an Ascend910B'):
        subgraph_910b(*inputs)


def test_950_interface_rejects_cpu_inputs():
    with pytest.raises(RuntimeError, match='NPU'):
        subgraph_950(torch.tensor([0]), torch.empty(0, dtype=torch.long),
                     torch.empty(0, dtype=torch.long))


@pytest.mark.skipif(not torch.npu.is_available(), reason='NPU unavailable')
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_950_hubs_partial_warp_and_nondefault_stream(dtype):
    if '950' not in torch.npu.get_device_name():
        pytest.skip('950 hardware required')
    degrees = torch.full((97,), 2, dtype=dtype)
    degrees[3], degrees[35], degrees[70] = 257, 513, 129
    rowptr = torch.cat((torch.zeros(1, dtype=dtype), degrees.cumsum(0).to(dtype)))
    generator = torch.Generator().manual_seed(950)
    nodes = torch.randperm(97, generator=generator).to(dtype)
    for stream in (torch.npu.default_stream(), torch.npu.Stream(), torch.npu.Stream()):
        with torch.npu.stream(stream):
            for iteration in range(3):
                col = torch.randint(97, (int(rowptr[-1]),), generator=generator, dtype=dtype)
                current_nodes = nodes.roll(iteration * 31)
                expected = subgraph(rowptr, col, current_nodes)
                inputs = tuple(t.npu() for t in (rowptr, col, current_nodes))
                actual = subgraph_950(*inputs)
                for index, (a, b) in enumerate(zip(actual, expected)):
                    value = a.cpu()
                    assert torch.equal(value, b), (str(stream), iteration, index,
                                                  value[:100], b[:100])
