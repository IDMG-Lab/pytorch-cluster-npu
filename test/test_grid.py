from itertools import product

import pytest
import torch
from torch_cluster import grid_cluster
from torch_cluster.testing import devices, dtypes, tensor

tests = [{
    'pos': [2, 6],
    'size': [5],
    'cluster': [0, 0],
}, {
    'pos': [2, 6],
    'size': [5],
    'start': [0],
    'cluster': [0, 1],
}, {
    'pos': [[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]],
    'size': [5, 5],
    'cluster': [0, 5, 3, 0, 1],
}, {
    'pos': [[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]],
    'size': [5, 5],
    'end': [19, 19],
    'cluster': [0, 6, 4, 0, 1],
}]

devices = [torch.device('cpu')]

if hasattr(torch, "npu") and torch.npu.is_available():
    devices.append(torch.device('npu:0'))

@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_grid_cluster(test, dtype, device):
    if dtype == torch.bfloat16 and device == torch.device('cuda:0'):
        return

    pos = tensor(test['pos'], dtype, device)
    size = tensor(test['size'], dtype, device)
    start = tensor(test.get('start'), dtype, device)
    end = tensor(test.get('end'), dtype, device)

    cluster = grid_cluster(pos, size, start, end)

    print("\n===== Grid Debug =====")
    print("devices:", devices)
    print(f"Device: {device}, dtype: {dtype}")
    print("pos:", pos)
    print("size:", size)
    print("start:", start)
    print("end:", end)
    print("output:", cluster)
    print("expected:", test['cluster'])
    print("output device:", cluster.device)
    print("=====================\n")

    assert cluster.tolist() == test['cluster']

    jit = torch.jit.script(grid_cluster)
    assert torch.equal(jit(pos, size, start, end), cluster)
