from itertools import product
import pytest
import torch
from torch_cluster import grid_cluster

tests = [
    {
        'pos': [2, 6],
        'size': [5],
        'cluster': [0, 1],
    },
    {
        'pos': [1, 4, 9, 12, 16],
        'size': [5],
        'start': [0],
        'cluster': [0, 0, 1, 2, 3],
    }
]


if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("NPU not available", allow_module_level=True)

devices = [torch.device("npu:0")]
dtypes = [torch.float32]

@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_grid_cluster_npu_1d(test, dtype, device):

    # 1. tensor
    pos = torch.tensor(test['pos'], dtype=dtype, device=device)
    size = torch.tensor(test['size'], dtype=dtype, device=device)
    start = torch.tensor(test.get('start'), dtype=dtype, device=device) if test.get('start') else None
    end = torch.tensor(test.get('end'), dtype=dtype, device=device) if test.get('end') else None

    print("\n===== NPU 1D Grid Debug =====")
    print("pos:", pos)
    print("size:", size)
    print("start:", start)
    print("end:", end)

    # 2. run op
    cluster = grid_cluster(pos, size, start, end)

    # 3. sync
    torch.npu.synchronize()

    output = cluster.tolist()

    print("output:", output)
    print("expected:", test['cluster'])
    print("=============================\n")

    # 4. check
    assert output == test['cluster']

    # 5. JIT check (optional but useful)
    jit = torch.jit.script(grid_cluster)
    assert torch.equal(jit(pos, size, start, end), cluster)