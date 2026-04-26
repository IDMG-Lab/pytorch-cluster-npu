from itertools import product
import pytest
import torch
import time
from torch_cluster import grid_cluster

# ===== 测试用例（增强版）=====
tests = [
    {
        'name': 'mod_64',
        'pos': [i * 3 % 97 for i in range(64)],
        'size': [5],
        'start': [0],
        'cluster': [(i * 3 % 97) // 5 for i in range(64)],
    },
    {
        'name': 'range_64',
        'pos': list(range(64)),
        'size': [5],
        'start': [0],
        'cluster': [i // 5 for i in range(64)],
    },
    {
        'name': 'small_7',
        'pos': list(range(7)),
        'size': [3],
        'start': [0],
        'cluster': [i // 3 for i in range(7)],
    },
    {
        'name': 'unaligned_33',
        'pos': list(range(33)),
        'size': [4],
        'start': [0],
        'cluster': [i // 4 for i in range(33)],
    },
    {
        'name': 'cross_65',
        'pos': list(range(65)),
        'size': [5],
        'start': [0],
        'cluster': [i // 5 for i in range(65)],
    },
    {
        'name': 'random_50',
        'pos': [i * 7 % 53 for i in range(50)],
        'size': [6],
        'start': [0],
        'cluster': [(i * 7 % 53) // 6 for i in range(50)],
    },
]

# ===== 检查 NPU =====
if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("NPU not available", allow_module_level=True)

devices = [torch.device("npu:0"), torch.device("cpu")]
dtypes = [torch.float32]


# ===== 性能测试函数 =====
def benchmark(func, *args, repeat=50, device=None):
    # 预热
    for _ in range(10):
        func(*args)

    # 同步（仅 NPU）
    if device is not None and device.type == "npu":
        torch.npu.synchronize()

    start = time.time()

    for _ in range(repeat):
        out = func(*args)

    if device is not None and device.type == "npu":
        torch.npu.synchronize()

    end = time.time()

    return (end - start) / repeat


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_grid_cluster_npu_1d(test, dtype, device):

    print(f"\n===== Test: {test['name']} | Device: {device} =====")

    # ===== 1. tensor =====
    pos = torch.tensor(test['pos'], dtype=dtype, device=device)
    size = torch.tensor(test['size'], dtype=dtype, device=device)
    start = torch.tensor(test.get('start'), dtype=dtype, device=device) if test.get('start') else None
    end = torch.tensor(test.get('end'), dtype=dtype, device=device) if test.get('end') else None

    print("pos:", pos)
    print("size:", size)
    print("start:", start)
    print("end:", end)

    # ===== 2. run op =====
    cluster = grid_cluster(pos, size, start, end)

    # ===== 3. sync（仅 NPU）=====
    if device.type == "npu":
        torch.npu.synchronize()

    output = cluster.tolist()

    print("output  :", output)
    print("expected:", test['cluster'])
    print("=============================")

    # ===== 4. 正确性检查 =====
    assert output == test['cluster']

    # ===== 5. JIT 检查 =====
    jit = torch.jit.script(grid_cluster)
    assert torch.equal(jit(pos, size, start, end), cluster)

    # ===== 6. 性能测试 =====
    runtime = benchmark(grid_cluster, pos, size, start, end, device=device)

    print(f"[Benchmark] {device} time: {runtime * 1e6:.2f} us")

    # ===== 7. CPU vs NPU 对比（只在 NPU case 打印）=====
    if device.type == "npu":
        pos_cpu = pos.cpu()
        size_cpu = size.cpu()
        start_cpu = start.cpu() if start is not None else None

        cpu_time = benchmark(grid_cluster, pos_cpu, size_cpu, start_cpu, end, device=torch.device("cpu"))

        print(f"[Benchmark] CPU time: {cpu_time * 1e6:.2f} us")
        print(f"[Benchmark] Speedup: {cpu_time / runtime:.2f}x")

    print("\n")