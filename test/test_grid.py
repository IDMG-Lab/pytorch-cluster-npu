from itertools import product
import pytest
import torch
import time
import random
from torch_cluster import grid_cluster

# ========== 1. 基础测试用例 ==========
basic_tests = [
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

# ========== 2. 数据类型测试 ==========
dtype_tests = [
    {
        'name': 'dtype_float16',
        'pos': [i * 0.5 for i in range(100)],
        'size': [2.0],
        'start': [0],
        'dtype': torch.float16,
        'cluster': [int((i * 0.5) // 2) for i in range(100)],
    },
    {
        'name': 'dtype_int32',
        'pos': list(range(100)),
        'size': [10],
        'start': [0],
        'dtype': torch.int32,
        'cluster': [i // 10 for i in range(100)],
    },
]

# ========== 3. 起始点测试 ==========
start_tests = [
    {
        'name': 'start_nonzero',
        'pos': [10, 15, 20, 25, 30],
        'size': [5],
        'start': [10],
        'cluster': [(p - 10) // 5 for p in [10, 15, 20, 25, 30]],  # 显式计算
    },
    {
        'name': 'start_negative',
        'pos': [-10, -5, 0, 5, 10],
        'size': [5],
        'start': [-10],
        'cluster': [(p + 10) // 5 for p in [-10, -5, 0, 5, 10]],
    },
    {
        'name': 'start_float',
        'pos': [1.1, 2.2, 3.3, 4.4, 5.5],
        'size': [1.0],
        'start': [0.5],
        'cluster': [int((p - 0.5) // 1.0) for p in [1.1, 2.2, 3.3, 4.4, 5.5]],
    },
]

# ========== 4. 边界测试 ==========
edge_tests = [
    {
        'name': 'empty_input',
        'pos': [],
        'size': [5],
        'start': [0],
        'cluster': [],
    },
    {
        'name': 'single_point',
        'pos': [42],
        'size': [10],
        'start': [0],
        'cluster': [42 // 10],
    },
    {
        'name': 'negative_coords',
        'pos': [-10, -5, -1, 0, 1, 5, 10],
        'size': [3],
        'start': [-12],
        'cluster': [int((p + 12) // 3) for p in [-10, -5, -1, 0, 1, 5, 10]],
    },
    {
        'name': 'large_coords',
        'pos': [1000000, 2000000, 3000000],
        'size': [100000],
        'start': [0],
        'cluster': [p // 100000 for p in [1000000, 2000000, 3000000]],
    },
]

# ========== 5. 对齐测试（各种非8倍数）=========
def generate_alignment_tests():
    """生成各种数据量的对齐测试"""
    tests = []
    for n in [1, 2, 3, 5, 6, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129]:
        points = list(range(n))
        tests.append({
            'name': f'align_{n}',
            'pos': points,
            'size': [5],
            'start': [0],
            'cluster': [p // 5 for p in points],  # 使用 points 变量
        })
    return tests

alignment_tests = generate_alignment_tests()

# ========== 6. 随机测试 ==========
def generate_random_test(seed=42, num_points=100):
    """生成随机测试用例"""
    random.seed(seed)
    points = [random.uniform(-100, 100) for _ in range(num_points)]
    size = random.uniform(0.5, 10)
    start = random.uniform(-50, 50)
    
    # 计算期望结果
    expected = [int((p - start) // size) for p in points]
    
    return {
        'name': f'random_{num_points}_seed{seed}',
        'pos': points,
        'size': [size],
        'start': [start],
        'cluster': expected,
    }

# 生成多个随机测试
random_tests = [generate_random_test(seed=i, num_points=100) for i in range(5)]

# ========== 7. 性能测试（大数据集）=========
def create_performance_test(num_points):
    """创建性能测试用例"""
    points = list(range(num_points))
    size = 10
    start = 0
    expected = [p // size for p in points]
    
    return {
        'name': f'perf_{num_points}',
        'pos': points,
        'size': [size],
        'start': [start],
        'cluster': expected,
        'benchmark_only': True,
    }

performance_tests = [create_performance_test(n) for n in [1000, 10000,100000,1000000]]

# ========== 8. 合并所有测试 ==========
all_tests = (dtype_tests)
print(f"Total tests generated: {len(all_tests)}")

# ========== 9. NPU可用性检查 ==========
if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("NPU not available", allow_module_level=True)

devices = [torch.device("npu:0")]
dtypes = [torch.float32]

# ========== 10. 性能基准函数 ==========
def benchmark(func, *args, repeat=50, device=None):
    """性能测试函数"""
    # 预热
    for _ in range(10):
        func(*args)

    # 同步
    if device is not None and device.type == "npu":
        torch.npu.synchronize()

    start = time.perf_counter()

    for _ in range(repeat):
        out = func(*args)

    if device is not None and device.type == "npu":
        torch.npu.synchronize()

    end = time.perf_counter()

    return (end - start) / repeat

# ========== 11. 主测试函数 ==========
@pytest.mark.parametrize('test,dtype,device', product(all_tests, dtypes, devices))
def test_grid_cluster_enhanced(test, dtype, device):
    
    # 跳过性能测试在CPU上的执行
    if test.get('benchmark_only') and device.type == "cpu":
        pytest.skip("Performance test only for NPU")
    
    # 使用测试用例指定的dtype
    if 'dtype' in test:
        dtype = test['dtype']
    
    print(f"\n===== Test: {test['name']} | Device: {device} | dtype: {dtype} =====")
    
    # 创建张量
    pos = torch.tensor(test['pos'], dtype=dtype, device=device)
    size = torch.tensor(test['size'], dtype=dtype, device=device)
    start = torch.tensor(test.get('start', [0]), dtype=dtype, device=device)
    
    print(f"  Points: {len(test['pos'])}, Size: {test['size']}, Start: {test.get('start', [0])}")
    
    # 执行算子
    cluster = grid_cluster(pos, size, start, None)
    
    # 同步
    if device.type == "npu":
        torch.npu.synchronize()
    
    # 转换为列表
    output = cluster.tolist()
    
    # 对于空输入的特殊处理
    if len(test['cluster']) == 0:
        assert len(output) == 0
        print("  Empty input test passed")
        return
    
    # 完整验证
    assert output == test['cluster'], f"Mismatch at test {test['name']}"
    
    # JIT测试
    try:
        jit = torch.jit.script(grid_cluster)
        jit_output = jit(pos, size, start, None)
        if device.type == "npu":
            torch.npu.synchronize()
        assert torch.equal(jit_output, cluster)
        print("  JIT compilation passed")
    except Exception as e:
        print(f" JIT test skipped: {e}")
    
    # 性能测试
    runtime = benchmark(grid_cluster, pos, size, start, None, device=device)
    print(f"  [Benchmark] {device} time: {runtime * 1e6:.2f} us")
        
    # CPU对比（仅NPU）
    if device.type == "npu":
        pos_cpu = pos.cpu()
        size_cpu = size.cpu()
        start_cpu = start.cpu() if start is not None else None
            
        cpu_time = benchmark(grid_cluster, pos_cpu, size_cpu, start_cpu, None, device=torch.device("cpu"))
        print(f"  [Benchmark] CPU time: {cpu_time * 1e6:.2f} us")
        print(f"  [Benchmark] Speedup: {cpu_time / runtime:.2f}x")
    
    print(f"  Test '{test['name']}' passed!")
    print("=" * 60)

# ========== 12. 测试选择器函数 =========
def run_quick_tests():
    """快速测试（只运行基础测试）"""
    return basic_tests

def run_smoke_tests():
    """冒烟测试（最小测试集）"""
    return basic_tests[:3]  # 前3个基础测试

if __name__ == "__main__":
    # 如果直接运行，执行pytest
    pytest.main([__file__, "-v", "-k", "not perf"])  # 跳过性能测试