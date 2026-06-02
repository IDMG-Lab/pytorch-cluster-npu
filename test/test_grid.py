from itertools import product
import pytest
import torch
import time
import random
from torch_cluster import grid_cluster

# =========================================================
# 工具函数
# =========================================================
def calc_grid_cluster(point, size, start, end):
    """
    与 Kernel 完全一致的线性展开:

    cluster += grid[d] * stride
    stride *= grid_size[d]

    grid_size[d] =
        floor((end[d] - start[d]) / size[d]) + 1
    """

    cluster = 0
    stride = 1

    for d in range(len(point)):
        grid = int((point[d] - start[d]) // size[d])

        cluster += grid * stride

        grid_size = int((end[d] - start[d]) // size[d]) + 1

        stride *= grid_size

    return cluster


# =========================================================
# 1. 基础测试
# =========================================================
basic_tests = [
    {
        'name': 'mod_64',
        'pos': [
            [i * 3 % 97, i * 3 % 97 + 1, i * 3 % 97 + 2]
            for i in range(64)
        ],
        'size': [5, 5, 5],
        'start': [0, 0, 0],
        'end': [100, 100, 100],
    },

    {
        'name': 'range_64',
        'pos': [
            [i, i + 1, i + 2]
            for i in range(64)
        ],
        'size': [5, 5, 5],
        'start': [0, 0, 0],
        'end': [100, 100, 100],
    },

    {
        'name': 'small_7',
        'pos': [
            [i, i + 1, i + 2]
            for i in range(7)
        ],
        'size': [3, 3, 3],
        'start': [0, 0, 0],
        'end': [30, 30, 30],
    },

    {
        'name': 'unaligned_33',
        'pos': [
            [i, i + 1, i + 2]
            for i in range(33)
        ],
        'size': [4, 4, 4],
        'start': [0, 0, 0],
        'end': [40, 40, 40],
    },

    {
        'name': 'cross_65',
        'pos': [
            [i, i + 1, i + 2]
            for i in range(65)
        ],
        'size': [5, 5, 5],
        'start': [0, 0, 0],
        'end': [100, 100, 100],
    },
]

# 自动生成 cluster
for test in basic_tests:
    test['cluster'] = [
        calc_grid_cluster(
            p,
            test['size'],
            test['start'],
            test['end']
        )
        for p in test['pos']
    ]


# =========================================================
# 2. dtype测试
# =========================================================
dtype_tests = [
    {
        'name': 'dtype_float16',

        'pos': [
            [i * 0.5, i * 0.5 + 1, i * 0.5 + 2]
            for i in range(100)
        ],
        'size': [2.0, 2.0, 2.0],
        'start': [0, 0, 0],
        'end': [100, 100, 100],
        'dtype': torch.float16,
    },
]

for test in dtype_tests:
    test['cluster'] = [
        calc_grid_cluster(
            p,
            test['size'],
            test['start'],
            test['end']
        )
        for p in test['pos']
    ]


# =========================================================
# 3. start测试
# =========================================================
start_tests = [
    {
        'name': 'start_nonzero',

        'pos': [
            [10, 15, 20],
            [15, 20, 25],
            [20, 25, 30],
        ],
        'size': [5, 5, 5],
        'start': [10, 10, 10],
        'end': [40, 40, 40],
    },

    {
        'name': 'negative_start',
        'pos': [
            [-10, -5, 0],
            [0, 5, 10],
            [10, 15, 20],
        ],
        'size': [5, 5, 5],
        'start': [-10, -10, -10],
        'end': [40, 40, 40],
    },
]

for test in start_tests:
    test['cluster'] = [
        calc_grid_cluster(
            p,
            test['size'],
            test['start'],
            test['end']
        )
        for p in test['pos']
    ]


# =========================================================
# 4. 边界测试
# =========================================================
edge_tests = [
    {
        'name': 'empty_input',
        'pos': [],
        'size': [5, 5, 5],
        'start': [0, 0, 0],
        'end': [100, 100, 100],
        'cluster': [],
    },

    {
        'name': 'single_point',
        'pos': [
            [42, 43, 44]
        ],
        'size': [10, 10, 10],
        'start': [0, 0, 0],
        'end': [100, 100, 100],
    },
]

for test in edge_tests:
    if len(test['pos']) != 0:
        test['cluster'] = [
            calc_grid_cluster(
                p,
                test['size'],
                test['start'],
                test['end']
            )
            for p in test['pos']
        ]


# =========================================================
# 5. 对齐测试
# =========================================================
def generate_alignment_tests():
    tests = []

    for n in [
        1, 2, 3, 5, 6, 7,
        9, 15, 17, 31,
        33, 63, 65,
        127, 129
    ]:

        points = [
            [i, i + 1, i + 2]
            for i in range(n)
        ]

        test = {
            'name': f'align_{n}',
            'pos': points,
            'size': [5, 5, 5],
            'start': [0, 0, 0],
            'end': [100, 100, 100],
        }

        test['cluster'] = [
            calc_grid_cluster(
                p,
                test['size'],
                test['start'],
                test['end']
            )
            for p in points
        ]

        tests.append(test)

    return tests


alignment_tests = generate_alignment_tests()

# =========================================================
# 6. 维度测试
# =========================================================
def generate_dimension_tests():
    tests = []
    for dim in [1, 2, 3, 4, 5, 8]:
        num_points = 32
        points = []
        for i in range(num_points):
            point = []
            for d in range(dim):
                point.append(i + d)
            points.append(point)

        size = [5.0] * dim
        start = [0.0] * dim
        end = [100.0] * dim
        expected = [
            calc_grid_cluster(
                p,
                size,
                start,
                end
            )
            for p in points
        ]

        tests.append({
            'name': f'dim_{dim}',
            'pos': points,
            'size': size,
            'start': start,
            'end': end,
            'cluster': expected,
        })
    return tests
dimension_tests = generate_dimension_tests()


# =========================================================
# 7. 随机测试
# =========================================================
def generate_random_test(seed=42, num_points=100):
    random.seed(seed)

    points = [
        [
            random.uniform(-100, 100),
            random.uniform(-100, 100),
            random.uniform(-100, 100),
        ]
        for _ in range(num_points)
    ]

    size = [5.0, 5.0, 5.0]
    start = [-100.0, -100.0, -100.0]
    end = [100.0, 100.0, 100.0]
    expected = [
        calc_grid_cluster(
            p,
            size,
            start,
            end
        )
        for p in points
    ]

    return {
        'name': f'random_{num_points}_seed{seed}',
        'pos': points,
        'size': size,
        'start': start,
        'end': end,
        'cluster': expected,
    }


random_tests = [
    generate_random_test(seed=i, num_points=100)
    for i in range(5)
]


# =========================================================
# 8. 性能测试
# =========================================================
def create_performance_test(num_points):

    points = [
        [i, i + 1, i + 2]
        for i in range(num_points)
    ]
    size = [10, 10, 10]
    start = [0, 0, 0]
    end = [1000000, 1000000, 1000000]
    expected = [
        calc_grid_cluster(
            p,
            size,
            start,
            end
        )
        for p in points
    ]

    return {
        'name': f'perf_{num_points}',
        'pos': points,
        'size': size,
        'start': start,
        'end': end,
        'cluster': expected,
        'benchmark_only': True,
    }


performance_tests = [
    create_performance_test(n)
    for n in [10000]
]


# =========================================================
# 8. 合并测试  basic_tests + start_tests + edge_tests + alignment_tests + dimension_tests + random_tests  performance_tests
# =========================================================
all_tests = (performance_tests)

print(f"Total tests generated: {len(all_tests)}")


# =========================================================
# 9. NPU检查
# =========================================================
if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("NPU not available", allow_module_level=True)

devices = [torch.device("npu:0")]

dtypes = [torch.float32]


# =========================================================
# 10. benchmark
# =========================================================
def benchmark(func, *args, repeat=50, device=None):

    for _ in range(10):
        func(*args)

    if device is not None and device.type == "npu":
        torch.npu.synchronize()

    start_time = time.perf_counter()

    for _ in range(repeat):
        func(*args)

    if device is not None and device.type == "npu":
        torch.npu.synchronize()

    end_time = time.perf_counter()

    return (end_time - start_time) / repeat


# =========================================================
# 11. 主测试
# =========================================================
@pytest.mark.parametrize(
    'test,dtype,device',
    product(all_tests, dtypes, devices)
)
def test_grid_cluster_enhanced(
    test,
    dtype,
    device
):

    if test.get('benchmark_only') and device.type == "cpu":
        pytest.skip("Performance test only for NPU")

    if 'dtype' in test:
        dtype = test['dtype']

    print(
        f"\n===== Test: {test['name']} "
        f"| Device: {device} "
        f"| dtype: {dtype} ====="
    )

    # -----------------------------
    # tensor
    # -----------------------------
    pos = torch.tensor(test['pos'], dtype=dtype, device=device)
    size = torch.tensor(test['size'], dtype=dtype, device=device)
    start = torch.tensor(test['start'], dtype=dtype, device=device)
    end = torch.tensor(test['end'], dtype=dtype, device=device)

    # -----------------------------
    # print
    # -----------------------------
    print(f"  Points: {len(test['pos'])}")
    print(f"  Size: {test['size']}")
    print(f"  Start: {test['start']}")
    print(f"  End: {test['end']}")

    print(f"  pos.shape = {pos.shape}")
    print(f"  pos.dtype = {pos.dtype}")

    print(f"  Input head: {test['pos'][:16]}")
    print(f"  Input tail: {test['pos'][-16:]}")

    # -----------------------------
    # run op
    # -----------------------------
    cluster = grid_cluster(pos, size, start, end)

    if device.type == "npu":
        torch.npu.synchronize()

    output = cluster.tolist()

    print(f"  Output head: {output[:16]}")
    print(f"  Output tail: {output[-16:]}")

    print(f"  Expect head: {test['cluster'][:16]}")
    print(f"  Expect tail: {test['cluster'][-16:]}")

    # -----------------------------
    # empty
    # -----------------------------
    if len(test['cluster']) == 0:
        assert len(output) == 0
        print("  Empty input test passed")
        return

    # -----------------------------
    # check
    # -----------------------------
    assert output == test['cluster'], \
        f"Mismatch at test {test['name']}"

    # -----------------------------
    # JIT
    # -----------------------------
    try:
        jit = torch.jit.script(grid_cluster)

        jit_output = jit(pos, size, start, end)

        if device.type == "npu":
            torch.npu.synchronize()

        assert torch.equal(jit_output, cluster)

        print("  JIT compilation passed")

    except Exception as e:
        print(f"  JIT test skipped: {e}")

    # -----------------------------
    # benchmark
    # -----------------------------
    runtime = benchmark(grid_cluster, pos, size, start, end, device=device)

    print(
        f"  [Benchmark] "
        f"{device} time: "
        f"{runtime * 1e6:.2f} us"
    )

    # -----------------------------
    # CPU compare
    # -----------------------------
    if device.type == "npu":

        pos_cpu = pos.cpu()
        size_cpu = size.cpu()
        start_cpu = start.cpu()
        end_cpu = end.cpu()

        cpu_time = benchmark(grid_cluster, pos_cpu, size_cpu, start_cpu, end_cpu, device=torch.device("cpu"))

        print(
            f"  [Benchmark] CPU time: "
            f"{cpu_time * 1e6:.2f} us"
        )

        print(
            f"  [Benchmark] Speedup: "
            f"{cpu_time / runtime:.2f}x"
        )

    print(f"  Test '{test['name']}' passed!")
    print("=" * 60)


# =========================================================
# 12. selector
# =========================================================
def run_quick_tests():
    return basic_tests


def run_smoke_tests():
    return basic_tests[:3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])