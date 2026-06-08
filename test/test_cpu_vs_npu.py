"""
VoxelGrid 算子 NPU vs CPU 性能对比测试

测试内容:
  1. 正确性验证 — NPU 输出与 CPU 参考实现逐元素比对
  2. 性能对比   — 多维度、多规模下的 NPU / CPU 延迟及加速比
  3. 冷启动开销 — 首次调用 (JIT 编译) 耗时单独统计

运行方式:
  python test_cpu_vs_npu.py
  python test_cpu_vs_npu --quick          # 只跑小规模
  python test_cpu_vs_npu.py --correctness-only  # 只做正确性
"""

import argparse
import math
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch_cluster import grid_cluster


# ============================================================================
# 0. 命令行参数
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="VoxelGrid NPU vs CPU 性能对比测试"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只运行小规模快速测试",
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="只运行正确性验证，跳过性能测量",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="warmup 迭代次数 (default: 10)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="性能测量迭代次数 (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    return parser.parse_args()


# ============================================================================
# 1. CPU 参考实现 (纯 Python + NumPy 风格向量化)
# ============================================================================
def cpu_grid_cluster_reference(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """
    CPU 参考实现 — 与 NPU kernel 算法完全一致.

    算法:
      grid[d]       = floor((pos[d] - start[d]) / size[d])
      grid_size[d]  = floor((end[d] - start[d]) / size[d]) + 1
      cluster       = Σ_d grid[d] * stride[d]
      stride[0]     = 1
      stride[d+1]   = stride[d] * grid_size[d]

    Args:
        pos:   [N, D] 点坐标
        size:  [D]    体素尺寸
        start: [D]    网格起点
        end:   [D]    网格终点

    Returns:
        cluster: [N] int64 体素索引
    """
    N, D = pos.shape
    cluster = torch.zeros(N, dtype=torch.int64, device="cpu")
    stride = 1

    for d in range(D):
        p_d = pos[:, d].to(torch.float64)
        s_d = float(size[d])
        st_d = float(start[d])
        ed_d = float(end[d])

        # grid[d] = floor((pos - start) / size)
        grid = ((p_d - st_d) / s_d).floor().to(torch.int64)

        # cluster += grid * stride
        cluster += grid * stride

        # grid_size = floor((end - start) / size) + 1
        grid_size = int((ed_d - st_d) // s_d) + 1
        stride *= grid_size

    return cluster


def cpu_grid_cluster_vectorized(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """
    CPU 向量化实现 — 一次性计算所有维度，减少 Python 循环.

    适用于 pos 较大时的快速验证.
    """
    D = pos.shape[1]

    # [N, D] → 统一转为 float64 保证精度
    p = pos.to(torch.float64)
    s = size.to(torch.float64)  # [D]
    st = start.to(torch.float64)  # [D]
    ed = end.to(torch.float64)  # [D]

    # grid[d] = floor((p[:, d] - st[d]) / s[d])
    grid = ((p - st) / s).floor().to(torch.int64)  # [N, D]

    # stride 按 D 维累积
    # grid_size[d] = (ed - st) // s + 1
    grid_sizes = ((ed - st) // s + 1).to(torch.int64)  # [D]

    strides = torch.cat([
        torch.tensor([1], dtype=torch.int64),
        torch.cumprod(grid_sizes[:-1], dim=0)
    ])  # [D], strides = [1, gs[0], gs[0]*gs[1], ...]

    # cluster = Σ_d grid[:, d] * strides[d]
    cluster = (grid * strides).sum(dim=1).to(torch.int64)

    return cluster


# ============================================================================
# 2. 正确性测试用例
# ============================================================================
def build_correctness_cases() -> List[Dict]:
    """构造正确性测试用例集合."""

    cases = []

    # ---------- 2.1 基础用例 ----------
    base = [
        {
            "name": "basic_3d_5pts",
            "pos": [[0, 0, 0], [1, 2, 3], [10, 10, 10], [5, 5, 5], [4, 4, 4]],
            "size": [2.0, 2.0, 2.0],
            "start": [0.0, 0.0, 0.0],
            "end": [20.0, 20.0, 20.0],
        },
        {
            "name": "basic_1d_5pts",
            "pos": [[0], [3], [7], [11], [15]],
            "size": [5.0],
            "start": [0.0],
            "end": [20.0],
        },
        {
            "name": "basic_2d_3pts",
            "pos": [[0, 0], [11, 9], [2, 8]],
            "size": [5.0, 5.0],
            "start": [0.0, 0.0],
            "end": [20.0, 20.0],
        },
        {
            "name": "basic_5d_8pts",
            "pos": [[i * d for d in range(5)] for i in range(8)],
            "size": [3.0] * 5,
            "start": [0.0] * 5,
            "end": [30.0] * 5,
        },
    ]
    cases.extend(base)

    # ---------- 2.2 边界用例 ----------
    edge = [
        {"name": "single_point", "pos": [[1, 2, 3]], "size": [5.0, 5.0, 5.0],
         "start": [0.0, 0.0, 0.0], "end": [100.0, 100.0, 100.0]},
        {"name": "two_points", "pos": [[0, 0, 0], [0, 0, 0]],
         "size": [5.0, 5.0, 5.0], "start": [0.0, 0.0, 0.0],
         "end": [100.0, 100.0, 100.0]},
    ]
    cases.extend(edge)

    # ---------- 2.3 非零 start ----------
    nonzero_start = [
        {"name": "start_offset_1", "pos": [[10, 15, 20], [15, 20, 25]],
         "size": [5.0, 5.0, 5.0], "start": [10.0, 10.0, 10.0],
         "end": [40.0, 40.0, 40.0]},
        {"name": "negative_coords", "pos": [[-10, -5, 0], [0, 5, 10]],
         "size": [5.0, 5.0, 5.0], "start": [-10.0, -10.0, -10.0],
         "end": [40.0, 40.0, 40.0]},
        {"name": "float_size", "pos": [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
         "size": [0.5, 0.5, 0.5], "start": [0.0, 0.0, 0.0],
         "end": [10.0, 10.0, 10.0]},
    ]
    cases.extend(nonzero_start)

    # ---------- 2.4 对齐边界 (触发 NPU 的 BUFFER_POINTS=64 边界) ----------
    for n in [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]:
        points = [[float(i), float(i + 1), float(i + 2)] for i in range(n)]
        expected = cpu_grid_cluster_reference(
            torch.tensor(points, dtype=torch.float32),
            torch.tensor([5.0, 5.0, 5.0]),
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([100.0, 100.0, 100.0]),
        ).tolist()

        cases.append({
            "name": f"align_n{n}",
            "pos": points,
            "size": [5.0, 5.0, 5.0],
            "start": [0.0, 0.0, 0.0],
            "end": [100.0, 100.0, 100.0],
        })

    # ---------- 2.5 随机用例 ----------
    random.seed(42)
    for i in range(3):
        N = random.choice([50, 200, 1000])
        D = random.choice([2, 3, 4])
        points = [[random.uniform(-100, 100) for _ in range(D)] for _ in range(N)]
        size = [random.uniform(1.0, 10.0) for _ in range(D)]
        start = [random.uniform(-110, -100) for _ in range(D)]
        end = [random.uniform(100, 110) for _ in range(D)]
        cases.append({
            "name": f"random_N{N}_D{D}_{i}",
            "pos": points,
            "size": size,
            "start": start,
            "end": end,
        })

    return cases


# ============================================================================
# 3. 性能测试用例
# ============================================================================
# 测试规模配置: (num_points, dim, label)
_PERF_SIZES_QUICK = [
    (1_000, 3, "1K"),
    (10_000, 3, "10K"),
]

_PERF_SIZES_FULL = [
    (1_000, 3, "1K"),
    (10_000, 3, "10K"),
    (100_000, 3, "100K"),
    (1_000_000, 3, "1M"),
    (10_000, 1, "10K_D1"),
    (10_000, 5, "10K_D5"),
    (10_000, 8, "10K_D8"),
]


def build_perf_cases(quick: bool = False) -> List[Dict]:
    """构造性能测试用例."""
    sizes = _PERF_SIZES_QUICK if quick else _PERF_SIZES_FULL
    cases = []
    for n, d, label in sizes:
        cases.append({
            "label": label,
            "num_points": n,
            "dim": d,
            "size": [5.0] * d,
            "start": [0.0] * d,
            "end": [1000.0] * d,
            "dtype": torch.float32,
        })
    return cases


def generate_random_points(num_points: int, dim: int) -> List[List[float]]:
    """生成随机点云."""
    random.seed(42)  # 固定种子保证可复现
    return [
        [random.uniform(0.0, 999.0) for _ in range(dim)]
        for _ in range(num_points)
    ]


# ============================================================================
# 4. 基准测试工具
# ============================================================================
class BenchmarkTimer:
    """计时器，支持 warmup + synchronize + 统计."""

    def __init__(self, warmup: int = 10, repeat: int = 100):
        self.warmup = warmup
        self.repeat = repeat

    def measure(
        self,
        fn: Callable,
        *args,
        device: Optional[torch.device] = None,
    ) -> Dict:
        """
        测量函数延迟.

        Returns:
            dict with keys: mean_us, std_us, min_us, max_us, median_us
        """
        # Warmup
        for _ in range(self.warmup):
            fn(*args)

        if device is not None and device.type == "npu":
            torch.npu.synchronize()

        # 计时
        times = []
        for _ in range(self.repeat):
            t0 = time.perf_counter()
            fn(*args)
            if device is not None and device.type == "npu":
                torch.npu.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)  # us

        times = torch.tensor(times)

        return {
            "mean_us": times.mean().item(),
            "std_us": times.std().item(),
            "min_us": times.min().item(),
            "max_us": times.max().item(),
            "median_us": times.median().item(),
        }


def format_time(us: float) -> str:
    """格式化时间显示."""
    if us < 1:
        return f"{us * 1000:.2f} ns"
    elif us < 1000:
        return f"{us:.2f} us"
    elif us < 1e6:
        return f"{us / 1000:.2f} ms"
    else:
        return f"{us / 1e6:.3f} s"


# ============================================================================
# 5. 正确性验证
# ============================================================================
def run_correctness_tests(args) -> Tuple[int, int]:
    """
    运行正确性测试.

    Returns:
        (passed, total)
    """
    print("=" * 72)
    print("  正确性验证")
    print("=" * 72)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[SKIP] NPU 不可用，无法进行正确性对比")
        return 0, 0

    device = torch.device("npu:0")
    cases = build_correctness_cases()
    passed = 0
    total = len(cases)

    for i, case in enumerate(cases):
        name = case["name"]
        pos_list = case["pos"]
        size_list = case["size"]
        start_list = case["start"]
        end_list = case["end"]

        # CPU reference
        pos_cpu = torch.tensor(pos_list, dtype=torch.float32)
        size_cpu = torch.tensor(size_list, dtype=torch.float32)
        start_cpu = torch.tensor(start_list, dtype=torch.float32)
        end_cpu = torch.tensor(end_list, dtype=torch.float32)

        # 用参考实现产生期望值
        expected = cpu_grid_cluster_vectorized(pos_cpu, size_cpu, start_cpu, end_cpu)

        # NPU
        pos_npu = torch.tensor(pos_list, dtype=torch.float32, device=device)
        size_npu = torch.tensor(size_list, dtype=torch.float32, device=device)
        start_npu = torch.tensor(start_list, dtype=torch.float32, device=device)
        end_npu = torch.tensor(end_list, dtype=torch.float32, device=device)

        try:
            cluster_npu = grid_cluster(pos_npu, size_npu, start_npu, end_npu)
        except Exception as e:
            print(f"  [{i + 1}/{total}] {name}: FAIL (exception: {e})")
            continue

        torch.npu.synchronize()

        cluster_cpu_result = cluster_npu.cpu()

        if cluster_cpu_result.numel() == 0:
            if expected.numel() == 0:
                passed += 1
                print(f"  [{i + 1}/{total}] {name}: PASS (empty)")
            else:
                print(f"  [{i + 1}/{total}] {name}: FAIL (got empty, expected non-empty)")
            continue

        # 逐元素比对
        match = torch.equal(cluster_cpu_result, expected)

        if match:
            passed += 1
            label = "PASS"
        else:
            # 找出第一个不匹配的位置
            mismatch = (cluster_cpu_result != expected).nonzero(as_tuple=True)[0]
            first_n = min(5, len(mismatch))
            detail = []
            for idx in mismatch[:first_n].tolist():
                detail.append(
                    f"idx={idx}: got={cluster_cpu_result[idx].item()}, "
                    f"expected={expected[idx].item()}"
                )
            label = f"FAIL ({', '.join(detail)})"

        N = cluster_cpu_result.numel()
        D = pos_cpu.shape[1]
        print(f"  [{i + 1}/{total}] {name}: {label}  [N={N}, D={D}]")

    print(f"\n  正确性结果: {passed}/{total} 通过")
    return passed, total


# ============================================================================
# 6. 性能对比
# ============================================================================
def run_performance_comparison(args) -> Dict:
    """
    运行 NPU vs CPU 性能对比.

    Returns:
        汇总结果 dict
    """
    print("\n" + "=" * 72)
    print("  性能对比: NPU vs CPU")
    print("=" * 72)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[SKIP] NPU 不可用")
        return {}

    device_npu = torch.device("npu:0")
    device_cpu = torch.device("cpu")

    timer = BenchmarkTimer(warmup=args.warmup, repeat=args.repeat)
    cases = build_perf_cases(quick=args.quick)

    results = []

    # 表头
    print(f"\n{'Case':<14} {'N':>8} {'D':>3}  "
          f"{'NPU mean':>12} {'NPU min':>12} {'NPU max':>12}  "
          f"{'CPU mean':>12} {'CPU min':>12} {'CPU max':>12}  "
          f"{'Speedup':>8}")
    print("-" * 100)

    for case in cases:
        label = case["label"]
        N = case["num_points"]
        D = case["dim"]
        dtype = case["dtype"]

        # 生成数据
        pos_list = generate_random_points(N, D)
        size_list = case["size"]
        start_list = case["start"]
        end_list = case["end"]

        # ---- CPU 基准 ----
        pos_cpu = torch.tensor(pos_list, dtype=dtype, device=device_cpu)
        size_cpu = torch.tensor(size_list, dtype=dtype, device=device_cpu)
        start_cpu = torch.tensor(start_list, dtype=dtype, device=device_cpu)
        end_cpu = torch.tensor(end_list, dtype=dtype, device=device_cpu)

        cpu_stats = timer.measure(
            grid_cluster, pos_cpu, size_cpu, start_cpu, end_cpu,
            device=device_cpu,
        )

        # ---- NPU 基准 ----
        pos_npu = torch.tensor(pos_list, dtype=dtype, device=device_npu)
        size_npu = torch.tensor(size_list, dtype=dtype, device=device_npu)
        start_npu = torch.tensor(start_list, dtype=dtype, device=device_npu)
        end_npu = torch.tensor(end_list, dtype=dtype, device=device_npu)

        npu_stats = timer.measure(
            grid_cluster, pos_npu, size_npu, start_npu, end_npu,
            device=device_npu,
        )

        # ---- 计算加速比 ----
        speedup = cpu_stats["mean_us"] / npu_stats["mean_us"] if npu_stats["mean_us"] > 0 else float("inf")

        print(
            f"{label:<14} {N:>8} {D:>3}  "
            f"{format_time(npu_stats['mean_us']):>12} "
            f"{format_time(npu_stats['min_us']):>12} "
            f"{format_time(npu_stats['max_us']):>12}  "
            f"{format_time(cpu_stats['mean_us']):>12} "
            f"{format_time(cpu_stats['min_us']):>12} "
            f"{format_time(cpu_stats['max_us']):>12}  "
            f"{speedup:>7.2f}x"
        )

        results.append({
            "label": label,
            "N": N,
            "D": D,
            "npu_us": npu_stats["mean_us"],
            "cpu_us": cpu_stats["mean_us"],
            "speedup": speedup,
        })

    # ---- 汇总 ----
    print("-" * 100)
    if results:
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        max_speedup = max(r["speedup"] for r in results)
        min_speedup = min(r["speedup"] for r in results)
        print(f"\n  加速比统计: 平均 {avg_speedup:.2f}x  |  "
              f"最大 {max_speedup:.2f}x  | 最小 {min_speedup:.2f}x")

        # 最大规模延迟
        largest = max(results, key=lambda r: r["N"])
        print(f"  最大规模 ({largest['label']}, N={largest['N']}): "
              f"NPU={format_time(largest['npu_us'])}, "
              f"CPU={format_time(largest['cpu_us'])}, "
              f"加速比={largest['speedup']:.2f}x")

    return {"results": results}


# ============================================================================
# 7. 冷启动 (首次调用 / JIT) 测试
# ============================================================================
def run_cold_start_test(args):
    """测量 NPU 首次调用的 JIT 编译开销."""
    print("\n" + "=" * 72)
    print("  冷启动 (首次调用) 测试")
    print("=" * 72)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[SKIP] NPU 不可用")
        return

    import subprocess
    import sys

    # 通过单独进程运行来测量真实的冷启动
    cold_start_script = """
import time
import torch
from torch_cluster import grid_cluster

pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], device="npu:0")
size = torch.tensor([5.0, 5.0, 5.0], device="npu:0")
start = torch.tensor([0.0, 0.0, 0.0], device="npu:0")
end = torch.tensor([100.0, 100.0, 100.0], device="npu:0")

t0 = time.perf_counter()
result = grid_cluster(pos, size, start, end)
torch.npu.synchronize()
t1 = time.perf_counter()

print(f"COLD_START_TIME_US={(t1 - t0) * 1e6:.1f}")
print(f"RESULT={result.cpu().tolist()}")
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", cold_start_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("COLD_START_TIME_US="):
                cold_us = float(line.split("=")[1])
                print(f"  首次调用耗时 (含 JIT): {format_time(cold_us)}")
            elif line.startswith("RESULT="):
                print(f"  输出: {line.split('=')[1]}")
        if proc.stderr:
            print(f"  stderr: {proc.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT] 冷启动超时 (>120s)")
    except Exception as e:
        print(f"  [ERROR] 冷启动测试失败: {e}")


# ============================================================================
# 8. 带宽和吞吐量分析
# ============================================================================
def run_bandwidth_analysis(args):
    """分析不同规模下的 NPU 吞吐量."""
    print("\n" + "=" * 72)
    print("  NPU 吞吐量分析")
    print("=" * 72)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[SKIP] NPU 不可用")
        return

    device = torch.device("npu:0")
    timer = BenchmarkTimer(warmup=args.warmup, repeat=args.repeat)

    # 多个规模的吞吐量
    point_counts = [100, 1_000, 10_000, 100_000]
    if not args.quick:
        point_counts.append(1_000_000)

    dim = 3
    dtype = torch.float32

    print(f"\n{'N':>10} {'Mean(us)':>12} {'Std(us)':>12} {'Points/us':>14} {'MB/s(in)':>12}")
    print("-" * 62)

    for N in point_counts:
        pos_list = generate_random_points(N, dim)
        size_list = [5.0] * dim
        start_list = [0.0] * dim
        end_list = [1000.0] * dim

        pos = torch.tensor(pos_list, dtype=dtype, device=device)
        size = torch.tensor(size_list, dtype=dtype, device=device)
        start = torch.tensor(start_list, dtype=dtype, device=device)
        end = torch.tensor(end_list, dtype=dtype, device=device)

        stats = timer.measure(
            grid_cluster, pos, size, start, end,
            device=device,
        )

        # 输入数据量: pos (N*D*4B) + size (D*4B) + start (D*4B) + end (D*4B)
        input_bytes = N * dim * 4 + dim * 4 * 3
        # 输出数据量: cluster (N*8B)
        output_bytes = N * 8

        throughput_pts = N / stats["mean_us"]  # points per us
        throughput_mb = (input_bytes + output_bytes) / stats["mean_us"] / 1e6 * 1e6  # MB/s

        print(
            f"{N:>10} {stats['mean_us']:>12.2f} {stats['std_us']:>12.2f} "
            f"{throughput_pts:>14.2f} {throughput_mb:>12.2f}"
        )

    print("-" * 62)
    print("  MB/s(in) 仅计算输入+输出数据量，不含中间 buffer")


# ============================================================================
# 9. 主入口
# ============================================================================
def main():
    args = parse_args()

    print("=" * 72)
    print("  VoxelGrid 算子 — NPU vs CPU 性能对比测试")
    print(f"  PyTorch: {torch.__version__}")
    npu_available = hasattr(torch, "npu") and torch.npu.is_available()
    print(f"  NPU available: {npu_available}")
    if npu_available:
        print(f"  NPU count: {torch.npu.device_count()}")
    print(f"  Warmup: {args.warmup}  Repeat: {args.repeat}")
    print(f"  Quick mode: {args.quick}")
    print("=" * 72)

    # ---- Step 1: 正确性 ----
    passed, total = run_correctness_tests(args)
    if total > 0 and passed < total:
        print("\n[WARNING] 部分正确性测试未通过，请先检查 kernel 实现！")

    if args.correctness_only:
        return

    # ---- Step 2: 性能对比 ----
    perf_results = run_performance_comparison(args)

    # ---- Step 3: 冷启动 ----
    run_cold_start_test(args)

    # ---- Step 4: 吞吐量 ----
    run_bandwidth_analysis(args)

    # ---- 最终结论 ----
    print("\n" + "=" * 72)
    print("  总结")
    print("=" * 72)
    if passed == total and total > 0:
        print(f"  正确性: {passed}/{total} 全部通过")
    else:
        print(f"  正确性: {passed}/{total} 通过 (有失败)")

    if perf_results.get("results"):
        results = perf_results["results"]
        print(f"  性能测试: {len(results)} 个规模完成")
        print(f"  最快加速比: {max(r['speedup'] for r in results):.2f}x")
        print(f"  最慢加速比: {min(r['speedup'] for r in results):.2f}x")
        # 在大规模下 NPU 优势更明显
        large_results = [r for r in results if r["N"] >= 10000]
        if large_results:
            avg_large_speedup = sum(r["speedup"] for r in large_results) / len(large_results)
            print(f"  N>=10K 平均加速比: {avg_large_speedup:.2f}x")

    print("=" * 72)
    print("  测试完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
