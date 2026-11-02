"""950 acceptance: integer-exact CSR reference and the uploaded pyg-lib shapes.

Run from repository root: python test/acceptance/subgraph950.py --accuracy
Performance reports both device events (raw capacity interface, excluding the
host read of output length) and public API wall time (uploaded benchmark method).
Baseline is the uploaded pyg_subgraph_cpu.log, confirmed by the user.
Relative performance = CPU baseline ms / NPU public API ms; target >= 0.2.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa
from torch_cluster import subgraph_950 as subgraph

M = 1024 * 1024
ACCURACY = [
    (6, 12, 4), (5, 8, 3), (8, 16, 3), (4, 6, 1), (128, 1024, 128),
    (128, 0, 3), (8, 8, 3), (6, 14, 3), (4, 9, 3), (6, 10, 4),
    (32, 992, 16), (100, 500, 20), (10000, 50000, 1000),
    (1000000, 4000000, 10000), (100, 500, 20), (4096, 128*1024, 410),
    (16384, 512*1024, 1600), (65536, 2*M, 6400),
    (262144, 8*M, 25600), (1000000, 32*M, 100000),
    (65536, 256*1024, 6400), (65536, 512*1024, 6400),
    (65536, 4*M, 6400), (65536, 16*M, 6400),
    (65536, 2*M, 64), (65536, 2*M, 640), (65536, 2*M, 32768),
    (65536, 2*M, 57600), (65536, 2*M, 65536), (65536, 2*M, 6400),
    (1000000, 8*M, 100000), (1000000, M, 100000),
]
PERFORMANCE = [ACCURACY[i] for i in (22, 23, 26, 27, 28, 29, 30, 31)]
HISTORICAL_CPU_MS = [2.292626, 5.074346, 9.317874, 4.587184,
                     3.315235, 1.330157, 11.919088, 3.807254]


def make_input(n, e, q, dtype, seed, skew=False):
    rng = np.random.default_rng(seed)
    if n == 0:
        return (torch.zeros(1, dtype=dtype), torch.empty(0, dtype=dtype),
                torch.empty(0, dtype=dtype))
    degrees = np.full(n, e // n, dtype=np.int64)
    degrees[:e % n] += 1
    if skew:
        # Empty rows plus one high-degree row exercise imbalanced scheduling.
        degrees[:] = 0
        degrees[0] = e // 2
        rest = e - degrees[0]
        degrees[1:] = rest // max(1, n - 1)
        degrees[1:1 + rest % max(1, n - 1)] += 1
    rp = np.concatenate(([0], degrees.cumsum()))
    col = rng.integers(n, size=e, dtype=np.int64)
    nodes = rng.permutation(n)[:q]
    if skew and q and 0 not in nodes:
        nodes[0] = 0
    return tuple(torch.from_numpy(x).to(dtype) for x in (rp, col, nodes))


def reference(inputs):
    rp, col, nodes = (x.numpy().astype(np.int64, copy=False) for x in inputs)
    mapping = np.full(len(rp) - 1, -1, dtype=np.int64)
    mapping[nodes] = np.arange(len(nodes))
    degrees = rp[nodes + 1] - rp[nodes]
    prefix = np.concatenate(([0], degrees.cumsum()))
    source = np.repeat(np.arange(len(nodes)), degrees)
    eid = np.repeat(rp[nodes] - prefix[:-1], degrees) + np.arange(prefix[-1])
    mapped = mapping[col[eid]]
    keep = mapped >= 0
    out_rp = np.concatenate(([0], np.bincount(source[keep], minlength=len(nodes)).cumsum()))
    return tuple(torch.from_numpy(x).to(inputs[0].dtype)
                 for x in (out_rp, mapped[keep], eid[keep]))


def accuracy():
    results = []
    for case, shape in enumerate(ACCURACY + [(0, 0, 0), (1, 1, 1), (3, 0, 0)]):
        for dtype in (torch.int32, torch.int64):
            for skew in (False, True):
                if shape[0] <= 1 and skew:
                    continue
                cpu = make_input(*shape, dtype, 950 + case, skew)
                expected = reference(cpu)
                device = tuple(x.npu() for x in cpu)
                for edge_id in (True, False):
                    actual = subgraph(*device, return_edge_id=edge_id)
                    want = expected if edge_id else expected[:2]
                    assert len(actual) == len(want)
                    for a, b in zip(actual, want):
                        assert a.dtype == b.dtype and a.device.type == 'npu'
                        assert torch.equal(a.cpu(), b), (case, shape, dtype, skew, edge_id)
                result = dict(case=case, shape=shape, dtype=str(dtype), skew=skew,
                              edge_id_modes=[True, False], exact=True)
                results.append(result)
                print(json.dumps(result), flush=True)
    return results


def performance(warmup, iterations, profile_kernels=False):
    results = []
    for case, shape in enumerate(PERFORMANCE):
        cpu = make_input(*shape, torch.int32, 20261102 + case)
        digest = hashlib.sha256()
        for tensor in cpu:
            digest.update(tensor.numpy().tobytes())
        inputs = tuple(x.npu() for x in cpu)
        expected = reference(cpu)
        assert all(torch.equal(a.cpu(), b) for a, b in zip(subgraph(*inputs), expected))
        with torch.no_grad():
            for _ in range(warmup):
                subgraph(*inputs)
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                subgraph(*inputs)
            torch.npu.synchronize()
            wall_ms = 1000 * (time.perf_counter() - start) / iterations
            starts = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
            for s, e in zip(starts, ends):
                s.record()
                output = torch.ops.torch_cluster.subgraph_950(*inputs)
                e.record()
            torch.npu.synchronize()
            device_ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iterations
        result = dict(case=case, shape=shape, dtype='int32', seed=20261102 + case,
                      input_sha256=digest.hexdigest(), warmup=warmup, iterations=iterations,
                      device_ms=device_ms, public_wall_ms=wall_ms,
                      historical_cpu_ms=HISTORICAL_CPU_MS[case],
                      historical_wall_ratio=wall_ms / HISTORICAL_CPU_MS[case],
                      cpu_relative_performance=HISTORICAL_CPU_MS[case] / wall_ms,
                      required_relative_performance=.2,
                      performance_pass=HISTORICAL_CPU_MS[case] / wall_ms >= .2)
        if profile_kernels:
            profile_dir = Path('validation') / f'case{case}-hardware-profile'
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=iterations, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
            ) as prof:
                for _ in range(iterations):
                    subgraph(*inputs)
                    prof.step()
            files = sorted(profile_dir.glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
            assert files, 'Profiler did not produce hardware kernel timings'
            with files[-1].open() as f:
                rows = [row for row in csv.DictReader(f) if 'Subgraph950' in row['Name']]
            assert len(rows) == iterations * 6, (case, len(rows), iterations)
            rows.sort(key=lambda row: float(row['Start Time(us)']))
            sums, spans = [], []
            for i in range(iterations):
                group = rows[i*6:(i+1)*6]
                sums.append(sum(float(row['Duration(us)']) for row in group))
                spans.append(float(group[-1]['Start Time(us)']) + float(group[-1]['Duration(us)'])
                             - float(group[0]['Start Time(us)']))
            result['hardware_kernel_sum_ms'] = sum(sums) / iterations / 1000
            result['hardware_operator_span_ms'] = sum(spans) / iterations / 1000
            result['hardware_span_baseline_ratio'] = result['hardware_operator_span_ms'] / HISTORICAL_CPU_MS[case]
            result['device_relative_performance'] = HISTORICAL_CPU_MS[case] / result['hardware_operator_span_ms']
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy', action='store_true')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--profile-kernels', action='store_true')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
                  device=torch.npu.get_device_name(0),
                  baseline='uploaded pyg_subgraph_cpu.log',
                  performance_rule='mean(CPU_ms / NPU_public_wall_ms) >= 0.2')
    root = Path(__file__).resolve().parents[2]
    report['implementation_sha256'] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in ('csrc/npu/950/subgraph950.cpp', 'csrc/npu/subgraph_950_npu.cpp',
                     'torch_cluster/subgraph.py', 'torch_cluster/libsubgraph950.so')
    }
    if args.accuracy:
        report['accuracy'] = accuracy()
    if args.performance:
        report['performance'] = performance(args.warmup, args.iterations, args.profile_kernels)
        report['mean_relative_performance'] = sum(
            r['cpu_relative_performance'] for r in report['performance']) / len(PERFORMANCE)
        report['performance_pass'] = report['mean_relative_performance'] >= .2
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
