"""Summary-only cases for pyg-lib geometry and graph sampling operators.

These operators have functional tests in test/ops, but no corresponding
performance file under benchmark/ops. Run from the repository root. The
summary's None batch/COO inputs are converted to the one-batch pointer and
CSR representations required by pyg-lib's fps, graclus and random_walk APIs.
"""

import argparse
import time

import torch
import pyg_lib

K = 1024
M = K * K
# Each case carries its own parameter tuple; variation rows follow their
# base shape row so that rows sharing a shape stay adjacent.
CASES = {
    # (n, d, batch_count, ratio)
    'fps': [
        (1024, 32, 1, .5), (1024, 32, 1, .125),
        (1024, 32, 1, .25), (1024, 32, 2, .5),
        (4096, 32, 1, .5), (4096, 32, 1, .25), (4096, 32, 2, .5),
        (4096, 64, 1, .5), (8192, 16, 1, .5),
        (8192, 32, 1, .5), (8192, 64, 1, .5),
    ],
    # (n, d, size)
    'grid_cluster': [
        (1*M, 64, .5), (1*M, 64, .125),
        (4*M, 32, .5), (4*M, 32, .125), (4*M, 32, .25),
        (4*M, 64, .5),
        (8*M, 32, .5), (8*M, 32, .125), (8*M, 32, .25),
        (8*M, 64, .5),
        (32*M, 32, .5), (32*M, 32, .125),
        (32*M, 64, .5),
    ],
    # (n, d, k, batched, y_half); y_half rows query a half-size y,
    # as in the functional tests
    'knn': [
        (4096, 128, 2, 0, 0), (4096, 128, 8, 0, 0),
        (4096, 128, 64, 0, 0),
        (8192, 256, 8, 0, 0),
        (20000, 128, 8, 0, 0), (32000, 3, 8, 0, 0),
        (4096, 32, 64, 0, 0), (8192, 32, 64, 0, 0),
        (16384, 32, 8, 0, 0), (16384, 32, 8, 1, 0),
    ],
    # (n, m, d, batched)
    'nearest': [
        (20000, 10000, 128, 0),
        (8192, 8192, 3, 0), (8192, 4096, 3, 0), (8192, 4096, 3, 1),
        (32768, 32768, 16, 0), (32768, 32768, 32, 0),
        (32768, 32768, 64, 0), (4096, 1024, 3, 0), (4096, 2048, 3, 1),
        (16384, 8192, 3, 0), (32768, 16384, 3, 0),
    ],
    # (n, e, weighted)
    'graclus_cluster': [
        (10000, 50000, True), (10000, 100000, True),
        (5000, 10000, True), (128, 16256, True),
        (256, 65280, True), (512, 261120, True),
        (1024, 1046528, True), (2048, 4192256, True),
        (4096, 16773120, True), (500000, 2000000, True),
        (1000000, 4000000, True),
    ],
    # (n, d, r, max_num_neighbors, ignore_same_index); randn data,
    # r aligned with the functional tests (base 1.0)
    'radius': [
        (32*K, 3, 1.0, 32, False), (32*K, 3, 2.0, 32, False),
        (64*K, 3, 1.0, 32, False), (64*K, 3, 3.0, 32, False),
        (64*K, 3, 1.0, 100, False), (64*K, 3, 1.0, 32, True),
        (256*K, 3, 1.0, 32, False), (256*K, 3, 2.0, 32, False),
        (256*K, 3, 1.0, 32, True),
    ],
    'random_walk': [(row_e, col_e, s*K, length) for row_e, col_e, s, length in [
        (512*K, 512*K, 32, 64), (512*K, 512*K, 32, 128),
        (512*K, 512*K, 32, 256),
        (1*M, 1*M, 64, 128), (1*M, 1*M, 64, 256), (1*M, 1*M, 64, 512),
        (2*M, 2*M, 128, 128), (2*M, 2*M, 128, 256),
        (4*M, 4*M, 128, 128), (4*M, 4*M, 128, 256),
        (4*M, 4*M, 256, 512)]],
}


def sync(device):
    if device.type == 'npu':
        torch.npu.synchronize(device)
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def csr(n, e, device, index_dtype=torch.int64):
    counts = torch.full((n,), e // n, dtype=index_dtype)
    counts[:e % n] += 1
    rowptr = torch.empty(n + 1, dtype=index_dtype)
    rowptr[0] = 0
    rowptr[1:] = counts.cumsum(0)
    col = torch.randint(n, (e,), dtype=index_dtype)
    return rowptr.to(device), col.to(device)


def undirected_csr(n, e, device, weight_dtype):
    if e % 2:
        raise ValueError('graclus requires an even edge count for symmetric pairs')
    half = e // 2
    src = torch.randint(n, (half,), dtype=torch.int64)
    dst = torch.randint(n - 1, (half,), dtype=torch.int64)
    dst += (dst >= src).to(torch.int64)
    rows = torch.cat((src, dst))
    cols = torch.cat((dst, src))
    weights = torch.rand(half, dtype=weight_dtype).repeat(2)
    order = rows.argsort()
    counts = torch.bincount(rows, minlength=n)
    rowptr = torch.empty(n + 1, dtype=torch.int64)
    rowptr[0] = 0
    rowptr[1:] = counts.cumsum(0)
    return rowptr.to(device), cols[order].to(device), weights[order].to(device)


def make_call(op, case, dtype, device):
    if op == 'fps':
        n, d, batches, ratio = case
        pos = torch.rand(n, d, dtype=dtype, device=device)
        ptr = torch.linspace(0, n, batches + 1, dtype=torch.int64,
                             device=device)
        return lambda: pyg_lib.ops.fps(pos, ptr, ratio, False)
    if op == 'grid_cluster':
        n, d, size = case
        pos = torch.rand(n, d, dtype=dtype, device=device)
        size = torch.full((d,), size, dtype=dtype, device=device)
        start = torch.zeros(d, dtype=dtype, device=device)
        end = torch.ones(d, dtype=dtype, device=device)
        return lambda: pyg_lib.ops.grid_cluster(pos, size, start, end)
    if op == 'knn':
        n, d, k, batched, y_half = case
        x = torch.rand(n, d, dtype=dtype, device=device)
        y = torch.rand(n // 2, d, dtype=dtype, device=device) \
            if y_half else x
        if batched:  # as in test_knn_batched
            ptr_x = torch.linspace(0, n, 3, dtype=torch.int64, device=device)
            ptr_y = torch.linspace(0, y.size(0), 3, dtype=torch.int64,
                                   device=device)
            return lambda: pyg_lib.ops.knn(x, y, k, ptr_x=ptr_x, ptr_y=ptr_y)
        return lambda: pyg_lib.ops.knn(x, y, k)
    if op == 'nearest':
        n, m, d, batched = case
        x = torch.rand(n, d, dtype=dtype, device=device)
        y = torch.rand(m, d, dtype=dtype, device=device)
        if batched:  # as in test_nearest_batched
            ptr_x = torch.linspace(0, n, 3, dtype=torch.int64, device=device)
            ptr_y = torch.linspace(0, m, 3, dtype=torch.int64, device=device)
            return lambda: pyg_lib.ops.nearest(x, y, ptr_x=ptr_x, ptr_y=ptr_y)
        return lambda: pyg_lib.ops.nearest(x, y)
    if op == 'radius':
        n, d, r, max_num_neighbors, ignore_same_index = case
        # randn data, as in the functional tests; r stays around 1.0
        x = torch.randn(n, d, dtype=dtype, device=device)
        return lambda: pyg_lib.ops.radius(
            x, x, r, max_num_neighbors=max_num_neighbors,
            ignore_same_index=ignore_same_index)
    if op == 'graclus_cluster':
        n, e, weighted = case
        rowptr, col, weight = undirected_csr(n, e, device, dtype)
        if weighted:
            return lambda: pyg_lib.ops.graclus_cluster(rowptr, col, weight)
        return lambda: pyg_lib.ops.graclus_cluster(rowptr, col)
    row_e, col_e, starts, length = case
    n = row_e // 4
    rowptr, col = csr(n, col_e, device, dtype)
    seed = torch.arange(starts, dtype=dtype, device=device) % n
    return lambda: pyg_lib.sampler.random_walk(rowptr, col, seed, length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', choices=('all', *CASES), default='all')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--case', type=int, help='zero-based row index')
    args = parser.parse_args()
    if args.warmup < 0 or args.iter <= 0:
        parser.error('--warmup must be nonnegative and --iter positive')
    device = torch.device(args.device)
    if device.type == 'npu':
        import torch_npu  # noqa: F401
    for op, cases in CASES.items():
        if args.op not in ('all', op):
            continue
        for row, case in enumerate(cases):
            if args.case is not None and args.case != row:
                continue
            dtypes = ((torch.int32,) if op == 'random_walk'
                      else (torch.float32,))
            for dtype in dtypes:
                print(f'START {op}\t{row}\t{case}\t{dtype}', flush=True)
                fn = make_call(op, case, dtype, device)
                print(f'READY {op}\t{row}\t{case}\t{dtype}', flush=True)
                try:
                    with torch.no_grad():
                        for _ in range(args.warmup):
                            fn()
                        sync(device)
                        start = time.perf_counter()
                        for _ in range(args.iter):
                            fn()
                        sync(device)
                except (RuntimeError, NotImplementedError) as exc:
                    print(f'{op}\t{row}\t{case}\tUNSUPPORTED: {exc}', flush=True)
                    continue
                ms = 1000 * (time.perf_counter() - start) / args.iter
                print(f'{op}\t{row}\t{case}\t{dtype}\t{ms:.6f} ms', flush=True)


if __name__ == '__main__':
    main()
