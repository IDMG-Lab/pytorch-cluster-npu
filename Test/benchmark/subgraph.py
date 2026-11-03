"""Documented subgraph cases for pyg-lib's existing sampler benchmark.

The previous dataset and torch_geometric comparison cases are disabled; this
file measures only the eight shapes in 性能测试用例总结.md.
"""

import argparse
import time

import torch
import pyg_lib

# Disabled original DIMACS10/citationCiteseer dataset cases and
# torch_geometric.utils.subgraph comparison.

M = 1024 * 1024
CASES = [
    (65536, 4*M, 6400), (65536, 16*M, 6400),
    (65536, 2*M, 32768), (65536, 2*M, 57600),
    (65536, 2*M, 65536), (65536, 2*M, 6400),
    (1000000, 8*M, 100000), (1000000, M, 100000),
]


def sync(device):
    if device.type == 'npu':
        torch.npu.synchronize(device)
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def make_input(n, e, selected, device, index_dtype):
    # CPU preparation ensures a legal CSR with exactly E edges.
    degrees = torch.full((n,), e // n, dtype=index_dtype)
    degrees[:e % n] += 1
    rowptr = torch.empty(n + 1, dtype=index_dtype)
    rowptr[0] = 0
    rowptr[1:] = degrees.cumsum(0)
    col = torch.randint(n, (e,), dtype=index_dtype)
    nodes = torch.randperm(n, dtype=torch.int64)[:selected].to(index_dtype)
    return rowptr.to(device), col.to(device), nodes.to(device)


def main():
    parser = argparse.ArgumentParser()
    # The repository currently registers pyg::subgraph for CPU only.
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--case', type=int, help='zero-based row index')
    args = parser.parse_args()
    if args.warmup < 0 or args.iter <= 0:
        parser.error('--warmup must be nonnegative and --iter positive')
    device = torch.device(args.device)
    if device.type == 'npu':
        import torch_npu  # noqa: F401
    for row, case in enumerate(CASES):
        if args.case is not None and args.case != row:
            continue
        for index_dtype in (torch.int32,):
            print(f'START subgraph\t{row}\t{case}\t{index_dtype}', flush=True)
            rowptr, col, nodes = make_input(*case, device, index_dtype)
            with torch.no_grad():
                for _ in range(args.warmup):
                    pyg_lib.sampler.subgraph(rowptr, col, nodes, True)
                sync(device)
                start = time.perf_counter()
                for _ in range(args.iter):
                    pyg_lib.sampler.subgraph(rowptr, col, nodes, True)
                sync(device)
            print(f'subgraph\t{row}\t{case}\t{index_dtype}\t'
                  f'{1000*(time.perf_counter()-start)/args.iter:.6f} ms',
                  flush=True)


if __name__ == '__main__':
    main()
