"""Integer-exact 910B CSR subgraph document matrix and independent reference."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch_npu
from torch_cluster import subgraph_910b as subgraph
M = 1024 * 1024
ACCURACY = [(6, 12, 4), (5, 8, 3), (8, 16, 3), (4, 6, 1), (128, 1024, 128), (128, 0, 3), (8, 8, 3), (6, 14, 3), (4, 9, 3), (6, 10, 4), (32, 992, 16), (100, 500, 20), (10000, 50000, 1000), (1000000, 4000000, 10000), (100, 500, 20), (4096, 128 * 1024, 410), (16384, 512 * 1024, 1600), (65536, 2 * M, 6400), (262144, 8 * M, 25600), (1000000, 32 * M, 100000), (65536, 256 * 1024, 6400), (65536, 512 * 1024, 6400), (65536, 4 * M, 6400), (65536, 16 * M, 6400), (65536, 2 * M, 64), (65536, 2 * M, 640), (65536, 2 * M, 32768), (65536, 2 * M, 57600), (65536, 2 * M, 65536), (65536, 2 * M, 6400), (1000000, 8 * M, 100000), (1000000, M, 100000)]

def make_input(n, e, q, dtype, seed, skew=False):
    rng = np.random.default_rng(seed)
    if n == 0:
        return (torch.zeros(1, dtype=dtype), torch.empty(0, dtype=dtype), torch.empty(0, dtype=dtype))
    degrees = np.full(n, e // n, dtype=np.int64)
    degrees[:e % n] += 1
    if skew:
        degrees[:] = 0
        degrees[0] = e // 2
        rest = e - degrees[0]
        degrees[1:] = rest // max(1, n - 1)
        degrees[1:1 + rest % max(1, n - 1)] += 1
    rp = np.concatenate(([0], degrees.cumsum()))
    col = rng.integers(n, size=e, dtype=np.int64)
    nodes = rng.permutation(n)[:q]
    if skew and q and (0 not in nodes):
        nodes[0] = 0
    return tuple((torch.from_numpy(x).to(dtype) for x in (rp, col, nodes)))

def reference(inputs):
    (rp, col, nodes) = (x.numpy().astype(np.int64, copy=False) for x in inputs)
    mapping = np.full(len(rp) - 1, -1, dtype=np.int64)
    mapping[nodes] = np.arange(len(nodes))
    degrees = rp[nodes + 1] - rp[nodes]
    prefix = np.concatenate(([0], degrees.cumsum()))
    source = np.repeat(np.arange(len(nodes)), degrees)
    eid = np.repeat(rp[nodes] - prefix[:-1], degrees) + np.arange(prefix[-1])
    mapped = mapping[col[eid]]
    keep = mapped >= 0
    out_rp = np.concatenate(([0], np.bincount(source[keep], minlength=len(nodes)).cumsum()))
    return tuple((torch.from_numpy(x).to(inputs[0].dtype) for x in (out_rp, mapped[keep], eid[keep])))

def accuracy():
    results = []
    for (case, shape) in enumerate(ACCURACY + [(0, 0, 0), (1, 1, 1), (3, 0, 0)]):
        for dtype in (torch.int32, torch.int64):
            for skew in (False, True):
                if shape[0] <= 1 and skew:
                    continue
                cpu = make_input(*shape, dtype, 950 + case, skew)
                expected = reference(cpu)
                device = tuple((x.npu() for x in cpu))
                for edge_id in (True, False):
                    actual = subgraph(*device, return_edge_id=edge_id)
                    want = expected if edge_id else expected[:2]
                    assert len(actual) == len(want)
                    for (a, b) in zip(actual, want):
                        assert a.dtype == b.dtype and a.device.type == 'npu'
                        assert torch.equal(a.cpu(), b), (case, shape, dtype, skew, edge_id)
                result = dict(case=case, shape=shape, dtype=str(dtype), skew=skew, edge_id_modes=[True, False], exact=True)
                results.append(result)
                print(json.dumps(result), flush=True)
    return results
