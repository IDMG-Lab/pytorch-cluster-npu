"""Document-driven precision audit of nearest's PUBLIC NPU interface.

All 39 document rows, all four floating dtypes, both prescribed random input
distributions. Index outputs use exact equality to a CPU float64 KD-tree.
The three rows with output None exercise missing/unsorted batch labels and
are reported as validation tests, separately from numerical precision.
Each result is flushed to JSONL before proceeding so partial runs are useful.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch_npu  # noqa
from torch_cluster import nearest

HERE = Path(__file__).resolve().parent
DTYPES = dict(float16=torch.float16, bfloat16=torch.bfloat16,
              float32=torch.float32, float64=torch.float64)


def shape(text):
    return tuple(int(x) for x in text.strip('[]').split(','))


def make_case(row, case, dtype, distribution):
    generator = torch.Generator().manual_seed(20261102 + case)
    xs, ys = shape(row[0]), shape(row[1])
    def points(size):
        if distribution == 'uniform':
            return (torch.rand(size, generator=generator, dtype=torch.float64) * 10 - 5).to(dtype)
        mu = torch.rand(1, generator=generator).item() * 10 - 5
        sigma = .1 + torch.rand(1, generator=generator).item() * 1.9
        return (torch.randn(size, generator=generator, dtype=torch.float64) * sigma + mu).to(dtype)
    x, y = points(xs), points(ys)
    bx = by = None
    if row[2] != 'None':
        bx = torch.arange(xs[0], dtype=torch.long) * 2 // xs[0]
        by = torch.arange(ys[0], dtype=torch.long) * 2 // ys[0]
    if row[-1] == 'None':
        if case == 19:
            by.zero_()  # Batch 1 occurs only in x.
        elif case == 20:
            bx.zero_()  # Batch 1 occurs only in y.
        elif case == 21:
            bx = bx.flip(0)  # Unsorted labels.
        else:
            raise ValueError(f'Unspecified invalid case {case}')
    return x, y, bx, by


def reference(x, y, bx, by):
    x = x.double().reshape(x.size(0), -1).numpy()
    y = y.double().reshape(y.size(0), -1).numpy()
    result = np.empty(x.shape[0], dtype=np.int64)
    if bx is None:
        bx = torch.zeros(len(x), dtype=torch.long)
        by = torch.zeros(len(y), dtype=torch.long)
    for batch in np.unique(bx.numpy()):
        rows = np.flatnonzero(bx.numpy() == batch)
        centers = np.flatnonzero(by.numpy() == batch)
        assert len(centers), 'Golden received unmatched batches'
        tree = cKDTree(y[centers])
        # Query two neighbors so ties are detected rather than silently taking
        # the KD-tree's arbitrary tie order as an exact index golden.
        k = min(2, len(centers))
        distance, index = tree.query(x[rows], k=k, workers=4)
        if k == 1:
            selected = centers[index]
        else:
            selected = centers[index[:, 0]]
            for local in np.flatnonzero(distance[:, 0] == distance[:, 1]):
                candidates = np.asarray(tree.query_ball_point(
                    x[rows[local]], np.nextafter(distance[local, 0], np.inf)))
                values = ((y[centers[candidates]] - x[rows[local]]) ** 2).sum(1)
                minimum = values.min()
                selected[local] = centers[candidates[values == minimum]].min()
        result[rows] = selected
    return torch.from_numpy(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=list(DTYPES))
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    tables = json.loads((HERE / 'document_tables.json').read_text(encoding='utf-8-sig'))
    table = next(t for t in tables if t['section'] == '功能验收场景汇总' and t['operator'] == 'Nearest')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with target.with_suffix('.jsonl').open('w') as log:
        for case, row in enumerate(table['rows'][1:], 1):
            if args.case is not None and case != args.case:
                continue
            for name, dtype in DTYPES.items():
                if args.dtype and name != args.dtype:
                    continue
                for distribution in ('uniform', 'normal'):
                    result = dict(operator='nearest', case=case, document_row=row,
                                  dtype=name, distribution=distribution, seed=20261102+case,
                                  kind='validation' if row[-1] == 'None' else 'precision')
                    start = time.perf_counter()
                    inputs = make_case(row, case, dtype, distribution)
                    digest = hashlib.sha256()
                    for t in inputs:
                        if t is not None:
                            digest.update(t.contiguous().view(torch.uint8).numpy().tobytes())
                    result['input_sha256'] = digest.hexdigest()
                    try:
                        device = tuple(t.npu() if t is not None else None for t in inputs)
                        output = nearest(*device)
                        torch.npu.synchronize()
                        if result['kind'] == 'validation':
                            result.update(status='failed', reason='Expected invalid batches to be rejected')
                        elif output.device.type != 'npu':
                            result.update(status='failed', reason='Public interface returned a CPU tensor')
                        elif output.dtype != torch.int64 or output.shape != (inputs[0].size(0),):
                            result.update(status='failed', reason='Wrong output shape or dtype')
                        else:
                            golden = reference(*inputs)
                            actual = output.cpu()
                            mismatch = torch.nonzero(actual != golden).flatten()
                            result.update(status='passed' if not mismatch.numel() else 'failed',
                                          rule='integer_exact', elements=golden.numel(),
                                          mismatches=mismatch.numel())
                            if mismatch.numel():
                                ids = mismatch[:10]
                                result.update(first_indices=ids.tolist(), actual=actual[ids].tolist(),
                                              golden=golden[ids].tolist())
                    except Exception as error:
                        message = str(error)
                        rejected = result['kind'] == 'validation' and isinstance(error, (ValueError, AssertionError))
                        result.update(status='validation_passed' if rejected else 'error',
                                      error_type=type(error).__name__, reason=message[:2000])
                    result['seconds'] = time.perf_counter() - start
                    results.append(result)
                    line = json.dumps(result, ensure_ascii=False)
                    log.write(line + '\n'); log.flush()
                    print(line, flush=True)
                    del inputs
    report = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
                  device=torch.npu.get_device_name(0), reference='CPU float64 cKDTree',
                  implementation_sha256=hashlib.sha256((HERE.parents[2] / 'torch_cluster/nearest.py').read_bytes()).hexdigest(),
                  results=results)
    root = HERE.parents[2]
    report['implementation_sha256'] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('torch_cluster/nearest.py', 'csrc/npu/nearest_npu.cpp',
                     'torch_cluster/_nearest_npu.so')
    }
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
