"""Radius's documented shapes and uploaded pyg-lib attribute combinations.

Radius can truncate to any max_num_neighbors valid neighbors, so compare the
integer edge contract, exact CPU-float64 distances and exact capped counts,
not an arbitrary KD-tree traversal order. Never apply a 99% tolerance to IDs.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch_npu
from torch_cluster import radius
from nearest import DTYPES
from walk import extent
from parameters import geometry_parameters


def verify(x, y, edges, r, cap, ignore):
    assert edges.dtype == torch.long and edges.dim() == 2 and edges.size(0) == 2
    src, dst = edges[0].numpy(), edges[1].numpy()
    n, m = len(x), len(y)
    assert np.all((src >= 0) & (src < m)) and np.all((dst >= 0) & (dst < n)), 'Edge index out of range'
    if ignore:
        assert np.all(src != dst), 'Self edge returned with ignore_same_index'
    xp, yp = x.double().numpy(), y.double().numpy()
    for first in range(0, len(src), 262144):
        a, b = src[first:first+262144], dst[first:first+262144]
        squared = ((xp[b]-yp[a])**2).sum(1)
        bad = np.flatnonzero(squared > r*r)
        if len(bad):
            i = bad[0]
            raise AssertionError(f'Nonexistent radius edge y={a[i]} x={b[i]} squared_distance={squared[i]:.17g} r2={r*r}')
    keys = src.astype(np.int64)*max(1, n)+dst
    assert len(np.unique(keys)) == len(keys), 'Duplicate edge in radius output'
    actual_counts = np.bincount(src, minlength=m)
    assert np.all(actual_counts <= cap), 'Too many neighbors returned'
    expected = np.zeros(m, dtype=np.int64)
    if n:
        tree = cKDTree(xp)
        for first in range(0, m, 8192):
            query = yp[first:first+8192]
            k = min(n, cap+int(ignore))
            distances, indices = tree.query(query, k=k, workers=4)
            distances = distances.reshape(len(query), k)
            indices = indices.reshape(len(query), k)
            valid = distances <= r
            if ignore:
                valid &= indices != np.arange(first, first+len(query))[:, None]
            expected[first:first+len(query)] = np.minimum(cap, valid.sum(1))
    assert np.array_equal(actual_counts, expected), 'Missing or extra neighbors relative to capped float64 oracle'
    return dict(edges=len(keys), queries=m, exact_distances=True, exact_counts=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=list(DTYPES))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    table = next(t for t in json.loads((here/'document_tables.json').read_text(encoding='utf-8-sig'))
                 if t['section'] == '功能验收场景汇总' and t['operator'] == 'Radius')
    parameters = geometry_parameters()
    results = []
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.jsonl').open('w') as log:
        for case, spec in enumerate(table['rows'][1:], 1):
            for name, dtype in DTYPES.items():
                if args.dtype and args.dtype != name:
                    continue
                for distribution in ('uniform', 'normal'):
                    generator = torch.Generator().manual_seed(20261102+case)
                    sizes = extent(spec[0])
                    if distribution == 'uniform':
                        x = (torch.rand(sizes, generator=generator, dtype=torch.float64)*10-5).to(dtype)
                    else:
                        mu = torch.rand(1, generator=generator).item()*10-5
                        sigma = .1+torch.rand(1, generator=generator).item()*1.9
                        x = (torch.randn(sizes, generator=generator, dtype=torch.float64)*sigma+mu).to(dtype)
                    y = x if spec[0] == spec[1] else x[:extent(spec[1])[0]]
                    for r, cap, ignore in parameters['radius_attributes']:
                        record = dict(operator='radius', case=case, document_row=spec, dtype=name,
                                      distribution=distribution, seed=20261102+case, r=r,
                                      max_num_neighbors=cap, ignore_same_index=ignore)
                        start = time.perf_counter()
                        try:
                            dx = x.npu()
                            dy = dx if y is x else y.npu()
                            output = radius(dx, dy, r, max_num_neighbors=cap, ignore_same_index=ignore)
                            assert output.device.type == 'npu'
                            details = verify(x, y, output.cpu(), r, cap, ignore)
                            record.update(status='passed', rule='exact_edge_validity_and_capped_completeness', details=details)
                        except Exception as error:
                            record.update(status='error', error_type=type(error).__name__, reason=str(error)[:2000])
                        record['seconds'] = time.perf_counter()-start
                        results.append(record)
                        line = json.dumps(record, ensure_ascii=False)
                        log.write(line+'\n'); log.flush(); print(line, flush=True)
    target.write_text(json.dumps(dict(device=torch.npu.get_device_name(0), parameters=parameters,
                                     results=results), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
