"""Graclus document cases, validating legal maximal weighted greedy matchings.

Graclus's random/parallel node order is not fixed by the public contract, so
cluster IDs need not equal one CPU RNG run. The independent CPU verifier
requires real reciprocal edges, groups of size <=2, canonical labels,
maximality, and a valid greedy elimination order respecting edge weights.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch_npu
from torch_cluster import graclus_cluster


def verify(row, col, weight, labels, n):
    assert labels.dtype == torch.long and tuple(labels.shape) == (n,)
    label = labels.numpy()
    if not n:
        return dict(groups=0, greedy_rounds=0)
    assert np.all((label >= 0) & (label < n)), 'Cluster label out of range'
    order = np.argsort(label, kind='stable')
    starts = np.flatnonzero(np.r_[True, np.diff(label[order]) != 0])
    counts = np.diff(np.r_[starts, n])
    assert np.all(counts <= 2), 'A cluster has more than two nodes'
    members = order[starts]
    assert np.array_equal(label[members], members), 'Labels are not minimum member IDs'
    mate = np.arange(n)
    pair_start = starts[counts == 2]
    first, second = order[pair_start], order[pair_start+1]
    mate[first], mate[second] = second, first
    src, dst = row.long().numpy(), col.long().numpy()
    values = (np.ones(len(src), dtype=np.int64) if weight is None else
              weight.numpy() if weight.dtype in (torch.int32, torch.int64) else weight.double().numpy())
    keys = src.astype(np.int64)*n+dst
    sorted_keys = np.sort(keys)
    reverse = dst.astype(np.int64)*n+src
    indices = np.searchsorted(sorted_keys, reverse)
    reciprocal = indices < len(sorted_keys)
    reciprocal[reciprocal] &= sorted_keys[indices[reciprocal]] == reverse[reciprocal]
    eligible = reciprocal & (src != dst) & (values >= 0)
    selected = eligible & (dst == mate[src])
    score = np.full(n, -1, dtype=values.dtype)
    np.maximum.at(score, src[selected], values[selected])
    assert np.all(score[first] >= 0) and np.all(score[second] >= 0), 'A pair has no reciprocal eligible edge'
    unmatched = mate == np.arange(n)
    unpaired_edge = np.flatnonzero(eligible & unmatched[src] & unmatched[dst])
    if len(unpaired_edge):
        edge = unpaired_edge[0]
        raise AssertionError(f'Matching is not maximal: unpaired nodes {src[edge]}, {dst[edge]}, weight={values[edge]}')
    if weight is None:
        return dict(groups=len(starts), paired_nodes=2*len(first), greedy_rounds=1)
    # Existing CSR input is sorted by source. reduceat gives an O(E) maximum
    # for each remaining node and keeps int64 weights as integers throughout.
    assert np.all(src[:-1] <= src[1:]), 'Verifier requires source-sorted COO'
    degree = np.bincount(src, minlength=n)
    offsets = np.r_[0, degree.cumsum()]
    nonempty = degree > 0
    remaining = np.ones(n, dtype=bool)
    rounds = 0
    while remaining.any():
        best = np.full(n, -1, dtype=values.dtype)
        if len(src):
            valid = eligible & remaining[src] & remaining[dst]
            active = np.where(valid, values, -1)
            best[nonempty] = np.maximum.reduceat(active, offsets[:-1][nonempty])
        ready = remaining & (mate != np.arange(n)) & (score >= best)
        removed = ready | ready[mate] | (remaining & unmatched & (best < 0))
        assert np.any(removed), 'Matching has no weight-respecting greedy order'
        remaining[removed] = False
        rounds += 1
    return dict(groups=len(starts), paired_nodes=2*len(first), greedy_rounds=rounds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32', 'float64', 'int32', 'int64'])
    parser.add_argument('--resume-from')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    table = next(t for t in json.loads((here/'document_tables.json').read_text(encoding='utf-8-sig'))
                 if t['section'] == '功能验收场景汇总' and t['operator'] == 'Graclus')
    results = json.loads(Path(args.resume_from).read_text())['results'] if args.resume_from else []
    if args.retry_errors:
        results = [r for r in results if r['status'] == 'passed']
    completed = {(r['case'], r['weight_dtype']) for r in results}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    dtypes = dict(float16=torch.float16, bfloat16=torch.bfloat16, float32=torch.float32,
                  float64=torch.float64, int32=torch.int32, int64=torch.int64)
    with target.with_suffix('.jsonl').open('w') as log:
        for record in results:
            log.write(json.dumps(record, ensure_ascii=False)+'\n')
        for case, spec in enumerate(table['rows'][1:], 1):
            e, n = int(spec[0].strip('[]')), int(spec[-1].strip('[]'))
            weighted = spec[2] != 'None'
            configs = {args.dtype: dtypes[args.dtype]} if args.dtype else dtypes
            if not weighted:
                configs = {'none': None}
            for name, dtype in configs.items():
                if (case, name) in completed:
                    continue
                generator = torch.Generator().manual_seed(20261102+case)
                if e % 2 == 0 and n > 1:
                    src = torch.randint(n, (e//2,), generator=generator)
                    dst = torch.randint(n-1, (e//2,), generator=generator)
                    dst += (dst >= src).long()
                    row, col = torch.cat((src, dst)), torch.cat((dst, src))
                    if weighted:
                        half = (torch.randint(8, (e//2,), generator=generator, dtype=dtype)
                                if dtype in (torch.int32, torch.int64) else
                                torch.rand(e//2, generator=generator, dtype=dtype))
                        weight = half.repeat(2)
                    else:
                        weight = None
                else:
                    row = torch.arange(e) % max(1, n)
                    col = (row+1) % max(1, n)
                    weight = torch.ones(e, dtype=dtype) if weighted else None
                order = torch.argsort(row, stable=True)
                row, col = row[order], col[order]
                if weight is not None:
                    weight = weight[order]
                record = dict(operator='graclus', case=case, document_row=spec, weight_dtype=name,
                              index_dtype='int64', seed=20261102+case, weighted=weighted,
                              data='pyg-lib symmetric random edges and uniform nonnegative weights')
                start = time.perf_counter()
                try:
                    output = graclus_cluster(row.npu(), col.npu(), weight.npu() if weight is not None else None, n)
                    assert output.device.type == 'npu'
                    details = verify(row, col, weight, output.cpu(), n)
                    record.update(status='passed', rule='exact_matching_and_weighted_greedy_contract', details=details)
                except Exception as error:
                    record.update(status='error', error_type=type(error).__name__, reason=str(error)[:2000])
                record['seconds'] = time.perf_counter()-start
                results.append(record)
                line = json.dumps(record, ensure_ascii=False)
                log.write(line+'\n'); log.flush(); print(line, flush=True)
    target.write_text(json.dumps(dict(device=torch.npu.get_device_name(0), results=results), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
