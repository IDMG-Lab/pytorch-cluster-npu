"""All document random-walk shapes, with exact transition and RNG checks.

CPU/NPU RNG sequences need not match. Check the stochastic operator's actual
contract: legal CSR edges, isolated-node behavior, initial states, dtype,
repeatability under seed reset, optional-return semantics, and uniform edge
slot frequencies for p=q=1. No floating tolerance is applied to node IDs.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch_npu
from torch_cluster import random_walk
from parameters import geometry_parameters


def extent(text):
    items = []
    for value in text.strip('[]').split(','):
        unit = 1024**2 if value.endswith('M') else 1024 if value.endswith('K') else 1
        items.append(int(value.rstrip('KM'))*unit)
    return tuple(items)


def validate(walk, eids, rp, col, start, length):
    assert walk.dtype == start.dtype and eids.dtype == start.dtype
    assert tuple(walk.shape) == (len(start), length+1)
    assert tuple(eids.shape) == (len(start), length)
    assert torch.equal(walk[:, 0], start)
    slots = None
    samples = 0
    rp, col = rp.long().numpy(), col.long().numpy()
    n = len(rp)-1
    regular = len(col)//n if n and len(col)%n == 0 else None
    if regular:
        slots = np.zeros(regular, dtype=np.int64)
    for first in range(0, len(start), 8192):
        nodes = walk[first:first+8192].long().numpy()
        edges = eids[first:first+8192].long().numpy()
        assert np.all((nodes >= 0) & (nodes < n)), 'Node ID out of range'
        src, dst = nodes[:, :-1], nodes[:, 1:]
        begin, end = rp[src], rp[src+1]
        isolated = begin == end
        assert np.all(edges[isolated] == -1), 'Isolated node used an edge'
        assert np.array_equal(dst[isolated], src[isolated]), 'Isolated node moved'
        real = ~isolated
        selected = edges[real]
        assert np.all((selected >= begin[real]) & (selected < end[real])), 'Edge ID outside source CSR row'
        assert np.array_equal(col[selected], dst[real]), 'Edge ID and destination disagree'
        if slots is not None:
            offset = selected - begin[real]
            slots += np.bincount(offset, minlength=regular)
            samples += len(selected)
    statistics = dict(samples=samples, uniformity_assessed=False)
    if slots is not None and samples >= 1000:
        probability = 1./regular
        tolerance = 6*np.sqrt(probability*(1-probability)/samples)+1e-4
        observed = slots/samples
        assert np.max(np.abs(observed-probability)) <= tolerance, 'Biased edge-slot frequencies'
        statistics.update(uniformity_assessed=True, observed=observed.tolist(),
                          expected=probability, tolerance=float(tolerance))
    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=['int32', 'int64'])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    table = next(t for t in json.loads((here/'document_tables.json').read_text(encoding='utf-8-sig'))
                 if t['section'] == '功能验收场景汇总' and t['operator'] == 'random_walk')
    parameters = geometry_parameters()
    results = []
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.jsonl').open('w') as log:
        for case, spec in enumerate(table['rows'][1:], 1):
            for name, dtype in dict(int32=torch.int32, int64=torch.int64).items():
                if args.dtype and args.dtype != name:
                    continue
                e = extent(spec[0])[0]
                n = max(1, e//4)  # Uploaded pyg-lib graph-construction rule.
                count = extent(spec[2])[0]
                length = extent(spec[3])[1]-1
                generator = torch.Generator().manual_seed(20261102+case)
                degrees = torch.full((n,), e//n, dtype=torch.long)
                degrees[:e % n] += 1
                rp = torch.cat((torch.zeros(1, dtype=dtype), degrees.cumsum(0).to(dtype)))
                row = torch.repeat_interleave(torch.arange(n, dtype=dtype), degrees)
                col = torch.randint(n, (e,), generator=generator, dtype=dtype)
                start = (torch.arange(count, dtype=torch.long) % n).to(dtype)
                result = dict(operator='random_walk', case=case, document_row=spec,
                              dtype=name, p=parameters['random_walk_p'], q=parameters['random_walk_q'],
                              walk_length=length, num_nodes=n, seed=20261102+case,
                              coalesced=False, optional_returns_tested=[False, True])
                start_time = time.perf_counter()
                try:
                    inputs = tuple(t.npu() for t in (row, col, start))
                    torch.npu.manual_seed(950+case)
                    walk, edges = random_walk(*inputs, length, p=result['p'], q=result['q'],
                                              coalesced=False, num_nodes=n, return_edge_indices=True)
                    assert walk.device.type == 'npu' and edges.device.type == 'npu'
                    statistics = validate(walk.cpu(), edges.cpu(), rp, col, start, length)
                    torch.npu.manual_seed(950+case)
                    plain = random_walk(*inputs, length, p=result['p'], q=result['q'],
                                        coalesced=False, num_nodes=n, return_edge_indices=False)
                    assert isinstance(plain, torch.Tensor) and plain.device.type == 'npu'
                    assert torch.equal(plain.cpu(), walk.cpu()), 'Seed repeatability/optional return changed the walk'
                    result.update(status='passed', rule='exact_transitions_and_stochastic_contract',
                                  uniformity=statistics)
                except Exception as error:
                    result.update(status='error', error_type=type(error).__name__, reason=str(error)[:2000])
                result['seconds'] = time.perf_counter()-start_time
                results.append(result)
                line = json.dumps(result, ensure_ascii=False)
                log.write(line+'\n'); log.flush(); print(line, flush=True)
    target.write_text(json.dumps(dict(device=torch.npu.get_device_name(0), parameters=parameters,
                                     results=results), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
