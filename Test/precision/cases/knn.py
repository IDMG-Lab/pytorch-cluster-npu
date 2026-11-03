"""Document KNN shapes x actual pyg-lib k attributes, with exact index checks."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch_npu
from torch_cluster import knn, knn_graph
from nearest import DTYPES, shape
from parameters import geometry_parameters


def reference(x, y, bx, by, k, graph):
    x, y = x.double().numpy(), y.double().numpy()
    if bx is None:
        bx, by = torch.zeros(len(x), dtype=torch.long), torch.zeros(len(y), dtype=torch.long)
    result = []
    for batch in np.unique(by.numpy()):
        refs = np.flatnonzero(bx.numpy() == batch)
        queries = np.flatnonzero(by.numpy() == batch)
        if not len(refs):
            continue
        tree = cKDTree(x[refs])
        required = min(k + int(graph), len(refs))
        count = min(required + 1, len(refs))
        distances, indices = tree.query(y[queries], k=count, workers=4)
        distances, indices = distances.reshape(len(queries), count), indices.reshape(len(queries), count)
        for local, query in enumerate(queries):
            candidates = refs[indices[local]]
            values = distances[local]
            if count > required and values[required-1] == values[required]:
                candidates = refs[np.asarray(tree.query_ball_point(
                    y[query], np.nextafter(values[required-1], np.inf)))]
                values = ((x[candidates] - y[query])**2).sum(1)
            order = np.lexsort((candidates, values))
            selected = candidates[order[:required]]
            if graph:
                selected = selected[selected != query]
                result.extend((int(v), int(query)) for v in selected)
            else:
                result.extend((int(query), int(v)) for v in selected)
    return torch.tensor(result, dtype=torch.long).reshape(-1, 2).t().contiguous()


def canonical(edges, n):
    if not edges.numel():
        return edges
    return edges[:, torch.argsort(edges[0] * (n + 1) + edges[1])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=list(DTYPES))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    table = next(t for t in json.loads((here / 'document_tables.json').read_text(encoding='utf-8-sig'))
                 if t['section'] == '功能验收场景汇总' and t['operator'] == 'knn')
    parameters = geometry_parameters()
    results = []
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.jsonl').open('w') as log:
        for case, row in enumerate(table['rows'][1:], 1):
            for name, dtype in DTYPES.items():
                if args.dtype and name != args.dtype:
                    continue
                for distribution in ('uniform', 'normal'):
                    generator = torch.Generator().manual_seed(20261102+case)
                    size = shape(row[0])
                    if distribution == 'uniform':
                        x = (torch.rand(size, generator=generator, dtype=torch.float64) * 10 - 5).to(dtype)
                    else:
                        mu = torch.rand(1, generator=generator).item() * 10 - 5
                        sigma = .1 + torch.rand(1, generator=generator).item() * 1.9
                        x = (torch.randn(size, generator=generator, dtype=torch.float64) * sigma + mu).to(dtype)
                    graph = row[1] == 'None'
                    # The uploaded pyg-lib script uses y=x for equal shapes.
                    y = x if graph or row[0] == row[1] else x[:shape(row[1])[0]]
                    if not graph and y.shape != shape(row[1]):
                        # The empty-x/nonempty-y cases cannot use an x slice.
                        if distribution == 'uniform':
                            y = (torch.rand(shape(row[1]), generator=generator, dtype=torch.float64) * 10 - 5).to(dtype)
                        else:
                            y = (torch.randn(shape(row[1]), generator=generator, dtype=torch.float64) * sigma + mu).to(dtype)
                    bx = by = None
                    if row[2] != 'None':
                        bx = torch.arange(len(x), dtype=torch.long) * 2 // max(1, len(x))
                        by = bx if graph else torch.arange(len(y), dtype=torch.long) * 2 // max(1, len(y))
                    inputs = (x, y, bx, by)
                    digest = hashlib.sha256()
                    for tensor in inputs:
                        if tensor is not None:
                            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                    for k in parameters['knn_k']:
                        result = dict(operator='knn_graph' if graph else 'knn', case=case,
                                      document_row=row, dtype=name, distribution=distribution,
                                      seed=20261102+case, k=k, input_sha256=digest.hexdigest())
                        start = time.perf_counter()
                        try:
                            dx, dy, dbx, dby = tuple(t.npu() if t is not None else None for t in inputs)
                            actual = knn_graph(dx, k, batch=dbx) if graph else knn(dx, dy, k, dbx, dby)
                            torch.npu.synchronize()
                            assert actual.device.type == 'npu', 'Expected NPU output'
                            assert actual.dtype == torch.long and actual.dim() == 2 and actual.size(0) == 2
                            golden = reference(*inputs, k, graph)
                            a, b = canonical(actual.cpu(), max(len(x), len(y))), canonical(golden, max(len(x), len(y)))
                            result.update(status='passed' if torch.equal(a, b) else 'failed',
                                          rule='exact_edge_multiset', actual_shape=list(a.shape),
                                          golden_shape=list(b.shape))
                            if a.shape == b.shape:
                                result['mismatched_elements'] = int((a != b).sum())
                        except Exception as error:
                            result.update(status='error', error_type=type(error).__name__, reason=str(error)[:2000])
                        result['seconds'] = time.perf_counter() - start
                        results.append(result)
                        line = json.dumps(result, ensure_ascii=False)
                        print(line, flush=True); log.write(line+'\n'); log.flush()
    report = dict(device=torch.npu.get_device_name(0), torch=torch.__version__,
                  torch_npu=torch_npu.__version__, parameters=parameters, results=results,
                  reference='CPU float64 cKDTree, exact edge multiset')
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
