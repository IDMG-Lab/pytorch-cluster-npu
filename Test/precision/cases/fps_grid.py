"""Document shapes and pyg-lib attributes for FPS/grid, independent CPU goldens."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch_npu
from torch_cluster import fps, grid_cluster
from nearest import DTYPES, shape
from parameters import geometry_parameters


def fps_reference(pos, ptr, ratio):
    points = pos.double().numpy()
    result = []
    for first, last in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        count = math.ceil((last-first)*ratio)
        if not count:
            continue
        segment = points[first:last]
        chosen = 0
        distance = np.full(len(segment), np.inf)
        for _ in range(count):
            result.append(first+chosen)
            delta = segment-segment[chosen]
            distance = np.minimum(distance, np.einsum('ij,ij->i', delta, delta))
            chosen = int(distance.argmax())
    return torch.tensor(result, dtype=torch.long)


def grid_reference(pos, size, start, end):
    if not len(pos):
        return torch.empty(0, dtype=torch.long)
    # The existing grid contract truncates toward zero (not floor) and
    # encodes multi-axis labels modulo 2**64. Keep integer math exact.
    start = pos.amin(0) if start is None else start
    end = pos.amax(0) if end is None else end
    a, b, s = (t.double().numpy() for t in (start, end, size))
    counts = np.trunc((b-a)/s).astype(np.int64)+1
    strides, value = [], 1
    for count in counts:
        strides.append(value)
        value = value*int(count) % 2**64
    strides = np.asarray(strides, dtype=np.uint64)
    output = np.empty(len(pos), dtype=np.int64)
    for first in range(0, len(pos), 65536):
        block = pos[first:first+65536].double().numpy()
        coordinates = np.trunc((block-a)/s).astype(np.int64).view(np.uint64)
        with np.errstate(over='ignore'):
            labels = (coordinates*strides).sum(1, dtype=np.uint64)
        output[first:first+len(block)] = labels.view(np.int64)
    return torch.from_numpy(output)


def parse_grid_shape(text):
    return tuple(int(x.replace('M', '*1048576').replace('K', '*1024').split('*')[0]) *
                 (1048576 if 'M' in x else 1024 if 'K' in x else 1)
                 for x in text.strip('[]').split(','))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', choices=['fps', 'grid_cluster'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=list(DTYPES))
    parser.add_argument('--case', type=int)
    parser.add_argument('--resume-jsonl')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    table = next(t for t in json.loads((here/'document_tables.json').read_text(encoding='utf-8-sig'))
                 if t['section'] == '功能验收场景汇总' and t['operator'] == ('Fps' if args.op == 'fps' else 'grid_cluster'))
    parameters = geometry_parameters()
    results = ([json.loads(line) for line in Path(args.resume_jsonl).read_text().splitlines() if line]
               if args.resume_jsonl else [])
    if args.retry_errors:
        results = [record for record in results if record['status'] != 'error']
    completed = {(r['case'], r['dtype'], r['distribution'], r.get('ratio', r.get('voxel_size')))
                 for r in results}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = target.with_suffix('.jsonl')
    mode = 'a' if args.resume_jsonl and Path(args.resume_jsonl).resolve() == log_path.resolve() else 'w'
    with log_path.open(mode) as log:
        if mode == 'w':
            for record in results:
                log.write(json.dumps(record, ensure_ascii=False)+'\n')
        for case, row in enumerate(table['rows'][1:], 1):
            if args.case is not None and case != args.case:
                continue
            for name, dtype in DTYPES.items():
                if args.dtype and name != args.dtype:
                    continue
                for distribution in ('uniform', 'normal'):
                    attrs = parameters['fps_ratios' if args.op == 'fps' else 'grid_sizes']
                    if all((case, name, distribution, attribute) in completed for attribute in attrs):
                        continue
                    generator = torch.Generator().manual_seed(20261102+case)
                    sizes = parse_grid_shape(row[0])
                    print(json.dumps(dict(event='generate', operator=args.op, case=case,
                                          shape=sizes, dtype=name, distribution=distribution)), flush=True)
                    if distribution == 'normal':
                        mu = torch.rand(1, generator=generator).item()*10-5
                        sigma = .1+torch.rand(1, generator=generator).item()*1.9
                    # Keep the full requested tensor, but bound generation
                    # scratch space; [32M,64] otherwise exceeds the 32-GiB
                    # container limit through full-size float64 temporaries.
                    pos = torch.empty(sizes, dtype=dtype)
                    for first in range(0, sizes[0], 65536):
                        block_shape = (min(65536, sizes[0]-first), *sizes[1:])
                        if distribution == 'uniform':
                            block = torch.rand(block_shape, generator=generator, dtype=torch.float64).mul_(10).sub_(5)
                        else:
                            block = torch.randn(block_shape, generator=generator, dtype=torch.float64).mul_(sigma).add_(mu)
                        pos[first:first+len(block)].copy_(block)
                        del block
                    for attribute in attrs:
                        if (case, name, distribution, attribute) in completed:
                            continue
                        result = dict(operator=args.op, case=case, document_row=row, dtype=name,
                                      distribution=distribution, seed=20261102+case)
                        start_time = time.perf_counter()
                        try:
                            if args.op == 'fps':
                                batches = (shape(row[2])[0]-1 if row[2] != 'None' else
                                           shape(row[3])[0] if row[3] != 'None' and shape(row[3])[0]>1 else
                                           2 if row[1] != 'None' else 1)
                                ptr = torch.linspace(0, len(pos), batches+1, dtype=torch.long)
                                batch = torch.arange(len(pos), dtype=torch.long)*batches//max(1, len(pos))
                                if row[2] == 'None' and row[1] != 'None':
                                    ptr = torch.cat((torch.zeros(1, dtype=torch.long),
                                                     torch.bincount(batch, minlength=batches).cumsum(0)))
                                ratio = torch.full(shape(row[3]), attribute, dtype=dtype) if row[3] != 'None' else attribute
                                result.update(ratio=attribute, random_start=False, batches=batches)
                                kwargs = dict(ratio=ratio.npu() if isinstance(ratio, torch.Tensor) else ratio,
                                              random_start=False)
                                if row[2] != 'None':
                                    kwargs['ptr'] = ptr.npu()
                                elif row[1] != 'None':
                                    kwargs.update(batch=batch.npu(), batch_size=batches)
                                output = fps(pos.npu(), **kwargs)
                                torch.npu.synchronize()
                                golden = fps_reference(pos, ptr, attribute)
                            else:
                                size = torch.full((sizes[1],), attribute, dtype=dtype)
                                start = torch.zeros(sizes[1], dtype=dtype) if row[2] != 'None' else None
                                end = torch.ones(sizes[1], dtype=dtype) if row[3] != 'None' else None
                                result['voxel_size'] = attribute
                                output = grid_cluster(pos.npu(), size.npu(),
                                                      start.npu() if start is not None else None,
                                                      end.npu() if end is not None else None)
                                torch.npu.synchronize()
                                golden = grid_reference(pos, size, start, end)
                            assert output.device.type == 'npu' and output.dtype == torch.long
                            actual = output.cpu()
                            same_shape = actual.shape == golden.shape
                            mismatches = int((actual != golden).sum()) if same_shape else None
                            result.update(status='passed' if same_shape and mismatches == 0 else 'failed',
                                          rule='integer_exact', actual_shape=list(actual.shape),
                                          golden_shape=list(golden.shape), mismatches=mismatches)
                            if same_shape and mismatches:
                                idx = torch.nonzero(actual != golden).flatten()[:10]
                                result.update(first_indices=idx.tolist(), actual=actual[idx].tolist(), golden=golden[idx].tolist())
                        except Exception as error:
                            result.update(status='error', error_type=type(error).__name__, reason=str(error)[:2000])
                        result['seconds'] = time.perf_counter()-start_time
                        results.append(result)
                        line = json.dumps(result, ensure_ascii=False)
                        log.write(line+'\n'); log.flush(); print(line, flush=True)
                    del pos
    report = dict(device=torch.npu.get_device_name(0), torch=torch.__version__,
                  torch_npu=torch_npu.__version__, parameters=parameters, results=results,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  resumed_from=args.resume_jsonl)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
