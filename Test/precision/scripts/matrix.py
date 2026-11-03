"""Document identities and expected counts for complete coverage auditing."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
FLOATS = ('float16', 'bfloat16', 'float32', 'float64')
DISTRIBUTIONS = ('uniform', 'normal')
COUNTS = dict(nearest=312, graclus=145, knn=936, fps=936, radius=880, grid_cluster=888, random_walk=82, subgraph=136)

def identity(op, row):
    keys = {'nearest': ('case', 'dtype', 'distribution'), 'graclus': ('case', 'weight_dtype'), 'knn': ('case', 'dtype', 'distribution', 'k'), 'fps': ('case', 'dtype', 'distribution', 'ratio'), 'radius': ('case', 'dtype', 'distribution', 'r', 'max_num_neighbors', 'ignore_same_index'), 'grid_cluster': ('case', 'dtype', 'distribution', 'voxel_size'), 'random_walk': ('case', 'dtype'), 'subgraph': ('case', 'dtype', 'skew')}[op]
    return tuple((row[k] for k in keys))

def expected_identities(op, tables):
    if op == 'subgraph':
        return {(case, dtype, skew) for case in range(35) for dtype in ('torch.int32', 'torch.int64') for skew in (False, True) if not (case in (32, 33) and skew)}
    names = dict(nearest='Nearest', graclus='Graclus', knn='knn', fps='Fps', radius='Radius', grid_cluster='grid_cluster', random_walk='Random_walk')
    choices = [t for t in tables if t['section'] == '功能验收场景汇总' and t['operator'].lower() == names[op].lower()]
    assert len(choices) == 1, (op, [t['operator'] for t in tables])
    rows = choices[0]['rows'][1:]
    keys = set()
    for (case, row) in enumerate(rows, 1):
        if op == 'graclus':
            for dtype in (*FLOATS, 'int32', 'int64') if row[2] != 'None' else ('none',):
                keys.add((case, dtype))
        elif op == 'random_walk':
            for dtype in ('int32', 'int64'):
                keys.add((case, dtype))
        else:
            for dtype in FLOATS:
                for distribution in DISTRIBUTIONS:
                    base = (case, dtype, distribution)
                    if op == 'nearest':
                        keys.add(base)
                    elif op == 'knn':
                        keys.update(((*base, k) for k in (2, 8, 64)))
                    elif op in ('fps', 'grid_cluster'):
                        keys.update(((*base, value) for value in (0.125, 0.25, 0.5)))
                    elif op == 'radius':
                        keys.update(((*base, r, cap, ignore) for (r, cap, ignore) in ((1.0, 32, False), (1.0, 32, True), (1.0, 100, False), (2.0, 32, False), (3.0, 32, False))))
    return keys
