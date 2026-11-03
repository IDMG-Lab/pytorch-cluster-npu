"""Read actual uploaded pyg-lib benchmark constants without importing pyg_lib."""
import ast
import hashlib
from pathlib import Path


def geometry_parameters():
    path = Path(__file__).resolve().parents[2] / 'benchmark/geometry.py'
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id in ('K', 'M', 'CASES')]
    if any(isinstance(node, (ast.Call, ast.Attribute)) for statement in assignments for node in ast.walk(statement)):
        raise ValueError('Benchmark constants contain executable calls; inspect before use')
    namespace = {'__builtins__': {}}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), 'exec'), namespace)
    cases = namespace['CASES']
    return dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                fps_ratios=sorted({row[3] for row in cases['fps']}),
                grid_sizes=sorted({row[2] for row in cases['grid_cluster']}),
                knn_k=sorted({row[2] for row in cases['knn']}),
                radius_attributes=sorted({row[2:] for row in cases['radius']}),
                random_walk_p=1., random_walk_q=1., fps_random_start=False,
                cases=cases)
