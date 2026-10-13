"""CPU-only contracts for the review-stage ZIP adapters; no NPU operator calls.

Run independently of the project's device parametrization:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m unittest discover -s test -p test_pyg_zip_audit.py -v
"""
import ast
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import pyg_zip_audit as audit
import pyg_zip_accuracy as accuracy
import pyg_zip_runner as runner
import geometry_npu_performance as geometry
import subgraph_npu_performance as subgraph
import torch


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            call = dict(name=name, args=args, kwargs=kwargs)
            self.calls.append(call)
            return call
        return record


def zip_functions(name, names, pyg=None):
    """Execute only pinned constructor functions, with CPU torch and fake ops.

    ZIP main(), imports and CUDA/NPU entrypoints are never executed.
    """
    with zipfile.ZipFile(audit.DEFAULT_ZIP) as archive:
        tree = ast.parse(archive.read(name))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'torch': torch, 'pyg_lib': pyg}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<ZIP CPU constructors>', 'exec'), scope)
    return scope


class StaticContracts(unittest.TestCase):
    def test_all_77_cases_and_reference_rows(self):
        cases, _, spline = audit.load_spec()
        self.assertEqual(len(cases), 77)
        self.assertEqual(tuple(sum(c.op == op for c in cases) for op in audit.OPS), audit.COUNTS)
        self.assertEqual(sum(spline.values()), 44)
        runner.check_adapter_cases(cases)
        for case in cases:
            if case.op == 'random_walk':
                n = case.shape[0] // 4
                self.assertEqual(case.shape[0] // n, 4)
            self.assertGreater(case.reference_ms, 0)

    def test_changed_zip_rejected_before_importing_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'changed.zip'
            with zipfile.ZipFile(audit.DEFAULT_ZIP) as old, zipfile.ZipFile(target, 'w') as new:
                for name in audit.ZIP_SOURCE_HASHES:
                    data = old.read(name)
                    if name == 'pyg/geometry.py':
                        data = data.replace(b'knn(x, x, 8)', b'knn(x, x, 1)')
                    new.writestr(name, data)
            with self.assertRaisesRegex(ValueError, 'ZIP source changed'):
                audit.load_spec(target)

    def test_shape_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.md'
            text = audit.REFERENCE.read_text().replace('(4096,128)', '(4097,128)', 1)
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                audit.load_spec(reference_path=path)

    def test_default_entrypoints_cannot_import_backend(self):
        guard = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'torch_npu', 'torch_cluster', 'pyg_lib'):
        raise RuntimeError('Backend import before review: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, 'benchmarks')
entry = sys.argv[1]
sys.argv = [entry]
runpy.run_path(entry, run_name='__main__')
"""
        for name in ('pyg_zip_audit.py', 'pyg_zip_accuracy.py',
                     'geometry_npu_performance.py', 'subgraph_npu_performance.py'):
            with self.subTest(entrypoint=name):
                result = subprocess.run([sys.executable, '-c', guard, 'benchmarks/' + name],
                                        cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('未调用 NPU 算子', result.stdout)

    def test_execution_requires_explicit_output_and_separate_accuracy(self):
        for name, flags in (
                ('pyg_zip_accuracy.py', ['--check-only']),
                ('geometry_npu_performance.py', ['--benchmark']),
                ('subgraph_npu_performance.py', ['--benchmark', '--output', '/unused'])):
            with self.subTest(entrypoint=name):
                with self.assertRaises(SystemExit) as caught, patch('sys.stderr', io.StringIO()):
                    if name == 'pyg_zip_accuracy.py':
                        accuracy.main(flags)
                    elif name == 'geometry_npu_performance.py':
                        geometry.main(flags)
                    else:
                        subgraph.main(flags)
                self.assertEqual(caught.exception.code, 2)

    def test_accuracy_and_performance_modes_do_not_mix(self):
        with self.assertRaises(SystemExit), patch('sys.stderr', io.StringIO()):
            accuracy.main(['--benchmark'])
        with self.assertRaises(SystemExit), patch('sys.stderr', io.StringIO()):
            geometry.main(['--check-only'])
        for module in (geometry, subgraph):
            tree = ast.parse(Path(module.__file__).read_text())
            timing = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'latency')
            self.assertNotIn('verify', ast.unparse(timing))
            self.assertNotIn('manual_seed', ast.unparse(timing))
            self.assertNotIn('.item(', ast.unparse(timing))
            native_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                            ast.unparse(n.func).startswith('torch.ops.')]
            self.assertTrue(native_calls)
            self.assertTrue(all(ast.unparse(n.func).startswith('torch.ops.torch_cluster.') for n in native_calls))

    def test_accuracy_manifest_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = {'build': 'reviewed'}
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            (folder / 'summary.json').write_text(json.dumps(dict(kind='accuracy', manifest_unchanged=True, device='npu:0')))
            row = dict(op='knn', index=0, status='accuracy_passed', seed=2042)
            (folder / 'results.json').write_text(json.dumps([row]))
            args = SimpleNamespace(accuracy_report=folder, device='npu:0')
            self.assertEqual(runner.accuracy_records(args, runner.fingerprint(manifest))[('knn', 0)], row)
            with self.assertRaisesRegex(ValueError, 'stale'):
                runner.accuracy_records(args, runner.fingerprint({'build': 'changed'}))
            args.device = 'npu:1'
            with self.assertRaisesRegex(ValueError, 'device mismatch'):
                runner.accuracy_records(args, runner.fingerprint(manifest))

    def test_failed_accuracy_skips_performance_without_starting_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            vendor = folder / 'vendor'
            (vendor / 'op_api/lib').mkdir(parents=True)
            (vendor / 'op_api/lib/libcust_opapi.so').touch()
            case = audit.Case('knn', 0, (4096, 128), 'float32', 1.)
            args = SimpleNamespace(vendor=vendor, output=folder / 'performance', seed=42,
                                   device='npu:0', warmup=20, repeat=100)
            checked = {('knn', 0): {'status': 'accuracy_failed'}}
            with patch.object(runner, 'execution_manifest', return_value={'build': 'same'}), \
                    patch.object(runner, 'accuracy_records', return_value=checked), \
                    patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout='', stderr='')), \
                    patch.object(runner.subprocess, 'Popen') as popen, patch('sys.stdout', io.StringIO()):
                status = runner.campaign('performance', args, [case], {}, {})
            self.assertEqual(status, 1)
            popen.assert_not_called()
            result = json.loads((args.output / 'results.json').read_text())[0]
            self.assertEqual(result['status'], 'skipped_accuracy_not_passed')
            self.assertIsNone(result['npu_ms'])


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        geometry.torch = subgraph.torch = torch
        torch.set_num_threads(2)

    def test_inputs_and_parameters_match_zip_on_cpu(self):
        native, pyg_ops, pyg_sampler = Recorder(), Recorder(), Recorder()
        pyg = SimpleNamespace(ops=pyg_ops, sampler=pyg_sampler)
        source = zip_functions('pyg/geometry.py', {'csr', 'undirected_csr', 'make_call'}, pyg)
        cases = {'fps': (8, 3), 'grid_cluster': (8, 16), 'knn': (12, 3),
                 'nearest': (8, 5, 3), 'radius': (8, 3),
                 'graclus_cluster': (8, 56), 'random_walk': (32, 5, 3)}
        with patch.object(torch, 'ops', SimpleNamespace(torch_cluster=native)):
            for op, shape in cases.items():
                with self.subTest(op=op):
                    dtype = torch.int32 if op == 'random_walk' else torch.float32
                    torch.manual_seed(737)
                    adapted = geometry.make_call(op, shape, dtype, torch.device('cpu'))
                    actual = adapted()
                    torch.manual_seed(737)
                    original = source['make_call'](op, shape, dtype, torch.device('cpu'))()
                    count = len(adapted.inputs) - (1 if op == 'fps' else 0)
                    if op == 'nearest':
                        count = 2
                    for a, b in zip(adapted.inputs[:count], original['args'][:count]):
                        self.assertEqual(a.dtype, b.dtype)
                        self.assertTrue(torch.equal(a, b))
                    if op == 'fps':
                        self.assertEqual(actual['args'][2].item(), .5)
                        self.assertFalse(actual['args'][3])
                    elif op == 'knn':
                        self.assertEqual(actual['args'][2:], (None, None, 8, False, 1))
                        self.assertIs(adapted.inputs[0], adapted.inputs[1])
                    elif op == 'radius':
                        self.assertEqual(actual['args'][2:], (None, None, .5, 32, 1, False))
                    elif op == 'nearest':
                        self.assertEqual(actual['args'][2].tolist(), [0, 8])
                        self.assertEqual(actual['args'][3].tolist(), [0, 5])
                    elif op == 'random_walk':
                        self.assertEqual(actual['args'][3:], (3, 1., 1.))
                        self.assertEqual(adapted.inputs[2].tolist(), [0, 1, 2, 3, 4])
                        self.assertTrue(bool((adapted.inputs[0].diff() == 4).all()))
                    elif op == 'graclus_cluster':
                        rp, col, weight = adapted.inputs
                        rows = torch.repeat_interleave(torch.arange(8), rp.diff())
                        self.assertTrue(bool((rows != col).all()))
                        self.assertLess((rows * 8 + col).unique().numel(), 56)

    def test_subgraph_cpu_constructor_matches_zip(self):
        native = Recorder()
        original = zip_functions('pyg/subgraph.py', {'make_input'})['make_input']
        torch.manual_seed(99)
        with patch.object(torch, 'ops', SimpleNamespace(torch_cluster=native)):
            adapted = subgraph.make_call('subgraph', (13, 57, 6), torch.int32, torch.device('cpu'))
            call = adapted()
        torch.manual_seed(99)
        expected = original(13, 57, 6, torch.device('cpu'), torch.int32)
        self.assertEqual(call['name'], 'subgraph')
        self.assertEqual(len(call['args']), 3)
        for a, b in zip(adapted.inputs, expected):
            self.assertEqual(a.dtype, torch.int32)
            self.assertTrue(torch.equal(a, b))

    def test_timing_protocol_matches_zip_with_mock_clock(self):
        events = []
        fake = SimpleNamespace(no_grad=torch.no_grad)
        clock = iter((10., 10.012))
        for module in (geometry, subgraph):
            events.clear()
            clock = iter((10., 10.012))
            with patch.object(module, 'torch', fake), patch.object(module, 'sync', lambda device: events.append('sync')):
                with patch('time.perf_counter', lambda: next(clock)):
                    result = module.latency(lambda: events.append('call'), 'mock', 2, 3)
            self.assertEqual(events, ['call', 'call', 'sync', 'call', 'call', 'call', 'sync'])
            self.assertAlmostEqual(result, 4.)


class AccuracyOracles(unittest.TestCase):
    def check(self, op, shape, inputs, output):
        case = audit.Case(op, 0, shape, 'int32' if op in ('random_walk', 'subgraph') else 'float32', 1.)
        accuracy.verify(case, inputs, output, torch, expected_device='cpu')

    def test_fps_checks_greedy_selection(self):
        x = torch.tensor([[0., 0.], [1., 0.], [0., 2.], [3., 3.]])
        self.check('fps', (4, 2), (x,), torch.tensor([0, 3]))
        with self.assertRaisesRegex(ValueError, 'non-greedy'):
            self.check('fps', (4, 2), (x,), torch.tensor([0, 1]))

    def test_grid_int64_wrap_rejects_int32_truncation(self):
        x = torch.full((3, 16), .2)
        inputs = (x, torch.full((16,), .01), torch.zeros(16), torch.ones(16))
        number, stride = 0, 1
        for _ in range(16):
            number = (number + 20 * stride) % (2**64)
            stride = stride * 101 % (2**64)
        number = number if number < 2**63 else number - 2**64
        correct = torch.full((3,), number, dtype=torch.int64)
        self.check('grid_cluster', (3, 16), inputs, correct)
        with self.assertRaisesRegex(ValueError, 'int64/reference mismatch'):
            self.check('grid_cluster', (3, 16), inputs, correct.to(torch.int32).to(torch.int64))

    def test_knn_checks_all_eight_neighbors(self):
        x = torch.arange(12.).view(-1, 1)
        distances = torch.cdist(x.double(), x.double())
        ids = distances.topk(8, largest=False).indices
        edges = torch.stack((torch.arange(12).repeat_interleave(8), ids.reshape(-1)))
        self.check('knn', (12, 1), (x, x), edges)
        bad = edges.clone()
        bad[1, 1] = bad[1, 0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.check('knn', (12, 1), (x, x), bad)

    def test_nearest_exact_distance_check(self):
        x, y = torch.tensor([[0.], [3.]]), torch.tensor([[0.], [2.], [10.]])
        self.check('nearest', (2, 3, 1), (x, y), torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, 'reference mismatch'):
            self.check('nearest', (2, 3, 1), (x, y), torch.tensor([0, 2]))

    def test_graclus_rejects_nonadjacent_pair(self):
        inputs = (torch.tensor([0, 1, 2, 2, 2]), torch.tensor([1, 0]), torch.ones(2))
        self.check('graclus_cluster', (4, 2), inputs, torch.tensor([0, 0, 2, 3]))
        with self.assertRaisesRegex(ValueError, 'not adjacent'):
            self.check('graclus_cluster', (4, 2), inputs, torch.tensor([0, 1, 0, 3]))

    def test_graclus_rejects_trivial_singleton_output(self):
        inputs = (torch.tensor([0, 1, 2, 2, 2]), torch.tensor([1, 0]), torch.ones(2))
        with self.assertRaisesRegex(ValueError, 'not maximal'):
            self.check('graclus_cluster', (4, 2), inputs, torch.arange(4))

    def test_radius_checks_distances_counts_and_duplicates(self):
        x = torch.tensor([[0.], [.2], [2.]])
        edges = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]])
        self.check('radius', (3, 1), (x, x), edges)
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            self.check('radius', (3, 1), (x, x), edges[:, :-1])
        bad = edges.clone()
        bad[1, 1] = 0
        with self.assertRaisesRegex(ValueError, 'repeated edge'):
            self.check('radius', (3, 1), (x, x), bad)

    def test_rw_checks_native_edge_ids_and_all_transitions(self):
        inputs = (torch.tensor([0, 4, 8, 12], dtype=torch.int32),
                  torch.tensor([0, 1, 2, 0] * 3, dtype=torch.int32),
                  torch.tensor([0, 1], dtype=torch.int32))
        nodes = torch.tensor([[0, 1, 2, 0], [1, 0, 2, 0]], dtype=torch.int32)
        edges = torch.tensor([[1, 6, 11], [4, 2, 8]], dtype=torch.int32)
        self.check('random_walk', (12, 2, 3), inputs, (nodes, edges))
        bad = edges.clone()
        bad[0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'edge ID/node mismatch'):
            self.check('random_walk', (12, 2, 3), inputs, (nodes, bad))

    def test_radius_rejects_duplicates_across_copy_chunks(self):
        x = torch.zeros(32768, 1)
        query = torch.cat((torch.arange(32768).repeat(2), torch.tensor([0])))
        neighbor = torch.cat((torch.zeros(32768, dtype=torch.int64),
                              torch.ones(32768, dtype=torch.int64), torch.tensor([0])))
        with self.assertRaisesRegex(ValueError, 'repeated edge'):
            self.check('radius', (32768, 1), (x, x), torch.stack((query, neighbor)))

    def test_subgraph_checks_capacity_count_and_order(self):
        inputs = (torch.tensor([0, 2, 3, 5], dtype=torch.int32),
                  torch.tensor([1, 2, 0, 0, 2], dtype=torch.int32),
                  torch.tensor([2, 0], dtype=torch.int32))
        output = (torch.tensor([0, 2, 3], dtype=torch.int32),
                  torch.tensor([1, 0, 0, -999, -999], dtype=torch.int32),
                  torch.tensor([3, 4, 1, -999, -999], dtype=torch.int32),
                  torch.tensor([3], dtype=torch.int32))
        self.check('subgraph', (3, 5, 2), inputs, output)
        with self.assertRaisesRegex(ValueError, 'device count mismatch'):
            self.check('subgraph', (3, 5, 2), inputs, (*output[:3], torch.tensor([2], dtype=torch.int32)))

    def test_production_checker_rejects_cpu_fallback(self):
        case = audit.Case('fps', 0, (4, 2), 'float32', 1.)
        with self.assertRaisesRegex(ValueError, 'CPU fallback'):
            accuracy.verify(case, (torch.zeros(4, 2),), torch.tensor([0, 1]), torch)


if __name__ == '__main__':
    unittest.main()
