"""All uploaded pyg-lib performance cases; requires a reviewed precision gate."""
import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time
from types import SimpleNamespace

import torch
import torch_npu
import torch_cluster

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT


def load_benchmark(path,names):
    source=path.read_text()
    parsed=ast.parse(source)
    selected=[]
    for node in parsed.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):selected.append(node)
        if isinstance(node,ast.FunctionDef) and node.name in names:selected.append(node)
    ns=dict(torch=torch)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
    return ns,hashlib.sha256(source.encode()).hexdigest()


def baseline_rows(path):
    rows={}
    for line in path.read_text().splitlines():
        parts=line.split('\t')
        if len(parts)!=5 or not parts[-1].endswith(' ms'):continue
        rows[(parts[0],int(parts[1]))]=dict(case=ast.literal_eval(parts[2]),dtype=parts[3],ms=float(parts[4][:-3]))
    return rows


def nearest(x,y,ptr_x=None,ptr_y=None):
    if ptr_x is None:ptr_x=torch.tensor([0,len(x)],device=x.device)
    if ptr_y is None:ptr_y=torch.tensor([0,len(y)],device=y.device)
    return torch.ops.torch_cluster.nearest(x,y,ptr_x,ptr_y)


def knn(x,y,k,ptr_x=None,ptr_y=None):
    return torch.ops.torch_cluster.knn(x,y,ptr_x,ptr_y,k,False,1)


def verify_resume_manifest(output, manifest):
    path=output.with_suffix('.manifest.json')
    journal=output.with_suffix('.jsonl')
    if path.exists():
        assert json.loads(path.read_text())==manifest, 'Benchmark revision or baseline changed; use a new output path'
    else:
        assert not journal.exists() or not journal.stat().st_size, 'Unversioned benchmark journal; use a new output path'
        path.write_text(json.dumps(manifest,indent=2)+'\n')


def verify_tested_files(gate):
    assert gate['source_sha256'],'No precision-tested source fingerprint'
    for name,sha in gate['source_sha256'].items():
        assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha,('Changed since precision',name)
    assert gate['native_vendor_sha256'],'Fingerprint the tested native kernels before benchmarking'
    for name,sha in gate['native_vendor_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,('Native kernel changed since precision',name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--precision-gate',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    lease=(ROOT/'Test/device.lock').open('a')
    print(json.dumps(dict(event='waiting_for_exclusive_device')),flush=True)
    fcntl.flock(lease,fcntl.LOCK_EX)
    gate=json.loads(Path(args.precision_gate).read_text())
    assert gate['passed'] is True and gate['reviewed'] is True,'Complete reviewed precision acceptance first'
    assert Path(torch_cluster.__file__).resolve().parent==PROJECT/'torch_cluster'
    verify_tested_files(gate)
    geometry,geometry_sha=load_benchmark(ROOT/'Test/benchmark/geometry.py',
        {'K','M','CASES','sync','csr','undirected_csr','make_call'})
    subgraph,subgraph_sha=load_benchmark(ROOT/'Test/benchmark/subgraph.py',{'M','CASES','make_input'})
    geometry['pyg_lib']=SimpleNamespace(ops=SimpleNamespace(
        fps=lambda x,ptr,ratio,random:torch_cluster.fps(x,ptr=ptr,ratio=ratio,random_start=random),
        grid_cluster=torch_cluster.grid_cluster,knn=knn,nearest=nearest,
        graclus_cluster=torch.ops.torch_cluster.graclus,radius=torch_cluster.radius),
        sampler=SimpleNamespace(random_walk=lambda rp,col,seed,length:torch.ops.torch_cluster.random_walk(rp,col,seed,length,1.,1.)))
    baselines={**baseline_rows(ROOT/'Test/benchmark/baseline/geometry_cuda.tsv'),
               **baseline_rows(ROOT/'Test/benchmark/baseline/subgraph_cpu.tsv')}
    jobs=[(op,row,case) for op,cases in geometry['CASES'].items() for row,case in enumerate(cases)]
    jobs += [('subgraph',row,case) for row,case in enumerate(subgraph['CASES'])]
    assert len(jobs)==84 and len(baselines)==84,(len(jobs),len(baselines))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);journal=output.with_suffix('.jsonl')
    manifest=dict(precision_gate_sha256=hashlib.sha256(Path(args.precision_gate).read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  geometry_script_sha256=geometry_sha,subgraph_script_sha256=subgraph_sha,
                  baseline_sha256={name:hashlib.sha256((ROOT/'Test/benchmark/baseline'/name).read_bytes()).hexdigest()
                                   for name in ('geometry_cuda.tsv','subgraph_cpu.tsv')},
                  device=torch.npu.get_device_name(),architecture=platform.machine(),
                  torch=torch.__version__,torch_npu=torch_npu.__version__,warmup=20,iterations=100)
    verify_resume_manifest(output,manifest)
    records=[json.loads(line) for line in journal.read_text().splitlines() if line] if journal.exists() else []
    done={(r['op'],r['row']) for r in records}
    expected={(op,row) for op,row,_ in jobs}
    assert len(done)==len(records) and done<=expected,'Duplicate or unknown benchmark case in journal'
    with journal.open('a') as log:
        for op,row,case in jobs:
            if (op,row) in done:continue
            assert tuple(case)==tuple(baselines[(op,row)]['case']),(op,row,case,baselines[(op,row)])
            seed=20261102+row;torch.manual_seed(seed);torch.npu.manual_seed_all(seed)
            dtype=torch.int32 if op in ('random_walk','subgraph') else torch.float32
            assert baselines[(op,row)]['dtype']==str(dtype),('Baseline dtype mismatch',op,row)
            item=dict(op=op,row=row,case=case,dtype=str(dtype),seed=seed,warmup=20,iterations=100,
                      historical_pyg_ms=baselines[(op,row)]['ms'],required_relative_performance=.1,
                      baseline_device='CPU' if op=='subgraph' else 'CUDA')
            print(json.dumps(dict(event='start',**item)),flush=True)
            try:
                if op=='subgraph':
                    data=subgraph['make_input'](*case,torch.device('npu'),dtype)
                    fn=lambda:torch_cluster.subgraph_910b(*data,True)
                else:fn=geometry['make_call'](op,case,dtype,torch.device('npu'))
                with torch.no_grad():
                    for _ in range(20):fn()
                    torch.npu.synchronize();start=time.perf_counter()
                    for _ in range(100):fn()
                    torch.npu.synchronize();ms=1000*(time.perf_counter()-start)/100
                ratio=item['historical_pyg_ms']/ms
                item.update(status='measured',npu_public_wall_ms=ms,relative_performance=ratio,performance_pass=ratio>=.1)
                del fn
                if op=='subgraph':del data
                torch.npu.empty_cache()
            except Exception as error:item.update(status='error',error_type=type(error).__name__,reason=str(error),performance_pass=False)
            records.append(item);line=json.dumps(item);print(line,flush=True);log.write(line+'\n');log.flush()
    verify_tested_files(gate)
    assert all(hashlib.sha256((ROOT/'Test/benchmark/baseline'/name).read_bytes()).hexdigest()==sha
               for name,sha in manifest['baseline_sha256'].items()),'Baseline changed during benchmarking'
    report=dict(device=torch.npu.get_device_name(),architecture=platform.machine(),torch=torch.__version__,torch_npu=torch_npu.__version__,
                cases_expected=84,results=records,all_cases_pass=len(records)==84 and all(r['performance_pass'] for r in records),
                geometry_script_sha256=geometry_sha,subgraph_script_sha256=subgraph_sha,precision_gate=gate,
                run_manifest=manifest,source_unchanged=True,
                limitation='Historical CUDA/CPU logs did not record input hashes or seeds. Same scripts, shapes, attributes, dtypes and timing protocol; identical random bytes cannot be established.')
    output.write_text(json.dumps(report,indent=2)+'\n')
