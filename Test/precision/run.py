"""Precision entry for the synchronized project; helpers retain accepted rules."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ACCEPTANCE=HERE.parent
PROJECT=ACCEPTANCE.parent
SCRIPTS=HERE/'scripts'
OPS=('nearest','graclus','knn','fps','radius','grid_cluster','random_walk','subgraph')


def setup_worker():
    sys.path[:0]=[str(PROJECT),str(PROJECT/'Test/precision/cases'),str(SCRIPTS)]
    import torch_npu
    import torch_cluster
    assert Path(torch_cluster.__file__).resolve().parent==PROJECT/'torch_cluster'


def worker(op,output,probe=False):
    setup_worker()
    cases=PROJECT/'Test/precision/cases'
    if op in ('nearest','knn','fps','radius','grid_cluster'):
        path=SCRIPTS/'geometry.py';tree=ast.parse(path.read_text())
        if probe:
            function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='prepare_source')
            argument=function.args.args[0].arg
            statement=argument+' = '+argument+'.replace('+repr("table['rows'][1:]")+','+repr("table['rows'][1:2]")+')'
            function.body.insert(0,ast.parse(statement).body[0]);ast.fix_missing_locations(tree)
        sys.argv=['geometry.py','--project',str(PROJECT),'--op',op,'--output',str(output)]
        exec(compile(tree,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
    elif op in ('graclus','random_walk'):
        path=cases/('graclus.py' if op=='graclus' else 'walk.py')
        sys.argv=[str(path),'--output',str(output)]
        source=path.read_text()
        if probe:source=source.replace("table['rows'][1:]","table['rows'][1:2]")
        exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
    elif op=='subgraph':
        import torch
        import subgraph
        if probe:subgraph.ACCURACY=subgraph.ACCURACY[:1]
        output.write_text(json.dumps(dict(device=torch.npu.get_device_name(),
            backend='torch_cluster.subgraph_910b',accuracy=subgraph.accuracy()),indent=2)+'\n')
    elif op=='radius_int32':
        sys.argv=['radius_int32.py','--project',str(PROJECT),'--output',str(output)]
        if probe:sys.argv+=['--case','1']
        runpy.run_path(str(SCRIPTS/'radius_int32.py'),run_name='__main__')
    else:raise ValueError(op)



def verify_files():
    manifest=json.loads((ACCEPTANCE/'sync.json').read_text())
    expected={**manifest['tested_source_sha256'],**manifest['native_sha256'],**manifest['harness_sha256']}
    for name,value in expected.items():
        assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==value,('Changed deployed revision',name)


def check_controls(rows):
    n=[r for r in rows if r['op']=='nearest'];g=[r for r in rows if r['op']=='grid_cluster']
    c=[r for r in rows if r['op']=='graclus']
    return (len(rows)==7 and len(n)==3 and {r['case'] for r in n}=={14,15,34} and
        all(r['identical_to_950_input'] and len(r['checks'])==3 and
            all(t['mismatches']==0 for t in r['checks']) for r in n) and
        len(g)==2 and all(len(r['current_repeated'])==5 and
            all(v==r['expected'] for v in r['current_repeated']) for r in g) and
        len(c)==2 and all(len(r['checks'])==3 and all(t['passed'] for t in r['checks']) for r in c))


def regressions(out):
    command=[sys.executable,'-m','pytest','Test/precision/regression/test_repairs.py','Test/precision/regression/test_boundaries.py',
             '-q','--junitxml='+str(out/'regression.xml')]
    subprocess.run(command,cwd=PROJECT,check=True)
    cases=list(ET.parse(out/'regression.xml').getroot().iter('testcase'))
    assert len(cases)==327 and all(not any(c.tag in ('failure','error','skipped') for c in t) for t in cases)
    env={**os.environ,'ACCEPTANCE_PROJECT':str(PROJECT),'ACCEPTANCE_RESULTS':str(out),
         'ACCEPTANCE_FIXTURES':str(HERE/'fixtures')}
    subprocess.run([sys.executable,str(SCRIPTS/'controls.py')],cwd=PROJECT,env=env,check=True)
    controls=json.loads((out/'controls.json').read_text())['results']
    assert check_controls(controls),'Control fixture failure'
    return dict(regressions=dict(records=327,passed=True),controls=dict(records=7,passed=True))


def full_summary(out):
    sys.path.insert(0,str(SCRIPTS))
    from matrix import COUNTS,expected_identities,identity
    tables=json.loads((PROJECT/'Test/precision/cases/document_tables.json').read_text(encoding='utf-8-sig'))
    operators={}
    for op in OPS:
        data=json.loads((out/(op+'.json')).read_text());rows=data.get('results',data.get('accuracy',[]))
        keys=[identity(op,row) for row in rows]
        assert len(keys)==len(set(keys))==COUNTS[op] and set(keys)==expected_identities(op,tables),op
        assert all(row.get('exact') and row.get('edge_id_modes')==[True,False] if op=='subgraph'
                   else row['status'] in ('passed','validation_passed') for row in rows),op
        operators[op]=dict(records=len(rows),passed=True)
    rows=json.loads((out/'radius_int32.json').read_text())['results']
    expected={(case,'int32',dist,r,cap,ignore) for case,_,dist,r,cap,ignore in expected_identities('radius',tables)}
    keys=[identity('radius',row) for row in rows]
    assert len(keys)==len(set(keys))==220 and set(keys)==expected
    assert all(row['status']=='passed' and row['details']['strict_distance_passed'] for row in rows)
    return dict(operators=operators,matrix_records=4315,int32_radius_records=220,passed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=('regression','full'),default='regression')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--probe',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--worker',choices=(*OPS,'radius_int32'),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        assert args.output is not None
        args.output.parent.mkdir(parents=True,exist_ok=True);worker(args.worker,args.output,args.probe);return
    out=(args.output or HERE/'results/rerun').resolve()
    if args.suite=='full':assert not out.exists() or not any(out.iterdir()),'Use an empty --output directory'
    out.mkdir(parents=True,exist_ok=True)
    with (ACCEPTANCE/'device.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        verify_files()
        result=regressions(out)
        if args.suite=='full':
            for op in (*OPS,'radius_int32'):
                with (out/(op+'.log')).open('w') as log:
                    subprocess.run([sys.executable,str(Path(__file__)),'--worker',op,
                        '--output',str(out/(op+'.json'))],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            result.update(full_summary(out))
        verify_files();result.update(source_and_binaries_verified=True,project=str(PROJECT))
        (out/'run_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
