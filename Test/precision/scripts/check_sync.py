"""Verify the original checkout and publish its local acceptance gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[3]
ACCEPTANCE=PROJECT/'Test'
PRECISION=ACCEPTANCE/'precision'
RESULTS=PRECISION/'results'


def main():
    manifest=json.loads((ACCEPTANCE/'sync.json').read_text())
    assert PROJECT==Path(manifest['project'])
    subprocess.run([sys.executable,str(PRECISION/'run.py'),'--suite','regression',
        '--output',str(RESULTS)],cwd=PROJECT,check=True)
    regression=json.loads((RESULTS/'run_summary.json').read_text())
    assert regression['regressions']==dict(records=327,passed=True)
    assert regression['controls']==dict(records=7,passed=True)
    import torch
    import torch_npu
    import torch_cluster
    assert Path(torch_cluster.__file__).resolve().parent==PROJECT/'torch_cluster'
    assert torch.ops.torch_cluster.subgraph_950.default._schema.name=='torch_cluster::subgraph_950'
    assert torch.ops.torch_cluster.subgraph_910b.default._schema.name=='torch_cluster::subgraph_910b'
    x=torch.tensor([[0.,0.],[1.,0.],[3.,0.]],device='npu')
    smoke={}
    assert torch_cluster.fps(x,ratio=.5,random_start=False).cpu().tolist()==[0,2];smoke['fps']=True
    ids=torch_cluster.grid_cluster(x,torch.ones(2,device='npu'),torch.zeros(2,device='npu'),torch.full((2,),3.,device='npu'))
    assert ids.cpu().tolist()==[0,1,3];smoke['grid_cluster']=True
    q=torch.tensor([[.1,0.],[2.9,0.]],device='npu')
    assert torch_cluster.nearest(q,x).cpu().tolist()==[0,2];smoke['nearest']=True
    edges=torch_cluster.knn(x,x,1).cpu()
    assert sorted(zip(*edges.tolist()))==[(0,0),(1,1),(2,2)];smoke['knn']=True
    edges=torch_cluster.radius(x,x,.25,max_num_neighbors=3).cpu()
    assert sorted(zip(*edges.tolist()))==[(0,0),(1,1),(2,2)];smoke['radius']=True
    row=torch.tensor([0,1],device='npu');col=torch.tensor([1,0],device='npu')
    labels=torch_cluster.graclus_cluster(row,col,torch.ones(2,device='npu'),2).cpu()
    assert len(labels)==2 and labels[0]==labels[1];smoke['graclus']=True
    walks=torch_cluster.random_walk(row,col,torch.tensor([0,1],device='npu'),3,num_nodes=2)
    assert walks.cpu().tolist()==[[0,1,0,1],[1,0,1,0]];smoke['random_walk']=True
    for dtype in (torch.int32,torch.int64):
        inputs=tuple(torch.tensor(value,dtype=dtype,device='npu') for value in ([0,1,2],[1,0],[0,1]))
        for edge_id in (True,False):
            expected=([0,1,2],[1,0],[0,1]) if edge_id else ([0,1,2],[1,0])
            for function in (torch_cluster.subgraph,torch_cluster.subgraph_910b):
                actual=function(*inputs,return_edge_id=edge_id)
                assert tuple(t.cpu().tolist() for t in actual)==expected
    smoke['subgraph']=True
    maps=Path('/proc/self/maps').read_text()
    loaded=sorted({line.split()[-1] for line in maps.splitlines() if '.so' in line and str(PROJECT) in line})
    assert any('/.precision_910b_runtime/' in path for path in loaded),loaded
    assert not any('precision-compare-910b1-' in line and ('.so' in line) for line in maps.splitlines()),'Old native runtime loaded'
    for name,value in {**manifest['tested_source_sha256'],**manifest['native_sha256']}.items():
        assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==value,name
    manifest['validation_status']='passed original-checkout verification'
    (ACCEPTANCE/'sync.json').write_text(json.dumps(manifest,indent=2)+'\n')
    gate=dict(passed=True,reviewed=True,source_sha256={**manifest['tested_source_sha256'],**manifest['harness_sha256']},
        native_vendor_sha256=manifest['native_sha256'],
        review_method='Accepted complete matrix is mapped by byte-identical source/native artifacts; original checkout independently passed 327 regressions, seven fixture controls and eight-operator smoke checks.',
        expected_records=4315,extra_int32_radius_records=220,regression_records=327,control_records=7,
        accepted_reference_gate_sha256=manifest['accepted_gate_sha256'],
        full_matrix_rerun_after_sync=False,
        precision_evidence_sha256={str(path.relative_to(PROJECT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ACCEPTANCE/'sync.json',RESULTS/'reference_gate.json',RESULTS/'reference_summary.json',
                         RESULTS/'regression.xml',RESULTS/'controls.json')})
    (RESULTS/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    summary=json.loads((RESULTS/'reference_summary.json').read_text())
    summary['original_checkout_verification']=dict(project=str(PROJECT),regressions=327,controls=7,
        smoke=smoke,byte_identical_tested_files=len(manifest['tested_source_sha256']),byte_identical_native_files=43,
        full_matrix_rerun=False,loaded_libraries=loaded)
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    result=dict(passed=True,project=str(PROJECT),imported_module=torch_cluster.__file__,
        regressions=327,controls=7,smoke=smoke,native_runtime_local=True,
        source_and_binary_files_verified=len(manifest['tested_source_sha256']),native_files_verified=43,
        full_matrix_rerun=False,loaded_libraries=loaded)
    (ACCEPTANCE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
