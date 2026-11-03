"""Replay verified x86-generated inputs and deterministic structural probes."""
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch_npu

import os
ROOT=Path(__file__).resolve().parent
PROJECT=Path(os.environ['ACCEPTANCE_PROJECT'])
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'Test/precision/cases'))
from torch_cluster import nearest,grid_cluster,graclus_cluster
from nearest import reference as nearest_reference
from graclus import verify as graclus_verify
import torch_cluster
assert Path(torch_cluster.__file__).resolve().parent == PROJECT/'torch_cluster'
torch.set_num_threads(8)
records=[]
output=Path(os.environ['ACCEPTANCE_RESULTS'])/'controls.json'
output.parent.mkdir(parents=True,exist_ok=True)
with output.with_suffix('.jsonl').open('w') as log:
    def emit(value):
        records.append(value);line=json.dumps(value)
        print(line,flush=True);log.write(line+'\n');log.flush()
    for dtype in (torch.float32,torch.float16):
        inputs=(torch.tensor([[-.5,1.9],[-1.5,-2.1]],dtype=dtype).npu(),
                torch.ones(2,dtype=dtype).npu(),torch.zeros(2,dtype=dtype).npu(),
                torch.full((2,),3.,dtype=dtype).npu())
        values=[grid_cluster(*inputs).cpu().tolist() for _ in range(5)]
        emit(dict(op='grid_cluster',dtype=str(dtype),expected=[4,-9],
                  baseline950_first=[5,-10],baseline950_later=[4,-9],current_repeated=values))
    baseline=json.loads((Path(os.environ['ACCEPTANCE_FIXTURES'])/'nearest_reference.json').read_text())['results']
    for fixture in torch.load(Path(os.environ['ACCEPTANCE_FIXTURES'])/'nearest.pt',map_location='cpu',weights_only=True):
        inputs=fixture['inputs'];digest=hashlib.sha256()
        for tensor in inputs:
            if tensor is not None:digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        old=next(r for r in baseline if r['case']==fixture['case'] and r['status']=='failed')
        assert digest.hexdigest()==fixture['sha256']==old['input_sha256']
        golden=nearest_reference(*inputs)
        device=tuple(t.npu() if t is not None else None for t in inputs)
        checks=[]
        for _ in range(3):
            actual=nearest(*device).cpu()
            ids=torch.nonzero(actual!=golden).flatten()
            checks.append(dict(mismatches=len(ids),first_indices=ids[:10].tolist(),
                               actual=actual[ids[:10]].tolist(),golden=golden[ids[:10]].tolist(),
                               original_query_outputs=actual[old['first_indices']].tolist()))
        emit(dict(op='nearest',case=old['case'],dtype=old['dtype'],input_sha256=digest.hexdigest(),
                  identical_to_950_input=True,baseline950_actual=old['actual'],
                  baseline950_golden=old['golden'],checks=checks))
    for case,n,e,dtype in [(28,128,16256,torch.float16),(29,256,65280,torch.float32)]:
        g=torch.Generator().manual_seed(20261102+case)
        src=torch.randint(n,(e//2,),generator=g);dst=torch.randint(n-1,(e//2,),generator=g)
        dst+=(dst>=src).long()
        row,col=torch.cat((src,dst)),torch.cat((dst,src))
        weight=torch.rand(e//2,generator=g,dtype=dtype).repeat(2)
        order=torch.argsort(row,stable=True);row,col,weight=row[order],col[order],weight[order]
        sha=hashlib.sha256()
        for tensor in (row,col,weight):sha.update(tensor.view(torch.uint8).numpy().tobytes())
        checks=[]
        device=tuple(t.npu() for t in (row,col,weight))
        for _ in range(3):
            actual=graclus_cluster(*device,n).cpu()
            try:detail=graclus_verify(row,col,weight,actual,n);checks.append(dict(passed=True,details=detail))
            except AssertionError as error:checks.append(dict(passed=False,reason=str(error)))
        emit(dict(op='graclus',case=case,dtype=str(dtype),input_sha256=sha.hexdigest(),checks=checks,
                  input_provenance='ARM regeneration of uniform/integer construction; historical 950 graph hash absent'))
output.write_text(json.dumps(dict(device=torch.npu.get_device_name(),torch=torch.__version__,
                                  torch_npu=torch_npu.__version__,results=records),indent=2)+'\n')
