"""Run document inputs with explicit decision rules; retain strict diagnostics.

The original source and baseline reports remain untouched. Transformations
only add decision metrics, journal resume and reuse a CPU KD-tree reference
query across k values. The latter is self-tested against the original oracle.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import torch

ROOT=Path(__file__).resolve().parent
FILES=dict(nearest='nearest.py',knn='knn.py',fps='fps_grid.py',
           grid_cluster='fps_grid.py',radius='radius.py')


def record_key(row):
    return tuple(row.get(k) for k in ('operator','case','dtype','distribution','k',
                                     'ratio','voxel_size','r','max_num_neighbors','ignore_same_index'))


def cached_knn_reference(original,max_k):
    held=None;cached=None
    def reference(x,y,bx,by,k,graph):
        nonlocal held,cached
        args=(x,y,bx,by)
        if held is None or not all(a is b for a,b in zip(args,held)):
            held=args
            cached=original(x,y,bx,by,max_k+1,False)
        q=cached[0];counts=torch.bincount(q,minlength=len(y))
        starts=counts.cumsum(0)-counts
        rank=torch.arange(len(q))-starts[q]
        edges=cached[:,rank<k+int(graph)]
        if graph:
            edges=edges[:,edges[0]!=edges[1]].flip(0)
        return edges
    return reference


def selftest_knn_cache(original):
    x=torch.tensor([[0.],[0.],[1.],[-1.],[2.],[10.],[10.],[11.]],dtype=torch.float64)
    batch=torch.tensor([0]*5+[1]*3)
    for batches in (False,True):
        for graph in (False,True):
            for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
                points=x.to(dtype);b=batch if batches else None
                cached=cached_knn_reference(original,64)
                for k in (2,8,64):
                    assert torch.equal(original(points,points,b,b,k,graph),cached(points,points,b,b,k,graph)),(batches,graph,dtype,k)


def cached_fps_reference(original,max_ratio=.5):
    held=None;held_ptr=None;cached=None;limit=0.
    def reference(pos,ptr,ratio):
        nonlocal held,held_ptr,cached,limit
        if held is not pos or held_ptr is None or not torch.equal(ptr,held_ptr) or ratio>limit:
            held=pos;held_ptr=ptr.clone();limit=max(max_ratio,ratio)
            cached=original(pos,ptr,limit)
        chunks=[];offset=0
        for first,last in zip(ptr[:-1].tolist(),ptr[1:].tolist()):
            full=int(math.ceil((last-first)*limit))
            count=int(math.ceil((last-first)*ratio))
            chunks.append(cached[offset:offset+count]);offset+=full
        return torch.cat(chunks) if chunks else torch.empty(0,dtype=torch.long)
    return reference


def selftest_fps_cache(original):
    for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        for offsets in ([0,9],[0,4,9],[0,0,4,4,9],[0,0,0]):
            ptr=torch.tensor(offsets);n=offsets[-1]
            x=torch.arange(n,dtype=torch.float64).reshape(-1,1).to(dtype)
            cached=cached_fps_reference(original)
            for ratio in (0.,.125,.25,.5,1.):
                assert torch.equal(original(x,ptr,ratio),cached(x,ptr,ratio)),(dtype,offsets,ratio)


def prepare_source(source,op,review=True):
    if op in ('nearest','knn','radius'):
        assert source.count('    results = []')==1
        source=source.replace('    results = []','    results = list(resume_records)')
        var='record' if op=='radius' else 'result'
        pattern=r'(?m)^( +)start = time.perf_counter\(\)$'
        source,count=re.subn(pattern,lambda m:m[1]+f'if record_key({var}) in completed: continue\n'+m[0],source)
        assert count==1,count
    source=source.replace("target.with_suffix('.jsonl').open('w')","target.with_suffix('.jsonl').open('a')")
    if op=='nearest' and review:
        needle='                                              golden=golden[ids].tolist())'
        assert source.count(needle)==1
        source=source.replace(needle,needle+'''
                            result['strict_status']=result['status']
                            review=nearest_review(*inputs,actual,golden)
                            result.update(decision_precision=review,status='passed' if review['passed'] else 'failed',rule='distance_decision_and_exact_structure')''')
    elif op=='knn' and review:
        needle='                        except Exception as error:'
        assert source.count(needle)==1
        source=source.replace(needle,'''                            result['strict_status']=result['status']
                            review=knn_review(x,y,bx,by,k,graph,actual.cpu(),golden)
                            result.update(decision_precision=review,status='passed' if review['passed'] else 'failed',rule='distance_decision_and_exact_structure')
'''+needle)
    elif op=='fps' and review:
        needle='                        except Exception as error:'
        assert source.count(needle)==1
        source=source.replace(needle,'''                            result['strict_status']=result['status']
                            if result['status']=='passed':
                                review=dict(passed=True,rule='optimality_proved_by_exact_reference_sequence')
                            else:
                                review=fps_review(pos,ptr,attribute,actual)
                            result.update(decision_precision=review,status='passed' if review['passed'] else 'failed',rule='conditional_farthest_score_and_exact_structure')
'''+needle)
    elif op=='radius' and review:
        source=source.replace("rule='exact_edge_validity_and_capped_completeness'","rule='distance_boundary_and_exact_structure'")
    return source


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',default='repair-project')
    parser.add_argument('--op',required=True,choices=list(FILES))
    parser.add_argument('--output',required=True)
    parser.add_argument('--dtype')
    parser.add_argument('--case',type=int)
    parser.add_argument('--strict-only',action='store_true')
    args=parser.parse_args()
    project=ROOT/args.project;directory=project/'Test/precision/cases';script=directory/FILES[args.op]
    sys.path[:0]=[str(project),str(directory)]
    from decision_precision import nearest_review,knn_review,fps_review,radius_review
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    journal=output.with_suffix('.jsonl')
    records=[json.loads(line) for line in journal.read_text().splitlines() if line] if journal.exists() else []
    source=script.read_text();rewritten=prepare_source(source,args.op,review=not args.strict_only)
    # Baseline resume can retain its original strict decision; the wrapper's
    # mixed review is deliberately disabled there.
    namespace=dict(__name__='precision_worker',__file__=str(script),resume_records=records,
                   completed={record_key(r) for r in records},record_key=record_key,
                   nearest_review=nearest_review,knn_review=knn_review,fps_review=fps_review)
    exec(compile(rewritten,str(script), 'exec'),namespace)
    if args.op=='knn':
        selftest_knn_cache(namespace['reference'])
        namespace['reference']=cached_knn_reference(namespace['reference'],64)
    elif args.op=='fps':
        selftest_fps_cache(namespace['fps_reference'])
        namespace['fps_reference']=cached_fps_reference(namespace['fps_reference'])
    elif args.op=='radius' and not args.strict_only:
        def verify(*inputs):
            review=radius_review(*inputs)
            assert review['passed'],review
            return review
        namespace['verify']=verify
    manifest=dict(original_script_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rewritten_sha256=hashlib.sha256(rewritten.encode()).hexdigest(),
                  project=str(project),operator=args.op,resumed_records=len(records),
                  strict_only=args.strict_only,
                  decision_rule_sha256=hashlib.sha256((ROOT/'decision_precision.py').read_bytes()).hexdigest(),
                  rule=('Original exact indices and original business rules; CPU KD-tree reuse and journal resume only' if args.strict_only else
                        'Float32 distance decisions: CANN mixed tolerance; integer structure: exact; Double scores: exact'),
                  kd_tree_reuse_selftest=args.op=='knn',fps_prefix_reuse_selftest=args.op=='fps')
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sys.argv=[str(script),'--output',str(output)]
    if args.op in ('fps','grid_cluster'):
        sys.argv+=['--op',args.op]
        if records:sys.argv+=['--resume-jsonl',str(journal)]
    if args.dtype:sys.argv+=['--dtype',args.dtype]
    if args.case is not None:sys.argv+=['--case',str(args.case)]
    namespace['main']()
