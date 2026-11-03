"""Cover the native Radius registration's previously omitted Int32 inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
LEGACY_WRAPPER_SHA256='d46b81666795d40b773f37a1bac3a98e80efe5e97ed81303f176bcaa76b1c4f1'


class UniqueQueryKDTree:
    """Query each identical Int32 coordinate once; keep the original KD-tree.

    Returned indices still address the full reference array, including all
    duplicates. The original verifier recomputes strict Double distances and
    applies the per-query self exclusion after expanding the query results.
    """
    def __init__(self, data):
        from scipy.spatial import cKDTree
        self.tree=cKDTree(data)

    def query(self, points, k, workers=4):
        import numpy as np
        unique,inverse=np.unique(points,axis=0,return_inverse=True)
        distances,indices=self.tree.query(unique,k=k,workers=workers)
        return distances[inverse],indices[inverse]


def record_key(record):
    return tuple(record[name] for name in
        ('case','dtype','distribution','r','max_num_neighbors','ignore_same_index'))


def previous_records(output, project, script):
    """Retain only versioned, strictly passing rows; never discard a failure."""
    journal=output.with_suffix('.jsonl');manifest=output.with_suffix('.manifest.json')
    if not journal.exists() or not journal.stat().st_size:return [],None
    assert manifest.exists(),'Unversioned Int32 journal'
    old_bytes=manifest.read_bytes();old=json.loads(old_bytes)
    assert old['project']==str(project) and old['dtype']=='int32' and old['expected_records']==220
    assert old['original_source_sha256']==hashlib.sha256(script.read_bytes()).hexdigest()
    assert old['decision_rule_sha256']==hashlib.sha256((ROOT/'decision_precision.py').read_bytes()).hexdigest()
    current=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert old['wrapper_sha256'] in (LEGACY_WRAPPER_SHA256,current),'Unreviewed wrapper revision'
    raw=journal.read_bytes();assert raw.endswith(b'\n'),'Incomplete journal record'
    rows=[json.loads(line) for line in raw.splitlines() if line]
    assert len(rows)==len({record_key(row) for row in rows}),'Duplicate Int32 record'
    for row in rows:
        assert row['status']=='passed' and row['details']['strict_distance_passed'],row
    info=dict(records=len(rows),journal_prefix_bytes=len(raw),
        journal_prefix_sha256=hashlib.sha256(raw).hexdigest(),
        previous_manifest_sha256=hashlib.sha256(old_bytes).hexdigest(),
        previous_wrapper_sha256=old['wrapper_sha256'])
    archived=manifest.with_name(manifest.stem+'.previous.'+info['previous_manifest_sha256'][:12]+'.json')
    if archived.exists():assert archived.read_bytes()==old_bytes
    else:archived.write_bytes(old_bytes)
    return rows,info


def prepare_source(source, selected_case=None):
    replacements=[
        ("rule='exact_edge_validity_and_capped_completeness'",
         "rule='exact_int32_coordinates_and_capped_completeness'"),
        ('    results = []','    results = list(prior_records)'),
        ("with target.with_suffix('.jsonl').open('w') as log:",
         "with target.with_suffix('.jsonl').open('a') as log:"),
        ('                        start = time.perf_counter()',
         '                        if record_key(record) in prior_keys: continue\n'
         '                        start = time.perf_counter()')]
    if selected_case is not None:
        replacements.append(("for case, spec in enumerate(table['rows'][1:], 1):",
            "for case, spec in enumerate(table['rows'][1:], 1):\n"
            '            if case != selected_case: continue'))
    for before,after in replacements:
        assert source.count(before)==1,before
        source=source.replace(before,after)
    return source

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',default='repair-project')
    parser.add_argument('--output',required=True);parser.add_argument('--case',type=int)
    args=parser.parse_args();project=ROOT/args.project
    sys.path[:0]=[str(project),str(project/'Test/precision/cases')]
    import torch
    import torch_npu
    import torch_cluster
    import nearest
    import decision_precision
    from decision_precision import radius_review
    assert Path(torch_cluster.__file__).resolve().parent==project/'torch_cluster'
    # Radius imports this dictionary from nearest. This process only runs
    # Int32 Radius; the original files and the other matrices are unchanged.
    nearest.DTYPES={'int32':torch.int32}
    script=project/'Test/precision/cases/radius.py'
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    prior,resumed=previous_records(output,project,script)
    if args.case is not None:assert all(row['case']==args.case for row in prior)
    source=prepare_source(script.read_text(),args.case)
    namespace=dict(__name__='int32_radius_worker',__file__=str(script),
        prior_records=prior,prior_keys={record_key(row) for row in prior},
        record_key=record_key,selected_case=args.case)
    exec(compile(source,str(script),'exec'),namespace)
    decision_precision.cKDTree=UniqueQueryKDTree
    def verify(*inputs):
        review=radius_review(*inputs)
        assert review['passed'] and review['strict_distance_passed'],review
        return review
    namespace['verify']=verify
    output.with_suffix('.manifest.json').write_text(json.dumps(dict(
        project=str(project),dtype='int32',document_shapes=22,attributes=5,distributions=2,
        expected_records=220,original_source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decision_rule_sha256=hashlib.sha256((ROOT/'decision_precision.py').read_bytes()).hexdigest(),
        reference_optimization='Identical Int32 query coordinates share the unchanged cKDTree query; full reference multiplicities, Double distances and index/self checks retained',
        resumed_from=resumed,
        rule='Exact CPU Double geometry for Int32 coordinates; strict < r^2; exact edge constraints'),indent=2)+'\n')
    sys.argv=[str(script),'--dtype','int32','--output',str(output)]
    namespace['main']()
