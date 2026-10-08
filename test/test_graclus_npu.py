"""NPU Graclus: upstream matching invariants, exact phase specification, dtype safety."""
from pathlib import Path
import sys
import pytest
import torch
import torch_npu  # noqa: F401
import torch_cluster
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from graclus_reference import assert_matching, matching_reference, phase_reference

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason='NPU unavailable')
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int32, torch.int64]


def complete(n, dtype):
    rowptr = torch.arange(n + 1) * (n - 1)
    rows, columns, values = [], [], []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            rows.append(u); columns.append(v)
            lo, hi = min(u, v), max(u, v)
            values.append((lo * 17 + hi * 11) % 13)
    return rowptr, torch.tensor(columns, dtype=torch.long), torch.tensor(values, dtype=dtype)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('weighted', [False, True])
def test_matches_independent_npu_algorithm_reference(dtype, weighted):
    rp, col, weight = complete(7, dtype)
    if not weighted:
        weight = None
    expected = matching_reference(rp, col, weight)
    actual = torch.ops.torch_cluster.graclus(rp.npu(), col.npu(), weight.npu() if weighted else None).cpu()
    assert_matching(rp, col, weight, actual)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('dtype', DTYPES)
def test_public_coo_interface_and_jit(dtype):
    row = torch.tensor([0,0,1,1,1,2,2,2,3,3,0])
    col = torch.tensor([1,2,0,2,3,0,1,3,1,2,0])
    weight = torch.tensor([1,2,1,3,2,2,3,1,2,1,9], dtype=dtype)
    degree = torch.bincount(row[:-1], minlength=4)
    rp = torch.cat([torch.zeros(1,dtype=torch.long),degree.cumsum(0)])
    for fn in (torch_cluster.graclus_cluster, torch.jit.script(torch_cluster.graclus_cluster)):
        got = fn(row.npu(), col.npu(), weight.npu(), 4)
        assert got.device.type == 'npu' or got.device.type == 'privateuseone'
        assert_matching(rp, col[:-1], weight[:-1], got)


@pytest.mark.parametrize('num_nodes', [0,1,100])
def test_isolated_nodes(num_nodes):
    rp, col = torch.zeros(num_nodes+1,dtype=torch.long),torch.empty(0,dtype=torch.long)
    got = torch.ops.torch_cluster.graclus(rp.npu(),col.npu(),None).cpu()
    assert torch.equal(got,torch.arange(num_nodes))


@pytest.mark.parametrize('dtype', [torch.float64,torch.int64])
def test_wide_weights_do_not_round_to_float32(dtype):
    rp,col,weight=complete(6,dtype)
    if dtype == torch.float64:
        weight=1.0+weight*2.0**-40
    else:
        weight=2**53+weight
    expected=matching_reference(rp,col,weight)
    got=torch.ops.torch_cluster.graclus(rp.npu(),col.npu(),weight.npu()).cpu()
    assert torch.equal(got,expected)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_special_weights(dtype):
    rp,col,weight=complete(7,dtype)
    weight[::5]=float('nan');weight[::7]=float('inf');weight[::11]=-1
    expected=matching_reference(rp,col,weight)
    got=torch.ops.torch_cluster.graclus(rp.npu(),col.npu(),weight.npu()).cpu()
    assert_matching(rp,col,weight,got,maximal=False)
    assert torch.equal(got,expected)


def test_directed_cycle_terminates_with_legal_singletons():
    rp=torch.arange(4);col=torch.tensor([1,2,0])
    got=torch.ops.torch_cluster.graclus(rp.npu(),col.npu(),None).cpu()
    assert_matching(rp,col,None,got,maximal=False)
    assert torch.equal(got,matching_reference(rp,col))


@pytest.mark.parametrize('n',[255,256,257,24575,24576,24577,1048575,1048576,1048577])
def test_tile_residency_and_core_boundaries(n):
    generator=torch.Generator().manual_seed(817)
    u=torch.randint(n,(1301,),generator=generator)
    v=torch.randint(n-1,(1301,),generator=generator);v += v>=u
    row,col=torch.cat([u,v]),torch.cat([v,u])
    order=torch.argsort(row,stable=True);row,col=row[order],col[order]
    rp=torch.cat([torch.zeros(1,dtype=torch.long),torch.bincount(row,minlength=n).cumsum(0)])
    for _ in range(3):
        got=torch.ops.torch_cluster.graclus(rp.npu(),col.npu(),None).cpu()
        assert_matching(rp,col,None,got)


def test_multiedges_do_not_change_adjacency_validation():
    rp,col,w=complete(7,torch.float32)
    row=torch.repeat_interleave(torch.arange(7),rp[1:]-rp[:-1])
    row,col,w=row.repeat_interleave(3),col.repeat_interleave(3),w.repeat_interleave(3)
    rp=torch.cat([torch.zeros(1,dtype=torch.long),torch.bincount(row,minlength=7).cumsum(0)])
    got=torch.ops.torch_cluster.graclus(rp.npu(),col.npu(),w.npu()).cpu()
    assert_matching(rp,col,w,got)
    assert torch.equal(got,matching_reference(rp,col,w))


@pytest.mark.parametrize('rp,col', [([1,1],[0]),([0,2,1],[0]),([0,1],[1]),([0,-1],[]),([1],[])])
def test_invalid_csr_raises(rp,col):
    with pytest.raises(RuntimeError,match='invalid CSR'):
        torch.ops.torch_cluster.graclus(torch.tensor(rp,dtype=torch.long).npu(),torch.tensor(col,dtype=torch.long).npu(),None)


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('partial',[False,True])
def test_actual_select_and_commit_match_phase_reference(dtype,partial):
    rp,col,weight=complete(7,dtype)
    if dtype==torch.float64: weight=1.0+weight*2.0**-40
    if dtype==torch.int64: weight=2**53+weight
    state=torch.full((7,),-1,dtype=torch.int32)
    if partial: state[0]=0
    expected_candidate,expected_state=phase_reference(rp,col,weight,state.tolist())
    candidate,after,counts=torch.ops.torch_cluster._graclus_phase(rp.npu(),col.npu(),weight.npu(),state.npu())
    assert candidate.cpu().tolist()==expected_candidate
    assert after.cpu().tolist()==expected_state


@pytest.mark.parametrize('row_length',[63,64,65,127,128,129,255,256,257,1023,1024,1025])
def test_weighted_long_row_vector_padding(row_length):
    # A star exercises long CSR rows without constructing a dense test graph.
    # Include a self-loop, masked vertices, NaNs, infinity, negatives and ties.
    n=row_length
    col=torch.cat([torch.arange(n),torch.zeros(n-1,dtype=torch.long)])
    rp=torch.cat([torch.zeros(1,dtype=torch.long),torch.arange(n,2*n)])
    weight=(torch.arange(n)%19).float()
    weight[::7]=float('nan');weight[::13]=float('inf');weight[::17]=-1
    weight[0]=float('inf')
    weight=torch.cat([weight,weight[1:]])
    state=torch.full((n,),-1,dtype=torch.int32)
    state[11::11]=torch.arange(11,n,11,dtype=torch.int32)
    expected_candidate,expected_state=phase_reference(rp,col,weight,state.tolist())
    candidate,after,_=torch.ops.torch_cluster._graclus_phase(rp.npu(),col.npu(),weight.npu(),state.npu())
    assert candidate.cpu().tolist()==expected_candidate
    assert after.cpu().tolist()==expected_state


def test_noncontiguous_csr_and_wide_weight():
    rp,col,weight=complete(7,torch.float64)
    weight=1.0+weight*2.0**-40
    def strided(tensor):
        backing=torch.empty(tensor.numel()*2,dtype=tensor.dtype,device='npu')
        view=backing[::2]
        view.copy_(tensor.npu())
        return view
    got=torch.ops.torch_cluster.graclus(strided(rp),strided(col),strided(weight)).cpu()
    assert torch.equal(got,matching_reference(rp,col,weight))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.int64])
def test_actual_bitmap_phase_with_sparse_active_rows(dtype):
    n=24577
    active=torch.tensor([0,7,63,64,255,512,24575,24576])
    small_rp,small_col,weight=complete(active.numel(),dtype)
    row=active[torch.repeat_interleave(torch.arange(active.numel()),small_rp[1:]-small_rp[:-1])]
    col=active[small_col]
    if dtype==torch.float64: weight=1.0+weight*2.0**-40
    if dtype==torch.int64: weight=2**53+weight
    rp=torch.cat([torch.zeros(1,dtype=torch.long),torch.bincount(row,minlength=n).cumsum(0)])
    state=torch.arange(n,dtype=torch.int32);state[active]=-1
    expected_candidate,expected_state=phase_reference(rp,col,weight,state.tolist())
    if hasattr(torch.ops.torch_cluster,'_graclus_phase_bitmap'):
        candidate,after,_,bitmap=torch.ops.torch_cluster._graclus_phase_bitmap(rp.npu(),col.npu(),weight.npu(),state.npu())
    else:
        candidate,after,_=torch.ops.torch_cluster._graclus_phase(rp.npu(),col.npu(),weight.npu(),state.npu())
        bitmap=None
    assert candidate.cpu().tolist()==expected_candidate
    assert after.cpu().tolist()==expected_state
    if bitmap is None: return
    words=bitmap[:(n+31)//32].cpu().to(torch.long)&0xffffffff
    nodes=torch.arange(n)
    flags=(words[nodes//32]>>(nodes%32))&1
    assert torch.equal(flags.bool(),torch.tensor(expected_state)>=0)
    assert int(words[-1]>>(n%32))==0


@pytest.mark.parametrize('dtype',DTYPES)
def test_empty_graph_preserves_weight_contract(dtype):
    got=torch.ops.torch_cluster.graclus(torch.zeros(1,dtype=torch.long).npu(),
        torch.empty(0,dtype=torch.long).npu(),torch.empty(0,dtype=dtype).npu())
    assert got.dtype==torch.long and got.numel()==0


@pytest.mark.parametrize('weight',[torch.empty(0,dtype=torch.bool),torch.ones(1)])
def test_empty_graph_rejects_invalid_weight(weight):
    with pytest.raises(RuntimeError,match='unsupported weight dtype|weight must be'):
        torch.ops.torch_cluster.graclus(torch.zeros(1,dtype=torch.long).npu(),
            torch.empty(0,dtype=torch.long).npu(),weight.npu())


@pytest.mark.parametrize('edges',[6822,6824,6826])
def test_fused_core_boundaries(edges):
    n=1024;generator=torch.Generator().manual_seed(472)
    u=torch.randint(n,(edges//2,),generator=generator)
    v=torch.randint(n-1,(edges//2,),generator=generator);v+=v>=u
    row,col=torch.cat([u,v]),torch.cat([v,u])
    values=torch.rand(edges//2,generator=generator)
    weights=torch.cat([values,values])
    order=torch.argsort(row,stable=True);row,col,weights=row[order],col[order],weights[order]
    rp=torch.cat([torch.zeros(1,dtype=torch.long),torch.bincount(row,minlength=n).cumsum(0)])
    got=torch.ops.torch_cluster.graclus(rp.npu(),col.npu(),weights.npu()).cpu()
    assert_matching(rp,col,weights,got)
    assert torch.equal(got,matching_reference(rp,col,weights))
    # The two adjacent aligned footprints straddle the generic 96 KiB budget.
    align=lambda size:(size+31)//32*32
    footprint=align((n+1)*8)+align(edges*8)+align(edges*4)+2*align(n*4)
    counts=torch.ops.torch_cluster._graclus_phase_counts()
    if footprint<=96*1024:
        assert counts==[1,0,0,0,0]
    else:
        assert counts[1]>0 and counts[1]==counts[2]
