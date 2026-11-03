import sys
from pathlib import Path
import pytest
import torch
import torch_npu
from torch_cluster import fps, grid_cluster, knn, nearest, radius

project=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(project/'Test/precision/cases'),str(project/'Test/precision/scripts')]
from fps_grid import fps_reference,grid_reference


def test_double_grid_preserves_values_below_integer_boundaries():
    x=torch.tensor([[1.-2.**-52],[1.+2.**-52],[-1.+2.**-52],[-1.-2.**-52]],dtype=torch.float64)
    actual=grid_cluster(x.npu(),torch.ones(1,dtype=x.dtype).npu(),
                        torch.zeros(1,dtype=x.dtype).npu(),torch.full((1,),3.,dtype=x.dtype).npu())
    assert actual.device.type=='npu'
    assert actual.cpu().tolist()==[0,1,0,-1]


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_grid_inferred_bounds_use_device_high_precision(dtype):
    x=torch.tensor([[-.5,1.9],[-1.5,-2.1],[2.5,3.1]],dtype=dtype)
    size=torch.full((2,),.25,dtype=dtype)
    expected=grid_reference(x,size,None,None)
    actual=grid_cluster(x.npu(),size.npu())
    assert actual.device.type=='npu'
    assert torch.equal(actual.cpu(),expected)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_grid_labels_keep_all_64_bits(dtype):
    x=torch.ones((2,64),dtype=dtype);size=torch.full((64,),.5,dtype=dtype)
    start=torch.zeros(64,dtype=dtype);end=torch.ones(64,dtype=dtype)
    expected=(3**64-1)%2**64
    if expected>=2**63:expected-=2**64
    actual=grid_cluster(x.npu(),size.npu(),start.npu(),end.npu())
    assert actual.device.type=='npu'
    assert actual.cpu().tolist()==[expected,expected]


def test_fps_double_does_not_collapse_close_coordinates():
    x=torch.tensor([[0.],[1.],[1.+2.**-30],[1.+2.**-29],[2.]],dtype=torch.float64)
    expected=fps_reference(x,torch.tensor([0,5]),1.)
    actual=fps(x.npu(),ratio=1.,random_start=False)
    assert actual.device.type=='npu'
    assert torch.equal(actual.cpu(),expected)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_fps_low_precision_input_still_uses_true_double_scores(dtype):
    x=torch.tensor([[0.,0.],[16384.,0.],[16384.,1.]],dtype=dtype)
    actual=fps(x.npu(),ratio=.5,random_start=False)
    assert actual.cpu().tolist()==[0,2]


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_fps_cardinality_uses_integer_batch_size(dtype):
    n=257;x=torch.arange(n).to(dtype).reshape(-1,1)
    actual=fps(x.npu(),ratio=.5,random_start=False)
    expected=fps_reference(x,torch.tensor([0,n]),.5)
    assert actual.device.type=='npu' and len(actual)==129
    assert torch.equal(actual.cpu(),expected)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_neighbors_compare_true_double_scores(dtype):
    refs=torch.tensor([[16384.,1.],[16384.,0.]],dtype=dtype).npu()
    query=torch.zeros((1,2),dtype=dtype).npu()
    assert nearest(query,refs).cpu().tolist()==[1]
    assert knn(refs,query,1).cpu().tolist()==[[0],[1]]


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_neighbors_ties_and_short_batches(dtype):
    refs=torch.tensor([[0.],[0.],[2.],[10.]],dtype=dtype).npu()
    query=torch.tensor([[0.],[10.]],dtype=dtype).npu()
    bx=torch.tensor([0,0,0,1],device='npu');by=torch.tensor([0,1],device='npu')
    assert nearest(query,refs,by,bx).cpu().tolist()==[0,3]
    assert knn(refs,query,2,bx,by).cpu().tolist()==[[0,0,1],[0,1,3]]


def test_double_neighbors_do_not_collapse_tiny_coordinate_difference():
    refs=torch.tensor([[1.+2.**-40],[1.]],dtype=torch.float64).npu()
    query=torch.zeros((1,1),dtype=torch.float64).npu()
    assert nearest(query,refs).cpu().tolist()==[1]
    assert knn(refs,query,1).cpu().tolist()==[[0],[1]]


def test_neighbor_shortlist_uses_full_double_search_when_float_overflows():
    refs=(torch.arange(128,dtype=torch.float64)*1.e20).reshape(-1,1).npu()
    query=torch.tensor([[1.2e20],[98.9e20]],dtype=torch.float64).npu()
    assert nearest(query,refs).cpu().tolist()==[1,99]
    assert knn(refs,query,1).cpu().tolist()==[[0,1],[1,99]]


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_radius_strict_boundary_matches_cpu_cuda_contract(dtype):
    x=torch.tensor([[0.],[1.],[-1.],[.5]],dtype=dtype).npu()
    y=torch.zeros((1,1),dtype=dtype).npu()
    result=radius(x,y,1.,max_num_neighbors=32)
    assert sorted(zip(*result.cpu().tolist()))==[(0,0),(0,3)]


def test_radius_double_preserves_sub_float32_boundary_distinctions():
    x=torch.tensor([[1.-2.**-40],[1.+2.**-40]],dtype=torch.float64).npu()
    y=torch.zeros((1,1),dtype=torch.float64).npu()
    assert radius(x,y,1.).cpu().tolist()==[[0],[0]]


def test_radius_retries_a_crowded_uncertain_boundary():
    # The first 128 references round to 1.0 in Float32 and are all outside;
    # the valid neighbor follows them. A fixed shortlist would miss it.
    x=torch.cat((torch.full((128,1),1.+2.**-40,dtype=torch.float64),
                 torch.tensor([[1.-2.**-40]],dtype=torch.float64))).npu()
    y=torch.zeros((1,1),dtype=torch.float64).npu()
    assert radius(x,y,1.,max_num_neighbors=1).cpu().tolist()==[[0],[128]]


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_radius_global_self_exclusion_and_short_batches(dtype):
    x=torch.tensor([[0.],[.2],[10.]],dtype=dtype).npu()
    y=torch.tensor([[0.],[10.]],dtype=dtype).npu()
    bx=torch.tensor([0,0,1],device='npu');by=torch.tensor([0,1],device='npu')
    result=radius(x,y,.5,bx,by,max_num_neighbors=2,ignore_same_index=True)
    assert sorted(zip(*result.cpu().tolist()))==[(0,1),(1,2)]
