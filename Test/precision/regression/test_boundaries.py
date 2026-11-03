"""Additional contracts beyond the document matrix; run on an isolated NPU build.

Geometry coordinates are finite real points. Invalid NaN/Inf coordinates must
be rejected explicitly, rather than silently yielding plausible integer IDs.
Graclus weights retain the existing semantics: NaN/negative edges ineligible,
positive infinity outranks every finite weight.
"""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch_npu
from torch_cluster import fps,grid_cluster,knn,nearest,radius,graclus_cluster

PROJECT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(PROJECT/'Test/precision/cases'),str(PROJECT/'Test/precision/scripts')]
from fps_grid import grid_reference
from graclus import verify as verify_graclus
from decision_precision import fps_review,score_check

FLOATS=(torch.float16,torch.bfloat16,torch.float32,torch.float64)


@pytest.mark.parametrize('dtype',FLOATS)
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),'mixed_inf'])
@pytest.mark.parametrize('op',['fps','grid_cluster','knn','nearest','radius'])
def test_nonfinite_geometry_has_explicit_rejection(dtype,value,op):
    # Different shapes across special values exercise scalar-sized and wider
    # feature rows, without mixing special cases into random value products.
    mixed=isinstance(value,str)
    d=9 if mixed else 1 if np.isnan(value) else 3 if value>0 else 7
    x=torch.zeros((3,d),dtype=dtype)
    if mixed:x[1,0]=float('inf');x[2,-1]=-float('inf')
    else:x[1,0]=value
    x=x.npu()
    y=torch.zeros((2,d),dtype=dtype).npu()
    with pytest.raises((RuntimeError,ValueError),match='finite'):
        if op=='fps':fps(x,ratio=.5,random_start=False)
        elif op=='grid_cluster':grid_cluster(x,torch.ones(d,dtype=dtype).npu())
        elif op=='knn':knn(x,y,1)
        elif op=='nearest':nearest(x,y)
        else:radius(x,y,1.)


@pytest.mark.parametrize('dtype',FLOATS)
def test_fps_random_start_and_multiple_batches(dtype):
    g=torch.Generator().manual_seed(22471)
    x=torch.rand((33,7),generator=g,dtype=torch.float64).to(dtype)
    ptr=torch.tensor([0,16,33]);ratio=.5
    for _ in range(3):
        actual=fps(x.npu(),ptr=ptr.npu(),ratio=ratio,random_start=True).cpu()
        # Relabel each CPU batch so its DUT-selected first point occupies the
        # canonical starting position. This preserves all pairwise distances
        # and lets the independent conditional farthest-point oracle replay it.
        points=x.clone();mapped=actual.clone();offset=0
        for first,last in zip(ptr[:-1].tolist(),ptr[1:].tolist()):
            count=int(np.ceil((last-first)*ratio));chosen=actual[offset:offset+count]
            assert len(chosen)==count and ((chosen>=first)&(chosen<last)).all()
            initial=int(chosen[0]);points[first],points[initial]=x[initial],x[first]
            segment=mapped[offset:offset+count]
            segment[chosen==first]=initial;segment[chosen==initial]=first
            offset+=count
        assert fps_review(points,ptr,ratio,mapped)['passed']


@pytest.mark.parametrize('dtype',FLOATS)
@pytest.mark.parametrize('ratio',[0.,1.])
def test_fps_ratio_boundaries(dtype,ratio):
    x=torch.tensor([[0.,0.],[2.,0.],[0.,4.],[2.,3.]],dtype=dtype)
    actual=fps(x.npu(),ratio=ratio,random_start=False).cpu()
    assert fps_review(x,torch.tensor([0,4]),ratio,actual)['passed']


@pytest.mark.parametrize('dtype',FLOATS)
def test_knn_cosine_and_noncontiguous_inputs(dtype):
    g=torch.Generator().manual_seed(8127)
    x=torch.randn((7,33),generator=g,dtype=torch.float64).t().to(dtype)
    y=torch.randn((7,17),generator=g,dtype=torch.float64).t().to(dtype)
    result=knn(x.npu(),y.npu(),4,cosine=True).cpu()
    assert result.shape==(2,68)
    xp,yp=x.double().numpy(),y.double().numpy()
    xn=xp/np.linalg.norm(xp,axis=1,keepdims=True)
    yn=yp/np.linalg.norm(yp,axis=1,keepdims=True)
    scores=((yn[:,None,:]-xn[None,:,:])**2).sum(2)
    actual=[];golden=[]
    for q in range(len(y)):
        selected=result[1,result[0]==q].numpy()
        assert len(selected)==4 and len(np.unique(selected))==4
        actual.extend(np.sort(scores[q,selected]));golden.extend(np.sort(scores[q])[:4])
    assert score_check(actual,golden,dtype)['passed']


@pytest.mark.parametrize('shape',[(33,17,7,4),(65,33,3,2),(129,31,7,3),(97,65,3,8),(65,31,128,4)])
@pytest.mark.parametrize('cosine',[False,True])
def test_native_knn_query_ownership_under_allocator_reuse(shape,cosine):
    n,m,d,k=shape;g=torch.Generator().manual_seed(9182)
    x=torch.randn((n,d),generator=g);y=torch.randn((m,d),generator=g)
    xp,yp=x.double().numpy(),y.double().numpy()
    if cosine:
        xp=xp/np.linalg.norm(xp,axis=1,keepdims=True)
        yp=yp/np.linalg.norm(yp,axis=1,keepdims=True)
    scores=((yp[:,None,:]-xp[None,:,:])**2).sum(2)
    dx,dy=x.npu(),y.npu()
    for _ in range(5):
        scratch=torch.full((2*m*k+17,),-918273,dtype=torch.long,device='npu')
        del scratch
        edges=knn(dx,dy,k,cosine=cosine).cpu()
        assert edges.shape==(2,m*k)
        assert torch.equal(torch.bincount(edges[0],minlength=m),torch.full((m,),k))
        a=[];b=[]
        for query in range(m):
            ids=edges[1,edges[0]==query].numpy()
            assert np.all((ids>=0)&(ids<n)) and len(np.unique(ids))==k
            a.extend(np.sort(scores[query,ids]));b.extend(np.sort(scores[query])[:k])
        assert score_check(a,b,torch.float32)['passed']


@pytest.mark.parametrize('dtype',FLOATS)
def test_grid_int64_wrap_includes_native_float32(dtype):
    x=torch.ones((2,64),dtype=dtype);size=torch.full((64,),.5,dtype=dtype)
    start=torch.zeros(64,dtype=dtype);end=torch.ones(64,dtype=dtype)
    expected=grid_reference(x,size,start,end)
    assert torch.equal(grid_cluster(x.npu(),size.npu(),start.npu(),end.npu()).cpu(),expected)


def test_radius_int32_preserves_neighbor_at_large_coordinate_offset():
    x=torch.tensor([[2**24],[2**24+1],[2**24+2]],dtype=torch.int32).npu()
    y=torch.tensor([[2**24+1]],dtype=torch.int32).npu()
    assert radius(x,y,.5,max_num_neighbors=3).cpu().tolist()==[[0],[1]]


def test_graclus_positive_infinity_outranks_max_finite_weight():
    row=torch.tensor([0,0,1,1,2,2,3,3]);col=torch.tensor([1,3,0,2,1,3,0,2])
    maximum=torch.finfo(torch.float32).max
    # If +inf is collapsed to max-finite, global tie priority pairs (2,3)
    # first, then (0,1). Every chosen pair is worse than an available +inf
    # edge, so no valid weighted greedy order can produce that matching.
    weight=torch.tensor([maximum,float('inf'),maximum,float('inf'),
                         float('inf'),maximum,float('inf'),maximum])
    labels=graclus_cluster(row.npu(),col.npu(),weight.npu(),4).cpu()
    verify_graclus(row,col,weight,labels,4)


@pytest.mark.parametrize('dtype',FLOATS)
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),'mixed_inf'])
def test_graclus_special_weight_semantics(dtype,value):
    row=torch.tensor([0,0,1,1,2,2,3,3]);col=torch.tensor([1,3,0,2,1,3,0,2])
    if isinstance(value,str):
        weight=torch.tensor([float('inf'),float('nan'),float('inf'),1.,
                             1.,-float('inf'),float('nan'),-float('inf')],dtype=dtype)
    else:weight=torch.full((8,),value,dtype=dtype)
    labels=graclus_cluster(row.npu(),col.npu(),weight.npu(),4).cpu()
    verify_graclus(row,col,weight,labels,4)


@pytest.mark.parametrize('dtype',[*FLOATS,torch.int32,torch.int64])
@pytest.mark.parametrize('distribution',['uniform','normal'])
@pytest.mark.parametrize('n',[32,127,512])
def test_graclus_signed_weight_distributions(dtype,distribution,n):
    g=torch.Generator().manual_seed(5573+n);half=n*(128 if n>=512 else 16)
    src=torch.randint(n,(half,),generator=g);dst=torch.randint(n-1,(half,),generator=g)
    dst+=(dst>=src).long();row=torch.cat((src,dst));col=torch.cat((dst,src))
    if distribution=='uniform':weight=torch.rand(half,generator=g,dtype=torch.float64)*10-5
    else:
        mu=torch.rand(1,generator=g).item()*10-5;sigma=.1+torch.rand(1,generator=g).item()*1.9
        weight=torch.randn(half,generator=g,dtype=torch.float64)*sigma+mu
    weight=weight.to(dtype).repeat(2);order=torch.argsort(row,stable=True)
    row,col,weight=row[order],col[order],weight[order]
    labels=graclus_cluster(row.npu(),col.npu(),weight.npu(),n).cpu()
    verify_graclus(row,col,weight,labels,n)


def test_graclus_int64_weight_order_above_exact_double_integer_range():
    row=torch.tensor([0,0,1,1,2,2,3,3]);col=torch.tensor([1,3,0,2,1,3,0,2])
    low=2**53;high=low+1
    weight=torch.tensor([low,high,low,high,high,low,high,low],dtype=torch.int64)
    labels=graclus_cluster(row.npu(),col.npu(),weight.npu(),4).cpu()
    verify_graclus(row,col,weight,labels,4)


@pytest.mark.parametrize('dtype',FLOATS)
@pytest.mark.parametrize('length',[0,2])
def test_fps_rejects_empty_or_mismatched_ratio_tensor(dtype,length):
    x=torch.ones((4,3),dtype=dtype).npu();ratio=torch.full((length,),.5,dtype=dtype).npu()
    with pytest.raises(RuntimeError,match='ratio'):
        fps(x,ptr=torch.tensor([0,4],device='npu'),ratio=ratio,random_start=False)


@pytest.mark.parametrize('dtype',FLOATS)
def test_fps_rejects_empty_ptr(dtype):
    with pytest.raises(RuntimeError,match='ptr'):
        fps(torch.ones((4,3),dtype=dtype).npu(),ptr=torch.empty(0,dtype=torch.long).npu(),ratio=.5)


@pytest.mark.parametrize('dtype',FLOATS)
@pytest.mark.parametrize('which',['empty_x','empty_y','different_batches'])
def test_nearest_rejects_invalid_ptr_metadata(dtype,which):
    x=torch.ones((4,3),dtype=dtype).npu();y=torch.zeros((2,3),dtype=dtype).npu()
    px=torch.tensor([] if which=='empty_x' else [0,2,4] if which=='different_batches' else [0,4],dtype=torch.long).npu()
    py=torch.tensor([] if which=='empty_y' else [0,2],dtype=torch.long).npu()
    with pytest.raises(RuntimeError,match='ptr'):
        torch.ops.torch_cluster.nearest(x,y,px,py)


@pytest.mark.parametrize('dtype',FLOATS)
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('op',['knn','nearest','radius'])
def test_nonfinite_query_coordinates_are_rejected(dtype,value,op):
    x=torch.zeros((3,3),dtype=dtype).npu();y=torch.zeros((2,3),dtype=dtype);y[0,0]=value;y=y.npu()
    with pytest.raises(RuntimeError,match='finite'):
        if op=='knn':knn(x,y,1)
        elif op=='nearest':nearest(x,y)
        else:radius(x,y,1.)


@pytest.mark.parametrize('dtype',FLOATS)
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_fps_nonfinite_ratio_is_rejected(dtype,value):
    with pytest.raises(RuntimeError,match='finite'):
        fps(torch.ones((4,3),dtype=dtype).npu(),ratio=torch.tensor(value,dtype=dtype).npu(),random_start=False)


@pytest.mark.parametrize('dtype',FLOATS)
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('which',['size','start','end'])
def test_grid_nonfinite_metadata_is_rejected(dtype,value,which):
    data=dict(size=torch.ones(3,dtype=dtype),start=torch.zeros(3,dtype=dtype),end=torch.ones(3,dtype=dtype))
    data[which][0]=value
    with pytest.raises(RuntimeError,match='finite'):
        grid_cluster(torch.ones((2,3),dtype=dtype).npu(),**{k:v.npu() for k,v in data.items()})


@pytest.mark.parametrize('dtype',FLOATS)
@pytest.mark.parametrize('which',['size','start','end'])
def test_grid_empty_metadata_is_rejected(dtype,which):
    data=dict(size=torch.ones(3,dtype=dtype),start=torch.zeros(3,dtype=dtype),end=torch.ones(3,dtype=dtype))
    data[which]=torch.empty(0,dtype=dtype)
    with pytest.raises(RuntimeError,match='size/start/end'):
        grid_cluster(torch.ones((2,3),dtype=dtype).npu(),**{k:v.npu() for k,v in data.items()})


@pytest.mark.parametrize('dtype',[*FLOATS,torch.int32])
@pytest.mark.parametrize('which',['zero_radius','zero_capacity'])
def test_radius_zero_boundaries_return_no_edges(dtype,which):
    x=torch.ones((3,3),dtype=dtype).npu()
    output=radius(x,x,0. if which=='zero_radius' else 1.,
                  max_num_neighbors=0 if which=='zero_capacity' else 32)
    assert output.dtype==torch.long and tuple(output.shape)==(2,0)


def test_double_neighbor_order_resolves_one_ulp_coordinate_difference():
    upper=float(np.nextafter(1.,np.inf))
    refs=torch.tensor([[upper],[1.]],dtype=torch.float64).npu()
    query=torch.zeros((1,1),dtype=torch.float64).npu()
    assert nearest(query,refs).cpu().tolist()==[1]
    assert knn(refs,query,1).cpu().tolist()==[[0],[1]]


def test_double_fps_resolves_one_ulp_coordinate_difference():
    upper=float(np.nextafter(1.,np.inf))
    points=torch.tensor([[0.],[1.],[upper]],dtype=torch.float64).npu()
    assert fps(points,ratio=.5,random_start=False).cpu().tolist()==[0,2]
