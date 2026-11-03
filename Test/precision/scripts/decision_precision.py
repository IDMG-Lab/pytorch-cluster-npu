"""Explicit business rules for floating decisions that return integer IDs.

Preserve the original exact-index diagnostic separately. CANN mixed tolerance
is applied to Float32 distance decision quality, never to integer IDs. Double
decisions have no CANN tolerance entry and use exact distance comparisons.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree


def score_check(actual, golden, dtype):
    from precision import check_precision
    a=torch.as_tensor(np.asarray(actual).copy(),dtype=torch.float64)
    g=torch.as_tensor(np.asarray(golden).copy(),dtype=torch.float64)
    compute_dtype=torch.float32 if dtype==torch.float32 else torch.float64
    return check_precision(a.to(compute_dtype),g)


def nearest_review(x,y,bx,by,actual,golden):
    a=actual.numpy();g=golden.numpy()
    assert a.shape==(len(x),) and np.all((a>=0)&(a<len(y)))
    if bx is not None:assert np.array_equal(bx.numpy(),by.numpy()[a]),'Wrong nearest batch'
    # The public API treats a rank-one point tensor as N scalar coordinates.
    xp=(x.double().reshape(-1,1) if x.dim()==1 else x.double()).numpy()
    yp=(y.double().reshape(-1,1) if y.dim()==1 else y.double()).numpy()
    ad=((xp-yp[a])**2).sum(1);gd=((xp-yp[g])**2).sum(1)
    return score_check(ad,gd,x.dtype)


def knn_review(x,y,bx,by,k,graph,actual,golden):
    assert actual.shape==golden.shape and actual.dtype==torch.long
    a=actual.numpy();g=golden.numpy()
    query,ref=(a[1],a[0]) if graph else (a[0],a[1])
    gquery,gref=(g[1],g[0]) if graph else (g[0],g[1])
    assert np.all((query>=0)&(query<len(y))) and np.all((ref>=0)&(ref<len(x)))
    if graph:assert np.all(query!=ref),'KNN graph contains self edges'
    if bx is not None:assert np.array_equal(by.numpy()[query],bx.numpy()[ref]),'Wrong KNN batch'
    keys=query.astype(np.int64)*max(1,len(x))+ref
    assert len(np.unique(keys))==len(keys),'Duplicate KNN edge'
    ac=np.bincount(query,minlength=len(y));gc=np.bincount(gquery,minlength=len(y))
    assert np.array_equal(ac,gc),'Wrong KNN cardinality'
    xp,yp=x.double().numpy(),y.double().numpy()
    ad=((xp[ref]-yp[query])**2).sum(1);gd=((xp[gref]-yp[gquery])**2).sum(1)
    # Match equal ranks within each query, independent of neighbor ordering.
    ao=np.lexsort((ad,query));go=np.lexsort((gd,gquery))
    return score_check(ad[ao],gd[go],x.dtype)


def fps_review(pos,ptr,ratio,actual):
    x=pos.double().numpy();a=actual.numpy();out_offset=0;ad=[];gd=[]
    for first,last in zip(ptr[:-1].tolist(),ptr[1:].tolist()):
        count=int(np.ceil((last-first)*ratio))
        chosen=a[out_offset:out_offset+count];out_offset+=count
        assert len(chosen)==count and np.all((chosen>=first)&(chosen<last)),'Wrong FPS batch/count'
        if not count:continue
        assert chosen[0]==first,'Wrong FPS start with random_start=False'
        points=x[first:last];distance=np.full(last-first,np.inf)
        for i,index in enumerate(chosen):
            local=int(index-first)
            if i:
                ad.append(float(distance[local]));gd.append(float(distance.max()))
            delta=points-points[local]
            distance=np.minimum(distance,np.einsum('ij,ij->i',delta,delta))
    assert out_offset==len(a),'Wrong FPS total count'
    return score_check(ad,gd,pos.dtype)


def radius_review(x,y,edges,r,cap,ignore):
    assert edges.dtype==torch.long and edges.dim()==2 and edges.size(0)==2
    query,ref=edges.numpy();n,m=len(x),len(y)
    assert np.all((query>=0)&(query<m)) and np.all((ref>=0)&(ref<n)),'Invalid radius index'
    if ignore:assert np.all(query!=ref),'Radius self edge'
    keys=query.astype(np.int64)*max(1,n)+ref
    assert len(np.unique(keys))==len(keys),'Duplicate radius edge'
    counts=np.bincount(query,minlength=m)
    assert np.all(counts<=cap),'Radius capacity exceeded'
    xp,yp=x.double().numpy(),y.double().numpy();r2=r*r
    rtol=2.**-13 if x.dtype==torch.float32 else 0.
    band=rtol*(1+abs(r2)) if r else 0.
    low=np.zeros(m,dtype=np.int64);high=low.copy();strict_counts=low.copy()
    if n and m and cap:
        tree=cKDTree(xp);width=min(n,cap+int(ignore))
        for first in range(0,m,4096):
            ds,ids=tree.query(yp[first:first+4096],k=width,workers=4)
            ds=np.asarray(ds).reshape(-1,width);ids=np.asarray(ids).reshape(-1,width)
            # Recompute direct squared scores to avoid sqrt rounding at r.
            sq=((xp[ids]-yp[first:first+len(ids),None,:])**2).sum(2)
            eligible=np.ones_like(sq,dtype=bool)
            if ignore:eligible &= ids!=np.arange(first,first+len(ids))[:,None]
            low[first:first+len(ids)]=np.minimum(cap,((sq<r2-band)&eligible).sum(1))
            high[first:first+len(ids)]=np.minimum(cap,((sq<r2+band)&eligible).sum(1))
            strict_counts[first:first+len(ids)]=np.minimum(cap,((sq<r2)&eligible).sum(1))
    assert np.all((counts>=low)&(counts<=high)),'Radius completeness outside tolerance boundary band'
    ad=[];gd=[];strict_invalid=0
    for first in range(0,len(query),262144):
        q=query[first:first+262144];v=ref[first:first+262144]
        sq=((xp[v]-yp[q])**2).sum(1)
        strict_invalid+=int(np.sum(sq>=r2))
        if not band:assert np.all(sq<r2),'Radius strict boundary violated'
        ad.append(sq);gd.append(np.minimum(sq,r2))
    detail=score_check(np.concatenate(ad) if ad else [],np.concatenate(gd) if gd else [],x.dtype)
    return dict(**detail,edges=len(query),queries=m,boundary_band=band,
                strict_indices=True,strict_capacity=True,strict_batch=True,
                strict_distance_passed=not strict_invalid and np.array_equal(counts,strict_counts),
                strict_invalid_edges=strict_invalid,
                strict_count_mismatches=int(np.sum(counts!=strict_counts)))
