#include "kernel_operator.h"
using namespace AscendC;

// 910B: coarse AIV row partitioning, SIMD tiles, snapshot reads, single-owner writes.
// No SIMT threads, coloring, cross-core neighbor writes, or spin barriers.
class KernelGraclus {
 public:
  __aicore__ inline uint32_t Align32(uint64_t bytes) {return (bytes+31)/32*32;}
  __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR weight,
      GM_ADDR state, GM_ADDR candidate, GM_ADDR next, GM_ADDR proposals,
      GM_ADDR counters, GM_ADDR out, const GraclusTilingData& d) {
    n=d.numNodes; e=d.numEdges; phase=d.phase; kind=d.weightKind;
    weighted=d.weighted; cores=d.cores; tile=d.tile; resident=d.resident;
    id=GetBlockIdx();
    const uint32_t unit=resident==2?512:16;
    const uint32_t units=(n+unit-1)/unit;
    begin=uint64_t(units)*id/cores*unit;
    end=uint64_t(units)*(id+1)/cores*unit;
    if(end>n) end=n;
    rp.SetGlobalBuffer((__gm__ int64_t*)rowptr,n+1);
    ci.SetGlobalBuffer((__gm__ int64_t*)col,e);
    wf.SetGlobalBuffer((__gm__ float*)weight,e);
    wi.SetGlobalBuffer((__gm__ int64_t*)weight,e);
    old.SetGlobalBuffer((__gm__ int32_t*)state,n);
    cand.SetGlobalBuffer((__gm__ int32_t*)candidate,n);
    dst.SetGlobalBuffer((__gm__ int32_t*)next,n);
    prop.SetGlobalBuffer((__gm__ int32_t*)proposals,n);
    count.SetGlobalBuffer((__gm__ int32_t*)counters,1024);
    output.SetGlobalBuffer((__gm__ int32_t*)out,uint64_t(n)*2);
    pipe.InitBuffer(edges,8*tile);
    pipe.InitBuffer(weightBuf,8*tile);
    pipe.InitBuffer(indexBuf,4*tile);
    pipe.InitBuffer(scoreBuf,4*tile);
    pipe.InitBuffer(reduceBuf,4*tile);
    pipe.InitBuffer(offsetBuf,8*tile);
    pipe.InitBuffer(outBuf,8*tile);
    pipe.InitBuffer(nodeBuf,4*tile);
    pipe.InitBuffer(candidateBuf,4*tile);
    pipe.InitBuffer(rowBuf,8*257);
    const uint32_t residentBytes=resident==1 && (phase==1 || phase==2)?((n+7)/8)*32:
      resident==2 && phase==1?((n+255)/256)*32:
      resident==3 && phase==0?Align32((uint64_t(n)+1)*8)+Align32(uint64_t(e)*8)+
        Align32(uint64_t(e)*4)+2*Align32(uint64_t(n)*4):32;
    pipe.InitBuffer(residentBuf,residentBytes);
    pipe.InitBuffer(resultBuf,32);
    pipe.InitBuffer(rowSelBuf,32768);
    offsets=offsetBuf.Get<uint32_t>();
    if(phase==1) {
      auto offsetIndices=offsets.ReinterpretCast<int32_t>();
      CreateVecIndex(offsetIndices,int32_t(0),tile);
      PipeBarrier<PIPE_V>();
      Muls(offsetIndices,offsetIndices,int32_t(8),tile);
      PipeBarrier<PIPE_ALL>();
    }
    cache=residentBuf.Get<int32_t>();
    if(resident==1 && (phase==1 || phase==2)) {
      const auto source=phase==1?old:cand;
      // Copy whole state in pieces; DMA parameter widths come from installed SDK.
      for(uint32_t off=0;off<n;off+=tile) {
        uint32_t len=n-off<tile?n-off:tile;
        CopyIn(cache[off],source[off],len);
      }
      PipeBarrier<PIPE_ALL>();
    } else if(resident==2 && phase==1) {
      const uint32_t words=(n+31)/32;
      for(uint32_t off=0;off<words;off+=tile) {
        uint32_t len=words-off<tile?words-off:tile;
        CopyIn(cache[off],cand[off],len);
      }
      PipeBarrier<PIPE_ALL>();
    }
  }
  template<class T>
  __aicore__ inline void CopyIn(LocalTensor<T> a,GlobalTensor<T> b,uint32_t len) {
    DataCopyExtParams p{1,uint32_t(len*sizeof(T)),0,0,0};
    DataCopyPadExtParams<T> pad{false,0,0,0};
    DataCopyPad(a,b,p,pad);
  }
  template<class T>
  __aicore__ inline void CopyOut(GlobalTensor<T> a,LocalTensor<T> b,uint32_t len) {
    DataCopyExtParams p{1,uint32_t(len*sizeof(T)),0,0,0}; DataCopyPad(a,b,p);
  }
  __aicore__ inline bool TieBetter(uint32_t u,uint32_t v,int32_t chosen) {
    if(chosen<0) return true;
    const uint32_t c=uint32_t(chosen);
    const uint32_t hi1=u<v?v:u, lo1=u<v?u:v;
    const uint32_t hi0=u<c?c:u, lo0=u<c?u:c;
    if(hi1!=hi0) return hi1>hi0;
    return lo1>lo0;
  }
  template<uint32_t Mode>
  __aicore__ inline int32_t Status(uint32_t v) {
    if constexpr(Mode==2)
      return (uint32_t(cache.GetValue(v>>5))&(1U<<(v&31)))?0:-1;
    if constexpr(Mode==1) return cache.GetValue(v);
    return old.GetValue(v);
  }
  __aicore__ inline bool WideEligible(uint64_t value) {
    if(kind==1) return int64_t(value)>=0;
    const uint64_t mag=value&0x7fffffffffffffffULL;
    return mag<=0x7ff0000000000000ULL && (!(value>>63) || mag==0);
  }
  __aicore__ inline uint64_t WideKey(uint64_t value) {
    return kind==2?(value&0x7fffffffffffffffULL):value;
  }
  __aicore__ inline void Count(uint32_t remaining,uint32_t changed,uint32_t error=0,uint32_t complete=0) {
    const uint32_t first=64*id/cores,last=64*(id+1)/cores,len=(last-first)*16;
    auto buf=outBuf.Get<int32_t>();
    PipeBarrier<PIPE_ALL>();
    const uint32_t cp=(len+63)/64*64;
    auto fv=weightBuf.Get<float>();
    auto vbuf=edges.Get<float>();
    auto idxI=scoreBuf.Get<int32_t>();
    auto idxF=reduceBuf.Get<float>();
    auto mk=candidateBuf.Get<uint8_t>();
    Duplicate(fv,float(0),cp);PipeBarrier<PIPE_V>();
    CreateVecIndex(idxI,int32_t(0),cp);PipeBarrier<PIPE_V>();
    Cast(idxF,idxI,RoundMode::CAST_NONE,cp);PipeBarrier<PIPE_V>();
    Duplicate(vbuf,float(int32_t(remaining)),cp);PipeBarrier<PIPE_V>();
    CompareScalar(mk,idxF,float(0),CMPMODE::EQ,cp);PipeBarrier<PIPE_V>();
    AscendC::Select(fv,mk,vbuf,fv,SELMODE::VSEL_TENSOR_TENSOR_MODE,cp);PipeBarrier<PIPE_V>();
    Duplicate(vbuf,float(int32_t(changed)),cp);PipeBarrier<PIPE_V>();
    CompareScalar(mk,idxF,float(1),CMPMODE::EQ,cp);PipeBarrier<PIPE_V>();
    AscendC::Select(fv,mk,vbuf,fv,SELMODE::VSEL_TENSOR_TENSOR_MODE,cp);PipeBarrier<PIPE_V>();
    Duplicate(vbuf,float(int32_t(error)),cp);PipeBarrier<PIPE_V>();
    CompareScalar(mk,idxF,float(2),CMPMODE::EQ,cp);PipeBarrier<PIPE_V>();
    AscendC::Select(fv,mk,vbuf,fv,SELMODE::VSEL_TENSOR_TENSOR_MODE,cp);PipeBarrier<PIPE_V>();
    Duplicate(vbuf,float(int32_t(complete)),cp);PipeBarrier<PIPE_V>();
    CompareScalar(mk,idxF,float(3),CMPMODE::EQ,cp);PipeBarrier<PIPE_V>();
    AscendC::Select(fv,mk,vbuf,fv,SELMODE::VSEL_TENSOR_TENSOR_MODE,cp);PipeBarrier<PIPE_V>();
    Cast(buf,fv,RoundMode::CAST_RINT,len);PipeBarrier<PIPE_ALL>();
    CopyOut(count[first*16],buf,len);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Prepare() {
    auto fill=outBuf.Get<int32_t>();
    Duplicate(fill,int32_t(-1),tile);PipeBarrier<PIPE_ALL>();
    uint32_t error=0;
    {
    auto rows=rowBuf.Get<int64_t>();
    auto lowCur=indexBuf.Get<int32_t>();
    auto highCur=scoreBuf.Get<int32_t>();
    auto lowNxt=reduceBuf.Get<int32_t>();
    auto highNxt=nodeBuf.Get<int32_t>();
    auto wf=weightBuf.Get<float>();
    auto lcf=wf,hcf=wf[256],lnf=wf[512],hnf=wf[768];
    auto diff=candidateBuf.Get<float>();
    auto red=resultBuf.Get<float>();
    auto tmpw=edges.Get<float>();
    for(uint32_t off=begin;off<end;off+=256) {
      uint32_t len=end-off<256?end-off:256;
      const uint32_t padded=(len+63)/64*64;
      CopyIn(rows,rp[off],len+1);PipeBarrier<PIPE_ALL>();
      Duplicate(lowCur,int32_t(0),padded);Duplicate(highCur,int32_t(0),padded);
      Duplicate(lowNxt,int32_t(0),padded);Duplicate(highNxt,int32_t(0),padded);
      auto off32=offsets.ReinterpretCast<int32_t>();
      CreateVecIndex(off32,int32_t(0),padded);PipeBarrier<PIPE_V>();
      Muls(off32,off32,int32_t(8),padded);PipeBarrier<PIPE_V>();
      Gather(lowCur,rows.ReinterpretCast<int32_t>(),offsets,0,len);PipeBarrier<PIPE_V>();
      Adds(off32,off32,int32_t(4),padded);PipeBarrier<PIPE_V>();
      Gather(highCur,rows.ReinterpretCast<int32_t>(),offsets,0,len);PipeBarrier<PIPE_V>();
      Adds(off32,off32,int32_t(4),padded);PipeBarrier<PIPE_V>();
      Gather(lowNxt,rows.ReinterpretCast<int32_t>(),offsets,0,len);PipeBarrier<PIPE_V>();
      Adds(off32,off32,int32_t(4),padded);PipeBarrier<PIPE_V>();
      Gather(highNxt,rows.ReinterpretCast<int32_t>(),offsets,0,len);PipeBarrier<PIPE_V>();
      Cast(lcf,lowCur,RoundMode::CAST_NONE,padded);
      Cast(hcf,highCur,RoundMode::CAST_NONE,padded);
      Cast(lnf,lowNxt,RoundMode::CAST_NONE,padded);
      Cast(hnf,highNxt,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
      ReduceMin(red,lcf,tmpw,padded);PipeBarrier<PIPE_ALL>(); if(red.GetValue(0)<float(0)) error=1;
      ReduceMax(red,lcf,tmpw,padded);PipeBarrier<PIPE_ALL>(); if(red.GetValue(0)>float(int32_t(e))) error=1;
      ReduceMin(red,hcf,tmpw,padded);PipeBarrier<PIPE_ALL>(); if(red.GetValue(0)!=float(0)) error=1;
      ReduceMax(red,hcf,tmpw,padded);PipeBarrier<PIPE_ALL>(); if(red.GetValue(0)!=float(0)) error=1;
      Sub(diff,lnf,lcf,padded);PipeBarrier<PIPE_V>();
      ReduceMin(red,diff,tmpw,padded);PipeBarrier<PIPE_ALL>(); if(red.GetValue(0)<float(0)) error=1;
      ReduceMin(red,hnf,tmpw,padded);PipeBarrier<PIPE_ALL>(); if(red.GetValue(0)!=float(0)) error=1;
      ReduceMax(red,hnf,tmpw,padded);PipeBarrier<PIPE_ALL>(); if(red.GetValue(0)!=float(0)) error=1;
      CopyOut(dst[off],fill,len);
    }
    if(begin==0 || end==n) {
      auto eb=candidateBuf.Get<int64_t>();
      if(begin==0) { CopyIn(eb,rp[0],1); }
      if(end==n)   { CopyIn(eb[8],rp[n],1); }
      PipeBarrier<PIPE_ALL>();
      if(begin==0 && eb.GetValue(0)!=0) error=1;
      if(end==n   && eb.GetValue(8)!=e) error=1;
      PipeBarrier<PIPE_ALL>();
    }
    }
    auto columns=edges.Get<int64_t>();
    auto lowInt=indexBuf.Get<int32_t>();
    auto highInt=scoreBuf.Get<int32_t>();
    auto lowFl=reduceBuf.Get<float>();
    auto highFl=nodeBuf.Get<float>();
    auto red=resultBuf.Get<float>();
    auto tmp=outBuf.Get<float>();
    const uint32_t first=uint64_t(e)*id/cores,last=uint64_t(e)*(id+1)/cores;
    for(uint32_t off=first;off<last;off+=tile) {
      uint32_t len=last-off<tile?last-off:tile;
      const uint32_t padded=(len+63)/64*64;
      CopyIn(columns,ci[off],len);PipeBarrier<PIPE_ALL>();
      Duplicate(lowInt,int32_t(0),padded);Duplicate(highInt,int32_t(0),padded);
      auto off32=offsets.ReinterpretCast<int32_t>();
      CreateVecIndex(off32,int32_t(0),padded);PipeBarrier<PIPE_V>();
      Muls(off32,off32,int32_t(8),padded);PipeBarrier<PIPE_V>();
      Gather(lowInt,columns.ReinterpretCast<int32_t>(),offsets,0,len);PipeBarrier<PIPE_V>();
      Adds(off32,off32,int32_t(4),padded);PipeBarrier<PIPE_V>();
      Gather(highInt,columns.ReinterpretCast<int32_t>(),offsets,0,len);PipeBarrier<PIPE_V>();
      Cast(lowFl,lowInt,RoundMode::CAST_NONE,padded);
      Cast(highFl,highInt,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
      ReduceMin(red,lowFl,tmp,padded);PipeBarrier<PIPE_ALL>();
      if(red.GetValue(0)<float(0)) error=1;
      ReduceMax(red,lowFl,tmp,padded);PipeBarrier<PIPE_ALL>();
      if(red.GetValue(0)>=float(int32_t(n))) error=1;
      ReduceMin(red,highFl,tmp,padded);PipeBarrier<PIPE_ALL>();
      if(red.GetValue(0)!=float(0)) error=1;
      ReduceMax(red,highFl,tmp,padded);PipeBarrier<PIPE_ALL>();
      if(red.GetValue(0)!=float(0)) error=1;
    }
    if(resident==2) {
      auto bits=scoreBuf.Get<int32_t>();
      Duplicate(bits,int32_t(0),tile);PipeBarrier<PIPE_ALL>();
      const uint32_t first=begin/32,last=(end+31)/32;
      for(uint32_t off=first;off<last;off+=tile) {
        uint32_t len=last-off<tile?last-off:tile;
        CopyOut(prop[off],bits,len);
      }
      PipeBarrier<PIPE_ALL>();
    }
    if(e==0) {
      PipeBarrier<PIPE_ALL>();
      auto source=edges.Get<int32_t>(),expanded=outBuf.Get<int32_t>();
      auto wordIndices=weightBuf.Get<int32_t>();
      auto byteOffsets=offsets.ReinterpretCast<int32_t>();
      for(uint32_t off=begin;off<end;off+=tile) {
        const uint32_t len=end-off<tile?end-off:tile;
        Duplicate(source,int32_t(0),len+1);PipeBarrier<PIPE_V>();
        CreateVecIndex(source,int32_t(off),len);PipeBarrier<PIPE_V>();
        CreateVecIndex(wordIndices,int32_t(0),len*2);PipeBarrier<PIPE_V>();
        ShiftRight(byteOffsets,wordIndices,int32_t(1),len*2);PipeBarrier<PIPE_V>();
        Muls(expanded,byteOffsets,int32_t(2),len*2);PipeBarrier<PIPE_V>();
        Sub(wordIndices,wordIndices,expanded,len*2);PipeBarrier<PIPE_V>();
        Muls(expanded,wordIndices,int32_t(-1),len*2);PipeBarrier<PIPE_V>();
        Adds(expanded,expanded,int32_t(1),len*2);PipeBarrier<PIPE_V>();
        Muls(byteOffsets,byteOffsets,int32_t(4),len*2);PipeBarrier<PIPE_V>();
        Mul(byteOffsets,byteOffsets,expanded,len*2);PipeBarrier<PIPE_V>();
        Muls(wordIndices,wordIndices,int32_t(4*len),len*2);PipeBarrier<PIPE_V>();
        Add(byteOffsets,byteOffsets,wordIndices,len*2);PipeBarrier<PIPE_V>();
        Gather(expanded,source,offsets,0,len*2);PipeBarrier<PIPE_ALL>();
        CopyOut(output[uint64_t(off)*2],expanded,len*2);PipeBarrier<PIPE_ALL>();
      }
    }
    Count(end-begin,0,error);
  }
  __aicore__ inline void Fused() {
    const uint32_t rowBytes=Align32((uint64_t(n)+1)*8),colBytes=Align32(uint64_t(e)*8);
    const uint32_t weightBytes=Align32(uint64_t(e)*4),stateBytes=Align32(uint64_t(n)*4);
    auto rows=residentBuf.Get<int64_t>();
    auto columns=residentBuf.Get<int64_t>()[rowBytes/8];
    auto weights=residentBuf.Get<float>()[(rowBytes+colBytes)/4];
    auto state=residentBuf.Get<int32_t>()[(rowBytes+colBytes+weightBytes)/4];
    auto proposal=residentBuf.Get<int32_t>()[(rowBytes+colBytes+weightBytes+stateBytes)/4];
    CopyIn(rows,rp,n+1);CopyIn(columns,ci,e);
    if(weighted) CopyIn(weights,wf,e);
    Duplicate(state,int32_t(-1),n);PipeBarrier<PIPE_ALL>();
    uint32_t error=rows.GetValue(0)!=0 || rows.GetValue(n)!=e;
    for(uint32_t u=0;u<n;u++) {
      int64_t a=rows.GetValue(u),b=rows.GetValue(u+1);
      if(a<0 || b<a || b>e) error=1;
    }
    for(uint32_t i=0;i<e;i++) {
      int64_t v=columns.GetValue(i);if(v<0 || v>=n) error=1;
    }
    if(error) {Count(0,0,error,1);return;}
    bool finished=false;
    for(uint32_t iteration=0;iteration<64;iteration++) {
      for(uint32_t u=0;u<n;u++) {
        int32_t chosen=-1;float best=-1;
        if(state.GetValue(u)<0) {
          for(uint32_t edge=rows.GetValue(u);edge<uint32_t(rows.GetValue(u+1));edge++) {
            uint32_t v=columns.GetValue(edge);
            if(v==u || state.GetValue(v)>=0) continue;
            float value=weighted?weights.GetValue(edge):0;
            if(!(value>=0)) continue;
            if((!weighted && TieBetter(u,v,chosen)) ||
                (weighted && (value>best || (value==best && TieBetter(u,v,chosen))))) {
              chosen=v;best=value;
            }
          }
        }
        proposal.SetValue(u,chosen);
      }
      PipeBarrier<PIPE_ALL>();
      uint32_t remaining=0,changed=0;
      for(uint32_t u=0;u<n;u++) {
        int32_t value=state.GetValue(u),v=proposal.GetValue(u);
        if(value<0) {
          if(v<0) {value=u;changed++;}
          else if(proposal.GetValue(v)==int32_t(u)) {value=u<uint32_t(v)?u:uint32_t(v);changed++;}
          state.SetValue(u,value);
        }
        if(value<0) remaining++;
      }
      PipeBarrier<PIPE_ALL>();
      if(!remaining) {finished=true;break;}
      if(!changed) break;
    }
    if(!finished) {
      for(uint32_t u=0;u<n;u++) {
        if(state.GetValue(u)>=0) continue;
        int32_t chosen=-1;float best=-1;
        for(uint32_t edge=rows.GetValue(u);edge<uint32_t(rows.GetValue(u+1));edge++) {
          uint32_t v=columns.GetValue(edge);
          if(v==u || state.GetValue(v)>=0) continue;
          float value=weighted?weights.GetValue(edge):0;
          if(!(value>=0)) continue;
          bool reverse=false;
          for(uint32_t j=rows.GetValue(v);j<uint32_t(rows.GetValue(v+1));j++) {
            if(columns.GetValue(j)==u && (!weighted || weights.GetValue(j)>=0)) {reverse=true;break;}
          }
          if(reverse && ((!weighted && TieBetter(u,v,chosen)) ||
              (weighted && (value>best || (value==best && TieBetter(u,v,chosen)))))) {
            chosen=v;best=value;
          }
        }
        uint32_t v=chosen<0?u:uint32_t(chosen),label=u<v?u:v;
        state.SetValue(u,label);state.SetValue(v,label);
      }
    }
    PipeBarrier<PIPE_ALL>();
    auto source=edges.Get<int32_t>(),expanded=outBuf.Get<int32_t>();
    for(uint32_t off=0;off<n;off+=tile) {
      const uint32_t len=n-off<tile?n-off:tile;
      for(uint32_t j=0;j<len;j++) {
        source.SetValue(j,state.GetValue(off+j));
        offsets.SetValue(2*j,4*j);offsets.SetValue(2*j+1,4*len);
      }
      source.SetValue(len,0);PipeBarrier<PIPE_ALL>();
      Gather(expanded,source,offsets,0,len*2);PipeBarrier<PIPE_ALL>();
      CopyOut(output[uint64_t(off)*2],expanded,len*2);PipeBarrier<PIPE_ALL>();
    }
    Count(0,0,0,1);
  }
  __aicore__ inline void SelectSparse16(uint32_t block,uint32_t rowsCount,
      uint32_t blockFirst,uint32_t edgeCount,uint64_t matchedBits) {
    // A bounded 64x16 ragged CSR tile becomes two SIMD row reductions.
    // Only float32-weight bitmap mode enters here; larger rows retain Select.
    constexpr uint32_t cells=1024;
    const uint32_t count=((rowsCount+3)/4)*64;
    auto arena=rowSelBuf.Get<int32_t>();
    auto values=arena.ReinterpretCast<float>();
    auto vertices=arena[cells].ReinterpretCast<float>();
    auto positions=arena[cells*2];
    auto limits=arena[cells*3];
    auto scratch=arena[cells*4];
    auto mask=arena[cells*5].ReinterpretCast<uint8_t>();
    auto maxima=candidateBuf.Get<float>();
    auto rows=rowBuf.Get<int64_t>();
    auto originalIndices=indexBuf.Get<int32_t>();
    auto originalScores=scoreBuf.Get<float>();
    // positions = (rowptr[row]-blockFirst)+slot, limits = rowptr[row+1]-blockFirst.
    CreateVecIndex(positions,int32_t(0),count);PipeBarrier<PIPE_V>();
    ShiftRight(scratch,positions,int32_t(4),count);PipeBarrier<PIPE_V>();
    Muls(scratch,scratch,int32_t(8),count);PipeBarrier<PIPE_V>();
    Gather(limits,rows.ReinterpretCast<int32_t>(),scratch.ReinterpretCast<uint32_t>(),8,count);
    Gather(values.ReinterpretCast<int32_t>(),rows.ReinterpretCast<int32_t>(),scratch.ReinterpretCast<uint32_t>(),0,count);
    PipeBarrier<PIPE_V>();
    Muls(scratch,scratch,int32_t(2),count);PipeBarrier<PIPE_V>();
    Sub(positions,positions,scratch,count);
    Adds(values.ReinterpretCast<int32_t>(),values.ReinterpretCast<int32_t>(),-int32_t(blockFirst),count);
    Adds(limits,limits,-int32_t(blockFirst),count);PipeBarrier<PIPE_V>();
    Add(positions,positions,values.ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
    Cast(values,positions,RoundMode::CAST_NONE,count);
    Cast(limits.ReinterpretCast<float>(),limits,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
    Compare(mask,values,limits.ReinterpretCast<float>(),CMPMODE::LT,count);PipeBarrier<PIPE_V>();
    Maxs(positions,positions,int32_t(0),count);PipeBarrier<PIPE_V>();
    Mins(positions,positions,int32_t(edgeCount-1),count);PipeBarrier<PIPE_V>();
    Muls(positions,positions,int32_t(4),count);PipeBarrier<PIPE_V>();
    Gather(values,originalScores,positions.ReinterpretCast<uint32_t>(),0,count);
    Gather(vertices.ReinterpretCast<int32_t>(),originalIndices,positions.ReinterpretCast<uint32_t>(),0,count);
    PipeBarrier<PIPE_V>();
    AscendC::Select(values,mask,values,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,count);
    PipeBarrier<PIPE_V>();
    Cast(vertices,vertices.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);
    CreateVecIndex(scratch,int32_t(0),count);PipeBarrier<PIPE_V>();
    ShiftRight(scratch,scratch,int32_t(4),count);PipeBarrier<PIPE_V>();
    Adds(scratch,scratch,int32_t(block),count);PipeBarrier<PIPE_V>();
    Cast(limits.ReinterpretCast<float>(),scratch,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
    Compare(mask,vertices,limits.ReinterpretCast<float>(),CMPMODE::NE,count);PipeBarrier<PIPE_V>();
    AscendC::Select(values,mask,values,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,count);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(maxima,values,16,rowsCount,1,1,2,ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    CreateVecIndex(positions,int32_t(0),count);PipeBarrier<PIPE_V>();
    ShiftRight(positions,positions,int32_t(4),count);PipeBarrier<PIPE_V>();
    Muls(positions,positions,int32_t(4),count);PipeBarrier<PIPE_V>();
    Gather(limits.ReinterpretCast<float>(),maxima,positions.ReinterpretCast<uint32_t>(),0,count);
    PipeBarrier<PIPE_V>();
    Compare(mask,values,limits.ReinterpretCast<float>(),CMPMODE::EQ,count);PipeBarrier<PIPE_V>();
    AscendC::Select(vertices,mask,vertices,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,count);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask,values,float(0),CMPMODE::GE,count);PipeBarrier<PIPE_V>();
    AscendC::Select(vertices,mask,vertices,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,count);
    PipeBarrier<PIPE_V>();
    // For fixed u, the existing (max(u,v),min(u,v)) tie order is exactly v order.
    WholeReduceMax(maxima,vertices,16,rowsCount,1,1,2,ReduceOrder::ORDER_ONLY_VALUE);
    Sync<HardEvent::V_S>();
    auto selected=outBuf.Get<int32_t>();
    for(uint32_t r=0;r<rowsCount;r++)
      selected.SetValue(r,((matchedBits>>r)&1)?-1:int32_t(maxima.GetValue(r)));
    PipeBarrier<PIPE_ALL>();CopyOut(prop[block],selected,rowsCount);PipeBarrier<PIPE_ALL>();
  }

  template<bool ShortVectors=false>
  __aicore__ inline void FilterBarrier() {
    if constexpr(ShortVectors) {PipeBarrier<PIPE_V>();}
    else {PipeBarrier<PIPE_ALL>();}
  }
  template<bool Weighted,bool Wide,uint32_t Mode,bool ShortVectors=false>
  __aicore__ inline void Select() {
    auto columns=edges.Get<int64_t>();
    auto indices=indexBuf.Get<int32_t>();
    auto floatWeight=weightBuf.Get<float>();
    auto wideWeight=weightBuf.Get<int64_t>();
    auto scores=scoreBuf.Get<float>();
    auto reduced=resultBuf.Get<float>();
    auto tmp=reduceBuf.Get<float>();
    auto selected=outBuf.Get<int32_t>();
    auto rows=rowBuf.Get<int64_t>();
    const uint32_t rowTile=Mode==2 || (Mode==1 && Weighted && !Wide &&
      uint64_t(e)<=uint64_t(n)*8)?64:256;
    for(uint32_t block=begin;block<end;block+=rowTile) {
      uint32_t rowsCount=end-block<rowTile?end-block:rowTile;
      uint64_t matchedBits=0;
      if constexpr(Mode==2) {
        matchedBits=uint32_t(cache.GetValue(block>>5));
        if(rowsCount>32)matchedBits|=uint64_t(uint32_t(cache.GetValue((block>>5)+1)))<<32;
        const uint64_t validBits=rowsCount==64?~uint64_t(0):(uint64_t(1)<<rowsCount)-1;
        if((matchedBits&validBits)==validBits) {
          Duplicate(selected,int32_t(-1),rowsCount);PipeBarrier<PIPE_ALL>();
          CopyOut(prop[block],selected,rowsCount);PipeBarrier<PIPE_ALL>();
          continue;
        }
      }
      CopyIn(rows,rp[block],rowsCount+1);PipeBarrier<PIPE_ALL>();
      const uint32_t blockFirst=rows.GetValue(0),blockLast=rows.GetValue(rowsCount);
      bool sparse16=false;
      bool batch=(Mode==2 || (Mode==1 && Weighted && !Wide)) &&
        blockLast>blockFirst && blockLast-blockFirst<=tile;
      if(batch) {
        if constexpr(Mode==2) {
          // Two bitmap words describe the whole 64-row block. Avoid
          // repeated scalar UB status loads, including fully matched blocks.
          auto firstRow=indexBuf.Get<int32_t>(),lastRow=scoreBuf.Get<int32_t>();
          auto lengths=nodeBuf.Get<int32_t>();
          Gather(firstRow,rows.ReinterpretCast<int32_t>(),offsets,0,rowsCount);PipeBarrier<PIPE_V>();
          Gather(lastRow,rows.ReinterpretCast<int32_t>(),offsets,8,rowsCount);PipeBarrier<PIPE_V>();
          Sub(lengths,lastRow,firstRow,rowsCount);PipeBarrier<PIPE_V>();
          Cast(floatWeight,lengths,RoundMode::CAST_NONE,rowsCount);PipeBarrier<PIPE_V>();
          ReduceMax(reduced,floatWeight,tmp,rowsCount,false);PipeBarrier<PIPE_ALL>();
          batch=reduced.GetValue(0)<float(64);
          sparse16=reduced.GetValue(0)<=float(16);
        } else {
          bool active=false;
          for(uint32_t r=0;r<rowsCount;r++) {
            if(rows.GetValue(r+1)-rows.GetValue(r)>=64) batch=false;
            if(Status<Mode>(block+r)<0) active=true;
          }
          batch=batch && active;
        }
      }
      if(batch) {
        const uint32_t len=blockLast-blockFirst;
        if constexpr((Mode==1 || Mode==2) && Weighted && !Wide) {
          const uint32_t padded=(len+63)/64*64;
          Duplicate(indices,int32_t(0),padded);
          Duplicate(floatWeight,float(-1),padded);PipeBarrier<PIPE_ALL>();
        }
        CopyIn(columns,ci[blockFirst],len);
        if(Weighted) {
          if(Wide) CopyIn(wideWeight,wi[blockFirst],len);
          else CopyIn(floatWeight,wf[blockFirst],len);
        }
        PipeBarrier<PIPE_ALL>();
        Gather(indices,columns.ReinterpretCast<int32_t>(),offsets,0,len);PipeBarrier<PIPE_ALL>();
        if constexpr(Mode==2 && Weighted && !Wide) {
          const uint32_t padded=(len+63)/64*64;
          auto byteOffsets=candidateBuf.Get<int32_t>();
          auto flags=reduceBuf.Get<int32_t>();
          auto flagFloat=nodeBuf.Get<float>();
          auto bitmask=rowSelBuf.Get<int32_t>();
          auto table=bitmask[512];
          auto mask=edges.Get<uint8_t>();
          for(uint32_t b=0;b<32;b++)table.SetValue(b,int32_t(1U<<b));
          PipeBarrier<PIPE_ALL>();
          ShiftRight(byteOffsets,indices,int32_t(5),padded);PipeBarrier<PIPE_V>();
          Muls(byteOffsets,byteOffsets,int32_t(4),padded);PipeBarrier<PIPE_V>();
          Gather(flags,cache,byteOffsets.ReinterpretCast<uint32_t>(),0,padded);PipeBarrier<PIPE_V>();
          ShiftRight(byteOffsets,indices,int32_t(5),padded);PipeBarrier<PIPE_V>();
          Muls(byteOffsets,byteOffsets,int32_t(32),padded);PipeBarrier<PIPE_V>();
          Sub(byteOffsets,indices,byteOffsets,padded);PipeBarrier<PIPE_V>();
          Muls(byteOffsets,byteOffsets,int32_t(4),padded);PipeBarrier<PIPE_V>();
          Gather(bitmask,table,byteOffsets.ReinterpretCast<uint32_t>(),0,padded);PipeBarrier<PIPE_V>();
          And(flags.ReinterpretCast<uint16_t>(),flags.ReinterpretCast<uint16_t>(),
              bitmask.ReinterpretCast<uint16_t>(),padded*2);PipeBarrier<PIPE_V>();
          Cast(flagFloat,flags,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
          CompareScalar(mask,flagFloat,float(0),CMPMODE::EQ,padded);PipeBarrier<PIPE_V>();
          AscendC::Select(scores,mask,floatWeight,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
          PipeBarrier<PIPE_V>();
          CompareScalar(mask,scores,float(0),CMPMODE::GE,padded);PipeBarrier<PIPE_V>();
          AscendC::Select(scores,mask,scores,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
          Sync<HardEvent::V_S>();
        }
        if constexpr(Mode==1 && Weighted && !Wide) {
          const uint32_t padded=(len+63)/64*64;
          auto byteOffsets=candidateBuf.Get<int32_t>();
          auto flags=reduceBuf.Get<int32_t>();
          auto flagFloat=nodeBuf.Get<float>();
          auto mask=edges.Get<uint8_t>();
          // SIMD filter across the CSR block amortizes short-row DMA and
          // status lookups. Per-row winner and tie selection remain exact.
          Muls(byteOffsets,indices,int32_t(4),padded);PipeBarrier<PIPE_V>();
          Gather(flags,cache,byteOffsets.ReinterpretCast<uint32_t>(),0,padded);
          PipeBarrier<PIPE_V>();
          Cast(flagFloat,flags,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
          CompareScalar(mask,flagFloat,float(0),CMPMODE::LT,padded);PipeBarrier<PIPE_V>();
          AscendC::Select(scores,mask,floatWeight,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
          PipeBarrier<PIPE_V>();
          CompareScalar(mask,scores,float(0),CMPMODE::GE,padded);PipeBarrier<PIPE_V>();
          AscendC::Select(scores,mask,scores,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
          Sync<HardEvent::V_S>();
        }
      }
      if constexpr(Mode==2 && Weighted && !Wide) {
        if(batch && sparse16 && matchedBits==0) {
          SelectSparse16(block,rowsCount,blockFirst,blockLast-blockFirst,matchedBits);
          continue;
        }
      }
      // Each row owns one int32 candidate. Scalar UB stores avoid three
      // full-vector masked updates (and barriers) for every short CSR row.
      Duplicate(selected,int32_t(-1),rowsCount);
      PipeBarrier<PIPE_ALL>();
      for(uint32_t r=0;r<rowsCount;r++) {
        const uint32_t u=block+r;
        int32_t chosen=-1;
        float best=-1;
        uint64_t bestWide=0;
        bool haveWide=false;
        if constexpr(Mode==2) {
          if((matchedBits>>r)&1)continue;
        } else {
          if(Status<Mode>(u)>=0) continue;
        }
        const uint32_t first=rows.GetValue(r),last=rows.GetValue(r+1);
        for(uint32_t off=first;off<last;off+=tile) {
          uint32_t len=last-off<tile?last-off:tile;
          const uint32_t edgeBase=batch?off-blockFirst:0;
          const uint32_t padded=(len+63)/64*64;
          const bool vectorFilter=Weighted && !Wide && Mode==1 && len>=(ShortVectors?64:128);
          if(vectorFilter) {
            Duplicate(indices,int32_t(0),padded);
            Duplicate(floatWeight,float(-1),padded);
            PipeBarrier<PIPE_ALL>();
          }
          if(!batch) {
            CopyIn(columns,ci[off],len);
            if(Weighted) {
              if(Wide) CopyIn(wideWeight,wi[off],len);
              else CopyIn(floatWeight,wf[off],len);
            }
            PipeBarrier<PIPE_ALL>();
            Gather(indices,columns.ReinterpretCast<int32_t>(),offsets,0,len);PipeBarrier<PIPE_ALL>();
          }
          if(Weighted && !Wide && len<64) {
            for(uint32_t j=0;j<len;j++) {
              const uint32_t v=indices.GetValue(edgeBase+j);
              const float value=((Mode==1 || Mode==2) && batch)?scores.GetValue(edgeBase+j):floatWeight.GetValue(edgeBase+j);
              if(v==u || !(value>=0) || (!((Mode==1 || Mode==2) && batch) && Status<Mode>(v)>=0)) continue;
              if(value>best || (value==best && TieBetter(u,v,chosen))) {
                best=value;chosen=v;
              }
            }
          } else if(Weighted && !Wide) {
            if(vectorFilter) {
              auto byteOffsets=candidateBuf.Get<int32_t>();
              auto flags=reduceBuf.Get<int32_t>();
              auto flagFloat=nodeBuf.Get<float>();
              auto mask=edges.Get<uint8_t>();
              // Cache bounds are <2^24, so index-to-float is exact here.
              // CompareScalar on 910B requires complete 64-float repeats.
              Muls(byteOffsets,indices,int32_t(4),padded);FilterBarrier<ShortVectors>();
              Gather(flags,cache,byteOffsets.ReinterpretCast<uint32_t>(),0,padded);
              FilterBarrier<ShortVectors>();
              Cast(flagFloat,flags,RoundMode::CAST_NONE,padded);FilterBarrier<ShortVectors>();
              CompareScalar(mask,flagFloat,float(0),CMPMODE::LT,padded);FilterBarrier<ShortVectors>();
              AscendC::Select(scores,mask,floatWeight,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
              FilterBarrier<ShortVectors>();
              Cast(flagFloat,indices,RoundMode::CAST_NONE,padded);FilterBarrier<ShortVectors>();
              CompareScalar(mask,flagFloat,float(int32_t(u)),CMPMODE::NE,padded);FilterBarrier<ShortVectors>();
              AscendC::Select(scores,mask,scores,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
              FilterBarrier<ShortVectors>();
              CompareScalar(mask,scores,float(0),CMPMODE::GE,padded);FilterBarrier<ShortVectors>();
              AscendC::Select(scores,mask,scores,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
              FilterBarrier<ShortVectors>();
            } else {
              Duplicate(scores,float(-1),padded);PipeBarrier<PIPE_ALL>();
              for(uint32_t j=0;j<len;j++) {
                uint32_t v=indices.GetValue(edgeBase+j);
                float w=floatWeight.GetValue(edgeBase+j);
                if(v!=u && Status<Mode>(v)<0 && w>=0) scores.SetValue(j,w);
              }
            }
            PipeBarrier<PIPE_ALL>();
            ReduceMax(reduced,scores,tmp,padded,false);PipeBarrier<PIPE_ALL>();
            float maximum=reduced.GetValue(0);
            if(maximum<0 || maximum<best) continue;
            if(vectorFilter) {
              auto mask=edges.Get<uint8_t>();
              CompareScalar(mask,scores,maximum,CMPMODE::EQ,padded);PipeBarrier<PIPE_ALL>();
              auto words=mask.ReinterpretCast<uint32_t>();
              for(uint32_t word=0;word<(len+31)/32;word++) {
                uint32_t bits=words.GetValue(word);
                for(uint32_t b=0;bits && b<32;b++,bits>>=1) {
                  uint32_t j=word*32+b;
                  if(!(bits&1) || j>=len) continue;
                  uint32_t v=indices.GetValue(edgeBase+j);
                  if(maximum>best || TieBetter(u,v,chosen)) {best=maximum;chosen=v;}
                }
              }
            } else {
              for(uint32_t j=0;j<len;j++) {
                if(scores.GetValue(j)!=maximum) continue;
                uint32_t v=indices.GetValue(edgeBase+j);
                if(maximum>best || TieBetter(u,v,chosen)) {best=maximum;chosen=v;}
              }
            }
          } else {
            for(uint32_t j=0;j<len;j++) {
              uint32_t v=indices.GetValue(edgeBase+j);
              if(v==u || Status<Mode>(v)>=0) continue;
              if(Wide) {
                uint64_t value=uint64_t(wideWeight.GetValue(edgeBase+j));
                if(!WideEligible(value)) continue;
                uint64_t key=WideKey(value);
                if(!haveWide || key>bestWide || (key==bestWide && TieBetter(u,v,chosen))) {
                  haveWide=true;bestWide=key;chosen=v;
                }
              } else if(TieBetter(u,v,chosen)) chosen=v;
            }
          }
        }
        selected.SetValue(r,chosen);
      }
      PipeBarrier<PIPE_ALL>();CopyOut(prop[block],selected,rowsCount);PipeBarrier<PIPE_ALL>();
    }
  }
  template<HardEvent Event>
  __aicore__ inline void Sync() {SetFlag<Event>(0);WaitFlag<Event>(0);}

  __aicore__ inline void CommitResident() {
    // Resident candidate labels are <24576, so FP32 label operations are exact.
    // Weight comparisons keep their original precision in Select.
    auto state=nodeBuf.Get<int32_t>(),p=candidateBuf.Get<int32_t>();
    auto stateFloat=weightBuf.Get<float>(),candidateFloat=edges.Get<float>();
    auto ids=indexBuf.Get<float>(),labels=scoreBuf.Get<float>();
    auto result=outBuf.Get<float>(),tmp=reduceBuf.Get<float>();
    auto reduced=resultBuf.Get<float>();
    auto mask=candidateBuf.Get<uint8_t>();
    auto byteOffsets=offsets.ReinterpretCast<int32_t>();
    uint32_t remaining=0,changed=0;
    for(uint32_t off=begin;off<end;off+=tile) {
      const uint32_t len=end-off<tile?end-off:tile,padded=(len+63)/64*64;
      Duplicate(state,int32_t(0),padded);Duplicate(p,int32_t(-1),padded);
      Sync<HardEvent::V_MTE2>();
      CopyIn(state,old[off],len);CopyIn(p,cand[off],len);
      Sync<HardEvent::MTE2_V>();
      Cast(stateFloat,state,RoundMode::CAST_NONE,padded);
      Cast(candidateFloat,p,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
      Maxs(labels,candidateFloat,float(0),padded);PipeBarrier<PIPE_V>();
      Cast(byteOffsets,labels,RoundMode::CAST_RINT,padded);PipeBarrier<PIPE_V>();
      Muls(byteOffsets,byteOffsets,int32_t(4),padded);PipeBarrier<PIPE_V>();
      Gather(state,cache,offsets,0,padded);PipeBarrier<PIPE_V>();
      Cast(result,state,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
      CreateVecIndex(ids,float(int32_t(off)),padded);PipeBarrier<PIPE_V>();
      Min(labels,ids,candidateFloat,padded);PipeBarrier<PIPE_V>();
      Compare(mask,result,ids,CMPMODE::EQ,padded);PipeBarrier<PIPE_V>();
      AscendC::Select(result,mask,labels,stateFloat,SELMODE::VSEL_TENSOR_TENSOR_MODE,padded);
      PipeBarrier<PIPE_V>();
      CompareScalar(mask,candidateFloat,float(0),CMPMODE::LT,padded);PipeBarrier<PIPE_V>();
      AscendC::Select(result,mask,ids,result,SELMODE::VSEL_TENSOR_TENSOR_MODE,padded);
      PipeBarrier<PIPE_V>();
      CompareScalar(mask,stateFloat,float(0),CMPMODE::LT,padded);PipeBarrier<PIPE_V>();
      AscendC::Select(result,mask,result,stateFloat,SELMODE::VSEL_TENSOR_TENSOR_MODE,padded);
      PipeBarrier<PIPE_V>();
      Duplicate(labels,float(1),padded);PipeBarrier<PIPE_V>();
      AscendC::Select(labels,mask,labels,float(0),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
      PipeBarrier<PIPE_V>();
      ReduceSum(reduced,labels,tmp,len);Sync<HardEvent::V_S>();
      const uint32_t active=uint32_t(int32_t(reduced.GetValue(0)));
      CompareScalar(mask,result,float(0),CMPMODE::LT,padded);PipeBarrier<PIPE_V>();
      Duplicate(labels,float(1),padded);PipeBarrier<PIPE_V>();
      AscendC::Select(labels,mask,labels,float(0),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
      PipeBarrier<PIPE_V>();
      ReduceSum(reduced,labels,tmp,len);Sync<HardEvent::V_S>();
      const uint32_t pending=uint32_t(int32_t(reduced.GetValue(0)));
      remaining+=pending;changed+=active-pending;
      Cast(state,result,RoundMode::CAST_RINT,len);Sync<HardEvent::V_MTE3>();
      CopyOut(dst[off],state,len);Sync<HardEvent::MTE3_V>();
    }
    Count(remaining,changed);
  }
  __aicore__ inline void CommitBitmap() {
    // Bitmap mode bounds labels to 2^20, so FP32 labels remain exact.
    // Weight comparisons keep their original precision in Select.
    auto state=nodeBuf.Get<int32_t>(),p=candidateBuf.Get<int32_t>();
    auto stateFloat=weightBuf.Get<float>(),candidateFloat=edges.Get<float>();
    auto ids=indexBuf.Get<float>(),labels=scoreBuf.Get<float>();
    auto result=outBuf.Get<float>(),tmp=reduceBuf.Get<float>();
    auto reduced=resultBuf.Get<float>();
    auto mask=candidateBuf.Get<uint8_t>();
    auto byteOffsets=offsets.ReinterpretCast<int32_t>();
    uint32_t remaining=0,changed=0;
    for(uint32_t off=begin;off<end;off+=tile) {
      const uint32_t len=end-off<tile?end-off:tile,padded=(len+63)/64*64;
      Duplicate(state,int32_t(0),padded);Duplicate(p,int32_t(-1),padded);
      Sync<HardEvent::V_MTE2>();
      CopyIn(state,old[off],len);CopyIn(p,cand[off],len);
      PipeBarrier<PIPE_ALL>();
      auto replies=rowSelBuf.Get<int32_t>();
      Duplicate(replies,int32_t(-2),padded*8);PipeBarrier<PIPE_ALL>();
      const __ubuf__ int32_t* selected=(__ubuf__ int32_t*)p.GetPhyAddr();
      const __ubuf__ int32_t* previous=(__ubuf__ int32_t*)state.GetPhyAddr();
      __ubuf__ int32_t* receive=(__ubuf__ int32_t*)replies.GetPhyAddr();
      __gm__ int32_t* source=(__gm__ int32_t*)cand.GetPhyAddr();
      if ASCEND_IS_AIV {
        for(uint32_t j=0;j<len;j++) {
          if(previous[j]<0 && selected[j]>=0)
            copy_gm_to_ubuf_align_b32(receive+j*8,source+selected[j],0,1,4,0,0,0,0);
        }
      }
      PipeBarrier<PIPE_ALL>();
      Cast(stateFloat,state,RoundMode::CAST_NONE,padded);
      Cast(candidateFloat,p,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
      CreateVecIndex(byteOffsets,int32_t(0),padded);PipeBarrier<PIPE_V>();
      Muls(byteOffsets,byteOffsets,int32_t(32),padded);PipeBarrier<PIPE_V>();
      Gather(state,replies,offsets,0,padded);PipeBarrier<PIPE_V>();
      Cast(result,state,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
      CreateVecIndex(ids,float(int32_t(off)),padded);PipeBarrier<PIPE_V>();
      Min(labels,ids,candidateFloat,padded);PipeBarrier<PIPE_V>();
      Compare(mask,result,ids,CMPMODE::EQ,padded);PipeBarrier<PIPE_V>();
      AscendC::Select(result,mask,labels,stateFloat,SELMODE::VSEL_TENSOR_TENSOR_MODE,padded);
      PipeBarrier<PIPE_V>();
      CompareScalar(mask,candidateFloat,float(0),CMPMODE::LT,padded);PipeBarrier<PIPE_V>();
      AscendC::Select(result,mask,ids,result,SELMODE::VSEL_TENSOR_TENSOR_MODE,padded);
      PipeBarrier<PIPE_V>();
      CompareScalar(mask,stateFloat,float(0),CMPMODE::LT,padded);PipeBarrier<PIPE_V>();
      AscendC::Select(result,mask,result,stateFloat,SELMODE::VSEL_TENSOR_TENSOR_MODE,padded);
      PipeBarrier<PIPE_V>();
      Duplicate(labels,float(1),padded);PipeBarrier<PIPE_V>();
      AscendC::Select(labels,mask,labels,float(0),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
      PipeBarrier<PIPE_V>();
      ReduceSum(reduced,labels,tmp,len);Sync<HardEvent::V_S>();
      const uint32_t active=uint32_t(int32_t(reduced.GetValue(0)));
      CompareScalar(mask,result,float(0),CMPMODE::LT,padded);PipeBarrier<PIPE_V>();
      Duplicate(labels,float(1),padded);PipeBarrier<PIPE_V>();
      AscendC::Select(labels,mask,labels,float(0),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
      PipeBarrier<PIPE_V>();
      ReduceSum(reduced,labels,tmp,len);Sync<HardEvent::V_S>();
      const uint32_t pending=uint32_t(int32_t(reduced.GetValue(0)));
      remaining+=pending;changed+=active-pending;
      Cast(state,result,RoundMode::CAST_RINT,len);Sync<HardEvent::V_MTE3>();
      CopyOut(dst[off],state,len);Sync<HardEvent::MTE3_V>();
      Duplicate(labels,float(-1),padded);PipeBarrier<PIPE_V>();
      // Labels and node indices are exactly representable in bitmap mode.
      Adds(labels,result,float(0),len);PipeBarrier<PIPE_V>();
      CompareScalar(mask,labels,float(0),CMPMODE::GE,padded);PipeBarrier<PIPE_ALL>();
      CopyOut(prop[off/32],mask.ReinterpretCast<int32_t>(),(len+31)/32);
      PipeBarrier<PIPE_ALL>();
    }
    Count(remaining,changed);
  }
  __aicore__ inline void Commit() {
    if(resident==1) {CommitResident();return;}
    if(resident==2) {CommitBitmap();return;}
    auto state=nodeBuf.Get<int32_t>(),p=candidateBuf.Get<int32_t>(),result=outBuf.Get<int32_t>();
    uint32_t remaining=0,changed=0;
    for(uint32_t off=begin;off<end;off+=tile) {
      uint32_t len=end-off<tile?end-off:tile;
      CopyIn(state,old[off],len);CopyIn(p,cand[off],len);PipeBarrier<PIPE_ALL>();
      for(uint32_t j=0;j<len;j++) {
        uint32_t u=off+j;int32_t value=state.GetValue(j),v=p.GetValue(j);
        if(value<0) {
          if(v<0) {value=u;changed++;}
          else {
            int32_t back=resident==1?cache.GetValue(v):cand.GetValue(v);
            if(back==int32_t(u)) {value=u<uint32_t(v)?u:uint32_t(v);changed++;}
          }
        }
        if(value<0) remaining++;
        result.SetValue(j,value);
      }
      PipeBarrier<PIPE_ALL>();CopyOut(dst[off],result,len);PipeBarrier<PIPE_ALL>();
      if(resident==2) {
        auto flags=reduceBuf.Get<float>();
        auto mask=scoreBuf.Get<uint8_t>();
        const uint32_t words=(len+31)/32;
        const uint32_t padded=(len+63)/64*64;
        Duplicate(flags,float(-1),padded);PipeBarrier<PIPE_ALL>();
        Cast(flags,result,RoundMode::CAST_NONE,len);PipeBarrier<PIPE_ALL>();
        CompareScalar(mask,flags,float(0),CMPMODE::GE,padded);PipeBarrier<PIPE_ALL>();
        CopyOut(prop[off/32],mask.ReinterpretCast<int32_t>(),words);PipeBarrier<PIPE_ALL>();
      }
    }
    Count(remaining,changed);
  }
  __aicore__ inline void Finalize() {
    auto src=nodeBuf.Get<int32_t>(),expanded=outBuf.Get<int32_t>();
    // Gather from [labels, zero] into interleaved low/high words (little endian).
    auto wordIndices=weightBuf.Get<int32_t>();
    auto byteOffsets=offsets.ReinterpretCast<int32_t>();
    auto source=edges.Get<int32_t>();
    // Reserved 32B-aligned zero slot beyond CopyIn padding reach.
    Duplicate(source[tile+8],int32_t(0),8);PipeBarrier<PIPE_V>();
    for(uint32_t off=begin;off<end;off+=tile) {
      uint32_t len=end-off<tile?end-off:tile;
      CopyIn(source,old[off],len);PipeBarrier<PIPE_ALL>();
      // Construct [0,4*Z,4,4*Z,...] with SIMD integer arithmetic (Z=zero slot).
      CreateVecIndex(wordIndices,int32_t(0),len*2);PipeBarrier<PIPE_V>();
      ShiftRight(byteOffsets,wordIndices,int32_t(1),len*2);PipeBarrier<PIPE_V>();
      Muls(expanded,byteOffsets,int32_t(2),len*2);PipeBarrier<PIPE_V>();
      Sub(wordIndices,wordIndices,expanded,len*2);PipeBarrier<PIPE_V>();
      Muls(expanded,wordIndices,int32_t(-1),len*2);PipeBarrier<PIPE_V>();
      Adds(expanded,expanded,int32_t(1),len*2);PipeBarrier<PIPE_V>();
      Muls(byteOffsets,byteOffsets,int32_t(4),len*2);PipeBarrier<PIPE_V>();
      Mul(byteOffsets,byteOffsets,expanded,len*2);PipeBarrier<PIPE_V>();
      Muls(wordIndices,wordIndices,int32_t(4*(tile+8)),len*2);PipeBarrier<PIPE_V>();
      Add(byteOffsets,byteOffsets,wordIndices,len*2);PipeBarrier<PIPE_V>();
      Gather(expanded,source,offsets,0,len*2);PipeBarrier<PIPE_ALL>();
      CopyOut(output[uint64_t(off)*2],expanded,len*2);PipeBarrier<PIPE_ALL>();
    }
  }
  __aicore__ inline void Finish() {
    auto buffer=outBuf.Get<int32_t>();
    for(uint32_t off=0;off<n;off+=tile) {
      uint32_t len=n-off<tile?n-off:tile;
      CopyIn(buffer,old[off],len);PipeBarrier<PIPE_ALL>();
      CopyOut(dst[off],buffer,len);PipeBarrier<PIPE_ALL>();
    }
    // One AIV only: finish the remaining bidirectional eligible graph safely.
    for(uint32_t u=0;u<n;u++) {
      if(dst.GetValue(u)>=0) continue;
      int32_t chosen=-1;float best=-1;uint64_t bestWide=0;bool haveWide=false;
      for(int64_t edge=rp.GetValue(u);edge<rp.GetValue(u+1);edge++) {
        uint32_t v=ci.GetValue(edge);
        if(v==u || dst.GetValue(v)>=0) continue;
        float value=0;uint64_t key=0;
        if(weighted) {
          if(kind==0) {value=wf.GetValue(edge);if(!(value>=0)) continue;}
          else {uint64_t bits=uint64_t(wi.GetValue(edge));if(!WideEligible(bits)) continue;key=WideKey(bits);}
        }
        bool reverse=false;
        for(int64_t j=rp.GetValue(v);j<rp.GetValue(v+1);j++) {
          if(ci.GetValue(j)!=u) continue;
          if(!weighted || (kind==0?wf.GetValue(j)>=0:WideEligible(uint64_t(wi.GetValue(j))))) {reverse=true;break;}
        }
        if(!reverse) continue;
        bool better=!weighted?TieBetter(u,v,chosen):kind==0?
          (value>best || (value==best && TieBetter(u,v,chosen))):
          (!haveWide || key>bestWide || (key==bestWide && TieBetter(u,v,chosen)));
        if(better) {chosen=v;best=value;bestWide=key;haveWide=true;}
      }
      uint32_t v=chosen<0?u:uint32_t(chosen),label=u<v?u:v;
      dst.SetValue(u,label);dst.SetValue(v,label);
    }
    DataCacheCleanAndInvalid<int32_t,CacheLine::ENTIRE_DATA_CACHE>(dst);
  }
 private:
  TPipe pipe;
  TBuf<QuePosition::VECCALC> edges,weightBuf,indexBuf,scoreBuf,reduceBuf,offsetBuf,outBuf,nodeBuf,candidateBuf,rowBuf,residentBuf,resultBuf,rowSelBuf;
  GlobalTensor<int64_t> rp,ci,wi;
  GlobalTensor<float> wf;
  GlobalTensor<int32_t> old,cand,dst,prop,count,output;
  LocalTensor<int32_t> cache;
  LocalTensor<uint32_t> offsets;
  uint32_t n,e,phase,kind,weighted,cores,tile,resident,id,begin,end;
};
extern "C" __global__ __aicore__ void graclus(GM_ADDR rowptr,GM_ADDR col,GM_ADDR weight,
    GM_ADDR state,GM_ADDR candidate,GM_ADDR next,GM_ADDR proposals,GM_ADDR counters,
    GM_ADDR out,GM_ADDR workspace,GM_ADDR tiling) {
  GET_TILING_DATA(data,tiling);
  KernelGraclus op;
  op.Init(rowptr,col,weight,state,candidate,next,proposals,counters,out,data);
  if(TILING_KEY_IS(0)) op.Prepare();
  else if(TILING_KEY_IS(7)) op.Fused();
  else if(TILING_KEY_IS(1)) op.Select<true,false,0>();
  else if(TILING_KEY_IS(2)) op.Select<false,false,0>();
  else if(TILING_KEY_IS(3)) op.Select<true,true,0>();
  else if(TILING_KEY_IS(11)) op.Select<true,false,1>();
  else if(TILING_KEY_IS(12)) op.Select<false,false,1>();
  else if(TILING_KEY_IS(13)) op.Select<true,true,1>();
  else if(TILING_KEY_IS(21)) op.Select<true,false,2>();
  else if(TILING_KEY_IS(22)) op.Select<false,false,2>();
  else if(TILING_KEY_IS(23)) op.Select<true,true,2>();
  else if(TILING_KEY_IS(31)) op.Select<true,false,1,true>();
  else if(TILING_KEY_IS(4)) op.Commit();
  else if(TILING_KEY_IS(5)) op.Finalize();
  else if(TILING_KEY_IS(6)) op.Finish();
}
