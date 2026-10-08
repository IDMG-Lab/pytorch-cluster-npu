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
    offsets=offsetBuf.Get<uint32_t>();
    for(uint32_t i=0;i<tile;i++) offsets.SetValue(i,i*8);
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
  __aicore__ inline uint32_t Priority(uint32_t u,uint32_t v) {
    uint32_t lo=u<v?u:v, hi=u<v?v:u;
    return (lo*0x9e3779b1U)^(hi*0x85ebca77U);
  }
  __aicore__ inline bool TieBetter(uint32_t u,uint32_t v,int32_t chosen) {
    if(chosen<0) return true;
    uint32_t a=Priority(u,v), b=Priority(u,uint32_t(chosen));
    if(a!=b) return a>b;
    uint32_t lo=u<v?u:v,hi=u<v?v:u;
    uint32_t oldLo=u<uint32_t(chosen)?u:uint32_t(chosen);
    uint32_t oldHi=u<uint32_t(chosen)?uint32_t(chosen):u;
    return lo<oldLo || (lo==oldLo && hi<oldHi);
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
    Duplicate(buf,int32_t(0),len);PipeBarrier<PIPE_ALL>();
    buf.SetValue(0,remaining);buf.SetValue(1,changed);buf.SetValue(2,error);
    buf.SetValue(3,complete);
    SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
    // Contiguous cache-line-owned counters, one DMA per AIV; all slots defined.
    CopyOut(count[first*16],buf,len);
    SetFlag<HardEvent::MTE3_S>(0);WaitFlag<HardEvent::MTE3_S>(0);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Prepare() {
    auto fill=outBuf.Get<int32_t>();
    Duplicate(fill,int32_t(-1),tile);PipeBarrier<PIPE_ALL>();
    uint32_t error=0;
    auto rows=rowBuf.Get<int64_t>();
    for(uint32_t off=begin;off<end;off+=256) {
      uint32_t len=end-off<256?end-off:256;
      CopyIn(rows,rp[off],len+1);PipeBarrier<PIPE_ALL>();
      for(uint32_t i=0;i<len;i++) {
        int64_t a=rows.GetValue(i),b=rows.GetValue(i+1);
        if(a<0 || b<a || b>e || (off+i==0 && a!=0) || (off+i+1==n && b!=e)) error=1;
      }
      CopyOut(dst[off],fill,len);
    }
    auto columns=edges.Get<int64_t>();
    const uint32_t first=uint64_t(e)*id/cores,last=uint64_t(e)*(id+1)/cores;
    for(uint32_t off=first;off<last;off+=tile) {
      uint32_t len=last-off<tile?last-off:tile;
      CopyIn(columns,ci[off],len);PipeBarrier<PIPE_ALL>();
      for(uint32_t i=0;i<len;i++) {
        int64_t v=columns.GetValue(i);if(v<0 || v>=n) error=1;
      }
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
      for(uint32_t off=begin;off<end;off+=tile) {
        const uint32_t len=end-off<tile?end-off:tile;
        for(uint32_t j=0;j<len;j++) {
          source.SetValue(j,int32_t(off+j));
          offsets.SetValue(2*j,4*j);offsets.SetValue(2*j+1,4*len);
        }
        source.SetValue(len,0);PipeBarrier<PIPE_ALL>();
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
  template<bool Weighted,bool Wide,uint32_t Mode>
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
    constexpr uint32_t rowTile=Mode==2?64:256;
    for(uint32_t block=begin;block<end;block+=rowTile) {
      uint32_t rowsCount=end-block<rowTile?end-block:rowTile;
      CopyIn(rows,rp[block],rowsCount+1);PipeBarrier<PIPE_ALL>();
      const uint32_t blockFirst=rows.GetValue(0),blockLast=rows.GetValue(rowsCount);
      bool batch=Mode==2 && blockLast>blockFirst && blockLast-blockFirst<=tile;
      if(batch) {
        bool active=false;
        for(uint32_t r=0;r<rowsCount;r++) {
          if(rows.GetValue(r+1)-rows.GetValue(r)>=128) batch=false;
          if(Status<Mode>(block+r)<0) active=true;
        }
        batch=batch && active;
      }
      if(batch) {
        const uint32_t len=blockLast-blockFirst;
        CopyIn(columns,ci[blockFirst],len);
        if(Weighted) {
          if(Wide) CopyIn(wideWeight,wi[blockFirst],len);
          else CopyIn(floatWeight,wf[blockFirst],len);
        }
        PipeBarrier<PIPE_ALL>();
        Gather(indices,columns.ReinterpretCast<int32_t>(),offsets,0,len);PipeBarrier<PIPE_ALL>();
      }
      for(uint32_t r=0;r<rowsCount;r++) {
        const uint32_t u=block+r;
        int32_t chosen=-1;
        float best=-1;
        uint64_t bestWide=0;
        bool haveWide=false;
        if(Status<Mode>(u)>=0) {selected.SetValue(r,-1);continue;}
        const uint32_t first=rows.GetValue(r),last=rows.GetValue(r+1);
        for(uint32_t off=first;off<last;off+=tile) {
          uint32_t len=last-off<tile?last-off:tile;
          const uint32_t edgeBase=batch?off-blockFirst:0;
          const uint32_t padded=(len+63)/64*64;
          const bool vectorFilter=Weighted && !Wide && Mode==1 && len>=128;
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
              const float value=floatWeight.GetValue(edgeBase+j);
              if(v==u || Status<Mode>(v)>=0 || !(value>=0)) continue;
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
              Muls(byteOffsets,indices,int32_t(4),padded);PipeBarrier<PIPE_ALL>();
              Gather(flags,cache,byteOffsets.ReinterpretCast<uint32_t>(),0,padded);
              PipeBarrier<PIPE_ALL>();
              Cast(flagFloat,flags,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_ALL>();
              CompareScalar(mask,flagFloat,float(0),CMPMODE::LT,padded);PipeBarrier<PIPE_ALL>();
              AscendC::Select(scores,mask,floatWeight,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
              PipeBarrier<PIPE_ALL>();
              Cast(flagFloat,indices,RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_ALL>();
              CompareScalar(mask,flagFloat,float(int32_t(u)),CMPMODE::NE,padded);PipeBarrier<PIPE_ALL>();
              AscendC::Select(scores,mask,scores,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
              PipeBarrier<PIPE_ALL>();
              CompareScalar(mask,scores,float(0),CMPMODE::GE,padded);PipeBarrier<PIPE_ALL>();
              AscendC::Select(scores,mask,scores,float(-1),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
              PipeBarrier<PIPE_ALL>();
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
  __aicore__ inline void Commit() {
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
    auto zeros=scoreBuf.Get<int32_t>();
    auto source=edges.Get<int32_t>();
    for(uint32_t off=begin;off<end;off+=tile) {
      uint32_t len=end-off<tile?end-off:tile;
      CopyIn(source,old[off],len);PipeBarrier<PIPE_ALL>();
      source.SetValue(len,0);
      for(uint32_t j=0;j<len;j++) {offsets.SetValue(2*j,4*j);offsets.SetValue(2*j+1,4*len);}
      PipeBarrier<PIPE_ALL>();
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
  TBuf<QuePosition::VECCALC> edges,weightBuf,indexBuf,scoreBuf,reduceBuf,offsetBuf,outBuf,nodeBuf,candidateBuf,rowBuf,residentBuf,resultBuf;
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
  else if(TILING_KEY_IS(4)) op.Commit();
  else if(TILING_KEY_IS(5)) op.Finalize();
  else if(TILING_KEY_IS(6)) op.Finish();
}
