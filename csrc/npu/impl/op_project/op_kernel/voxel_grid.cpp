#include "kernel_operator.h"
using namespace AscendC;
#ifndef DTYPE_POS
#define DTYPE_POS float
#endif

// Eight exact byte accumulators implement modulo-2^64 grid labels on AIV.
// No int64 vector multiply is required; the final low/high words are gathered
// directly into int64 output storage. Unusually large coordinates use the
// general scalar definition rather than silently losing integer precision.
class KernelVoxelGrid {
public:
  __aicore__ inline void Init(GM_ADDR pos,GM_ADDR size,GM_ADDR start,GM_ADDR end,
      GM_ADDR out,const VoxelGridTilingData* d) {
    dim=d->dim;n=d->numPoints;
    first=GetBlockIdx()*d->blockPoints;
    count=d->blockPoints+(GetBlockIdx()==GetBlockNum()-1?d->tailPoints:0);
    pg.SetGlobalBuffer((__gm__ DTYPE_POS*)pos);sg.SetGlobalBuffer((__gm__ DTYPE_POS*)size);
    bg.SetGlobalBuffer((__gm__ DTYPE_POS*)start);eg.SetGlobalBuffer((__gm__ DTYPE_POS*)end);
    og.SetGlobalBuffer((__gm__ int64_t*)out);
    pipe.InitBuffer(inBuf,T*sizeof(DTYPE_POS));pipe.InitBuffer(posBuf,T*4);
    pipe.InitBuffer(coordBuf,T*4);pipe.InitBuffer(cfBuf,T*4);pipe.InitBuffer(tmpBuf,T*4);
    pipe.InitBuffer(carryBuf,T*4);pipe.InitBuffer(absBuf,T*4);pipe.InitBuffer(limbBuf,T*8*4);
    pipe.InitBuffer(packedBuf,T*2*4);pipe.InitBuffer(outBuf,T*2*4);pipe.InitBuffer(indexBuf,T*2*4);
    pipe.InitBuffer(workBuf,T*4);pipe.InitBuffer(resultBuf,32);
    auto indices=indexBuf.Get<int32_t>(),tmp=outBuf.Get<int32_t>(),scratch=packedBuf.Get<int32_t>();
    CreateVecIndex(indices,int32_t(0),T*2);PipeBarrier<PIPE_V>();
    ShiftRight(tmp,indices,int32_t(1),T*2);PipeBarrier<PIPE_V>();
    Muls(scratch,tmp,int32_t(2),T*2);PipeBarrier<PIPE_V>();
    Sub(indices,indices,scratch,T*2);PipeBarrier<PIPE_V>();
    Muls(indices,indices,int32_t(T*4),T*2);PipeBarrier<PIPE_V>();
    Muls(tmp,tmp,int32_t(4),T*2);PipeBarrier<PIPE_V>();
    Add(indices,indices,tmp,T*2);PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Process() {
    auto input=inBuf.Get<DTYPE_POS>();auto pos=posBuf.Get<float>();
    auto coord=coordBuf.Get<int32_t>();auto cf=cfBuf.Get<float>(),tmp=tmpBuf.Get<float>();
    auto carry=carryBuf.Get<float>(),magnitude=absBuf.Get<float>(),limbs=limbBuf.Get<float>();
    auto packed=packedBuf.Get<int32_t>(),words=outBuf.Get<int32_t>();
    auto offsets=indexBuf.Get<uint32_t>();
    for(uint32_t off=0;off<count;off+=T) {
      const uint32_t len=count-off<T?count-off:T;
      for(uint32_t b=0;b<8;++b)Duplicate(limbs[b*T],float(0),len);
      Duplicate(magnitude,float(0),len);PipeBarrier<PIPE_ALL>();
      uint64_t stride=1;
      for(uint32_t axis=0;axis<dim;++axis) {
        const float size=static_cast<float>(sg.GetValue(axis));
        const float start=static_cast<float>(bg.GetValue(axis));
        const float end=static_cast<float>(eg.GetValue(axis));
        DataCopyPad(input,pg[uint64_t(axis)*n+first+off],DataCopyExtParams{1,uint32_t(len*sizeof(DTYPE_POS)),0,0,0},DataCopyPadExtParams<DTYPE_POS>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();
        if constexpr(sizeof(DTYPE_POS)==sizeof(float)) {
          Adds(pos,input.ReinterpretCast<float>(),-start,len);
        } else {
          Cast(pos,input,RoundMode::CAST_NONE,len);PipeBarrier<PIPE_V>();
          Adds(pos,pos,-start,len);
        }
        PipeBarrier<PIPE_V>();Muls(pos,pos,1.0f/size,len);PipeBarrier<PIPE_V>();
        Cast(coord,pos,RoundMode::CAST_TRUNC,len);PipeBarrier<PIPE_V>();
        Cast(cf,coord,RoundMode::CAST_NONE,len);PipeBarrier<PIPE_V>();
        Abs(tmp,cf,len);PipeBarrier<PIPE_V>();Add(magnitude,magnitude,tmp,len);PipeBarrier<PIPE_V>();
        for(uint32_t b=0;b<8;++b) {
          const uint32_t byte=(stride>>(b*8))&255;
          if(byte) {
            Muls(tmp,cf,float(int32_t(byte)),len);PipeBarrier<PIPE_V>();
            Add(limbs[b*T],limbs[b*T],tmp,len);PipeBarrier<PIPE_V>();
          }
        }
        stride*=uint64_t(int64_t((end-start)/size)+1);
      }
      PipeBarrier<PIPE_ALL>();
      auto reduced=resultBuf.Get<float>();
      ReduceMax(reduced,magnitude,workBuf.Get<float>(),int32_t(len),false);PipeBarrier<PIPE_ALL>();
      if(!(reduced.GetValue(0)<=32768.0f)) {
        auto output=outBuf.Get<int64_t>();
        for(uint32_t i=0;i<len;++i) {
          uint64_t label=0,k=1;
          for(uint32_t axis=0;axis<dim;++axis) {
            const float s=static_cast<float>(sg.GetValue(axis)),a=static_cast<float>(bg.GetValue(axis));
            const float e=static_cast<float>(eg.GetValue(axis));
            const int64_t c=int64_t((static_cast<float>(pg.GetValue(uint64_t(axis)*n+first+off+i))-a)/s);
            label+=uint64_t(c)*k;k*=uint64_t(int64_t((e-a)/s)+1);
          }
          output.SetValue(i,int64_t(label));
        }
      } else {
        Duplicate(carry,float(0),len);PipeBarrier<PIPE_V>();
        for(uint32_t b=0;b<8;++b) {
          Add(tmp,limbs[b*T],carry,len);PipeBarrier<PIPE_V>();
          Muls(carry,tmp,1.0f/256.0f,len);PipeBarrier<PIPE_V>();
          Cast(coord,carry,RoundMode::CAST_FLOOR,len);PipeBarrier<PIPE_V>();
          Cast(carry,coord,RoundMode::CAST_NONE,len);PipeBarrier<PIPE_V>();
          Muls(cf,carry,256.0f,len);PipeBarrier<PIPE_V>();
          Sub(limbs[b*T],tmp,cf,len);PipeBarrier<PIPE_V>();
        }
        Duplicate(packed,int32_t(0),len);Duplicate(packed[T],int32_t(0),len);PipeBarrier<PIPE_V>();
        for(uint32_t b=0;b<8;++b) {
          Cast(coord,limbs[b*T],RoundMode::CAST_TRUNC,len);PipeBarrier<PIPE_V>();
          Muls(coord,coord,int32_t(uint32_t(1)<<((b%4)*8)),len);PipeBarrier<PIPE_V>();
          Add(packed[(b/4)*T],packed[(b/4)*T],coord,len);PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();Gather(words,packed,offsets,0,len*2);PipeBarrier<PIPE_ALL>();
      }
      PipeBarrier<PIPE_ALL>();
      DataCopyPad(og[first+off],outBuf.Get<int64_t>(),DataCopyExtParams{1,len*8,0,0,0});
      PipeBarrier<PIPE_ALL>();
    }
  }
private:
  static constexpr uint32_t T=2048;
  TPipe pipe;
  TBuf<QuePosition::VECCALC> inBuf,posBuf,coordBuf,cfBuf,tmpBuf,carryBuf,absBuf,limbBuf,packedBuf,outBuf,indexBuf,workBuf,resultBuf;
  GlobalTensor<DTYPE_POS> pg,sg,bg,eg;GlobalTensor<int64_t> og;
  uint32_t dim,n,first,count;
};
extern "C" __global__ __aicore__ void voxel_grid(GM_ADDR pos,GM_ADDR size,GM_ADDR start,
    GM_ADDR end,GM_ADDR out,GM_ADDR workspace,GM_ADDR tiling) {
  GET_TILING_DATA(data,tiling);
  KernelVoxelGrid op;op.Init(pos,size,start,end,out,&data);op.Process();
}
