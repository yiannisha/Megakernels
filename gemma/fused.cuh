#pragma once
// All linear operators use the upstream three-stage TMA/consumer/storer ring.
template<int Opcode> struct FusedLinear {
 static constexpr int opcode=Opcode;
 static constexpr bool normalized=Opcode==QkvRope || Opcode==GatedGelu || Opcode==NormHead;
 struct Instruction {
  int index,start,tiles,parts,iters;
  __device__ Instruction(Cfg::instruction_t& x):index(x[1]),start(x[2]),tiles(x[3]),parts(x[4]),iters(x[3]*x[4]){}
  __device__ Instruction(megakernel::state<Cfg>& s):Instruction(s.instruction()){}
 };
 struct Specifics;
 using Pipeline=matvec_pipeline<Cfg,Globals,Instruction,Specifics>;
 static constexpr int sem_count=Pipeline::SEM_COUNT+(normalized?1:0);
 static_assert((Cfg::DYNAMIC_SEMAPHORES-sem_count)*sizeof(kittens::semaphore)>=64);
 static __device__ float* temporary(megakernel::state<Cfg>& s){return reinterpret_cast<float*>(&s.semaphores()[sem_count]);}
 struct Specifics {
  static __device__ void load_iter(megakernel::state<Cfg>& s,const Globals& g,Instruction& in,int iter,int col,kittens::st_bf<16,512>& tile,kittens::semaphore& sem){
   int row=Opcode==GatedGelu?(in.start+iter/2)*32+(iter%2)*16:(in.start+iter/in.parts)*16;
   int k=Opcode==DownResidual?(iter%in.parts)*2048:0;
   kittens::tma::load_async<kittens::dim::ROW,kittens::cache_policy::EVICT_FIRST>(tile,g.weights[in.index],kittens::coord<>{0,row,k+col*512},sem);
  }
  static __device__ void store(megakernel::state<Cfg>& s,const Globals& g,Instruction& in,int iter,int ring){
   int lane=kittens::laneid();Stage op=g.stage(in.index);
   float* scratch=reinterpret_cast<float*>(Pipeline::get_output_start(s,ring));
   float value=0;
   if(lane<16){
    #pragma unroll
    for(int w=0;w<16;w++)value+=scratch[w*16+lane];
   }
   if constexpr(Opcode==QkvRope){
    int tile=in.start+iter,head=tile/16,dim=(tile%16)*8+lane%8+(lane/8)*128;
    float x=float(bf(value));float peer=__shfl_xor_sync(0xffffffff,x,8);
    if(lane<16){
     bf result=bf(x);
     if(head<9){int offset=(*g.raw.pos)*128+dim%128;
      float co=float(g.raw.cos[offset]),si=float(g.raw.sin[offset]);
      result=bf(float(bf(x*co))+float(bf((lane<8?-peer:peer)*si)));
     }
     if(head<8)op.output[head*256+dim]=result;
     else op.kv[(head-8)*op.context*256+(*g.raw.pos)*256+dim]=result;
    }
    __threadfence();__syncwarp();
    if(lane==0){signal_flag(g,in.index,head,1);if(iter+1==in.iters)finish_stage(g,in.index,in.tiles);}
   }else if constexpr(Opcode==GatedGelu){
    if(lane<16){
     float x=float(bf(value));
     if(iter%2==0)temporary(s)[lane]=float(bf(.5f*x*(1.f+erff(x*.7071067811865475f))));
     else op.output[(in.start+iter/2)*16+lane]=bf(temporary(s)[lane]*x);
    }
    __threadfence();__syncwarp();
    if(lane==0 && iter%2==1){signal_flag(g,in.index,(in.start+iter/2)/128,1);if(iter+1==in.iters)finish_stage(g,in.index,in.tiles);}
   }else if constexpr(Opcode==DownResidual){
    if(lane<16){
     if(iter%in.parts==0)temporary(s)[lane]=value;else temporary(s)[lane]+=value;
     if(iter%in.parts==in.parts-1){int row=(in.start+iter/in.parts)*16+lane;
      op.output[row]=bf(float(bf(temporary(s)[lane]))+float(op.residual[row]));}
    }
    __threadfence();__syncwarp();
    if(lane==0 && iter+1==in.iters)finish_stage(g,in.index,in.tiles);
   }else if constexpr(Opcode==OutputResidual){
    if(lane<16){int row=(in.start+iter)*16+lane;op.output[row]=bf(float(bf(value))+float(op.residual[row]));}
    __threadfence();__syncwarp();if(lane==0 && iter+1==in.iters)finish_stage(g,in.index,in.tiles);
   }else if constexpr(Opcode==NormHead){
    if(lane<16)op.logits[(in.start+iter)*16+lane]=value;
    __threadfence();__syncwarp();if(lane==0 && iter+1==in.iters)finish_stage(g,in.index,in.tiles);
   }
  }
 };
 struct controller {
  static __device__ int release_lid(const Globals& g,Cfg::instruction_t& i,int& q){return Pipeline::release_lid(g,i,q);}
  static __device__ int init_semaphores(const Globals&,megakernel::state<Cfg>& s){
   Pipeline::init_semaphores(s);if constexpr(normalized)kittens::init_semaphore(s.semaphores()[Pipeline::SEM_COUNT],1);return sem_count;
  }
 };
 struct loader {
  static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){
   Instruction in(s);
   if constexpr(normalized){if(kittens::laneid()==0){
    s.wait_page_ready(Pipeline::get_activation_page(s));
    auto& norm=*reinterpret_cast<kittens::sv_bf<2048>*>(s.pages[Pipeline::get_activation_page(s)].ptr(4096));
    auto& sem=s.semaphores()[Pipeline::SEM_COUNT];
    kittens::tma::expect(sem,norm);kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(norm,g.norms[in.index],{},sem);
   }}
   Pipeline::loader_loop(s,g);
  }
 };
 using launcher=megakernel::NoOp<Cfg>::launcher;
 struct consumer {
  static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){
   Instruction in(s);Stage op=g.stage(in.index);
   if(threadIdx.x==0)s.wait_page_ready(Pipeline::get_activation_page(s));
   kittens::group<16>::sync(3);
   if constexpr(Opcode==DownResidual){/* Chunk dependencies are consumed below. */}
   else if constexpr(Opcode==OutputResidual){if(kittens::laneid()==0)wait_flag(g,op.producer,kittens::warpid()/2,1);}
   else {if(threadIdx.x==0)wait_stage(g,op.producer);kittens::group<16>::sync(3);}
   kittens::warp::sync();
   using SV=kittens::sv_bf<128>;using RV=kittens::rv_fl<128>;
   RV values;
   if constexpr(Opcode!=DownResidual){
    SV& input=reinterpret_cast<SV*>(&Pipeline::get_activations(s))[kittens::warpid()];
    kittens::warp::load(input,g.activations[in.index],kittens::coord<>{kittens::warpid()*128});kittens::warp::sync();
    kittens::warp::load(values,input);
    if constexpr(normalized){
     RV squared;kittens::warp::mul(squared,values,values);
     float sum=kittens::warp::sum(squared);
     if(kittens::laneid()==0)temporary(s)[kittens::warpid()]=sum;
     kittens::group<16>::sync(3);
     float total=0;for(int w=0;w<16;w++)total+=temporary(s)[w];
     float inv=rsqrtf(total/2048.f+1e-6f);
     kittens::wait(s.semaphores()[Pipeline::SEM_COUNT],0);
     auto& norm=reinterpret_cast<SV*>(s.pages[Pipeline::get_activation_page(s)].ptr(4096))[kittens::warpid()];
     RV scale;kittens::warp::load(scale,norm);kittens::warp::add(scale,scale,1.f);
     kittens::warp::mul(values,values,inv);kittens::warp::mul(values,values,scale);
     kittens::rv_bf<128> rounded;kittens::warp::copy(rounded,values);kittens::warp::copy(values,rounded);
    }
   }
   kittens::warp::sync();s.warp_finish_page(Pipeline::get_activation_page(s),1);
   Pipeline::consumer_loop(s,g,values,[&] __device__(int iter,RV& v){
    if constexpr(Opcode==DownResidual){int part=iter%in.parts;
     if(kittens::laneid()==0)wait_flag(g,op.producer,part,128);kittens::warp::sync();
     kittens::warp::load(v,g.activations[in.index],{part*16+kittens::warpid()});
    }
   });
  }
 };
 struct storer {static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){Pipeline::storer_loop(s,g);}};
};

struct EmbedOp {
 static constexpr int opcode=Embed;
 using controller=megakernel::NoOp<Cfg>::controller;using loader=megakernel::NoOp<Cfg>::loader;
 using launcher=megakernel::NoOp<Cfg>::launcher;using storer=megakernel::NoOp<Cfg>::storer;
 struct consumer {static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){
  Stage op=g.stage(s.instruction()[1]);
  if(kittens::warpid()==0){for(int i=kittens::laneid();i<2048;i+=32)op.output[i]=bf(float(op.weight[(*g.raw.token)*2048+i])*float(bf(sqrtf(2048.f))));
   __threadfence();__syncwarp();if(kittens::laneid()==0)finish_stage(g,s.instruction()[1],1);}
  kittens::group<16>::sync(5);
 }};
};
struct SampleOp {
 static constexpr int opcode=Sample;
 using controller=megakernel::NoOp<Cfg>::controller;using loader=megakernel::NoOp<Cfg>::loader;
 using launcher=megakernel::NoOp<Cfg>::launcher;using storer=megakernel::NoOp<Cfg>::storer;
 struct consumer {static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){
  int index=s.instruction()[1];Stage op=g.stage(index);
  if(threadIdx.x==0)wait_stage(g,op.producer);kittens::group<16>::sync(5);
  float value=-INFINITY;int id=0;
  for(int i=threadIdx.x;i<256000;i+=512)if(op.logits[i]>value){value=op.logits[i];id=i;}
  for(int d=16;d;d/=2){float v=__shfl_down_sync(0xffffffff,value,d);int k=__shfl_down_sync(0xffffffff,id,d);if(v>value || (v==value && k<id)){value=v;id=k;}}
  float* vals=reinterpret_cast<float*>(s.scratch());int* ids=reinterpret_cast<int*>(vals+16);
  if(kittens::laneid()==0){vals[kittens::warpid()]=value;ids[kittens::warpid()]=id;}kittens::group<16>::sync(5);
  if(threadIdx.x==0){value=vals[0];id=ids[0];for(int i=1;i<16;i++)if(vals[i]>value || (vals[i]==value && ids[i]<id)){value=vals[i];id=ids[i];}
   *g.raw.token=id;*g.raw.pos+=1;__threadfence();finish_stage(g,index,1);}
  kittens::group<16>::sync(5);
 }};
};
