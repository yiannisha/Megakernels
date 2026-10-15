#pragma once
// Split MQA: double-buffered TMA K/V tiles, tensor-core QK, online softmax,
// then a dependency-counter-triggered split combine. No standalone CUDA ops.
struct AttentionOp {
 static constexpr int opcode=AttentionPartial;
 static constexpr int ring_size=2;
 static constexpr int sem_count=5;
 static __device__ int iterations(const Globals& g,const Stage& op,int task){
  int tiles=(*g.raw.pos+32)/32,split=task%op.splits;
  int first=tiles*split/op.splits,last=tiles*(split+1)/op.splits;return last-first;
 }
 static __device__ int first_tile(const Globals& g,const Stage& op,int task){return ((*g.raw.pos+32)/32)*(task%op.splits)/op.splits;}
 static __device__ auto& ready(megakernel::state<Cfg>& s,int ring){return s.semaphores()[ring];}
 static __device__ auto& finished(megakernel::state<Cfg>& s,int ring){return s.semaphores()[2+ring];}
 static __device__ auto& output_ready(megakernel::state<Cfg>& s){return s.semaphores()[4];}
 static __device__ auto& keys(megakernel::state<Cfg>& s,int ring){return *reinterpret_cast<kittens::st_bf<32,256>*>(s.pages[s.pid(ring*2)].ptr());}
 static __device__ auto& values(megakernel::state<Cfg>& s,int ring){return *reinterpret_cast<kittens::st_bf<32,256>*>(s.pages[s.pid(ring*2+1)].ptr());}
 struct controller {
  static __device__ int release_lid(const Globals& g,Cfg::instruction_t& i,int& query){
   Stage op=g.stage(i[1]);int n=iterations(g,op,i[2]),used=2*min(n,2);
   if(query<13-used)return used+query;
   int q=query-(13-used);int ring=n==1?0:(n-2+q/2)%2;return ring*2+q%2;
  }
  static __device__ int init_semaphores(const Globals&,megakernel::state<Cfg>& s){
   for(int i=0;i<2;i++){kittens::init_semaphore(ready(s,i),1);kittens::init_semaphore(finished(s,i),16);}
   kittens::init_semaphore(output_ready(s),16);return sem_count;
  }
 };
 struct loader {static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){
  int index=s.instruction()[1],task=s.instruction()[2];Stage op=g.stage(index);int n=iterations(g,op,task),first=first_tile(g,op,task);
  if(kittens::laneid()==0){
   if(n==0){int page=s.pid(0);s.wait_page_ready(page);s.finish_page(page,16);}
   // K/V append must be visible before TMA is allowed to fetch cache tiles.
   wait_flag(g,op.producer,8,16);wait_flag(g,op.producer,9,16);
   for(int i=0;i<n;i++){
    int ring=i%2;kittens::wait(finished(s,ring),(i%4)<2);
    if(i<2)for(int p=0;p<2;p++)s.wait_page_ready(s.pid(ring*2+p));
    auto& sem=ready(s,ring);kittens::tma::expect_bytes(sem,32768);
    kittens::tma::load_async<kittens::dim::ROW,kittens::cache_policy::EVICT_LAST>(keys(s,ring),g.keys[index],kittens::coord<>{(first+i)*32,0},sem);
    kittens::tma::load_async<kittens::dim::ROW,kittens::cache_policy::EVICT_LAST>(values(s,ring),g.values[index],kittens::coord<>{(first+i)*32,0},sem);
   }
  }else if(kittens::laneid()>=2*min(n,2) && kittens::laneid()<13){int p=s.pid(kittens::laneid());s.wait_page_ready(p);s.finish_page(p,16);}
 }};
 using launcher=megakernel::NoOp<Cfg>::launcher;
 struct consumer {static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){
  int index=s.instruction()[1],task=s.instruction()[2];Stage op=g.stage(index);int head=task/op.splits,n=iterations(g,op,task),first=first_tile(g,op,task);
  float* scratch=reinterpret_cast<float*>(s.scratch());float* scores=scratch;float* state=scratch+64;float* output=scratch+128;
  if(threadIdx.x==0){wait_flag(g,op.producer,head,16);state[0]=-INFINITY;state[1]=0;}
  kittens::group<16>::sync(6);
  kittens::rv_fl<256> query;
  if(kittens::warpid()<2)kittens::warp::load(query,g.activations[index],{head});
  float numerator=0;
  for(int i=0;i<n;i++){
   int ring=i%2;kittens::wait(ready(s,ring),(i%4)>=2);
   if(kittens::warpid()<2){
    auto tile=keys(s,ring).template subtile<16,256>({kittens::warpid(),0});
    auto& score=*reinterpret_cast<kittens::sv_fl<16>*>(scores+kittens::warpid()*16);
    matvec(score,tile,query);
   }
   kittens::group<16>::sync(6);
   if(threadIdx.x==0){
    float maximum=state[0];int valid=min(32,*g.raw.pos+1-(first+i)*32);
    for(int j=0;j<valid;j++)maximum=fmaxf(maximum,scores[j]*.0625f);
    float alpha=isinf(state[0])?0.f:expf(state[0]-maximum),sum=0;
    for(int j=0;j<32;j++){scores[j]=j<valid?expf(scores[j]*.0625f-maximum):0.f;sum+=scores[j];}
    state[0]=maximum;state[1]=state[1]*alpha+sum;state[2]=alpha;
   }
   kittens::group<16>::sync(6);
   if(threadIdx.x<256){
    numerator*=state[2];
    #pragma unroll 4
    for(int j=0;j<32;j++)numerator+=scores[j]*float(values(s,ring)[{j,(int)threadIdx.x}]);
   }
   kittens::group<16>::sync(6);kittens::warp::arrive(finished(s,ring));
   if(i>=n-2)for(int p=0;p<2;p++)s.warp_finish_page(s.pid(ring*2+p),1);
  }
  if(threadIdx.x<256)output[threadIdx.x]=numerator;
  kittens::group<16>::sync(6);kittens::warp::arrive(output_ready(s));
 }};
 struct storer {static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){
  int index=s.instruction()[1],task=s.instruction()[2];Stage op=g.stage(index);
  kittens::wait(output_ready(s),0);float* scratch=reinterpret_cast<float*>(s.scratch());
  for(int d=kittens::laneid();d<256;d+=32)op.partials[task*256+d]=scratch[128+d];
  if(kittens::laneid()==0){op.stats[task*2]=scratch[64];op.stats[task*2+1]=scratch[65];}
  __threadfence();__syncwarp();if(kittens::laneid()==0){signal_flag(g,index,task,1);finish_stage(g,index,1);}
 }};
};
struct CombineOp {
 static constexpr int opcode=AttentionCombine;
 using controller=megakernel::NoOp<Cfg>::controller;using loader=megakernel::NoOp<Cfg>::loader;
 using launcher=megakernel::NoOp<Cfg>::launcher;using storer=megakernel::NoOp<Cfg>::storer;
 struct consumer {static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){
  int index=s.instruction()[1],head=s.instruction()[2];Stage op=g.stage(index);
  if(threadIdx.x==0)for(int p=0;p<op.splits;p++)wait_flag(g,op.producer,head*op.splits+p,1);
  kittens::group<16>::sync(6);
  if(threadIdx.x<256){float maximum=-INFINITY;
   for(int p=0;p<op.splits;p++)maximum=fmaxf(maximum,op.stats[(head*op.splits+p)*2]);
   float sum=0,value=0;
   for(int p=0;p<op.splits;p++){int task=head*op.splits+p;float scale=expf(op.stats[task*2]-maximum);
    sum+=op.stats[task*2+1]*scale;value+=op.partials[task*256+threadIdx.x]*scale;}
   op.output[head*256+threadIdx.x]=bf(value/sum);
  }
  __threadfence();kittens::group<16>::sync(6);if(threadIdx.x==0){signal_flag(g,index,head,1);finish_stage(g,index,1);}
 }};
};
