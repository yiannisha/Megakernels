// Gemma operators in the unmodified Megakernels controller/worker VM.
#include "kittens.cuh"
#include "megakernel.cuh"
#define GEMMA_CONSUMER_SYNC() kittens::group<8>::sync(4)
#include "ops.cuh"
#include "../demos/low-latency-llama/utils.cuh"
#include "../demos/low-latency-llama/matvec_pipeline.cuh"
#include <cuda/atomic>
#include <vector>
#include <memory>

#ifndef GEMMA_TIMING
#define GEMMA_TIMING 0
#endif
struct NativeConfig : megakernel::default_config {
 static constexpr bool TIMING_RECORD_ENABLED=GEMMA_TIMING;
 static constexpr int SCRATCH_BYTES=3072;
 static constexpr int DYNAMIC_SHARED_MEMORY=13*16384+1024;
};
using Cfg=NativeConfig;
using Weight=kittens::gl<kittens::bf16,1,1,-1,-1,kittens::st_bf<16,512>>;
using Act=kittens::gl<kittens::bf16,1,1,1,-1>;
struct Globals {
 static constexpr int hidden_dim=2048;
 megakernel::instruction_layout<Cfg> instructions;
 megakernel::timing_layout<Cfg> timings;
 State raw;
 Weight* weights;
 Act* activations;
 __device__ Stage stage(int i)const{return raw.stages[i];}
};
__device__ inline int read_counter(int* address) {
 return cuda::atomic_ref<int,cuda::thread_scope_device>(*address).load(cuda::memory_order_acquire);
}
__device__ inline void add_counter(int* address,int count) {
 cuda::atomic_ref<int,cuda::thread_scope_device>(*address).fetch_add(count,cuda::memory_order_release);
}
__device__ inline void wait_stage(const Globals& g,int index) {
 if(index<0)return;
 int expected=g.raw.total*g.stage(index).tasks;
 while(read_counter(g.raw.done+index)<expected)__nanosleep(20);
 __threadfence();
}
__device__ inline void wait_tiles(const Globals& g,int index,int start,int count) {
 int* flags=g.raw.done+g.raw.count+g.stage(index).begin;
 for(int i=start;i<start+count;i++)while(read_counter(flags+i)<g.raw.total)__nanosleep(20);
 __threadfence();
}
__device__ inline void producer_wait(const Globals& g,int index,int tile=-1) {
 if(index==0)return;
 Stage s=g.stage(index);
 if(tile>=0 && s.op==3){ // A query head needs only its own Q rows; K/V share one head.
  int width=g.stage(index-1).rows/g.stage(index-1).tasks;
  wait_tiles(g,index-1,tile*256/width,(tile==8?512:256)/width);
 }else if(tile>=0 && s.op==4){ // Q for this head plus the common K/V append.
  wait_tiles(g,index-1,tile,1);wait_tiles(g,index-1,8,1);
 }else if(tile>=0 && s.op==5){
  int rows_per_tile=g.stage(index-1).rows/g.stage(index-1).tasks;
  wait_tiles(g,index-1,tile*256/rows_per_tile,256/rows_per_tile);
 }else if(tile>=0 && s.op==6){
  int width=g.stage(index-1).rows/g.stage(index-1).tasks;
  wait_tiles(g,index-1,tile*256/width,256/width);wait_tiles(g,index-2,tile*256/width,256/width);
 }else if(s.op==2 && g.stage(index-1).op==2 && s.a==g.stage(index-1).a){
  // Gate and up projections independently consume the same normalized input.
  wait_stage(g,index-2);
 }else wait_stage(g,index-1);
}
__device__ inline void publish_tile(const Globals& g,int index,int tile) {
 if(g.stage(index).op==2 || g.stage(index).op==3)cuda::atomic_ref<int,cuda::thread_scope_device>(g.raw.done[g.raw.count+g.stage(index).begin+tile]).store(g.raw.total,cuda::memory_order_release);
}
struct GenericOp {
 static constexpr int opcode=1;
 using controller=megakernel::NoOp<Cfg>::controller;
 using loader=megakernel::NoOp<Cfg>::loader;
 using launcher=megakernel::NoOp<Cfg>::launcher;
 using storer=megakernel::NoOp<Cfg>::storer;
 struct consumer {
  static __device__ void run(const Globals& g,megakernel::state<Cfg>& s) {
   int index=s.instruction()[1],start=s.instruction()[2],count=s.instruction()[3];
   if(kittens::warpid()<8){
    for(int tile=start;tile<start+count;tile++){
     if(threadIdx.x==0)producer_wait(g,index,tile);
     GEMMA_CONSUMER_SYNC();
     execute(g.stage(index),tile,g.raw);
     __threadfence();GEMMA_CONSUMER_SYNC();
     if(threadIdx.x==0)publish_tile(g,index,tile);
     GEMMA_CONSUMER_SYNC();
    }
   }
   if(threadIdx.x==0)add_counter(g.raw.done+index,count);
   kittens::group<Cfg::NUM_CONSUMER_WARPS>::sync(5);
  }
 };
};
struct PipeOp {
 static constexpr int opcode=2;
 struct parsed_instruction {
  int index,start,iters,tiles,parts;
  __device__ parsed_instruction(Cfg::instruction_t& x):index(x[1]),start(x[2]),iters(x[3]*x[4]),tiles(x[3]),parts(x[4]){}
  __device__ parsed_instruction(megakernel::state<Cfg>& s):parsed_instruction(s.instruction()){}
 };
 struct specifics;
 using Pipeline=matvec_pipeline<Cfg,Globals,parsed_instruction,specifics>;
 struct specifics {
  static __device__ void load_iter(megakernel::state<Cfg>& s,const Globals& g,parsed_instruction& inst,int iter,int col,kittens::st_bf<16,512>& chunk,kittens::semaphore& sem){
   kittens::tma::load_async<kittens::dim::ROW,kittens::cache_policy::EVICT_FIRST>(chunk,g.weights[inst.index],kittens::coord<>{0,(inst.start+iter/inst.parts)*16,(iter%inst.parts)*2048+col*512},sem);
  }
  static __device__ void store(megakernel::state<Cfg>& s,const Globals& g,parsed_instruction& inst,int iter,int output_stage){
   float* scratch=reinterpret_cast<float*>(Pipeline::get_output_start(s,output_stage));
   Stage op=g.stage(inst.index);int lane=kittens::laneid();
   if(lane<16){float value=0;
    #pragma unroll
    for(int warp=0;warp<Cfg::NUM_CONSUMER_WARPS;warp++)value+=scratch[warp*16+lane];
    float* accum=reinterpret_cast<float*>(&s.semaphores()[Pipeline::SEM_COUNT]);
    if(iter%inst.parts==0)accum[lane]=value;else accum[lane]+=value;
    if(iter%inst.parts==inst.parts-1){
     int row=(inst.start+iter/inst.parts)*16+lane;
     if(op.op==8)op.logits[row]=accum[lane];else op.out[row]=bf(accum[lane]);
    }
   }
   __threadfence();__syncwarp();
   if(lane==0 && iter%inst.parts==inst.parts-1){publish_tile(g,inst.index,inst.start+iter/inst.parts);if(iter+1==inst.iters)add_counter(g.raw.done+inst.index,inst.tiles);}
  }
 };
 struct controller {
  static __device__ int release_lid(const Globals& g,Cfg::instruction_t& i,int& q){return Pipeline::release_lid(g,i,q);}
  static __device__ int init_semaphores(const Globals&,megakernel::state<Cfg>& s){return Pipeline::init_semaphores(s);}
 };
 struct loader {static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){Pipeline::loader_loop(s,g);}};
 using launcher=megakernel::NoOp<Cfg>::launcher;
 struct consumer {
  static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){
   parsed_instruction inst(s);
   if(threadIdx.x==0){s.wait_page_ready(Pipeline::get_activation_page(s));producer_wait(g,inst.index);}
   kittens::group<Cfg::NUM_CONSUMER_WARPS>::sync(3);
   using SV=kittens::sv_bf<Pipeline::REDUCTION_DIM_PER_WARP>;
   using RV=kittens::rv_fl<Pipeline::REDUCTION_DIM_PER_WARP>;
   SV& shared=reinterpret_cast<SV*>(&Pipeline::get_activations(s))[kittens::warpid()];
   Act activation=g.activations[inst.index];
   kittens::warp::load(shared,activation,kittens::coord<>{kittens::warpid()*Pipeline::REDUCTION_DIM_PER_WARP});
   kittens::warp::sync();RV values;kittens::warp::load(values,shared);kittens::warp::sync();
   s.warp_finish_page(Pipeline::get_activation_page(s),1);
   Pipeline::consumer_loop(s,g,values,[&] __device__ (int iter,RV& value){
    if(inst.parts>1){
     kittens::warp::load(value,activation,{(iter%inst.parts)*Cfg::NUM_CONSUMER_WARPS+kittens::warpid()});
    }
   });
  }
 };
 struct storer {static __device__ void run(const Globals& g,megakernel::state<Cfg>& s){Pipeline::storer_loop(s,g);}};
};
using NativeKernel=void;
struct Context {Globals g;Weight* weights;Act* activations;};
extern "C" void* create_native(State raw,void* instructions,int workers,int rows,void* timings){
 std::vector<Stage> stages(raw.count);cudaMemcpy(stages.data(),raw.stages,sizeof(Stage)*raw.count,cudaMemcpyDeviceToHost);
 std::vector<Weight> weights;weights.reserve(raw.count);
 bf* fallback=const_cast<bf*>(stages[0].w);
 for(auto s:stages)weights.emplace_back(reinterpret_cast<kittens::bf16*>(const_cast<bf*>(s.w?s.w:fallback)),nullptr,nullptr,(s.cols>=2048?s.rows:16),(s.cols>=2048?s.cols:2048));
 Weight* device_weights=nullptr;cudaMalloc(&device_weights,sizeof(Weight)*weights.size());cudaMemcpy(device_weights,weights.data(),sizeof(Weight)*weights.size(),cudaMemcpyHostToDevice);
 std::vector<Act> activations;for(auto s:stages)activations.emplace_back(reinterpret_cast<kittens::bf16*>(const_cast<bf*>(s.a?s.a:fallback)),nullptr,nullptr,nullptr,(s.cols>=2048?s.cols:2048));
 Act* device_activations=nullptr;cudaMalloc(&device_activations,sizeof(Act)*activations.size());cudaMemcpy(device_activations,activations.data(),sizeof(Act)*activations.size(),cudaMemcpyHostToDevice);
 Globals g{megakernel::instruction_layout<Cfg>((int*)instructions,nullptr,workers,rows,nullptr),megakernel::timing_layout<Cfg>((int*)timings,nullptr,workers,rows,nullptr),raw,device_weights,device_activations};
 return new Context{g,device_weights,device_activations};
}
extern "C" void destroy_native(void* handle){if(handle){auto* ctx=(Context*)handle;cudaFree(ctx->weights);cudaFree(ctx->activations);delete ctx;}}
extern "C" int launch(State raw,int blocks,void* stream){
 auto* ctx=(Context*)raw.next;ctx->g.raw=raw;
 auto kernel=megakernel::mk<Cfg,Globals,GenericOp,PipeOp>;
 cudaError_t e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);if(e!=cudaSuccess)return e;
 kernel<<<blocks,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,(cudaStream_t)stream>>>(ctx->g);
 return cudaGetLastError();
}
extern "C" int capacity(){int device;cudaDeviceProp p;cudaGetDevice(&device);cudaGetDeviceProperties(&p,device);return p.multiProcessorCount;}
