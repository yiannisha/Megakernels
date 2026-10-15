#include "kittens.cuh"
#include "megakernel.cuh"
#include "ops.cuh"
#include "../demos/low-latency-llama/utils.cuh"
#include "../demos/low-latency-llama/matvec_pipeline.cuh"
#include <cuda/atomic>
#include <vector>
#ifndef GEMMA_TIMING
#define GEMMA_TIMING 0
#endif
struct NativeConfig:megakernel::default_config {
 static constexpr bool TIMING_RECORD_ENABLED=GEMMA_TIMING;
 static constexpr int SCRATCH_BYTES=3072;
 static constexpr int DYNAMIC_SHARED_MEMORY=13*16384+1024;
};
using Cfg=NativeConfig;
using Weight=kittens::gl<kittens::bf16,1,1,-1,-1,kittens::st_bf<16,512>>;
using Act=kittens::gl<kittens::bf16,1,1,1,-1>;
using Norm=kittens::gl<kittens::bf16,1,1,1,2048,kittens::sv_bf<2048>>;
using Cache=kittens::gl<kittens::bf16,1,1,-1,256,kittens::st_bf<32,256>>;
struct Globals {
 static constexpr int hidden_dim=2048;
 megakernel::instruction_layout<Cfg> instructions;megakernel::timing_layout<Cfg> timings;
 State raw;Weight* weights;Act* activations;Norm* norms;Cache* keys;Cache* values;
 __device__ Stage stage(int i)const{return raw.stages[i];}
};
__device__ inline int acquire(int* p){return cuda::atomic_ref<int,cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);}
__device__ inline void release_add(int* p,int n){cuda::atomic_ref<int,cuda::thread_scope_device>(*p).fetch_add(n,cuda::memory_order_release);}
__device__ inline void wait_stage(const Globals& g,int index){
 if(index<0)return;int expected=g.raw.epoch*g.stage(index).tasks;
 while(acquire(g.raw.done+index)<expected)__nanosleep(20);
}
__device__ inline void wait_flag(const Globals& g,int stage,int flag,int arrivals){
 int* p=g.raw.done+g.raw.count+g.stage(stage).begin+flag;
 while(acquire(p)<g.raw.epoch*arrivals)__nanosleep(20);
}
__device__ inline void signal_flag(const Globals& g,int stage,int flag,int arrivals){release_add(g.raw.done+g.raw.count+g.stage(stage).begin+flag,arrivals);}
__device__ inline void finish_stage(const Globals& g,int stage,int n){release_add(g.raw.done+stage,n);}
#include "fused.cuh"
#include "attention.cuh"
struct Context {Globals g;};
template<class T> T* upload(const std::vector<T>& data){T* out=nullptr;cudaMalloc(&out,sizeof(T)*data.size());cudaMemcpy(out,data.data(),sizeof(T)*data.size(),cudaMemcpyHostToDevice);return out;}
extern "C" void* create_native(State raw,void* instructions,int workers,int rows,void* timings){
 std::vector<Stage> stages(raw.count);cudaMemcpy(stages.data(),raw.stages,sizeof(Stage)*raw.count,cudaMemcpyDeviceToHost);
 std::vector<Weight> weights;std::vector<Act> activations;std::vector<Norm> norms;std::vector<Cache> keys,values;
 bf* fallback=const_cast<bf*>(stages[0].weight);
 bf* cache=nullptr;int context=0;for(auto s:stages)if(s.kv){cache=s.kv;context=s.context;break;}
 for(auto s:stages){
  weights.emplace_back(reinterpret_cast<kittens::bf16*>(const_cast<bf*>(s.weight?s.weight:fallback)),nullptr,nullptr,s.op==GatedGelu?s.rows*2:(s.cols>=2048?s.rows:16),s.cols>=2048?s.cols:2048);
  activations.emplace_back(reinterpret_cast<kittens::bf16*>(const_cast<bf*>(s.input?s.input:fallback)),nullptr,nullptr,nullptr,s.op==AttentionPartial?2048:(s.cols>=2048?s.cols:2048));
  norms.emplace_back(reinterpret_cast<kittens::bf16*>(const_cast<bf*>(s.norm?s.norm:fallback)),nullptr,nullptr,nullptr,nullptr);
  keys.emplace_back(reinterpret_cast<kittens::bf16*>(s.kv?s.kv:cache),nullptr,nullptr,s.kv?s.context:context,nullptr);
  values.emplace_back(reinterpret_cast<kittens::bf16*>((s.kv?s.kv:cache)+(s.kv?s.context:context)*256),nullptr,nullptr,s.kv?s.context:context,nullptr);
 }
 Globals g{megakernel::instruction_layout<Cfg>((int*)instructions,nullptr,workers,rows,nullptr),megakernel::timing_layout<Cfg>((int*)timings,nullptr,workers,rows,nullptr),raw,upload(weights),upload(activations),upload(norms),upload(keys),upload(values)};
 return new Context{g};
}
extern "C" void destroy_native(void* handle){if(handle){auto* ctx=(Context*)handle;cudaFree(ctx->g.weights);cudaFree(ctx->g.activations);cudaFree(ctx->g.norms);cudaFree(ctx->g.keys);cudaFree(ctx->g.values);delete ctx;}}
extern "C" int launch(State raw,int blocks,void* stream){
 auto* ctx=(Context*)raw.context;ctx->g.raw=raw;
 auto kernel=megakernel::mk<Cfg,Globals,EmbedOp,FusedLinear<QkvRope>,AttentionOp,CombineOp,FusedLinear<OutputResidual>,FusedLinear<GatedGelu>,FusedLinear<DownResidual>,FusedLinear<NormHead>,SampleOp>;
 cudaError_t err=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);if(err!=cudaSuccess)return err;
 kernel<<<blocks,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,(cudaStream_t)stream>>>(ctx->g);return cudaGetLastError();
}
extern "C" int capacity(){int device;cudaDeviceProp p;cudaGetDevice(&device);cudaGetDeviceProperties(&p,device);return p.multiProcessorCount;}
