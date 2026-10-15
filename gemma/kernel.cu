// Static per-SM instruction streams, with operation completion barriers.
#include "ops.cuh"
__global__ void gemma(State state, unsigned long long* cycles) {
 auto grid=cooperative_groups::this_grid();
 for(int i=0;i<state.count;i++) {
  unsigned long long start=clock64();
  Stage s=state.stages[i];
  for(int tile=blockIdx.x;tile<s.tasks;tile+=gridDim.x){execute(s,tile,state);__syncthreads();}
  grid.sync();
  if(cycles && blockIdx.x==0 && threadIdx.x==0)cycles[i]+=clock64()-start;
 }
}
extern "C" int launch(State state,int blocks,void* stream) {
 unsigned long long* cycles=nullptr;void* args[]={&state,&cycles};return cudaLaunchCooperativeKernel((void*)gemma,blocks,256,args,0,(cudaStream_t)stream);
}
extern "C" int capacity() {
 int active=0,device=0;cudaDeviceProp p;cudaGetDevice(&device);cudaGetDeviceProperties(&p,device);
 cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,gemma,256,0);return active*p.multiProcessorCount;
}

extern "C" int launch_profile(State state,int blocks,void* stream,void* data) {
 void* args[]={&state,&data};return cudaLaunchCooperativeKernel((void*)gemma,blocks,256,args,0,(cudaStream_t)stream);
}
