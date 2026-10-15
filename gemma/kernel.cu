// Static per-SM instruction streams, with operation completion barriers.
#include "ops.cuh"
__global__ void gemma(State state) {
 auto grid=cooperative_groups::this_grid();
 for(int i=0;i<state.count;i++) {
  Stage s=state.stages[i];
  for(int tile=blockIdx.x;tile<s.tasks;tile+=gridDim.x){execute(s,tile,state);__syncthreads();}
  grid.sync();
 }
}
extern "C" int launch(State state,int blocks,void* stream) {
 void* args[]={&state};return cudaLaunchCooperativeKernel((void*)gemma,blocks,256,args,0,(cudaStream_t)stream);
}
extern "C" int capacity() {
 int active=0,device=0;cudaDeviceProp p;cudaGetDevice(&device);cudaGetDeviceProperties(&p,device);
 cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,gemma,256,0);return active*p.multiProcessorCount;
}
