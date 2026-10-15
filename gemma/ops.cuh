#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cmath>
using bf=__nv_bfloat16;
struct Stage { int op, rows, cols, tasks, begin; const bf *a,*w,*b; bf *out,*cache; float *logits; };
struct State { Stage* stages; int count,total; int *done,*next,*pos,*token; };
__device__ __forceinline__ float2 unpack(unsigned bits) {
 union { unsigned u; __nv_bfloat162 b; } v;v.u=bits;return __bfloat1622float2(v.b);
}
__device__ float warp_sum(float x) { for(int i=16;i;i/=2)x+=__shfl_down_sync(0xffffffff,x,i);return x; }
__device__ float block_sum(float x,float* sm) {
 x=warp_sum(x); if(threadIdx.x%32==0)sm[threadIdx.x/32]=x;__syncthreads();
 x=threadIdx.x<8?sm[threadIdx.x]:0; if(threadIdx.x<32)x=warp_sum(x);
 if(threadIdx.x==0)sm[0]=x;__syncthreads();return sm[0];
}
__device__ void execute(Stage s,int tile,State state) {
 __shared__ float sm[1024]; int t=threadIdx.x, lane=t%32,warp=t/32,p=*state.pos;
 if(s.op==0) { // scaled embedding
  for(int i=tile*256+t;i<min((tile+1)*256,2048);i+=256)
   s.out[i]=bf(float(s.w[(*state.token)*2048+i])*float(bf(sqrtf(2048.f))));
 } else if(s.op==1) { // Gemma offset RMSNorm
  float z=0;for(int i=t;i<2048;i+=256){float v=float(s.a[i]);z+=v*v;}
  float inv=rsqrtf(block_sum(z,sm)/2048.f+1e-6f);
  for(int i=t;i<2048;i+=256)s.out[i]=bf(float(s.a[i])*inv*(1.f+float(s.w[i])));
 } else if(s.op==2 || s.op==8) { // GEMV, one warp per row
  int row=tile*8+warp; if(row<s.rows){float acc=0;
   const uint4* w8=reinterpret_cast<const uint4*>(s.w+(long long)row*s.cols);
   const uint4* a8=reinterpret_cast<const uint4*>(s.a);
   float sums[4]={0,0,0,0};
   for(int c=lane;c<s.cols/8;c+=32){
    uint4 weights=w8[c], input=a8[c];
    float2 w=unpack(weights.x),x=unpack(input.x);sums[0]=fmaf(w.x,x.x,sums[0]);sums[0]=fmaf(w.y,x.y,sums[0]);
    w=unpack(weights.y);x=unpack(input.y);sums[1]=fmaf(w.x,x.x,sums[1]);sums[1]=fmaf(w.y,x.y,sums[1]);
    w=unpack(weights.z);x=unpack(input.z);sums[2]=fmaf(w.x,x.x,sums[2]);sums[2]=fmaf(w.y,x.y,sums[2]);
    w=unpack(weights.w);x=unpack(input.w);sums[3]=fmaf(w.x,x.x,sums[3]);sums[3]=fmaf(w.y,x.y,sums[3]);
   }
   acc=(sums[0]+sums[1])+(sums[2]+sums[3]);
   acc=warp_sum(acc);if(lane==0){if(s.op==8)s.logits[row]=acc;else s.out[row]=bf(acc);}
  }
 } else if(s.op==3) { // non-interleaved RoPE Q/K, append single KV head
  int head=tile; int d=t; float angle=p*powf(10000.f,-2.f*(d%128)/256.f);
  // HF rounds cos/sin to BF16 before multiplying, with BF16 products/addition.
  float co=float(bf(cosf(angle))),si=float(bf(sinf(angle)));
  float x=float(s.a[head*256+d]), y=float(s.a[head*256+(d+128)%256])*(d<128?-1.f:1.f);
  bf z=bf(float(bf(x*co))+float(bf(y*si)));
  if(head<8)s.out[head*256+d]=z;
  else {s.cache[p*256+d]=z;s.cache[s.cols*256+p*256+d]=s.a[2304+d];}
 } else if(s.op==4) { // MQA: one CTA per query head
  int h=tile;float localmax=-INFINITY;
  for(int k=warp;k<=p;k+=8){float dot=0;for(int d=lane;d<256;d+=32)dot+=float(s.a[h*256+d])*float(s.cache[k*256+d]);
   dot=warp_sum(dot);if(lane==0)sm[k]=dot*.0625f;}
  __syncthreads();
  for(int k=t;k<=p;k+=256)localmax=fmaxf(localmax,sm[k]);
  for(int i=16;i;i/=2)localmax=fmaxf(localmax,__shfl_down_sync(0xffffffff,localmax,i));
  __shared__ float red[8];if(lane==0)red[warp]=localmax;__syncthreads();
  float mx=-INFINITY;for(int i=0;i<8;i++)mx=fmaxf(mx,red[i]);
  float z=0;for(int k=t;k<=p;k+=256){sm[k]=expf(sm[k]-mx);z+=sm[k];}__syncthreads();
  __shared__ float tmp[8];float denom=block_sum(z,tmp);
  for(int k=t;k<=p;k+=256)sm[k]=float(bf(sm[k]/denom));__syncthreads();
  float out=0;for(int k=0;k<=p;k++)out+=sm[k]*float(s.cache[s.cols*256+k*256+t]);
  s.out[h*256+t]=bf(out);
 } else if(s.op==5) {for(int i=tile*256+t;i<min(s.rows,(tile+1)*256);i+=256)s.out[i]=bf(float(s.a[i])+float(s.b[i]));
 } else if(s.op==6) {for(int i=tile*256+t;i<min(s.rows,(tile+1)*256);i+=256){float x=float(s.a[i]);float gelu=.5f*x*(1.f+erff(x*.7071067811865475f));s.out[i]=bf(float(bf(gelu))*float(s.b[i]));}
 } else if(s.op==9) { // greedy sampler
  float v=-INFINITY;int idx=0;for(int i=t;i<s.rows;i+=256)if(s.logits[i]>v){v=s.logits[i];idx=i;}
  __shared__ float vals[256];__shared__ int ids[256];vals[t]=v;ids[t]=idx;__syncthreads();
  for(int stride=128;stride;stride/=2){if(t<stride && (vals[t+stride]>vals[t] || (vals[t+stride]==vals[t] && ids[t+stride]<ids[t]))){vals[t]=vals[t+stride];ids[t]=ids[t+stride];}__syncthreads();}
  if(t==0){*state.token=ids[0];*state.pos=p+1;}
 }
}
