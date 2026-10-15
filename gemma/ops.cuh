#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
using bf=__nv_bfloat16;
// Logical operators are also native instruction opcodes (zero is VM NoOp).
enum GemmaOp { Embed=1,QkvRope=2,AttentionPartial=3,AttentionCombine=4,
               OutputResidual=5,GatedGelu=6,DownResidual=7,NormHead=8,Sample=9 };
struct Stage {
 int op,rows,cols,tasks,begin,producer,splits,flags,context;
 const bf *input,*weight,*norm,*residual;
 bf *output,*kv;
 float *partials,*stats,*logits;
};
struct State {
 Stage* stages;int count,epoch;
 int *done;void* context;
 int *pos,*token;
 const bf *cos,*sin;
};
