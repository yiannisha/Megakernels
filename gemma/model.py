"""Original Gemma-2B: fused decode operators in the upstream Megakernels VM."""
import ctypes as C
import os
from pathlib import Path
import torch

EMBED,QKV,ATTENTION,COMBINE,OUTPUT,MLP,DOWN,HEAD,SAMPLE=range(1,10)
class Stage(C.Structure):
    _fields_=[(n,C.c_int) for n in ('op','rows','cols','tasks','begin','producer','splits','flags','context')]+[(n,C.c_void_p) for n in ('input','weight','norm','residual','output','kv','partials','stats','logits')]
class State(C.Structure):
    _fields_=[('stages',C.c_void_p),('count',C.c_int),('epoch',C.c_int)]+[(n,C.c_void_p) for n in ('done','context','pos','token','cos','sin')]
def ptr(x):return x.data_ptr() if x is not None else 0

class GemmaDecode:
    def __init__(self,model,max_context=8192):
        c=model.config
        if (c.model_type,c.hidden_size,c.intermediate_size,c.num_hidden_layers,c.num_attention_heads,c.num_key_value_heads,c.head_dim)!=('gemma',2048,16384,18,8,1,256):
            raise ValueError('This implementation targets original Gemma-2B')
        if c.hidden_act!='gelu' or c.rms_norm_eps!=1e-6 or c.rope_theta!=10000:raise ValueError('Unsupported Gemma numerics')
        if not (64<=max_context<=8192 and max_context%64==0):raise ValueError('KV capacity must be a multiple of 64, between 64 and 8192')
        if model.dtype!=torch.bfloat16 or not next(model.parameters()).is_cuda:raise ValueError('Weights must be CUDA BF16')
        self.model=model;self.max_context=max_context
        self.lib=C.CDLL(str(Path(__file__).with_name(os.environ.get('GEMMA_LIBRARY','kernel.so'))))
        self.lib.launch.argtypes=[State,C.c_int,C.c_void_p];self.lib.launch.restype=C.c_int
        self.lib.capacity.restype=C.c_int;self.blocks=self.lib.capacity()
        self.pos=torch.zeros(1,device='cuda',dtype=torch.int32);self.token=torch.zeros_like(self.pos)
        self.h=torch.empty(2048,device='cuda',dtype=torch.bfloat16)
        self.q=torch.empty_like(self.h);self.attn=torch.empty_like(self.h)
        self.act=torch.empty(16384,device='cuda',dtype=torch.bfloat16)
        self.kv=torch.zeros((18,2,max_context,256),device='cuda',dtype=torch.bfloat16)
        self.splits=4 if max_context<=512 else 8 if max_context<=2048 else 16
        self.partials=torch.empty((8*self.splits,256),device='cuda',dtype=torch.float32)
        self.stats=torch.empty((8*self.splits,2),device='cuda',dtype=torch.float32)
        self.logits=torch.empty(256000,device='cuda',dtype=torch.float32)
        inv=1.0/(10000.0**(torch.arange(0,256,2,device='cuda',dtype=torch.float32)/256))
        angles=torch.arange(max_context,device='cuda',dtype=torch.float32)[:,None]*inv[None,:]
        self.cos=angles.cos().bfloat16().contiguous();self.sin=angles.sin().bfloat16().contiguous()
        self.weights=[];stages=[]
        def add(op,rows,cols,tasks,producer=-1,flags=0,**tensors):
            args={name:ptr(tensors.get(name)) for name in ('input','weight','norm','residual','output','kv','partials','stats','logits')}
            stages.append(Stage(op,rows,cols,tasks,0,producer,self.splits,flags,max_context,**args));return len(stages)-1
        previous=add(EMBED,2048,2048,1,weight=model.model.embed_tokens.weight,output=self.h)
        for layer_index,layer in enumerate(model.model.layers):
            qkv=torch.cat([layer.self_attn.q_proj.weight,layer.self_attn.k_proj.weight,layer.self_attn.v_proj.weight])
            # Each 16-row tile holds eight non-interleaved RoPE dimension pairs.
            qkv=qkv.reshape(10,2,16,8,2048).permute(0,2,1,3,4).contiguous().reshape(2560,2048)
            gated=torch.stack([layer.mlp.gate_proj.weight.reshape(1024,16,2048),layer.mlp.up_proj.weight.reshape(1024,16,2048)],dim=1).reshape(32768,2048).contiguous()
            self.weights.extend([qkv,gated])
            q=add(QKV,2560,2048,160,previous,10,input=self.h,weight=qkv,norm=layer.input_layernorm.weight,output=self.q,kv=self.kv[layer_index])
            partial=add(ATTENTION,8*self.splits,256,8*self.splits,q,8*self.splits,input=self.q,kv=self.kv[layer_index],partials=self.partials,stats=self.stats)
            combined=add(COMBINE,2048,256,8,partial,8,output=self.attn,partials=self.partials,stats=self.stats)
            projected=add(OUTPUT,2048,2048,128,combined,input=self.attn,weight=layer.self_attn.o_proj.weight,residual=self.h,output=self.h)
            mlp=add(MLP,16384,2048,1024,projected,8,input=self.h,weight=gated,norm=layer.post_attention_layernorm.weight,output=self.act)
            previous=add(DOWN,2048,16384,128,mlp,input=self.act,weight=layer.mlp.down_proj.weight,residual=self.h,output=self.h)
        head=add(HEAD,256000,2048,16000,previous,input=self.h,weight=model.lm_head.weight,norm=model.model.norm.weight,logits=self.logits)
        add(SAMPLE,256000,0,1,head,logits=self.logits)
        total=0
        for stage in stages:stage.begin=total;total+=stage.flags
        self.stage_specs=stages
        self.stages=torch.frombuffer(bytearray(bytes((Stage*len(stages))(*stages))),dtype=torch.uint8).cuda()
        self.done=torch.zeros(len(stages)+total,device='cuda',dtype=torch.int32)
        # Cost-balanced per-SM instruction streams; all dependencies are device counters.
        streams=[[] for _ in range(self.blocks)];cost=[0.0]*self.blocks
        for index,stage in enumerate(stages):
            if stage.op in (ATTENTION,COMBINE,EMBED,SAMPLE):chunks=[(i,1) for i in range(stage.tasks)]
            else:
                chunk,remainder=divmod(stage.tasks,self.blocks)
                chunks=[(w*chunk+min(w,remainder),chunk+(w<remainder)) for w in range(self.blocks) if chunk+(w<remainder)>0]
            available=sorted(range(self.blocks),key=lambda w:(cost[w],w))
            for (start,count),worker in zip(chunks,available):
                parts=2 if stage.op==MLP else 8 if stage.op==DOWN else 1
                instruction=[0]*32;instruction[:5]=[stage.op,index,start,count,parts]
                streams[worker].append(instruction)
                estimate=stage.cols*16*2*count*parts if stage.op in (QKV,OUTPUT,MLP,DOWN,HEAD) else max_context*256*4/self.splits if stage.op==ATTENTION else 4096
                cost[worker]+=estimate
        length=max(map(len,streams))
        for stream in streams:stream.extend([[0]*32 for _ in range(length-len(stream))])
        self.instructions=torch.tensor(streams,device='cuda',dtype=torch.int32)
        self.timings=torch.zeros((self.blocks,length,128),device='cuda',dtype=torch.int32)
        self.state=State(ptr(self.stages),len(stages),0,ptr(self.done),0,ptr(self.pos),ptr(self.token),ptr(self.cos),ptr(self.sin))
        self.lib.create_native.argtypes=[State,C.c_void_p,C.c_int,C.c_int,C.c_void_p];self.lib.create_native.restype=C.c_void_p
        self.context=self.lib.create_native(self.state,ptr(self.instructions),self.blocks,length,ptr(self.timings))
        if not self.context:raise RuntimeError('Could not create native Gemma context')
        self.state.context=self.context
        self.lib.destroy_native.argtypes=[C.c_void_p];self.lib.destroy_native.restype=None

    def load_cache(self,cache,token,position):
        if not 0<=position<self.max_context:raise ValueError('Invalid KV position')
        self.position=position;self.state.epoch=0;self.done.zero_()
        for layer in range(18):
            k,v=cache[layer];self.kv[layer,0,:position].copy_(k[0,0,:position]);self.kv[layer,1,:position].copy_(v[0,0,:position])
        self.pos.fill_(position);self.token.fill_(token)

    def verify_counters(self):
        actual=self.done.cpu().tolist();epoch=self.state.epoch;count=0
        for i,stage in enumerate(self.stage_specs):
            assert actual[i]==epoch*stage.tasks,(i,actual[i],epoch*stage.tasks)
            expected=16 if stage.op==QKV else 128 if stage.op==MLP else 1
            flags=actual[len(self.stage_specs)+stage.begin:len(self.stage_specs)+stage.begin+stage.flags]
            assert all(x==epoch*expected for x in flags),(i,flags,epoch*expected)
            count+=len(flags)
        return dict(generation=epoch,stages=len(self.stage_specs),producer_counters=count)

    def close(self):
        if getattr(self,'context',None):torch.cuda.synchronize();self.lib.destroy_native(self.context);self.context=None
    def __del__(self):
        try:self.close()
        except Exception:pass
    def step(self):
        if not self.context:raise RuntimeError('Native Gemma is closed')
        if self.position>=self.max_context:raise ValueError('KV cache is full')
        self.position+=1;self.state.epoch+=1
        err=self.lib.launch(self.state,self.blocks,torch.cuda.current_stream().cuda_stream)
        if err:raise RuntimeError(f'Native Gemma launch failed: {err}')
        return self.token
