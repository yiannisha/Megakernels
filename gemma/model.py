"""Gemma-2B BF16 decode in one native Megakernels VM launch per token.

Prefill is performed by Transformers, with its actual KV cache transferred here.
The kernel uses the upstream VM, TMA matvec pipeline, and producer counters.
"""
import os
import ctypes as C
from pathlib import Path
import torch

class Stage(C.Structure):
    _fields_ = [(n,C.c_int) for n in ('op','rows','cols','tasks','begin')] + [(n,C.c_void_p) for n in ('a','w','b','out','cache','logits')]
class State(C.Structure):
    _fields_ = [('stages',C.c_void_p),('count',C.c_int),('total',C.c_int)] + [(n,C.c_void_p) for n in ('done','next','pos','token')]

def ptr(x): return x.data_ptr() if x is not None else 0

class GemmaDecode:
    def __init__(self, model, max_context=512):
        self.use_tma=os.environ.get("GEMMA_NATIVE_TMA", "1")=="1"
        c=model.config
        assert (c.model_type,c.hidden_size,c.intermediate_size,c.num_hidden_layers,c.num_attention_heads,c.num_key_value_heads,c.head_dim)==('gemma',2048,16384,18,8,1,256)
        assert c.hidden_act=='gelu' and c.rms_norm_eps==1e-6 and c.rope_theta==10000
        assert max_context<=1024
        self.model=model; self.max_context=max_context
        self.lib=C.CDLL(str(Path(__file__).with_name('kernel.so')))
        self.lib.launch.argtypes=[State,C.c_int,C.c_void_p];self.lib.launch.restype=C.c_int
        self.pos=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.token=torch.zeros_like(self.pos)
        self.h=torch.empty(2048,device='cuda',dtype=torch.bfloat16)
        self.norm=torch.empty_like(self.h);self.qkv=torch.empty(2560,device='cuda',dtype=torch.bfloat16)
        self.q=torch.empty_like(self.h);self.attn=torch.empty_like(self.h);self.proj=torch.empty_like(self.h)
        self.gate=torch.empty(16384,device='cuda',dtype=torch.bfloat16)
        self.up=torch.empty_like(self.gate);self.act=torch.empty_like(self.gate)
        self.kv=torch.zeros((18,2,max_context,256),device='cuda',dtype=torch.bfloat16)
        self.logits=torch.empty(c.vocab_size,device='cuda',dtype=torch.float32)
        self.weights=[];stages=[]
        def add(op,rows=2048,cols=0,a=None,w=None,b=None,out=None,cache=None,logits=None):
            tasks=1 if op in (1,9) else (rows+15)//16 if op in (2,8) and cols==2048 and self.use_tma else (rows+7)//8 if op in (2,8) else 9 if op==3 else 8 if op==4 else (rows+255)//256
            stages.append(Stage(op,rows,cols,tasks,0,ptr(a),ptr(w),ptr(b),ptr(out),ptr(cache),ptr(logits)))
        add(0,w=model.model.embed_tokens.weight,out=self.h)
        for l,layer in enumerate(model.model.layers):
            qkv=torch.cat([layer.self_attn.q_proj.weight,layer.self_attn.k_proj.weight,layer.self_attn.v_proj.weight]).contiguous()
            self.weights.append(qkv)
            add(1,a=self.h,w=layer.input_layernorm.weight,out=self.norm)
            add(2,2560,2048,a=self.norm,w=qkv,out=self.qkv)
            add(3,cols=max_context,a=self.qkv,out=self.q,cache=self.kv[l])
            add(4,cols=max_context,a=self.q,out=self.attn,cache=self.kv[l])
            add(2,2048,2048,a=self.attn,w=layer.self_attn.o_proj.weight,out=self.proj)
            add(5,a=self.h,b=self.proj,out=self.h)
            add(1,a=self.h,w=layer.post_attention_layernorm.weight,out=self.norm)
            add(2,16384,2048,a=self.norm,w=layer.mlp.gate_proj.weight,out=self.gate)
            add(2,16384,2048,a=self.norm,w=layer.mlp.up_proj.weight,out=self.up)
            add(6,16384,a=self.gate,b=self.up,out=self.act)
            add(2,2048,16384,a=self.act,w=layer.mlp.down_proj.weight,out=self.proj)
            add(5,a=self.h,b=self.proj,out=self.h)
        add(1,a=self.h,w=model.model.norm.weight,out=self.norm)
        add(8,c.vocab_size,2048,a=self.norm,w=model.lm_head.weight,logits=self.logits)
        add(9,c.vocab_size,logits=self.logits)
        total=0
        for s in stages:s.begin=total;total+=s.tasks
        packed=(Stage*len(stages))(*stages)
        self.stages=torch.frombuffer(bytearray(bytes(packed)),dtype=torch.uint8).cuda()
        self.blocks=self.lib.capacity()
        # The upstream instruction fetch indexes streams by physical SM ID.
        # One resident VM CTA per SM consumes its own packed instruction list.
        schedules=[[] for _ in range(self.blocks)]
        for index,stage in enumerate(stages):
            chunk=(stage.tasks+self.blocks-1)//self.blocks
            for worker,stream in enumerate(schedules):
                start=worker*chunk;count=min(chunk,stage.tasks-start)
                if count>0:
                    instruction=[0]*32
                    instruction[:4]=[2 if stage.op in (2,8) and stage.cols==2048 and self.use_tma else 1,index,start,count]
                    stream.append(instruction)
        length=max(map(len,schedules))
        for stream in schedules:stream.extend([[0]*32 for _ in range(length-len(stream))])
        self.instructions=torch.tensor(schedules,device='cuda',dtype=torch.int32)
        self.timings=torch.zeros((self.blocks,length,128),device='cuda',dtype=torch.int32)
        self.done=torch.zeros(len(stages)+total,device='cuda',dtype=torch.int32)
        self.state=State(ptr(self.stages),len(stages),0,ptr(self.done),0,ptr(self.pos),ptr(self.token))
        self.lib.create_native.argtypes=[State,C.c_void_p,C.c_int,C.c_int,C.c_void_p]
        self.lib.create_native.restype=C.c_void_p
        self.context=self.lib.create_native(self.state,ptr(self.instructions),self.blocks,length,ptr(self.timings))
        if not self.context:raise RuntimeError('Native VM context creation failed')
        self.state.next=self.context

    def load_cache(self, cache, token, position):
        assert 0 <= position < self.max_context
        self.position = position
        self.state.total=0
        self.done.zero_()
        for l in range(18):
            k,v=cache[l]
            self.kv[l,0,:position].copy_(k[0,0,:position])
            self.kv[l,1,:position].copy_(v[0,0,:position])
        self.pos.fill_(position);self.token.fill_(token)

    def step(self):
        if self.position >= self.max_context:
            raise ValueError('Gemma decode KV cache is full')
        self.position += 1
        self.state.total += 1
        err=self.lib.launch(self.state,self.blocks,torch.cuda.current_stream().cuda_stream)
        if err: raise RuntimeError(f'CUDA cooperative launch failed: {err}')
        return self.token
