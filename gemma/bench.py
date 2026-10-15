import json,sys,time,statistics,hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import GemmaDecode
ROOT=Path(__file__).resolve().parent / 'results'
torch.set_grad_enabled(False)
model=AutoModelForCausalLM.from_pretrained('google/gemma-2b',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
tok=AutoTokenizer.from_pretrained('google/gemma-2b',local_files_only=True)
prompt='The solar system contains eight planets. The largest planet is'
ids=tok(prompt,return_tensors='pt').input_ids.cuda()
pre=model(ids,use_cache=True)
first=int(pre.logits[0,-1].argmax());pos=ids.shape[1]
mk=GemmaDecode(model,max_context=512)
mk.load_cache(pre.past_key_values,first,pos)
ref=model(torch.tensor([[first]],device='cuda'),past_key_values=pre.past_key_values,use_cache=True)
mk.step();torch.cuda.synchronize()
a=mk.logits.float();b=ref.logits[0,-1].float()
err=(a-b).abs();cos=torch.nn.functional.cosine_similarity(a,b,dim=0).item()
validation=dict(max_abs_error=err.max().item(),mean_abs_error=err.mean().item(),cosine=cos,mk_token=int(a.argmax()),reference_token=int(b.argmax()))
print('VALIDATION',validation,flush=True)
(ROOT/'validation.json').write_text(json.dumps(validation,indent=2))
assert cos>.999 and int(a.argmax())==int(b.argmax()),validation
# Teacher-forced checks at several positions and prompts.
checks=[]
for text in ('The capital of France is', 'def fibonacci(n):', 'Explain why the sky is blue.'):
    z=tok(text,return_tensors='pt').input_ids.cuda()
    base=model(z,use_cache=True)
    for j in range(4):
        token=int(base.logits[0,-1].argmax());position=z.shape[1]+j
        mk.load_cache(base.past_key_values,token,position)
        target=model(torch.tensor([[token]],device='cuda'),past_key_values=base.past_key_values,use_cache=True)
        mk.step();torch.cuda.synchronize()
        cosine=torch.nn.functional.cosine_similarity(mk.logits.float(),target.logits[0,-1].float(),dim=0).item()
        row=dict(prompt=text,position=position,cosine=cosine,mk_token=int(mk.logits.argmax()),reference_token=int(target.logits[0,-1].argmax()))
        checks.append(row);assert cosine>.999 and row['mk_token']==row['reference_token'],row
        base=target
(ROOT/'correctness.json').write_text(json.dumps(checks,indent=2))
print('CORRECTNESS',checks,flush=True)
# Tune resident workers without running concurrently with vLLM.
pre=model(ids,use_cache=True)
capacity=mk.lib.capacity();sm=torch.cuda.get_device_properties(0).multi_processor_count
tuning=[]
for blocks in [sm,sm*2,sm*4,sm*8]:
    if blocks>capacity: continue
    mk.blocks=blocks;mk.load_cache(pre.past_key_values,first,pos)
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(8):mk.step()
    end.record();torch.cuda.synchronize()
    tuning.append(dict(blocks=blocks,gpu_ms_per_token=start.elapsed_time(end)/8))
mk.blocks=min(tuning,key=lambda x:x['gpu_ms_per_token'])['blocks']
(ROOT/'tuning.json').write_text(json.dumps(tuning,indent=2))
print('TUNING',tuning,'selected',mk.blocks,flush=True)
# Recreate immutable prompt cache; validation above mutated the HF cache.
pre=model(ids,use_cache=True)
rows=[];output=[]
for i in range(7):
 mk.load_cache(pre.past_key_values,first,pos)
 torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 output=[];t0=time.perf_counter();start.record()
 for j in range(127):
  mk.step();output.append(mk.token.clone())
 end.record();torch.cuda.synchronize();ms=(time.perf_counter()-t0)*1000
 row=dict(iteration=i,warmup=i<2,decode_tokens=127,wall_ms=ms,gpu_ms=start.elapsed_time(end))
 rows.append(row);print(row,flush=True)
output=[first]+[int(x) for x in output]
counters=mk.verify_counters()
print('COUNTERS',counters,flush=True)
source_hashes={name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest() for name in ['native.cu','ops.cuh','fused.cuh','attention.cuh','model.py','../demos/low-latency-llama/matvec_pipeline.cuh']}
result=dict(backend='fused-megakernel-native',counter_verification=counters,source_hashes=source_hashes,model='google/gemma-2b',prompt=prompt,prompt_ids=ids[0].tolist(),output_ids=output,text=tok.decode(output),runs=rows,validation=validation,blocks=mk.blocks)
(ROOT/'mk.json').write_text(json.dumps(result,indent=2));print(result['text'],flush=True)
