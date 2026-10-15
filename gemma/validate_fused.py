"""Independent full-model, native-prefill, context-boundary and reset checks."""
import json
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from model import GemmaDecode
root=Path(__file__).resolve().parent/'results'
torch.set_grad_enabled(False)
model=AutoModelForCausalLM.from_pretrained('google/gemma-2b',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
tok=AutoTokenizer.from_pretrained('google/gemma-2b',local_files_only=True)
engine=GemmaDecode(model,max_context=8192)
records=[]
# Native prefill is exercised with framework forward deliberately forbidden.
for text in ['The solar system contains eight planets. The largest planet is','The capital of France is','def fibonacci(n):']:
 ids=tok(text,return_tensors='pt').input_ids.cuda();gold=model(ids,use_cache=True)
 with patch.object(model,'forward',side_effect=AssertionError('Framework forward called during native prefill')):
  first=int(engine.prefill(ids));engine.step();torch.cuda.synchronize()
 assert first==int(gold.logits[0,-1].argmax()),(text,first,int(gold.logits[0,-1].argmax()))
 reference=model(torch.tensor([[first]],device='cuda'),past_key_values=gold.past_key_values,use_cache=True)
 cosine=torch.nn.functional.cosine_similarity(engine.logits,reference.logits[0,-1].float(),dim=0).item()
 row=dict(test='native_prefill',prompt=text,prompt_length=ids.shape[1],first_token=first,next_token=int(engine.logits.argmax()),reference_token=int(reference.logits[0,-1].argmax()),cosine=cosine,counters=engine.verify_counters())
 assert row['next_token']==row['reference_token'] and cosine>.999,row
 records.append(row);print(row,flush=True)
# Long real-model caches exercise split transitions and the last legal KV slot.
long=tok('The solar system contains eight planets. Jupiter is the largest planet. '*1200,return_tensors='pt').input_ids.cuda()
for position in [31,32,127,128,511,1023,2047,4095,8191]:
 pre=model(long[:,:position],use_cache=True);token=int(pre.logits[0,-1].argmax())
 engine.load_cache(pre.past_key_values,token,position)
 reference=model(torch.tensor([[token]],device='cuda'),past_key_values=pre.past_key_values,use_cache=True)
 engine.step();torch.cuda.synchronize()
 cosine=torch.nn.functional.cosine_similarity(engine.logits,reference.logits[0,-1].float(),dim=0).item()
 row=dict(test='context_boundary',position=position,token=int(engine.logits.argmax()),reference_token=int(reference.logits[0,-1].argmax()),cosine=cosine,counters=engine.verify_counters())
 assert row['token']==row['reference_token'] and cosine>.999,row
 records.append(row);print(row,flush=True)
 if position==8191:
  try:engine.step()
  except ValueError:pass
  else:raise AssertionError('KV overflow was not rejected')
(root/'fused_validation.json').write_text(json.dumps(records,indent=2));engine.close()
print('All fused validation checks passed',flush=True)
