"""Record the upstream VM's instruction events in a separate instrumented build."""
import json,os,statistics,subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from model import GemmaDecode
base=Path(__file__).resolve().parent
subprocess.run(['make','-C',str(base),'profile_kernel.so'],check=True)
os.environ['GEMMA_LIBRARY']='profile_kernel.so'
torch.set_grad_enabled(False)
m=AutoModelForCausalLM.from_pretrained('google/gemma-2b',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
t=AutoTokenizer.from_pretrained('google/gemma-2b',local_files_only=True)
z=t('The solar system contains eight planets. The largest planet is',return_tensors='pt').input_ids.cuda();o=m(z,use_cache=True)
k=GemmaDecode(m);k.load_cache(o.past_key_values,int(o.logits[0,-1].argmax()),z.shape[1])
for _ in range(16):k.step()
torch.cuda.synchronize()
events=k.timings.cpu().tolist();instructions=k.instructions.cpu().tolist();spans={}
for stream,times in zip(instructions,events):
 for instruction,event in zip(stream,times):
  if instruction[0] and event[4]>event[0]:
   opcode=str(k.stage_specs[instruction[1]].op)
   spans.setdefault(opcode,[]).append(event[4]-event[0])
assert spans,'Native VM did not record any instruction events'
result=dict(backend='upstream-vm',workers=k.blocks,counters=k.verify_counters(),note='Controller start/end instruction lifetimes overlap across workers and stages; not additive operation times.',instruction_lifetime_cycles={op:dict(count=len(v),median=statistics.median(v),min=min(v),max=max(v)) for op,v in spans.items()})
(base/'results/native_profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
k.close()
