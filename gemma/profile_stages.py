import ctypes as C,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from model import GemmaDecode,State,Stage
root=Path(__file__).resolve().parent/'results'
torch.set_grad_enabled(False)
m=AutoModelForCausalLM.from_pretrained('google/gemma-2b',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
t=AutoTokenizer.from_pretrained('google/gemma-2b',local_files_only=True)
z=t('The solar system contains eight planets. The largest planet is',return_tensors='pt').input_ids.cuda();o=m(z,use_cache=True)
k=GemmaDecode(m);k.load_cache(o.past_key_values,int(o.logits[0,-1].argmax()),z.shape[1])
c=torch.zeros(k.state.count,device='cuda',dtype=torch.int64)
k.lib.launch_profile.argtypes=[State,C.c_int,C.c_void_p,C.c_void_p];k.lib.launch_profile.restype=C.c_int
for _ in range(16):
 assert k.lib.launch_profile(k.state,k.blocks,torch.cuda.current_stream().cuda_stream,c.data_ptr())==0
torch.cuda.synchronize()
raw=k.stages.cpu().numpy().tobytes();stages=(Stage*k.state.count).from_buffer_copy(raw)
byop={}
for s,v in zip(stages,c.tolist()):byop[str(s.op)]=byop.get(str(s.op),0)+v
out=dict(blocks=k.blocks,cycles_by_opcode=byop,percent_by_opcode={op:100*v/sum(byop.values()) for op,v in byop.items()})
(root/'profile.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
