"""Run Gemma entirely through native prefill and fused Megakernels decode."""
import argparse,time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from model import GemmaDecode
parser=argparse.ArgumentParser()
parser.add_argument('--prompt',default='The solar system contains eight planets. The largest planet is')
parser.add_argument('--tokens',type=int,default=64)
parser.add_argument('--ignore-eos',action='store_true')
args=parser.parse_args()
if args.tokens<1:parser.error('--tokens must be positive')
torch.set_grad_enabled(False)
model=AutoModelForCausalLM.from_pretrained('google/gemma-2b',local_files_only=True,torch_dtype=torch.bfloat16).cuda().eval()
tokenizer=AutoTokenizer.from_pretrained('google/gemma-2b',local_files_only=True)
ids=tokenizer(args.prompt,return_tensors='pt').input_ids.cuda()
engine=GemmaDecode(model)
start=time.perf_counter();first=engine.prefill(ids);torch.cuda.synchronize();prefill_ms=(time.perf_counter()-start)*1000
output=[int(first)]
for _ in range(args.tokens-1):
 if not args.ignore_eos and output[-1]==model.config.eos_token_id:break
 output.append(int(engine.step()))
print(args.prompt+tokenizer.decode(output,skip_special_tokens=True))
print(f'Native prefill: {ids.numel()} tokens, {prefill_ms:.2f} ms; generated {len(output)} tokens')
print('Counter verification:',engine.verify_counters())
engine.close()
