import json,time
from pathlib import Path
import torch,vllm
from vllm import LLM,SamplingParams
root=Path(__file__).resolve().parent / 'results'
source=json.loads((root/'mk.json').read_text())
llm=LLM(model='/workspace/.cache/huggingface/hub/models--google--gemma-2b/snapshots/9cf48e52b224239de00d483ec8eb84fb8d0f3a3a',dtype='bfloat16',max_model_len=512,max_num_seqs=1,gpu_memory_utilization=.3,enable_prefix_caching=False,disable_log_stats=False,seed=42)
params=SamplingParams(temperature=0,max_tokens=128,ignore_eos=True)
rows=[]
for i in range(7):
 torch.cuda.synchronize();t=time.perf_counter()
 out=llm.generate([{'prompt_token_ids':source['prompt_ids']}],params,use_tqdm=False)[0]
 torch.cuda.synchronize();wall=(time.perf_counter()-t)*1000
 metrics=out.metrics
 assert metrics is not None and metrics.last_token_ts>metrics.first_token_ts,metrics
 row=dict(iteration=i,warmup=i<2,output_tokens=len(out.outputs[0].token_ids),decode_tokens=127,decode_ms=(metrics.last_token_ts-metrics.first_token_ts)*1000,wall_ms=wall)
 rows.append(row);print(row,flush=True)
result=dict(model='google/gemma-2b',vllm_version=vllm.__version__,prompt_ids=source['prompt_ids'],runs=rows,output_ids=list(out.outputs[0].token_ids),text=out.outputs[0].text)
(root/'vllm.json').write_text(json.dumps(result,indent=2));print(result['text'],flush=True)
