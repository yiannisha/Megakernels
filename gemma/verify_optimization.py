"""Compare preserved baseline and optimized code on the same GPU and model."""
import importlib.util,json,time,statistics,subprocess,tempfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from model import GemmaDecode
root=Path(__file__).resolve().parent/'results'
baseline_dir=tempfile.TemporaryDirectory(prefix='gemma-baseline-')
baseline_path=Path(baseline_dir.name)
repo=Path(__file__).resolve().parents[1]
for name in ['kernel.cu','ops.cuh','model.py']:
    (baseline_path/name).write_bytes(subprocess.check_output(['git','show',f'ba2c15d:gemma/{name}'],cwd=repo))
subprocess.run(['/usr/local/cuda-12.8/bin/nvcc','-O3','-maxrregcount=64','-std=c++17','-arch=sm_100','-shared','-Xcompiler','-fPIC',str(baseline_path/'kernel.cu'),'-o',str(baseline_path/'kernel.so')],check=True)
spec=importlib.util.spec_from_file_location('gemma_baseline',baseline_path/'model.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_grad_enabled(False)
m=AutoModelForCausalLM.from_pretrained('google/gemma-2b',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
t=AutoTokenizer.from_pretrained('google/gemma-2b',local_files_only=True)
base=module.GemmaDecode(m,max_context=1024);opt=GemmaDecode(m,max_context=1024)
checks=[]
long=t(('The solar system contains eight planets. Jupiter is the largest planet. '*150),return_tensors='pt').input_ids.cuda()
for position in [128,511,1023]:
 z=long[:,:position];pre=m(z,use_cache=True);token=int(pre.logits[0,-1].argmax())
 for engine in [base,opt]:engine.load_cache(pre.past_key_values,token,position)
 reference=m(torch.tensor([[token]],device='cuda'),past_key_values=pre.past_key_values,use_cache=True)
 base.step();opt.step();torch.cuda.synchronize()
 row=dict(position=position,baseline_token=int(base.logits.argmax()),optimized_token=int(opt.logits.argmax()),reference_token=int(reference.logits[0,-1].argmax()),cosine=torch.nn.functional.cosine_similarity(opt.logits,reference.logits[0,-1].float(),dim=0).item())
 assert row['baseline_token']==row['optimized_token']==row['reference_token'] and row['cosine']>.999,row
 checks.append(row);print(row,flush=True)
z=t('The solar system contains eight planets. The largest planet is',return_tensors='pt').input_ids.cuda();pre=m(z,use_cache=True);token=int(pre.logits[0,-1].argmax())
runs=[]
for iteration in range(7):
 for name,engine in [('baseline',base),('optimized',opt)]:
  engine.load_cache(pre.past_key_values,token,z.shape[1]);output=[]
  torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(127):engine.step();output.append(engine.token.clone())
  torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
  row=dict(iteration=iteration,warmup=iteration<2,version=name,wall_ms=elapsed,ms_per_token=elapsed/127)
  runs.append(row);print(row,flush=True)
  ids=[int(x) for x in output]
  if name=='baseline':baseline_ids=ids
  else:assert ids==baseline_ids,'Optimized output differs from baseline'
summary={name:statistics.median(x['ms_per_token'] for x in runs if not x['warmup'] and x['version']==name) for name in ['baseline','optimized']}
summary['speedup']=summary['baseline']/summary['optimized']
(root/'same_session_comparison.json').write_text(json.dumps(dict(checks=checks,runs=runs,summary=summary),indent=2));print(summary,flush=True)
