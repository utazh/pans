#!/bin/bash
# v31 many-class screen: prepare bundles -> prefix-length check -> build store/pcache/index -> dev screen
cd /home/panzihang/src/prism_layerkv; P=/home/panzihang/venvs/vllm-stable/bin/python; L=studies/v31/logs; mkdir -p $L
python3 scripts/v31_mc_prepare.py > $L/prepare.log 2>&1 || { echo PREPARE_FAILED >> $L/chainMC.log; exit 1; }
$P -c "
import json
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/data1/llm/Qwen/Qwen2.5-7B-Instruct')
out={}
for s in ['sst2','subj','trec']:
    r=json.loads(open(f'data/v31_mc/dev/{s}.jsonl').readline())
    out[s]=dict(prefix_tokens=len(tok(r['prefix_text'],add_special_tokens=False).input_ids), labels=len(r['labels']))
print(json.dumps(out)); assert all(v['prefix_tokens']<=16000 for v in out.values()), 'prefix too long'
" > $L/prefix_tokens.log 2>&1 || { echo TOKENS_FAILED >> $L/chainMC.log; exit 1; }
$P scripts/v31_mc_screen.py build 1 > $L/build.log 2>&1 || { echo BUILD_FAILED >> $L/chainMC.log; exit 1; }
$P scripts/v31_mc_screen.py screen 1,2,3 > $L/screen.log 2>&1 || { echo SCREEN_FAILED >> $L/chainMC.log; exit 1; }
echo CHAIN_MC_DONE >> $L/chainMC.log
