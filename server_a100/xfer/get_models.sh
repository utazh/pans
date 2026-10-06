#!/bin/bash
# Download Qwen2.5-{7B,14B,32B}-Instruct from hf-mirror into /home/shensicheng/qyd/pans/models (resumable).
B=/home/shensicheng/qyd/pans
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=$B/cache/hf TMPDIR=$B/tmp HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p $B/models $B/cache/hf
for m in Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct; do
  for try in 1 2 3 4 5; do
    $B/venv/bin/python - "$m" <<'EOF' && break
import sys
from huggingface_hub import snapshot_download
m = sys.argv[1]
p = snapshot_download(f'Qwen/{m}', local_dir=f'/home/shensicheng/qyd/pans/models/{m}', max_workers=8,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', 'merges.txt', 'vocab.json', 'tokenizer*', 'LICENSE'])
print('DONE', m, p, flush=True)
EOF
    echo "retry $m $try"; sleep 20
  done
done
du -sh $B/models/*
echo MODELS_DONE
