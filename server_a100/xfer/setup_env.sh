#!/bin/bash
# Build the PANS venv entirely inside /home/shensicheng/qyd/pans (no system / conda / other-user changes).
set -e
B=/home/shensicheng/qyd/pans
export UV_CACHE_DIR=$B/cache/uv TMPDIR=$B/tmp UV_PYTHON_INSTALL_DIR=$B/tools/python
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_HTTP_TIMEOUT=300
mkdir -p $B/tools $B/cache/uv $B/tmp
cd $B/tools
[ -x uv/uv ] || { mkdir -p uv && tar xzf $B/xfer/uv.tgz -C uv --strip-components=1; }
[ -x py310/bin/python3 ] || { mkdir -p py310 && tar xzf $B/xfer/py310.tgz -C py310 --strip-components=1; }
UV=$B/tools/uv/uv
$UV --version; $B/tools/py310/bin/python3 -V
[ -d $B/venv ] || $UV venv -p $B/tools/py310/bin/python3 $B/venv
export VIRTUAL_ENV=$B/venv
echo "== resolve vllm 0.19.0 / torch 2.10.0 (dry run)"
$UV pip install --dry-run vllm==0.19.0 torch==2.10.0 2>&1 | grep -i -E "torch|vllm|nvidia-cuda-runtime|nvidia-cublas|flashinfer|error" | head -20
echo "== install"
$UV pip install vllm==0.19.0 torch==2.10.0 transformers==4.57.6 numpy==2.2.6 safetensors==0.8.0 psutil==7.2.2 \
    tqdm requests sentencepiece==0.2.2 tokenizers==0.22.2 scipy==1.15.3 pandas pyarrow datasets huggingface_hub pytest
$B/venv/bin/python - <<'EOF'
import torch, transformers
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), torch.cuda.device_count())
from vllm.vllm_flash_attn import flash_attn_varlen_func
print('vllm flash_attn import ok')
EOF
echo SETUP_DONE
