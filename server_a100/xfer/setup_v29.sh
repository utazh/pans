#!/bin/bash
# 把 v29 流程接入 A100 仓库副本（本地分支 a100，不推送）。只写 /home/shensicheng/qyd/pans。
set -euo pipefail
B=/home/shensicheng/qyd/pans; S=$B/src; X=$B/xfer
cd $S
git rev-parse --verify a100 >/dev/null 2>&1 || git checkout -q -b a100
# 1) v29 脚本：git 里已有的保持 git 版本（并比对），其余加入
mkdir -p $B/tmp/v29s && tar xzf $X/v29_scripts.tgz -C $B/tmp/v29s
for f in $B/tmp/v29s/*; do b=$(basename $f)
  case $b in *.json) dst=configs/$b;; *) dst=scripts/$b;; esac
  if [ -f $dst ]; then cmp -s $f $dst && echo "same-in-git $b" || echo "DIFF-vs-git $b (keep git)"; else cp $f $dst; fi
done
# 2) 数据（JSON/JSONL 文本）与 reader 权重
tar xzf $X/pz_data.tgz -C $S
[ -f $X/pz_readers.tgz ] && tar xzf $X/pz_readers.tgz -C $S
# 3) 路径替换
mkdir -p $B/work/pans_kv $B/work/datasets
grep -rlE "/home/panzihang|/data1/" scripts src configs --include=*.py --include=*.sh --include=*.json 2>/dev/null | while read f; do
  sed -i -e "s#/home/panzihang/src/prism_layerkv#$S#g" \
         -e "s#/home/panzihang/venvs/vllm-stable/bin/python#$B/venv/bin/python#g" \
         -e "s#/data1/llm/Qwen/#$B/models/#g" \
         -e "s#/data1/pans/pans_kv#$B/work/pans_kv#g" \
         -e "s#/data1/pans/datasets#$B/work/datasets#g" \
         -e "s#/home/panzihang/src/contiguous_fuxian/data/paper_tasks#$B/work/data_raw#g" "$f"
done
# 4) 运行时目录链接到 A100 上重建的 7B 基础产物
mkdir -p runtime_data assets
ln -sfn $B/work/q7/prefix_store runtime_data/prefix_store
ln -sfn $B/work/q7/pcache_c16 runtime_data/pcache_c16
ln -sfn $B/work/q7/selector_index_k4_g32 assets/selector_index_k4_g32
# 5) 校验：数据成员哈希 + reader 安全加载 + checkpoint 哈希
$B/venv/bin/python - <<'EOF'
import json, hashlib, torch
from pathlib import Path
S = Path('/home/shensicheng/qyd/pans/src'); A = S / 'docs/layerkv_v4/audit'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dev = json.loads((A / 'v29_dev_membership.json').read_text())
for t, v in dev['tasks'].items():
    p = S / f'data/v29/p0/dev/{t}.jsonl'
    print('dev', t, 'OK' if sha(p) == v['file_sha256'] else 'MISMATCH')
bad = 0; n = 0
for p in sorted(S.glob('studies/**/readers/**/*.pt')):
    obj = torch.load(p, map_location='cpu', weights_only=True)   # 只允许张量/基本容器，拒绝任意对象
    n += 1
print('readers loaded safely (weights_only=True):', n)
fin = json.loads((A / 'v29_final_selection.json').read_text())['cells']
for cell, c in fin.items():
    if c['mode'] != 'shared': print('final', cell, c['config']); continue
    pts = sorted(Path(c['adapter_dir']).glob('*.pt'))
    ok = pts and sha(pts[0]) == c['checkpoint_sha256']
    print('final', cell, c['config'], 'OK' if ok else 'MISMATCH', pts[0].name if pts else 'missing')
EOF
git add -A scripts configs >/dev/null 2>&1 || true
git -c user.email=a100@local -c user.name=a100 commit -q -m "A100: add v29 scripts and local paths" || true
echo SETUP_V29_DONE
