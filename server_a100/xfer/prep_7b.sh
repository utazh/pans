#!/bin/bash
# 7B：权重从 /data0/wuping/models 只读复制并按 HF 官方 sha256 校验；小文件从 hf-mirror 重新下载并按 git blob sha1 校验。
# 之后继续用 hf-mirror 下载 14B/32B。只写 /home/shensicheng/qyd/pans。
set -u
B=/home/shensicheng/qyd/pans; M=$B/models/Qwen2.5-7B-Instruct; SRC=/data0/wuping/models/Qwen2.5-7B-Instruct
export TMPDIR=$B/tmp
rm -rf $M; mkdir -p $M
declare -A SHA=( [model-00001-of-00004.safetensors]=a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7
 [model-00002-of-00004.safetensors]=f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185
 [model-00003-of-00004.safetensors]=8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5
 [model-00004-of-00004.safetensors]=1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd )
declare -A GIT=( [config.json]=0178295f88afc3c7f279ed284f961f8c1be00654 [generation_config.json]=0eb3c536657dcd12626e09eca4b6198c0cbcde1e
 [merges.txt]=20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0 [model.safetensors.index.json]=14d037fdda5a1311dcc0275003a7a370d84114e6
 [tokenizer.json]=443909a61d429dff23010e5bddd28ff530edda00 [tokenizer_config.json]=07bfe0640cb5a0037f9322287fbfc682806cf672
 [vocab.json]=4783fe10ac3adce15ac8f358ef5462739852c569 [LICENSE]=6634c8cc3133b3848ec74b9f275acaaa1ea618ab )
ok=1
for f in "${!GIT[@]}"; do
  for t in 1 2 3; do curl -sfL -o $M/$f "https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/main/$f" && break; sleep 5; done
  g=$( (printf "blob %d\0" $(stat -c %s $M/$f); cat $M/$f) | sha1sum | cut -d" " -f1)
  [ "$g" = "${GIT[$f]}" ] && echo "OK  $f" || { echo "BAD $f $g"; ok=0; }
done
for f in "${!SHA[@]}"; do
  cp $SRC/$f $M/$f.part && mv $M/$f.part $M/$f
  h=$(sha256sum $M/$f | cut -d" " -f1)
  if [ "$h" = "${SHA[$f]}" ]; then echo "OK  $f (copied)"; else
    echo "BAD $f (copied) -> 从 hf-mirror 重新下载"; rm -f $M/$f
    curl -sfL -C - -o $M/$f "https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/main/$f"
    h=$(sha256sum $M/$f | cut -d" " -f1); [ "$h" = "${SHA[$f]}" ] && echo "OK  $f (downloaded)" || { echo "BAD $f final"; ok=0; }
  fi
done
[ $ok = 1 ] && echo MODEL7B_OK || echo MODEL7B_FAIL
du -sh $M
# 14B / 32B
sed -i 's/for m in Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct/for m in Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct/' $B/xfer/get_models.sh
setsid nohup bash $B/xfer/get_models.sh > $B/logs/get_models.log 2>&1 < /dev/null &
echo PREP7B_DONE
