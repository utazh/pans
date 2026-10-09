#!/bin/bash
# G1/G2 评测调度：每个作业 = 一个 (worktree, tag, cell) 的 dev（开发集选档）→ test（完整测试集），I 结果复用 --reuse-from。
# 作业表：每行 "<worktree> <tag> <reuse_from> <cell> <env>"；GPU 从候选里挑空闲显存 >= 20 GB 的卡（不碰他人进程）。
B=/home/shensicheng/qyd/pans; JOBS=$1; CAND=(4 5 6)
free_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v c="${CAND[*]}" \
  'BEGIN{split(c,a," ");for(i in a)ok[a[i]]=1} ($1 in ok)&&$2>20000{print $1; exit}'; }
n=0
while read -r wt tag reuse cell envf; do
  [ -z "$wt" ] && continue
  until g=$(free_gpu) && [ -n "$g" ]; do sleep 60; done
  n=$((n+1)); lo=$((40 + 4 * (n % 6))); hi=$((lo + 3))
  ( source $B/$envf; ulimit -n 1048576; cd $B/$wt
    $PANS_PY scripts/v29m_pipeline.py dev --cells $cell --gpu $g --cores $lo-$hi --tag $tag --reuse-from $reuse &&
    $PANS_PY scripts/v29m_pipeline.py test --cells $cell --gpu $g --cores $lo-$hi --tag $tag --reuse-from $reuse
    echo "JOB_DONE $wt $tag $cell rc=$?" ) > $B/logs/mlr_${wt}_${tag}_${cell}.log 2>&1 &
  echo "launched $wt $tag $cell gpu $g $(date +%H:%M)"
  sleep 120                                  # 等模型加载占上显存，再判断下一张卡
done < $JOBS
wait; echo QUEUE_MLR_DONE
