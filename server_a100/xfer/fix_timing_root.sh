#!/bin/bash
B=/home/shensicheng/qyd/pans
# 1) 停掉排队脚本与失败中的 14B 计时（只匹配我们自己的这两个脚本）
ps -u shensicheng -o pid,args | awk '/queue_timing_next.sh/ && !/awk/ {print $1}' | xargs -r kill
sleep 1
ps -u shensicheng -o pid,args | awk '/scripts\/v29m_timing.py a100_timing_q14/ && !/awk/ {print $1}' | xargs -r kill
sleep 3
ps -u shensicheng -o pid,args | awk '/run_evidence_v26.py/ && /a100_timing_q14/ && !/awk/ {print $1}' | xargs -r kill
sleep 2
mv $B/src/studies/v29/a100_timing_q14 $B/src/studies/v29/_a100_timing_q14_wrongroot_failed
mv $B/logs/a100_timing_q14.log $B/logs/a100_timing_q14.wrongroot.log
# 2) 计时库根目录可由 PANS_ROOT 覆盖（默认不变）
cd $B/src
python3 - <<'PY'
from pathlib import Path
p = Path('scripts/v29_timing_lib.py'); t = p.read_text()
a = "R = Path('/home/shensicheng/qyd/pans/src'); PY = '/home/shensicheng/qyd/pans/venv/bin/python'"
assert t.count(a) == 1, 'pattern'
t = t.replace(a, "R = Path(os.environ.get('PANS_ROOT', '/home/shensicheng/qyd/pans/src')); PY = '/home/shensicheng/qyd/pans/venv/bin/python'")
p.write_text(t); print('lib patched')
PY
git -c user.email=a100@local -c user.name=a100 commit -qam "timing lib: PANS_ROOT override for model worktrees"
for T in q14 q32; do (cd $B/src_$T && git merge -q a100 && grep -c PANS_ROOT scripts/v29_timing_lib.py); done
# 3) 重新排队 14B、32B（GPU 4，同一计时环境）
cat > $B/xfer/queue_timing_q14q32.sh <<'Q'
#!/bin/bash
B=/home/shensicheng/qyd/pans
for T in q14 q32; do
  cd $B/src_$T && . $B/env_timing.sh && . $B/env_$T.sh && export PANS_ROOT=$B/src_$T
  $B/venv/bin/python scripts/v29m_timing.py a100_timing_$T --tag s800 > $B/logs/a100_timing_$T.log 2>&1
done
echo ALL_TIMING_DONE
Q
setsid nohup bash $B/xfer/queue_timing_q14q32.sh > $B/logs/queue_timing_q14q32.log 2>&1 < /dev/null &
sleep 5; ps -u shensicheng -o pid,args | awk '/v29m_timing|queue_timing/ && !/awk/' | cut -c1-120
