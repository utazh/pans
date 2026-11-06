from pathlib import Path
import json,time,hashlib,subprocess
r=Path('/home/panzihang/src/prism_layerkv')
subprocess.run(['/home/panzihang/venvs/vllm-stable/bin/python','scripts/v28_final_results_audit.py'],cwd=r,check=True)
d=json.loads((r/'results/v28_summary/analysis.json').read_text())
q={(x['cell'],x['arm']):x for x in d['quality']}
t={(x['cell'],x['condition'],x['arm'],x['reference']):x for x in d['timing']}
lines=['# v2.8 四数据集 × 四预算实验结果','',
'固定各格 R4 reader，仅将推理原生块数切换为 R0；没有新增训练。所有结果来自服务器 prism_layerkv。','',
'## 主质量矩阵','',
'完整评估：SST-2 872、SUBJ 1003（修正后的自定义划分）、TREC 500、RTE 277。全部属于已观察历史划分，不是新的最终留出认证。','',
'| 数据集/预算 | I 准确率 | R4 准确率 | frozen-R0 准确率 | R0−R4 (pp) |',
'|---|---:|---:|---:|---:|']
cells=[f'{a}_b{b}' for a in ['sst2','subj','trec','rte'] for b in [5,10,25,50]]
for c in cells:
 vals=[q.get((c,a),{}).get('student_accuracy') for a in ['I','R4','R0']]
 fmt=lambda v:'缺失' if v is None else f'{100*v:.2f}%'
 delta=q.get((c,'R0_vs_R4'),{}).get('delta_pp')
 lines.append('| '+c+' | '+' | '.join(map(fmt,vals))+' | '+('缺失' if delta is None else f'{delta:+.2f}')+' |')
lines+=['','## SSD 冷缓存与普通缓存 TTFT','',
'每格六块 × 16 条不同请求，六种三臂顺序各一次；每臂独立进程。SSD-cold/preserve + reuse-prefetch，模型/reader 预热，正式请求逐次认证 payload 空和目标文件页不驻留。普通缓存另测。区间是探索性块级重采样，不是普遍稳定加速认证。','',
'| 单元 | 条件 | I ms | R4 ms | R0 ms | R4/I | R0/I | R0/R4 |','|---|---|---:|---:|---:|---:|---:|---:|']
for c in cells:
 for cond in ['preserve','warm']:
  a=t.get((c,cond,'R4','I'));b=t.get((c,cond,'R0','I'));e=t.get((c,cond,'R0','R4'))
  if all([a,b,e]):
   lines.append(f"| {c} | {cond} | {a['baseline_ms']:.2f} | {a['student_ms']:.2f} | {b['student_ms']:.2f} | {a['delta_percent']:+.2f}% | {b['delta_percent']:+.2f}% | {e['delta_percent']:+.2f}% |")
  else:lines.append(f'| {c} | {cond} | 未完成 | | | | | |')
lines+=['','## 层间基线质量','',
'KVSharer-Prefix 每格使用独立 30 条 FIT 搜索 G24/G20/G16，阈值不变。xKV-CLF/P8 的两档秩和两种精度均来自每任务同一精确 FP64 master。每格先检查 direct identity 和 FP64 全秩接入。外部桥仍恢复原生 payload，其 TTFT 不进入速度表；以下不是作者官方系统速度复现。','',
'| 单元 | KV G24 | KV G20 | KV G16 | CLF256/384 FP32 | BF16 | CLF384/576 FP32 | BF16 |','|---|---:|---:|---:|---:|---:|---:|---:|']
arms=['kvsharer_g24','kvsharer_g20','kvsharer_g16','clf_256_384_fp32','clf_256_384_bf16','clf_384_576_fp32','clf_384_576_bf16']
for c in cells:
 lines.append('| '+c+' | '+' | '.join('未达到/未完成' if (c,a) not in q else f"{q[c,a]['student_accuracy']*100:.2f}%" for a in arms)+' |')
lines+=['','## 结果与限制','',
'R0/R4 两条完整矩阵都保留，未根据测试分数逐格选择最优。质量翻转、保守下界、配对块区间见 results/v28_summary/analysis.json。外部资源表仅为 FIT 支持上的理论张量账，不是实际硬件 I/O、总显存或 SSD 速度。','',
f"缺失/失败项：{len(d['missing_or_incomplete'])}。详细列表见 analysis.json；没有用最佳值填补。",'',
'文件：results/v28_summary/quality.csv、timing.csv、external_theory_only.csv；studies/v28_frozen/jobs、gates、runs、logs 保存原始记录。']
(r/'EXPERIMENT_REPORT_V28_ZH.md').write_text('\n'.join(lines)+'\n')
print('REPORT_WRITTEN',len(d['quality']),len(d['timing']),len(d['missing_or_incomplete']))
