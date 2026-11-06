import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parents[1]
q=json.loads((r/'results/v22_quality_compare.json').read_text())['variants']
a=json.loads((r/'results/v22_analysis.json').read_text())
t=json.loads((r/'results/v22_timing_summary.json').read_text())
n=json.loads((r/'results/v22_nsight_summary.json').read_text())['variants']
v=json.loads((r/'results/v22_viz_summary.json').read_text())
inv=json.loads((r/'results/v22_invariants.json').read_text())
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
text=['# SharedKV v2.2 服务器实验报告','',
'本轮已按最新 GPT 方案完成：16 条候选边、4! 组匹配、等条件同编号控制、200 步蒸馏、完整模型回归、固定选择集合对照、私有 decode cache 审计、配对计时、Nsight 与 VizTracer。',
'',
'**结论：GQA 组对齐降低了局部平均读取误差，但在本轮 175 条 TREC 回归请求上没有带来准确率改善；所有候选仍未通过原局部门限。当前不能据此扩大共享范围或声称方法整体更强。**',
'',
'## 实施与保护边界','',
'- 全部安装、拟合、执行、结果与 profiler 文件位于服务器 /home/panzihang/src/prism_layerkv；没有在本地创建实验代码或结果。',
'- 上传包 pans_sharedkv_v22.zip 的 21 项 manifest 校验通过；8 个关键 v2 接口 Git blob 哈希与包内预期完全一致。服务器是无 .git 的独立目录，因此按内容身份审计后只增加文件，未伪造 Git 提交或删除运行 guard。',
'- 原 prism_pans 及此前实验源文件哈希检查通过，原目录本轮没有新增修改文件。具体范围见 audit/v22_before.json 和 results/v22_invariants.json。',
'- 主干冻结；当前层原生 Q 与 private query/generated KV 保留。只对短 Q、prefix 输出和 LSE 路由，长 source K/V 不重排。未修改 selector、P8 或 token 保留预算。',
'- 本次未修改包内算法源文件；新增服务器运行、分析、审计脚本与真实 Qwen 几何 CUDA 测试。','',
'## 数据与公平性','',
'模型 Qwen2.5-7B-Instruct，RTX 3090，28 层 / 28 Q heads / 4 KV heads / head_dim 128；TREC prefix 4940 tokens，P8，25% exact block budget。',
'64 fit 按 query 分成 48 edge-fit + 16 assignment-selection，64 val 用于局部报告和训练 checkpoint 选择。两种映射均测量完整 4×4 表、相同参数量、相同 seed / ridge / 学习率 / 训练步数，未使用 initial-reader。K/V 共用一对一映射。',
'**175 条端到端数据已在 v2.1 被观察，本轮只作回归/开发证据，不是新最终测试。** 它们与本轮 64 fit / 64 val 的 tokenized query hash 无交集。当前实验没有新任务或新 prefix 的泛化证明。',
'局部门限保持 output RMSE ≤0.03、max request ≤0.08、prefix mass P95 ≤0.03。所有产物 diagnostic_only=True，没有部署认证。','',
'## 局部验证：均值改善，但最坏请求并未一同改善','']
rows=[]
for target in (19,27):
 for steps in (0,200):
  for mode in ('diag','map'):
   d=a['local'][f'v22_{mode}{target}_s{steps}'];f=d['final']
   rows.append([target-1 if target==27 else 18,target,steps,mode,d['head_map'],f"{f['output_relative_rmse']:.6f}",f"{f['max_request_output_relative_rmse']:.6f}",f"{f['prefix_mass_p95']:.6f}",f"{f['mean_joint_attention_kl']:.6f}",d['local_gate_passed']])
text+=[table(['source','target','steps','mapping','π(target→source)','output RMSE','max request','mass P95','attention KL','local pass'],rows),'',
'diag 为同编号重新校准；map 为搜索组对应。200 步的 18→19 平均误差下降约 7.7%，26→27 下降约 15.3%；两者最坏请求误差均高于各自同编号控制。不能把均值下降写成所有请求都改善。','']
for target in (19,27):
 d=a['local'][f'v22_map{target}_s0']['assignment']
 text += [f'### target {target} 的 4×4 赋值代价','',table(['target组/source组','0','1','2','3'],[[h,*[f'{x:.6f}' for x in row]] for h,row in enumerate(d['cost_matrix'])]),'',f"选择 π={d['permutation']}，内部匹配代价下降 {d['relative_cost_reduction']*100:.2f}%。该百分比只属于 16 条内部匹配请求，不是任务质量提升。",'']
text+=['## 完整模型：175 条在线选择回归','']
rows=[]
for name in ('identity','diag19_s0','map19_s0','diag19_s200','map19_s200','diag27_s0','map27_s0','diag27_s200','map27_s200'):
 d=q['v22_reg_'+name]
 rows.append([name,f"{d['accuracy_pct']:.2f}%",round(d['accuracy_pct']*175/100),f"{d['first_token_mean_kl']:.6f}",f"{d['label_mean_kl']:.6f}",d['baseline_right_student_wrong'],d['baseline_wrong_student_right']])
text+=[table(['配置','准确率','正确数/175','首token KL vs identity','标签 KL vs identity','identity对→错','identity错→对'],rows),'',
'准确率按实际 runner 的标签续写**每 token 平均对数似然**计算；首 token 全词表 KL 与该指标不同。对照使用同一 hybrid / metadata-cache 后端，避免把执行优化收益归于组对齐。','']
rows=[]
for k,d in a['pairs'].items():
 rows.append([k,f"{d['baseline_accuracy']:.2f}%",f"{d['student_accuracy']:.2f}%",d['baseline_right_student_wrong'],d['baseline_wrong_student_right'],str(d['paired_bootstrap_95pct_delta_pp'])])
text += [table(['对照','diag准确率','map准确率','有害翻转','有益翻转','准确率差95%区间（百分点）'],rows),'',
'18→19 零步少对 3 题，200 步少对 2 题；26→27 零步正确数持平，200 步少对 1 题。这些小差异不能证明稳定退化，但本轮没有准确率增益证据。局部 surrogate 改善不能替代完整模型质量验收。26→27 的首 token KL 从同编号约 0.00163 进一步降到约 0.00068，这是分布接近度的正向证据；18→19 则从约 0.01615 上升至 0.02520。不能只保留有利指标。',
'',
'## 固定选择集合与 decode','',
'固定选择集合使用同一 identity 的实际逐层 token IDs，selector 计算仍计入；32 条单独比较，不能作为正式速度证据。',
table(['配置','准确率','首token KL','标签 KL'],[[k,f"{d['accuracy_pct']:.2f}%",f"{d['first_token_mean_kl']:.6f}",f"{d['label_mean_kl']:.6f}"] for k,d in json.loads((r/'results/v22_fixed_compare.json').read_text())['variants'].items()]),
'',
'固定32条只用于控制诊断，其结果与175条整体方向不完全相同，不能只取小样本中的有利结果。另跑 8 条、最多 16 token 生成的 identity 与两个映射配置，仅验证运行和私有 cache 边界，不据此宣称长生成质量。结果：'+json.dumps(a['decode'],ensure_ascii=False),
'',
'## 资源与开销','',
table(['配置','唯一 prefix bytes','reader bytes','packed副本 bytes','路由 bytes','净节省%（prefix+reader）'],[[k,d['prefix_buffer_bytes'],d['reader_bytes'],d['packed_extra_bytes'],d['routing_bytes'],f"{d['net_saving_pct']:.4f}"] for k,d in a['resources'].items()]),
'',
'组映射本身不进一步减少相同层对的 KV；其潜在价值应是让更多层通过质量验收，本轮尚未实现。请求 prefix buffer + reader 的净节省不是总模型显存节省，hybrid 原生层还有短期 prefix/tail 拼接 workspace。计算器没有改变保留率，也不是容量实验。',
'',
'## 正式 TTFT','',
'6 配置 ×4 轮 ×32 条相同回归请求，每配置 8 次 warmup、fresh Pcache；前/反/反/前顺序。复用冻结模型加载，设备2；GPU3 本任务空闲，其他用户 GPU0/1 作业继续，主机竞争无法完全消除。无 profiler、无 logits/trace 输出，新增 Q/output/LSE 路由在 TTFT 内。时间异常值全部保留。',
table(['配置','均值 ms','4轮均值 ms'],[[k,f"{d['mean_ttft_ms']:.3f}",', '.join(f"{b['mean_ttft_ms']:.2f}" for b in d['blocks'])] for k,d in t['variants'].items()]),
'',
table(['baseline','student','变化%（负为更快）','配对差值95%区间 ms'],[[d['baseline'],d['student'],f"{d['relative_latency_change_pct']:.2f}",str(d['block_bootstrap_delta95_ms'])] for d in t['comparisons']]),
'',
'本轮无共享均值276.946 ms、头对齐18→19为273.319 ms、头对齐26→27为278.474 ms；两种头对齐相对无共享的差值区间均跨零，未证明稳定加速。18→19头对齐比同编号控制慢9.75%，但此差值包含选择/缓存/调度变化，不能全部归因于几个路由算子。不同日期/运行的绝对TTFT不宜直接与v2.1比较。四轮 bootstrap 只能提供有限的轮次不确定性估计，不能用单个最佳点判断稳定加速。GPU常驻算子微基准中新增路由约 0.08–0.09 ms/consumer，该结果不是 TTFT，也不是硬件 copy 时间。',
'',
'## Nsight / VizTracer','',
'独立 profiler 请求，仅诊断。Nsight 统计严格限定 TTFT/logits_ready NVTX 范围，排除模型加载、warmup 和标签评分；H2D 含 payload/index/metadata，不能称为纯 KV 流量。重叠/嵌套事件时间不可相加当作关键路径。两条路径的 H2D 均约0.91ms；VizTracer中较大项仍包括线程等待、selector和缓存元数据维护。头对齐的 output_route NVTX 包含时间约0.182ms/请求，Q变换+路由约0.635ms；这些包含profiler开销，不能解释成全部端到端差值。',
table(['配置','H2D MB','H2D累计 ms','kernel数','GPU busy union ms','profiled TTFT ms'],[[k,f"{d['means']['h2d_bytes']/1e6:.4f}",f"{d['means']['h2d_sum_ms']:.4f}",f"{d['means']['gpu_kernel_count']:.1f}",f"{d['means']['gpu_busy_union_ms']:.3f}",f"{d['means']['ttft_profiled_ms']:.3f}"] for k,d in n.items()]),
'',
f"VizTracer 有 {v['trace_events']} 个函数事件，overflow={v['overflow']}，识别 {v['prefill_count']} 个首次 prefill。self 时间包含未跟踪库调用和阻塞，不等于纯 Python 计算。",
table(['函数','self ms/prefill'],[[x['function'],f"{x['self_ms_per_prefill']:.3f}"] for x in v['main_thread_by_self'][:10]]),
'',
'## 验证与后续决定','',
'- 416 项测试通过、5 项可选依赖测试跳过、87 个 subtests 通过；包括真实 CUDA / GQA 28:4 / d128 / prefix1228 / prefill与decode/private-tail数值和物理输入身份验证。',
'- 旧 reader 经 v21 / v22 新入口的 8 条固定计划桥接对照：logits bitwise相等、标签分数相等、逐层 token IDs 相等。',
f"- 物理 alias、源去重、private KV 边界、每请求4次 P8 selector、128校准与175回归 query隔离全部通过：{inv['all_passed']}；此前文件 {inv['prior_files_checked']} 项哈希未变。",
'- 局部平均误差改善是真实信号，但完整分类未改善，且所有局部门限仍失败。按本轮停止条件，不继续组合更多共享层，不放宽门限，不开展基于“更多合格共享层”的资源曲线。',
'- 目前只能说本模型的非同编号 head 有更低局部拟合代价；无法证明组错配是任务质量问题的主因。校准只观察少量 prefill 位置，最终评价涉及多 token 标签续写，目标不匹配是待验证解释，不能由本轮结果直接认定。',
'',
'## 文件与复现','',
'代码：src/contiguous_fuxian/sharedkv_v22/；运行清单：scripts/v22_*_jobs.json；拟合入口：scripts/screen_headalign_v22.py、scripts/train_headalign_v22.py；完整命令保存在各结果目录 command.json 和日志中。',
'主要汇总：results/v22_analysis.json、v22_quality_compare.json、v22_fixed_compare.json、v22_timing_summary.json、v22_nsight_summary.json、v22_viz_summary.json、v22_invariants.json。',
'原始 traces/readers/results 保留在服务器。profiles/v22/ 包含 .nsys-rep、SQLite 与 VizTracer JSON。详细包使用说明见 README_HEADALIGN_V22.md。',
'',
'## 白话版','',
'新方案确实给四组内容找到了更好读的对应关系，小范围检查也更接近原模型；但完整答题没有更准。翻译某一小段更准确，不代表整份答案更准确。当前不能据此放心共享更多层，所以先保留诊断结果，主流程继续保留质量更稳的配置。'
]
(r/'EXPERIMENT_REPORT_V22_ZH.md').write_text('\n'.join(text)+'\n')
print('Report written',r/'EXPERIMENT_REPORT_V22_ZH.md')
