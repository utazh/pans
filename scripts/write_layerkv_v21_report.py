#!/usr/bin/env python3
"""Write the final report from recorded evidence, after all jobs complete."""
from pathlib import Path
import json,statistics,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from contiguous_fuxian.sharedkv_v21.audit import paired_quality
def load(rel):return json.loads((ROOT/rel).read_text())
def scored(name):return [json.loads(x) for x in (ROOT/f'results/{name}/scored_records.jsonl').read_text().splitlines()]
def pair(a,b):return paired_quality([dict(x,query_hash=x['uid']) for x in scored(a)],[dict(x,query_hash=x['uid']) for x in scored(b)])
hold=load('results/v21_hold_compare.json')['variants'];reg=load('results/v21_reg_compare.json')['variants']
timing=load('results/v21_timing_summary.json');nsight=load('results/v21_nsight_summary.json');viz=load('results/v21_viz_summary.json')
pcomp={f'{a}__{b}':pair('v21_hold_'+a,'v21_hold_'+b) for a,b in [('identity','refit2_s200'),('identity','refit27_s200'),('identity','seq3'),('old_dense3','seq3'),('hybrid_identity','hybrid_refit27_s200'),('hybrid_identity','hybrid_seq3')]}
(ROOT/'results/v21_hold_paired_controls.json').write_text(json.dumps(pcomp,indent=2))
names=[('original','原系统'),('identity','vLLM split，无共享'),('old_dense1','旧单对 26→27'),('old_dense3','旧三对'),('refit2_s200','新单对 3→2'),('refit19_s200','新单对 18→19'),('refit27_s200','新单对 26→27'),('seq3','新顺序三对'),('hybrid_identity','hybrid，无共享'),('hybrid_refit27_s200','hybrid，新单对 26→27'),('hybrid_seq3','hybrid，新三对')]
lines=['# SharedKV v2.1 实验报告','',
'功能已经接入并完成真实 Qwen 实验。新校准明显缓解了旧三对共享的质量下降，但所有训练候选仍未通过预先设定的局部门限；以下结果不等同于部署认证或 1pp 非劣证明。','',
'工作目录：/home/panzihang/src/prism_layerkv。全部代码、训练、实验和 profiler 输出在服务器进行。原 prism_pans 和上一版实验代码保持不变。','',
'## 实验条件','',
'- Qwen2.5-7B-Instruct；28 层、28 Q heads、4 KV heads、head dim=128；RTX 3090；BF16 计算、FP16 payload；batch=1。',
'- TREC 固定 prefix=4940 tokens，P8 selector，约 25% 保留预算，GPU/CPU payload cache=55/131 MiB；保留原选择器和异步预取。',
'- 旧 128 条只用于回归/开发。新 fit=64、val=64、holdout=175，与旧 SharedKV 使用过的请求按 UID 和 query_text 排除；manifest 在 data/v21_splits。全部仍来自同一 TREC 数据池和同一 prefix。',
'- 最终 10 个主要质量对照在观察 holdout 前列明；额外 hybrid 三对用于执行路径数值核对。没有根据 holdout 调整 reader 参数或误差门限。','',
'## 独立评估：175 条','',
'标签按完整续写的平均 token log-probability 判定。下表 KL 为首 token 全词表 KL，相对原系统；它与分类正确率是不同指标。字节为实际恢复的 payload，原系统还含推测预取。','',
'| 配置 | 正确数/175 | 准确率 | 相对原系统 | 首 token KL | payload MB |',
'|---|---:|---:|---:|---:|---:|']
for suffix,label in names:
 r=hold['v21_hold_'+suffix]
 lines.append(f"| {label} | {r['student_correct']} | {r['accuracy_pct']:.2f}% | {r['delta_pp']:+.2f} pp | {r['first_token_mean_kl']:.5f} | {r['physical_payload_bytes']/1e6:.2f} |")
lines += ['',f"新三对相对旧三对提升 {pcomp['old_dense3__seq3']['delta_pp']:.2f} pp。回归集旧三对为 {reg['v21_reg_dense3']['accuracy_pct']:.2f}%，新三对为 {reg['v21_reg_seq3']['accuracy_pct']:.2f}%；独立集恢复没有完全保持到无共享水平。",'',
'同后端对照更能隔离共享本身的影响。下表 harmful 表示 baseline 对、student 错，beneficial 为反向翻转。','',
'| 配对 | 净变化 pp | harmful/beneficial | 配对 bootstrap 95% pp | 有害翻转率单侧 95% 上界 |',
'|---|---:|---:|---:|---:|']
for name,r in pcomp.items():
 lo,hi=r['paired_bootstrap_95pct_delta_pp']
 lines.append(f"| {name.replace('__',' → ')} | {r['delta_pp']:+.2f} | {r['baseline_right_student_wrong']}/{r['baseline_wrong_student_right']} | [{lo:.2f}, {hi:.2f}] | {r['harm_rate_exact_one_sided_upper95_pp']:.2f}% |")
lines+=['','这些区间以固定 checkpoint 和独立 query 假设为条件，没有多重选择/数据簇校正。175 条即使零 harmful，也不能用退化 bootstrap 宣称总体风险为零。完整逐层 token Jaccard、标签 margin、harmful UID 在 results/v21_hold_compare.json。','',
'## 旧七对隔离与校准','',
'| 共享对 | 旧 reader 回归准确率 | 新闭式校准 | 新 200 步蒸馏 |',
'|---|---:|---:|---:|']
for source,target in [(3,2),(6,7),(10,11),(15,14),(18,19),(22,23),(26,27)]:
 old=reg[f'v21_reg_isolated{target}']['accuracy_pct']
 vals=[f"{reg[f'v21_reg_refit{target}_s{s}']['accuracy_pct']:.2f}%" if f'v21_reg_refit{target}_s{s}' in reg else '—' for s in (0,200)]
 lines.append(f'| {source}→{target} | {old:.2f}% | {vals[0]} | {vals[1]} |')
lines+=['','teacher 为同一 vLLM FA2 后端的 identity。闭式校准固定 A，使用 student 的 prefix 权重、private tail 和 teacher 输出拟合 B/bV。蒸馏只更新 A/B/bK/bV，主干冻结，200 步、lr=1e-4、BF16 STE，使用独立 64 条 val 选择 checkpoint。','',
'顺序诊断采用 consumer 2→19→27，每步重跑真实 student fit/val，回放 teacher IDs，但仍执行 selector。上游 reader fingerprint 检查启用；未把 teacher 状态冒充已共享 student。所有 carry 都明确是未认证诊断候选。','',
'局部门限保持 output RMSE≤0.03、单请求最大 RMSE≤0.08、prefix mass P95≤0.03。','',
'| checkpoint | 输出相对 RMSE | 最大请求 RMSE | prefix mass P95 | local pass |',
'|---|---:|---:|---:|---|']
for name in ['v21_refit2_s0','v21_refit2_s200','v21_refit19_s0','v21_refit19_s200','v21_refit27_s0','v21_refit27_s200','v21_seq2','v21_seq3']:
 r=load(f'assets/{name}/trec.report.json')
 v=r['final'];lines.append(f"| {name} | {v['output_relative_rmse']:.4f} | {v['max_request_output_relative_rmse']:.4f} | {v['prefix_mass_p95']:.4f} | {r['local_gate_passed']} |")
lines+=['','局部 surrogate 误差下降不保证分类准确率单调上升。顺序 trace 的层间漂移、K-only/V-only 条件替换和非加权 logit 子空间诊断在 results/v21_operator_*.json；这些诊断不构成任务准确率或 KL 的理论下界。','',
'## 正式计时','',
'11 配置 × 4 轮，每组相同 32 条回归 query、8 条 warmup。顺序为正/倒/倒/正；每组重建 Pcache store，只复用冻结模型。所有 profiler/质量 GPU 任务退出后单独在 GPU2 运行；其他用户的 GPU0/1 负载仍存在，telemetry 已保存。正式组不带 profiler 或 logits 转储。','',
'TTFT 截止首 token logits ready，含最终 CUDA 同步；不含 benchmark 标签续写评分。P95 为每轮 P95 的四轮平均。','',
'| 配置 | 平均 TTFT ms | 平均 P95 ms |',
'|---|---:|---:|']
for name,r in timing['variants'].items():lines.append(f"| {name} | {r['mean_ttft_ms']:.2f} | {r['mean_p95_ttft_ms']:.2f} |")
lines+=['','| baseline → student | 平均延迟变化 | 四轮 block-bootstrap 95% 差值 ms |',
'|---|---:|---:|']
for r in timing['comparisons']:
 lo,hi=r['block_bootstrap_delta95_ms']
 lines.append(f"| {r['baseline']} → {r['student']} | {r['relative_latency_change_pct']:+.2f}% | [{lo:.2f}, {hi:.2f}] |")
lines+=['','原系统到 identity/hybrid 已经改变 attention 后端和执行组织；只有相同执行配置的 identity/shared 配对，才适合讨论跨层共享的额外收益。四轮重复只能提供有限的运行间证据，不能将同一 32 条 query 的重复当作新的独立质量样本。全部原始轮次和长尾保留。','',
'## 物理共享与固定计划','',
'单对唯一 selected prefix payload 从 70,647,808 降为 68,132,864 bytes（-3.5598%）；三对为 63,102,976 bytes（-10.6795%）。这是所选 prefix payload，不是模型总显存，也不是 SSD 文件压缩率。每个 dense reader 另占 264,192 bytes，预打包额外 133,120 bytes；峰值还包含临时 workspace 和标签评分。',
'',
'hybrid 的普通层会临时拼接 prefix/tail；拼接时间和分配均在运行中计入。去重的是持久 source bank，consumer 不重建自己的 prefix KV。', '',
'physical alias、唯一 source 数、private tail 长度、selector 调用数，以及 fixed shared/shadow/demand 的逐 query logits/计划一致性，见 results/v21_invariants.json。shadow-load 恢复 consumer 原生 KV 但不参与 attention；它会改变缓存驻留和队列，是诊断对照。','',
'## Nsight 与 VizTracer 瓶颈','',
'Nsight 统计只取 8 个正式请求的 TTFT/logits_ready 范围，排除加载、warmup 和标签评分。H2D 包含 payload、selector index 与 metadata，是 CUPTI 硬件拷贝记录，和 host 来源 payload 计数不同。下列时间来自 profiler，不作正式加速结论。','',
'| 配置 | profiled TTFT ms | kernel 数/请求 | GPU kernel sum ms | GPU busy union ms | H2D MB | H2D sum ms |',
'|---|---:|---:|---:|---:|---:|---:|']
for name,r in nsight['variants'].items():
 v=r['means'];lines.append(f"| {name} | {v['ttft_profiled_ms']:.2f} | {v['gpu_kernel_count']:.0f} | {v['gpu_kernel_sum_ms']:.2f} | {v['gpu_busy_union_ms']:.2f} | {v['h2d_bytes']/1e6:.2f} | {v['h2d_sum_ms']:.3f} |")
lines+=['',f"VizTracer 重采集获得 {viz['trace_events']:,} 个函数事件，overflow={viz['overflow']}；选取每个请求的第一次 decoder，排除标签 decode。self 时间包含未被追踪的库调用、C 调用及阻塞等待，不等于纯 Python 指令时间。",'',
'| 主线程函数 | self ms/prefill |',
'|---|---:|']
for r in viz['main_thread_by_self'][:10]:lines.append(f"| {r['function'].split(' (')[0]} | {r['self_ms_per_prefill']:.2f} |")
lines+=['','硬件 H2D 约 1 ms，GPU kernel 总时间约 30–32 ms，而 profiled TTFT 约 260–295 ms。结合主线程和 NVTX 记录，更值得优先处理的是 selector、缓存分数更新、分段/块元数据管理、KV 就绪等待和小算子提交。继续只压缩一份 prefix payload，不能直接消除这部分开销。跨线程 inclusive 时间存在重叠，不应相加估算关键路径。','',
'## 复现与完整性','',
'- 运行入口和参数说明：README_LAYERKV_V21.md。',
'- 新读者：assets/v21_refit*_s*/trec.pt、assets/v21_seq2/trec.pt、assets/v21_seq3/trec.pt；均为 diagnostic_only。',
'- 全部单测：380 passed、4 skipped、87 subtests passed，日志 logs/v21_all_tests.log；真实 vLLM GQA/hybrid/packed GPU 测试通过。',
'- 原工程 276 个文件 SHA256 未变，上一版代码未变；原包 manifest 和 8 个接口 hash 通过。见 audit/v21_*integrity.json、audit/v21_provenance.json。',
'- 适配修正和补充工具见 audit/v21_package_fixes.patch、README_LAYERKV_V21.md。',
'- 原始 Nsight：profiles/v21/nsys_*.nsys-rep；有效 VizTracer：profiles/v21/viz_hybrid_shared_retry/viztracer.json。',
'- 首次 profiler 因前一进程未退出发生加载 OOM，失败记录归档于 profiles/v21/failed_startup_overlap；首次空 VizTracer 文件仅保留作调试记录，未进入分析。',
'- 结果 CSV/JSON、原始逐请求记录、logits、trace、训练 history 和命令清单均保留在服务器。','']
lines+=['## Resident 微基准','',
'合成 GPU resident tensors，prefix=1228；prefill 的 Q/private=41/41，decode 几何为 1/42。每项 warmup=20、交替顺序采样 80 次。下面为同步 host wall 中位数；不含真实 KV 加载、selector、完整模型或服务排队，不能相加预测 TTFT。CUDA-event 间隔也保存在原始 JSON 中，可能包含主机提交间隙。','',
'| 算子路径 | Q=41/private=41 ms | Q=1/private=42 ms |','|---|---:|---:|']
m1=load('results/v21_micro_q41.json')['results'];m2=load('results/v21_micro_decode_q1_tail42.json')['results']
for name in ['native_cat_one_attention','identity_two_attention','v1_reader_two_attention','v21_reader_two_attention','v21_reader_cached_metadata','q_transform_offset_only','v_transform_merge_only']:
 lines.append(f"| {name} | {m1[name]['wall_ms_median']:.4f} | {m2[name]['wall_ms_median']:.4f} |")
lines+=['']
(ROOT/'EXPERIMENT_REPORT_V21_ZH.md').write_text('\n'.join(lines))
print('Wrote',ROOT/'EXPERIMENT_REPORT_V21_ZH.md')
