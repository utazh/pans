from pathlib import Path
import json,statistics
r=Path(__file__).resolve().parents[1];d=json.loads((r/'results/v24_final_analysis.json').read_text())
f=d['frozen_selection'];G=f['chosen_G']
out=['# v2.4 服务器执行报告',''];add=out.append
if f['last_all_seed_admitted_G'] is None:add('**本轮未通过预定的三种子开发集非劣条件，主线停在 G26（28→26个独立prefix KV来源），没有把当前共享组复制到更多层。** 其余种子、消融、独立最终集、历史完整TREC与两种SSD冷恢复策略均继续评估，保留失败证据。')
else:add(f'**按开发集门限，本轮推进到G{G}。** 最终集在训练和层数冻结后才评估，开发集准入不等同部署认证。')
old_kl=statistics.mean(d['quality']['dev'][f'group_s{s}']['paired_vs_identity']['online_score_fidelity']['candidate_kl'] for s in [24,25,26])
new_kl=statistics.mean(d['quality']['dev'][f'score_g26_s{s}']['paired_vs_identity']['online_score_fidelity']['candidate_kl'] for s in [24,25,26])
add(f"\n同一在线identity口径下，新评分校准的三种子开发集平均候选KL由{old_kl:.6f}降到{new_kl:.6f}（下降{100*(1-new_kl/old_kl):.1f}%）。这说明该校准配方改善了得分保真度，但不等于分类质量已通过验收。")
add('新旧配方同时改变目标及每步候选覆盖，实际训练工作量不同；不能把全部降幅归因于margin或某一损失项。')
for policy in ['native','preserve']:
 t=d['timing'][policy]['all'];lo,hi=t['block_bootstrap_relative95_pct']
 add(f"\nSSD冷恢复/{policy}：{t['baseline_mean_ms']:.2f}→{t['shared_mean_ms']:.2f} ms，相对变化{t['relative_change_pct']:+.2f}%，四轮块bootstrap区间[{lo:+.2f}, {hi:+.2f}]%。每种策略4轮×32条，同32条重复，均为冻结seed24；共512次正式请求全部通过冷认证。")
a=d['timing']['warm']['all'];b=d['warm_fresh_process_confirmation']['all']
add(f"\n普通缓存原同进程实验变化{a['relative_change_pct']:+.2f}%，事后独立进程复核变化{b['relative_change_pct']:+.2f}%。两组方向相反，尚不能宣称不受运行条件影响的稳定加速；复核不替换主结果。")

add('\n全部工作位于 /home/panzihang/src/prism_layerkv；原工程 /home/panzihang/src/prism_pans 未修改。开工前617个文件哈希均无变化。')
add('\n## 1. 实现与数据\n')
for s in [
'本层Q经A变换读取共享prefix KV，聚合输出经B变换；query及生成token KV仍28层各自保存。G26为source26服务25、27，其他层保留原生prefix KV。28层均参与计算。',
'新目标对齐完整候选续写平均log-probability，使用候选KL、中心化得分MSE、逐token MSE与教师top-margin；所有私有KV均由学生重算，主干冻结，仅训练A/B/bk/bv。',
'旧组边界目标、新评分目标、去margin目标各3个种子24/25/26，均200步、每25步验证，同fit64/val64与相同初值。checkpoint按各自校准验证目标选择，相同步数不代表相同计算量。',
'新dev768与final768从TREC train5500剩余问题划分，排除以往实验问题及prefix示例的规范化文本；与fit/val、历史dev、test500的token hash不重合。它们是新划分的训练池留出集，并非新官方测试集。',
'官方test500已在多轮观察，作为历史回归。固定4940-token prefix、25%精确块预算、在线P8、Qwen2.5-7B-Instruct。'
]:add('- '+s)
add('所有无共享基线使用上一轮优化后的hybrid/packed/metadata稀疏执行路径，保持相同25%预算和在线P8。')
add('\n## 2. 真实模型回放修复\n')
add('原包参考回放在64条验证请求触发未修改的原生评分门限：平均候选得分最大误差0.6223、逐token最大误差0.6848。保留拒绝报告，未放宽0.02/0.10门限。新增校准回放复用线上vLLM FA2前向、packed变换与逐token续写，并实现含LSE项的数学attention反向；修复后64条原生评分误差均为0。反向使用FP32数学softmax及原有BF16直通估计，不声称对有限精度量化精确求导。')
p=d['invariants']['shared_scorer_replay_parity']
add(f"最终共享模型在8条实际在线选择集合上核对：候选得分最大误差{p['max_mean_score_error']:.8g}，逐token最大误差{p['max_token_error']:.8g}。详见docs/sharedkv_v24/SERVER_REPLAY_REPAIR.md。未改线上attention、缓存管理及冷认证。")
add('这里的候选KL只比较TREC标签分布，不是此前报告的首token全词表KL；两者不能直接比较数值。旧组目标与新评分目标的损失也不是同一量纲。')
add('\n## 3. 开发集与扩层决定\n')
add('| 配置 | 正确/768 | 准确率 | 有害/有益翻转 | 净差pp | 保守下界pp | −1pp门限 |')
add('|---|---:|---:|---:|---:|---:|---|')
for arm,q in d['quality']['dev'].items():
 b,p=q['basic'],q['paired_vs_identity']
 add(f"| {arm} | {b['correct']}/{b['n']} | {b['accuracy_pct']:.2f}% | {p['harmful']}/{p['beneficial']} | {p['net_delta_pp']:+.3f} | {p['conservative_one_sided_lower_pp']:+.3f} | {'通过' if p['conservative_bound_passes'] else '未通过'} |")
add('\n准入规则在新开发集评估前保存：每个种子相对identity的保守配对下界≥−1个百分点，3个种子均满足才扩层。所附代码使用Bonferroni二项分位数组合下界；未通过不等于已证明真实退化超过1个百分点。开发集反复比较不构成家族错误率认证。')
for stage in f['stages']:add(f"- G{stage['G']}：{'三种子通过' if stage['all_seed_dev_gate_passed'] else '未全部通过，停止主线扩层'}。")
add('\n统一口径的开发集得分保真度（同一在线identity为教师；不用于额外挑模型）：\n')
add('| 配置 | 候选KL | 中心化得分MSE | 逐token MSE，先标签内后标签间平均 | 教师预测一致率 |')
add('|---|---:|---:|---:|---:|')
for arm,q in d['quality']['dev'].items():
 p=q['paired_vs_identity'];x=p['online_score_fidelity']
 add(f"| {arm} | {x['candidate_kl']:.6f} | {x['centered_score_mse']:.6f} | {x['per_label_mean_token_mse']:.6f} | {100*p['prediction_agreement']:.2f}% |")
add('候选分布是对平均续写log-probability做softmax的辅助分布，不等于模型已经校准的真实类别概率，也不是首token全词表KL。')
add('\n## 4. 冻结后的完整质量评估\n')
for split,title in [('final768','新最终留出集768'),('test500','历史完整TREC500')]:
 add('### '+title+'\n');add('| 配置 | 正确数 | 准确率 | 有害/有益翻转 | 净差pp | 保守下界pp |');add('|---|---:|---:|---:|---:|---:|')
 for arm,q in d['quality'][split].items():
  b,p=q['basic'],q['paired_vs_identity'];add(f"| {arm} | {b['correct']}/{b['n']} | {b['accuracy_pct']:.2f}% | {p['harmful']}/{p['beneficial']} | {p['net_delta_pp']:+.3f} | {p['conservative_one_sided_lower_pp']:+.3f} |")
 add('')
add('主计时模型预先固定seed24，不按最终集或测试集挑最好种子。历史500条不重新充当未观察过的验收集；剔除历史标记5条的strict495另存分析JSON。三个种子使用同一批问题，不能视为三倍独立样本。')
add('\n## 5. 正式TTFT\n')
add('| 缓存/读策略 | 无共享ms | 共享ms | 相对变化 | 配对块bootstrap95%区间 |');add('|---|---:|---:|---:|---:|')
for policy,g in d['timing'].items():
 t=g['all'];lo,hi=t['block_bootstrap_relative95_pct'];add(f"| {policy} | {t['baseline_mean_ms']:.2f} | {t['shared_mean_ms']:.2f} | {t['relative_change_pct']:+.2f}% | [{lo:+.2f}, {hi:+.2f}]% |")
t=d['warm_fresh_process_confirmation']['all'];lo,hi=t['block_bootstrap_relative95_pct']
add(f"| warm 独立进程复核（事后追加） | {t['baseline_mean_ms']:.2f} | {t['shared_mean_ms']:.2f} | {t['relative_change_pct']:+.2f}% | [{lo:+.2f}, {hi:+.2f}]% |")
add('独立进程普通缓存复核在发现混合套件残留mmap后追加，原四轮完整保留。两组处于不同阶段和进程生命周期，差异不能全部归因于mmap，也不能择优替换主结果。')
add('\n每种策略4轮×32条配对，顺序I-S、S-I、S-I、I-S，重复同32条历史请求。区间对4个配对块重采样，是探索性小样本计时区间，不能视为128个独立质量样本。顺序分层结果保留于JSON。本任务正式计时期间没有并发训练、评估或profiler；其他GPU既有任务未停止，配对设计不能消除全部主机负载影响。并行质量评估的耗时未混入此表。')
for s in ['warm：普通缓存路径且运行时已预热，不声称全部目标KV都在GPU。','native：请求前清空应用payload并验证所有目标文件页不驻留；保留原代码读后驱逐策略。','preserve：请求前同样认证冷页，但读取后保留文件页。只与同策略identity配对，避免把读策略收益算成共享收益。','模型/内核预热保留；约数秒/请求的冷准备单列、排除于TTFT。认证范围是应用缓存及OS文件页，不声称清空SSD设备内部缓存。']:add('- '+s)
for policy,arms in d['cold_certification'].items():
 for arm,x in arms.items():add(f"- {policy}/{arm}：{x['requests']}次冷认证全部通过；准备均值{x['mean_setup_ms_excluded']:.1f}ms（计时外）；首decoder进程read_bytes均值{x['mean_process_read_bytes_first_decoder']/1e6:.2f}MB，无payload写盘。")
add('本轮首次在同一进程从多轮普通缓存切入冷缓存时，mincore检出7884/69160页仍驻留，因而中止且没有有效冷TTFT；进程仍有19928条缓存文件映射。该失败保留在audit/v24_cold_restart.json及重命名的rejected记录中。随后每个正式冷arm使用独立进程、相同8次运行时预热及原封不动的v23冷准备/认证。两条独立进程探测先通过，再重跑全部冷arm。没有强制munmap存活tensor，也没有放宽零页条件。')
add('\n## 6. 资源与瓶颈\n')
res=d['resource'];add(f"所选request prefix KV减少{res['selected_payload_reduction_pct']:.2f}%；加入resident reader与packed副本后，该局部缓冲合计净减少{res['selected_buffer_plus_readers_net_reduction_pct']:.2f}%。这不是总模型显存或SSD占用降幅。持久prefix store仍保留原28层用于校准和对照，没有删除磁盘副本。")
add('\n| Nsight诊断，各8条 | H2D MB/请求 | H2D累计ms | GPU busy union ms | profiled TTFT ms |');add('|---|---:|---:|---:|---:|')
for name,x in d['nsight']['variants'].items():
 m=x['means'];add(f"| {name} | {m['h2d_bytes']/1e6:.2f} | {m['h2d_sum_ms']:.3f} | {m['gpu_busy_union_ms']:.2f} | {m['ttft_profiled_ms']:.2f} |")
add('\n硬件统计限定TTFT/logits_ready范围，含payload、索引、元数据；profiled耗时不进入正式TTFT。各累计时间可重叠，不可相加成关键路径。')
add('\n| 恢复路径诊断 | 预取读尝试/请求 | 晋升读尝试/请求 | 两路径均触及文件数/请求 |');add('|---|---:|---:|---:|')
for name,x in d['restore_diagnostic'].items():add(f"| {name} | {x['prefetch']['calls']:.1f} | {x['promotion']['calls']:.1f} | {x['mean_files_seen_in_both_paths']:.1f} |")
add('\n该审计含额外同步栅栏，仅做归因。文件读尝试及声明字节数不等于真实SSD或NAND流量，与/proc I/O及Nsight分开报告。VizTracer self时间含未跟踪库调用及阻塞等待，多线程inclusive时间不能相加。')

add('\nVizTracer主线程self时间前三项（每项均为8条profile请求均值，含阻塞与未跟踪调用）：\n')
add('| 条件 | 函数 | self ms/请求 |');add('|---|---|---:|')
for key in ['viz_identity','viz_shared','viz_warm_identity','viz_warm_shared']:
 if key not in d:continue
 for x in d[key]['main_thread_by_self'][:3]:
  name=x['function'].split(' (')[0].replace('|','/')
  add(f"| {key} | {name} | {x['self_ms_per_prefill']:.2f} |")

add('\n## 7. 训练成本与验证\n')
add('| 模型 | 最佳步 | 含加载总秒数 | 峰值已分配GiB | 顶层前向计数，含验证 |');add('|---|---:|---:|---:|---|')
for name,t in d['training'].items():add(f"| {name} | {t['best_step']} | {t['wall_seconds_total_including_load']:.1f} | {t['cuda_peak_allocated']/2**30:.2f} | {json.dumps(t['forward_calls_including_validation'])} |")
add('\n计数不含checkpoint反向内部重计算，其开销包含在墙钟中。旧目标计数单位为局部组，新目标为全模型候选分支，次数不能直接等同FLOPs。所有9个G26模型初值SHA256相同。失败的原生回放及一步smoke保留，不计入正式训练表。')
tests=(r/'logs/v24_all_gpu_tests.log').read_text().splitlines();add('完整GPU可用回归：'+next((s for s in reversed(tests) if 'passed' in s),'见日志')+'。')
add('物理KV别名、source数量、加载次数、28层私有生成KV实际存储、数据隔离、评分回放及原工程保护审计全部通过，详见audit/v24_final_invariants.json。')
add('\n## 8. 复现文件\n')
for x in ['audit/v24_preregistered_plan.json','audit/v24_frozen_selection.json','audit/v24_data_isolation.json','audit/v24_environment.json','results/v24_final_analysis.json','results/v24_quality_table.csv','figures/v24_results.png','configs/v24_formal_timing_jobs.json','scripts/run_v24_ladder_and_eval.py','scripts/run_v24_diagnostics_wait.py']:add('- '+x)
add('\n## 白话版\n')
add('这一轮让读取器直接模仿模型最终给各标签打分，而不只模仿中间层。能否扩大共享，仍取决于未参与校准的新问题上能否稳定保住质量。少存几份被选KV、减少冷恢复耗时属于另一组证据；更快不能替代质量验收。query和生成token的KV始终每层各自保存。')
(r/'EXPERIMENT_REPORT_V24_ZH.md').write_text('\n'.join(out)+'\n')
(r/'README_LAYERKV_V24_SERVER.md').write_text('# v2.4 server experiment\n\nSee EXPERIMENT_REPORT_V24_ZH.md and results/v24_final_analysis.json. All relative paths are under /home/panzihang/src/prism_layerkv. Original prism_pans and previous implementation files are unchanged.\n\nScripts/configs retain exact commands. Outputs refuse overwrite; reruns need fresh output paths. Use PYTHONPATH=src:runtime_deps_v24 and the existing vllm-stable Python. Raw profiler captures remain in profiles/v24. Delivery is an additive code/results overlay excluding weights, bulk KV and raw profiler traces.\n')
print('Report written')
