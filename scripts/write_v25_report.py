
from pathlib import Path
import json,re
r=Path(__file__).resolve().parents[1];d=json.loads((r/'results/v25_final_analysis.json').read_text());G=d['frozen_selection']['chosen_G'];L=[]
def add(s=''):L.append(s)
def table(headers,rows):
 add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
 for row in rows:add('| '+' | '.join(map(str,row))+' |')
 add()
def f(x):return f'{x:.2f}'
add('# 跨层共享 prefix KV v2.5：范围扩展与原生块保留实验')
add()
add(f'已完成 G=26、24、22、20、16、14 × R=0/4 的全部 12 个配置；最广配置有 14 个消费者读取跨层共享 prefix KV。按预先写定的开发集规则选择 G{G}，保留 R0/R4 两条路线，分别训练种子 24、25、26，再评估从未参与本轮选择的 1536 条留出数据。')
add()
fb=d['quality']['final']['identity']['basic'];fr0=d['quality']['final'][f'g{G}_r0_s24']['basic'];fr4=d['quality']['final'][f'g{G}_r4_s24']['basic'];res=d['resource'][f'g{G}_r4']
add(f'固定种子24的新留出准确率：无共享 {f(fb["accuracy_pct"])}%，R0 {f(fr0["accuracy_pct"])}%，R4 {f(fr4["accuracy_pct"])}%。R4 实际 selected prefix KV 减少 {f(100*res["selected_payload_reduction_fraction"])}%，计入 reader/packed 后这部分净减少 {f(res["selected_plus_readers_net_reduction_pct"])}%。')
add()
a0=[d['quality']['final'][f'g{G}_r0_s{s}']['basic']['accuracy_pct'] for s in [24,25,26]]
a4=[d['quality']['final'][f'g{G}_r4_s{s}']['basic']['accuracy_pct'] for s in [24,25,26]]
add(f'R0/R4 三种子准确率均值分别为 {f(sum(a0)/3)}% / {f(sum(a4)/3)}%；R4 相对同种子 R0 的差为 '+', '.join(f(y-x)+'pp' for x,y in zip(a0,a4))+'。这些是同一留出题集上的三次训练结果，不能按4608个独立问题计算置信区间。')
add()
tb=d['quality']['test500']['identity']['basic'];ts=d['quality']['test500'][f'g{G}_r4_s24']['basic']
add(f'历史完整 TREC 500 条，固定种子24：无共享 {tb["correct"]}/500（{f(tb["accuracy_pct"])}%），G{G}/R4 {ts["correct"]}/500（{f(ts["accuracy_pct"])}%）。')
add()
for policy in ['native','preserve']:
 tm=d['selected_timing'][policy]['r4'];lo,hi=tm['block_bootstrap_relative95_pct']
 add(f'SSD-cold/{policy}，R4：{f(tm["baseline_mean_ms"])} → {f(tm["shared_mean_ms"])} ms，变化 {f(tm["relative_change_pct"])}%，探索性块区间 [{f(lo)}, {f(hi)}]%；4轮重复同16题，不能视为完整数据集的独立请求统计。')
add()
passes=[R for R in [0,4] if d['certification'][f'r{R}']['all_three_fixed_seeds_pass']]
add('独立留出集三种子非劣验收：'+('、'.join(f'R={R}' for R in passes)+' 通过。' if passes else '**两条路线均未做到三个固定种子全部通过。**')+' 所有 checkpoint 仍标注 diagnostic，不自动替换原系统。v2.4 的验收失败不被本轮覆盖。')
add()
add('## 1. 范围、实现与可比性')
add()
add('Qwen2.5-7B-Instruct / RTX 3090，28 层，28 个 Q head、4 个 KV head、head dim 128；prefix 4940 token，块大小 16，预算 25%，P8 周期 8。主干冻结。各层正常产生 Q；消费者通过本层 Q 变换读取源层 prefix KV，再对聚合输出做本层变换。当前 query 与生成 token 的 KV 始终逐层独立，28 层计算均保留。')
add()
add('R=4：只从该层原本已选中的物理块中，把最后 4 块划为本层原生保留区；剩余部分走共享读取器。短末块使实际保留长度通常为 60 token，不能描述成新增 64 token 的滑动窗口。共享前段通过源缓冲区前缀视图读取，本层小块与私有 query/tail 拼接，仍由两个 attention 分支联合归一化；没有重建完整消费者 prefix KV。缓存准入仅接收消费者的小块 IDs。')
add()
add('实现为附加 v25 模块；v24 的 FA2/LSE-aware 反向、实际候选续写评分损失、推理内核和冷状态判定均保留。安装时 11 个基础接口哈希匹配；运行后再次核对。旧版→新版 identity 和旧 G26 读取器桥接各 8 条的首 token 全词表 logits、标签逐 token logprob、选择计划均精确一致。R4 实际在线与训练回放的初始 8 条冒烟验证：token 误差为 0，平均标签分数最大差约 4.77e-7。')
add()
graphs=[]
for g in [26,24,22,20,16,14]:
 p=json.loads((r/f'configs/v25_explore/g{g}.json').read_text());s=p['sources'];groups={}
 for i,src in enumerate(s):
  if i!=src:groups.setdefault(src,[]).append(i)
 graphs.append([g,28-g,'；'.join(f'{k}→'+','.join(map(str,v)) for k,v in groups.items())])
table(['完整来源数 G','共享消费者数','实际 source→consumer（0-based）'],graphs)
add('G14 涉及 7 组三层共享，共 21 层参与；其余 7 层原生读取。它不是所有层读取单一 KV。R4 还存在 14 份小型原生保留区，完整来源数 G 也不是总恢复调用次数。')
add()
add('## 2. 数据、训练与事先选择规则')
add()
add('拟合 64 条、校准验证 64 条复用既有 teacher；开发 768 条为此前已观察数据，仅做研究筛选。所有图/R/种子使用同一个冻结 v24 seed24 初始化，200 步、每 25 步校准验证；按相同目标选最佳步数，不以真实最终留出集选模型。R0/R4 的初始张量哈希逐一相同。')
add()
add('完整 TREC test500 是历史回归集。新 final1536 从 TREC train5500 未使用余集中固定种子抽取，排除旧训练/验证/开发/留出、prefix 示例及 test500 的规范化文本重复，模型 token hash 再查重。它是同任务新留出样本，不是新的官方 benchmark。每条数据只在质量统计中计一次，重复计时不扩大质量样本量。另附历史 test500 排除既有 5 个 prefix 重叠题后的 strict495。')
add()
sel=d['frozen_selection']
add(f'选择规则：G24/22/20/16/14 中，只要同一 G 的任一 R 在开发集准确率距 contemporaneous identity 不低于 1 个百分点，就进入候选，选其中最小 G；若全部不满足，固定回退到 G24 诊断。实际选择 G{G}，fallback={sel["fallback_used"]}。该点估计筛选不是非劣验收。两条 R 均保留，补种子 25/26，6 个 checkpoint 的 SHA 在 final1536 前冻结。')
add()
add('最终非劣容许差为 −1pp；每条 R 使用 alpha=0.025 的保守有害/有益翻转二项界，两条路线做 Bonferroni，且各自要求三个固定种子全部通过。没有挑选最好种子；开发网格和历史500的区间只作描述。')
add()
add('校准回放使用教师的固定选择计划；实际质量评估使用在线 P8 选择器。共享改变后续层输入后，也可能改变后续选择结果。配对 JSON 中的 selection_plan_exact_agreement 是完整计划哈希一致率，不是 token Jaccard，不能据此直接推断 token 重合比例。')
add()
add('## 3. 全覆盖质量曲线')
add()
qr=[]
for g in [26,24,22,20,16,14]:
 for R in [0,4]:
  a=d['quality']['dev'][f'g{g}_r{R}_s24'];b=d['quality']['test500'][f'g{g}_r{R}_s24']
  qr.append([g,R,f(a['basic']['accuracy_pct']),f(b['basic']['accuracy_pct']),b['basic']['correct'],f(b['paired_vs_identity']['net_delta_pp']),format(a['paired_vs_identity']['online_score_fidelity']['candidate_kl'],'.5f')])
add(f'无共享：开发 {f(d["quality"]["dev"]["identity"]["basic"]["accuracy_pct"])}%，历史500 {f(d["quality"]["test500"]["identity"]["basic"]["accuracy_pct"])}%。以下均 seed24。')
add()
table(['G','R','开发768准确率%','历史500准确率%','历史正确数','历史相对差pp','开发候选KL'],qr)
add('候选 KL 是完整标签续写平均 logprob 经 softmax 构造的辅助分布，既不是首 token 全词表 KL，也不是校准后的类别概率。局部校准误差降低不能替代上述真实分类结果。所有扩大范围配置均如实保留，没有因掉点删除。')
add()
add('## 4. 新留出 1536：固定三种子')
add()
b=d['quality']['final']['identity']['basic'];add(f'无共享正确 {b["correct"]}/1536，准确率 {f(b["accuracy_pct"])}%。')
add()
rr=[]
for R in [0,4]:
 for s in [24,25,26]:
  q=d['quality']['final'][f'g{G}_r{R}_s{s}'];p=q['paired_vs_identity']
  rr.append([R,s,q['basic']['correct'],f(q['basic']['accuracy_pct']),p['harmful'],p['beneficial'],f(p['net_delta_pp']),f(p['conservative_one_sided_lower_pp']),'通过' if p['conservative_bound_passes'] else '未通过'])
table(['R','种子','正确数','准确率%','有害翻转','有益翻转','差pp','保守单侧下界pp','−1pp非劣'],rr)
add('同一固定种子内，R4 相对 R0 的描述性对照如下；不额外宣称经过多重比较修正的优效验收。')
add()
table(['种子','R4相对R0准确率差pp','R0正确→R4错误','R0错误→R4正确'],[[s,f(d['final_R4_vs_R0'][str(s)]['net_delta_pp']),d['final_R4_vs_R0'][str(s)]['harmful'],d['final_R4_vs_R0'][str(s)]['beneficial']] for s in [24,25,26]])
add('本轮质量结论只覆盖 TREC 完整标签续写分类评分；多 token 私有 KV 审计验证运行语义，不等于自由生成质量评测。')
add()
add('## 5. 实际资源：完整源 + 原生保留区 + 读取器')
add()
rs=[]
for g in [26,24,22,20,16,14]:
 for R in [0,4]:
  x=d['resource'][f'g{g}_r{R}'];rs.append([g,R,f(x['shared_base_payload_bytes']/2**20),f(x['native_repair_payload_bytes']/2**20),f(100*x['selected_payload_reduction_fraction']),f((x['resident_reader_bytes']+x['packed_reader_extra_storage_bytes'])/2**20),f(x['selected_plus_readers_net_reduction_pct']),int(x['total_resolve_calls'])])
table(['G','R','源KV MiB','保留KV MiB','selected减少%','reader含packed MiB','selected+reader净减少%','恢复调用数'],rs)
add('口径是请求 selected 缓冲区。净减少另计常驻/packed reader；不包含主干权重、索引/selector、allocator保留、临时小块拼接和私有KV。持久化原始KV未删除，不能据此声称模型总显存或磁盘占用下降相同比例。R4 小缓冲区占用与恢复开销已实测计入；实际CPU/SSD/H2D流量另见 profiler，不等于这张逻辑/物理缓冲表。')
add()
add('## 6. TTFT：顺序配对、认证冷缓存')
add()
add('每个正式 arm 使用新进程，模型/内核先预热 8 次；冷请求前只清理本轮自有目标 prefix cache，并通过原控制器检查 GPU/CPU payload 空、无未完成请求、文件页 mincore 全不驻留，再计时到首 token logits 就绪。这里认证的是应用缓存和 OS 文件页冷态，不包含 SSD 控制器内部缓存。冷状态准备时间不计入 TTFT，模型启动不计入 TTFT。未清空全系统 page cache，也未触碰其他用户 GPU 作业。质量并行运行的 TTFT 不用于速度结论。')
add()
add('preserve 指受测文件读取时不主动追加原 native 路径的文件缓存驱逐；native 保留原系统 advice 行为。二者都在每个受测请求开始前认证目标文件页全冷，不能把 preserve 当成暖缓存。')
add()
tr=[]
for g in [26,24,22,20,16,14]:
 for R in [0,4]:
  x=d['curve_preserve'][f'g{g}_r{R}'];tr.append([g,R,f(x['baseline_mean_ms']),f(x['shared_mean_ms']),f(x['relative_change_pct'])])
table(['G','R','无共享ms','共享ms','变化%'],tr)
add('上表仅 2 个整网格正序/倒序块，每块同一组 16 题，是覆盖曲线诊断。两个块不足以给出强稳定性结论；不把 32 次重复请求当成 32 个独立质量样本。')
add()
tr=[]
for policy in ['preserve','native','warm']:
 for R in [0,4]:
  x=d['selected_timing'][policy][f'r{R}'];lo,hi=x['block_bootstrap_relative95_pct']
  tr.append([policy,R,f(x['baseline_mean_ms']),f(x['shared_mean_ms']),f(x['relative_change_pct']),f'[{f(lo)}, {f(hi)}]'])
table([f'选择 G{G} 的场景','R','无共享ms','共享ms','变化%','块bootstrap 95%范围%'],tr)
add('负值为降低。各场景 4 个顺序平衡块，冷每块16题、暖每块32题，主计时固定seed24。selected preserve 的前两块来自整网格、后两块是选定三臂，因此时间段和调度范围有差异。仅4块的重复查询区间是探索性区间；应连同 JSON 中每块的方向检查，不宣称覆盖所有工作负载的稳定收益。')
add()
add(f'正式冷状态认证：{sum(x["n"] for x in d["cold_certification"].values())} 个请求全部通过；每个首decoder阶段 process write_bytes=0。冷准备开销与进程 read_bytes 逐臂保存在结果 JSON。')
add()
add('## 7. Nsight / VizTracer / 恢复路径')
add()
nr=[]
for name,v in d['nsight']['variants'].items():
 x=v['means'];nr.append([name,v['n'],f(x['ttft_profiled_ms']),f(x['h2d_bytes']/1e6),f(x['h2d_sum_ms']),f(x['gpu_busy_union_ms'])])
table(['场景','请求数','profiled TTFT ms','H2D MB','H2D累计ms','GPU busy union ms'],nr)
add('这里仅 8 条 profiler 请求，范围为准确 TTFT/logits_ready NVTX；H2D 含 payload、选择索引及元数据。GPU累计时间、API等待、NVTX嵌套区间可能重叠，不能直接相加构造关键路径。Profiler会扰动执行，不能替代上一节正式计时。')
add()
ni=d['nsight']['variants']['nsys_native_identity']['means'];ns=d['nsight']['variants']['nsys_native_shared']['means']
wi=d['nsight']['variants']['nsys_warm_identity']['means'];ws=d['nsight']['variants']['nsys_warm_shared']['means']
add(f'冷态硬件 H2D 字节减少 {f(100*(1-ns["h2d_bytes"]/ni["h2d_bytes"]))}%，暖态减少 {f(100*(1-ws["h2d_bytes"]/wi["h2d_bytes"]))}%。冷态 H2D 累计时间只减少 {f(ni["h2d_sum_ms"]-ns["h2d_sum_ms"])} ms，远小于正式 native TTFT 的平均降幅 {f(d["selected_timing"]["native"]["r4"]["baseline_mean_ms"]-d["selected_timing"]["native"]["r4"]["shared_mean_ms"])} ms。')
add()
add('结合恢复路径的文件访问减少，这支持主机数据恢复、缓存管理和等待是主要收益来源；不能把 H2D 省时单独解释成全部加速。Profiler 中 GPU busy union 未见同量级变化。由于各阶段有重叠且 profiler 样本较小，这属于有实测支撑的机制解释，不是精确关键路径分解。')
add()
for key in ['native_identity','native_shared','warm_identity','warm_shared']:
 add(f'VizTracer {key} 主线程 self 时间前五项（每次首次 prefill，ms）：')
 add()
 for x in d['viz_'+key]['main_thread_by_self'][:5]:add(f'- `{x["function"].split(" (")[0]}`：{f(x["self_ms_per_prefill"])}')
 add()
add('VizTracer self 包含未追踪 C/库函数和阻塞等待，各线程重叠；所有4个trace均完整、无overflow。SSD prefetch/promotion 的调用、文件访问与耗时诊断另存 results/v25_final_analysis.json 的 restore_diagnostic，不能把嵌套耗时相加。')
add()
restore_rows=[]
for name,x in d['restore_diagnostic'].items():
 restore_rows.append([name,x['n'],int(x['prefetch']['unique_files']),int(x['promotion']['unique_files']),f(x['prefetch']['declared_file_bytes_requested']/1e6)])
table(['恢复策略/配置','请求数','预取文件数/请求','晋升文件数/请求','每路径请求文件内容 MB'],restore_rows)
add('同一文件可能在预取和晋升两条路径重复访问。表内文件内容字节数是调用所请求的文件大小，不能当成硬件 SSD 读取量或 H2D 量；实际进程 read_bytes 和 CUPTI 传输另行记录。文件访问与缓存管理减少为延迟收益提供了直接机制证据，但各阶段仍可能重叠。')
add()
add('## 8. 训练成本与摊销')
add()
tt=[]
for name,x in d['training'].items():
 tt.append([name,x['best_step'],f(x['wall_seconds_total_including_load']),f(x['cuda_peak_allocated']/2**30),format(x['validation']['candidate_kl'],'.5f')])
table(['配置','最佳步','总墙钟秒（含加载/验证）','峰值allocated GiB','校准候选KL'],tt)
for R in [0,4]:
 x=d['amortization_training_only'][f'r{R}'];v=x['queries_with_parent_training_only']
 add(f'G{G} R{R}：用 observed SSD-cold/preserve 平均差估算，当前训练加继承父reader训练的摊销请求数 '+(f'约 {v:.0f}' if v is not None else '不可给出正收益摊销点')+'。')
add()
add('以上仅训练成本示意，不含教师采集、12配置搜索、数据准备、额外种子与选择成本，不是总成本 break-even 保证；速度差存在有限样本不确定性。14B 未运行：本轮单卡24GiB、BF16权重与当前执行器不具备适配后的可比14B路径。')
add()
add('## 9. 审计、测试和复现')
add()
tests=(r/'logs/v25_all_gpu_tests.log').read_text(errors='replace');tail=[x for x in tests.splitlines() if ' passed' in x and (' in ' in x or 'skipped' in x)]
add('完整 GPU 回归：'+(tail[-1] if tail else '详见 logs/v25_all_gpu_tests.log')+'。')
add()
inv=d['invariants'];add(f'重新核对 {inv["prior_files_checked"]} 个既有受保护文件，变更数 0；原 prism_pans 目录没有本轮修改。实际在线审计覆盖源指针视图、R4原生值与存储一致、支持集不增不减、缓存准入仅小块、28层私有KV独立并生成多token。另有 G14/R4 单请求预检：14 个消费者、532 次 attention 调用、19 次 decoder 调用全部通过，私有缓存增长 15 token。最终两个R的在线/回放评分误差与采样统计见 audit/v25_final_invariants.json。')
add()
add('全部工作位于服务器 /home/panzihang/src/prism_layerkv。入口与复现实验参数位于 scripts/run_v25_experiment_pipeline.py、scripts/run_v25_timing.py、scripts/run_v25_diagnostics_wait.py 和 configs/v25*；数据隔离、事先规则、选择与冻结SHA位于 audit/v25*。结果逐题记录位于 results/v25_*，6个图/R的seed24及选中结构额外种子位于 assets/v25_*。')
add()
add('![覆盖率、质量和延迟曲线](figures/v25_coverage_quality_latency.png)')
add()
add('可交付压缩包 artifacts/pans_sharedkv_v25_server_impl.zip 含代码、配置、读取器、逐题结果、审计、报告和图；不包含主干模型、体积较大的KV与原始profile；重跑依赖既有 runtime_data/prefix_store、assets/selector_index_k4_g32 和 audit/v25_software_environment.json 记录的软件环境。原始Nsight/VizTracer保存在服务器 profiles/v25，清单在 audit/v25_raw_profile_inventory.json。ZIP每个成员SHA与整包SHA见 audit/v25_archive_validation.json / v25_delivery_manifest.json。')
add()
add('## 白话版')
add()
add('这次真的把共享范围扩大了：最广时，原来28份完整prefix KV只需要14份，但每层还会正常计算自己的Q和后续token KV。R4是在已有选择里留一点本层“原装记忆”，试图用小成本补偿共享误差。省下多少数据、掉不掉准确率、冷启动快不快是三件事，上面的表分别给出实测。没有通过独立三种子验收的配置，仍只是研究结果，不能称作质量保持的完成方案。')
(r/'EXPERIMENT_REPORT_V25_ZH.md').write_text('\n'.join(L)+'\n')
(r/'README_LAYERKV_V25_SERVER.md').write_text('# v2.5 server overlay\n\nSee EXPERIMENT_REPORT_V25_ZH.md and audit/v25_preregistered_plan.json.\n\nServer root: /home/panzihang/src/prism_layerkv. Original prism_pans and previous versions are protected.\nAll checkpoints remain diagnostic. Reproduction commands are recorded in configs/v25*_command.json and configs/v25*_jobs.json.\nUse fresh output names and owned cache copies; orchestration refuses overwriting existing artifacts.\nNo model weights, bulk KV, or raw profiles are included in the delivery ZIP.\n')
print('WROTE EXPERIMENT_REPORT_V25_ZH.md')
