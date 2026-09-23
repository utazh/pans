
from v26_common import *
r=ROOT
analysis=json.loads((r/'results/v26_matrix_analysis.json').read_text());rows=analysis['matrix']
primary=json.loads((r/'results/v26_primary_analysis.json').read_text())
mech=json.loads((r/'results/v26_mechanism_analysis.json').read_text())
audit=json.loads((r/'audit/v26_final_invariants.json').read_text())
hardware=json.loads((r/'results/v26_nsight_summary.json').read_text())
lines=['# v2.6 服务器实验报告','',
'本轮冻结 G16/R4、seed24，验证预取复用晋升，完成四任务×四KV预算及第二前缀复核。G16指16个完整prefix KV来源，12个消费者各保留最后4个已选原生块；28层的Q、当前query/生成KV、残差和MLP均保留。',
'全部产物位于服务器 /home/panzihang/src/prism_layerkv。原 prism_pans 与旧版本受保护文件未修改。',
'',
'## 实验设计','',
'- 主矩阵：SST-2、SUBJ、TREC、RTE，预算5%、10%、25%、50%；第二份独立示例前缀分别复核25%。',
'- 准确率：完整可评估划分 SST-2 dev 872、SUBJ修正后冻结自定义划分1003、TREC test 500、RTE dev 277。历史回归不能称为新最终留出集。',
'- 各任务/预算/前缀分别采集64 fit +64 validation无共享教师输出，冻结主干，200步读取器校准，seed24，每25步按预定validation目标选checkpoint。TREC/p0/25%复用冻结的v25主checkpoint。新采集的校准query限于240 tokens以内以适配256-token训练上限，完整评估不按长度裁剪；历史TREC初始化与新单元从prefix仿射拟合起步的差别保留。',
'- I/S采用相同在线P8选择政策，不强制两个模型实际选中相同token；source和consumer在组内必须位置一致。选择反馈属于本轮端到端结果，未单独隔离。',
'- 同P8窗口只有选择位置完全一致的层共享。原映射不兼容整数预算时，按预登记几何规则确定G16来源；没有按评估质量搜索图，没有修改选择器或原精确预算分配器。各单元映射见configs/v26_*_plan.json。',
'- 主计时：TREC/p0/25%，8个不重叠块×16请求，四臂位置平衡，每臂独立进程；SSD-cold/preserve与普通缓存各32臂。模型、读取器、K4索引预热，冷请求的目标payload应用缓存及文件页清空。',
'- 矩阵计时：每单元同一批128条请求，I-reuse对S-reuse，冷/普通各一对独立进程；跨单元交替臂顺序。这是描述性外推，不能称为每个单元均已取得顺序无关的稳定加速。',
'- 正式计时与本任务训练/质量评估隔离；GPU0/1外部负载未受控制。',
'',
'## SUBJ数据修正说明','','旧读取器用latin-1加str.splitlines()，把21个0x85字节误作换行。原始两文件按物理LF各5000行。补充修正将2条fit、1条validation、2条完整评估碎片接回唯一原始完整句子，保持原有UID、标签、前缀和划分成员，不按模型成绩筛样；全部角色重新核对无交叉。SUBJ五个读取器重新采集教师并从头校准，主表使用修正版本。自定义评估仍是1003个完整句子，不能称为标准1000条划分。原结果及原文件保留，详见audit/v26_subj_canonical_correction.json。','','## 四任务×四预算结果','',
'I/S均使用通过审计的reuse-prefetch晋升。准确率使用完整标签续写平均token log-probability，TTFT以首token logits ready为终点。',
'',
'| 任务 | 预算 | N | I准确率 | S准确率 | 差值pp | 冷 I→S ms | 冷变化 | 普通 I→S ms | 普通变化 |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
mainrows=[x for x in rows if x['prefix']=='p0'];negative=sum(x['delta_pp']<0 for x in mainrows)
cold=primary['preserve/shared/reuse'];warm=primary['warm/shared/reuse']
clo,chi=cold['block_bootstrap95_pct'];wlo,whi=warm['block_bootstrap95_pct']
headline=[f"主质量矩阵16格中{negative}格准确率下降，当前结果不支持跨任务/预算普遍无损。冻结TREC/p0/25%为68.60%→73.00%，其余完整结果见下表。",f"主配置在相同reuse-prefetch晋升下，SSD-cold/preserve TTFT变化{cold['delta_pct']:+.2f}%（块bootstrap95%区间[{clo:+.2f}%, {chi:+.2f}%]）；普通缓存变化{warm['delta_pct']:+.2f}%（[{wlo:+.2f}%, {whi:+.2f}%]）。此处是TREC/25%四臂主研究，不能代替每个矩阵单元的独立重复进程验证。"]
lines[2:2]=headline+['']
for x in rows:
 if x['prefix']!='p0':continue
 lines.append(f"| {x['task']} | {x['budget']}% | {x['n']} | {x['identity_accuracy']:.2f}% | {x['shared_accuracy']:.2f}% | {x['delta_pp']:+.2f} | {x['preserve_identity_ms']:.2f}→{x['preserve_shared_ms']:.2f} | {x['preserve_delta_pct']:+.2f}% | {x['warm_identity_ms']:.2f}→{x['warm_shared_ms']:.2f} | {x['warm_delta_pct']:+.2f}% |")
lines+=['','## 第二份独立前缀，25%预算','','| 任务 | I准确率 | S准确率 | 差值pp | 冷变化 | 普通变化 |','|---|---:|---:|---:|---:|---:|']
for x in rows:
 if x['prefix']=='p1':lines.append(f"| {x['task']} | {x['identity_accuracy']:.2f}% | {x['shared_accuracy']:.2f}% | {x['delta_pp']:+.2f} | {x['preserve_delta_pct']:+.2f}% | {x['warm_delta_pct']:+.2f}% |")
lines+=['','两前缀示例独立且与校准/评估隔离。第二前缀没有用于修改第一前缀读取器。不同前缀长度可能触发几何适配，因此该比较不是纯内容单变量实验。所有有害/有益翻转、保守界见v26_matrix_analysis.json；多单元历史回归不能替代新的预登记最终认证，v25非劣失败结论保留。',
'','## 缓存晋升四臂主计时','','| 条件/对照 | 基准 ms | 比较 ms | 变化 | 块bootstrap95%区间 |','|---|---:|---:|---:|---|']
for key,x in primary.items():
 if 'baseline_ms' in x:
  lo,hi=x['block_bootstrap95_pct'];lines.append(f"| {key} | {x['baseline_ms']:.2f} | {x['student_ms']:.2f} | {x['delta_pct']:+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] |")
lines+=['','shared/*比较同晋升策略下I→S；admission/I或S比较同模型原晋升→复用晋升。区间以8个运行块重采样，不把块内请求当成独立进程重复；时间块仍可能相关，因此区间属于探索性不确定性估计。晋升优化同时用于I和S，共享贡献比较同晋升策略下I/S。更完整晋升差值与顺序分层见JSON。',
'','## 固定checkpoint机制消融','','| 读取器/补丁 | 正确/N | 准确率 | 相对I差值pp |','|---|---:|---:|---:|']
for name,v in mech.items():
 q=v['vs_identity'];lines.append(f"| {name} | {q['student_correct']}/{q['n_unique_queries']} | {100*q['student_correct']/q['n_unique_queries']:.2f}% | {q['net_delta_pp']:+.2f} |")
lines+=['','unit_native为相同图和R4、单位A/B和零偏置的直接共享控制，不是完整KVSharer。learned_proxy仍加载原生补丁，仅把小块内容换成source仿射预测，不能用于速度结论；BF16与FP32参考差异见audit/v26_proxy_rounding.json，不能等同于重训R0。',
'','## 资源与离线成本','','| 前缀/任务/预算 | 选中payload减少 | 加reader与packed后净减少 | 构建秒 | 冷TTFT估算回本请求数 |','|---|---:|---:|---:|---:|']
for x in rows:
 cost='历史复用' if x['reader_build_seconds'] is None else f"{x['reader_build_seconds']:.1f}"
 pay='不适用' if x['cold_payback_requests'] is None else str(x['cold_payback_requests'])
 lines.append(f"| {x['prefix']}/{x['task']}/{x['budget']}% | {x['selected_kv_reduction_pct']:.2f}% | {x['net_selected_plus_reader_reduction_pct']:.2f}% | {cost} | {pay} |")
lines+=['','上述比例是所选prefix KV与读取器子系统字节数，不是模型总显存或持久SSD空间减少；完整28层KV仍保留。回本数仅用本轮教师采集+拟合/训练/验证成本除以描述性冷TTFT节省，不含研发搜索，不是吞吐保证。公共prefix构建两方法均需要；p1实际构建与索引成本另有日志。',
'','## Nsight','','| 臂 | H2D MB/请求 | H2D ms/请求 | GPU busy ms/请求 |','|---|---:|---:|---:|']
for name,v in hardware['variants'].items():
 x=v['means'];lines.append(f"| {name} | {x['h2d_bytes']/1e6:.3f} | {x['h2d_sum_ms']:.3f} | {x['gpu_busy_union_ms']:.3f} |")
lines+=['','每臂8请求，统计同步完成的TTFT NVTX区间，排除加载、warmup、标签评分。H2D包含payload、索引、元数据。NVTX/Python/API/kernel会重叠，不能相加当关键路径。VizTracer轨迹保留于profiles/v26。',
'','## 验证与复现','',
'- 完整回归：601 passed、4 skipped、87 subtests；既有readonly NumPy warning保留，见logs/v26_full_gpu_tests.log。',
'- 晋升GPU审计80请求含warmup：完整首token词表logits、全部候选token分数、ID、驻留内容、device_map、heap顺序及容量完全一致；晋升块独立storage，28层private KV独立。32个正式冷请求认证通过。',
f"- {audit['protected_files_unchanged']}个原有文件及v25归档保持不变；两个p0缓存副本逐文件校验一致且无硬链接，见audit/v26_final_invariants.json。",
'- 审计脚本的两次修复、新前缀完成标记补全及失败日志保留。标记补全时校验每块几何，索引构建再次逐层比较实际FP16源与BF16 prefix转换结果。',
'- 仅验证Qwen2.5-7B、单请求和当前四任务/预算；未验证14B，未实现完整外部KVSharer/xKV基线。',
'- 调度入口：scripts/v26_training_worker.py、v26_gpu3_sequence2.py、v26_quality_worker.py、v26_formal_sequence3.py、v26_diagnostics_sequence3.py。命令、SHA和拆分在configs、audit、data/v26，禁止覆盖已有输出。最终服务参数另记于*.effective_cli.json；继承的内层command.json仅是覆盖前模板。',
'- 汇总文件：results/v26_matrix.csv、v26_matrix_analysis.json、v26_primary_analysis.json、v26_mechanism_analysis.json、v26_offline_costs.json、v26_nsight_summary.json。'
]
lines+=['','## 本轮可追踪实验运行成本','','| 阶段 | 运行数 | 进程墙钟秒之和 |','|---|---:|---:|']
for stage,v in json.loads((r/'results/v26_run_cost_summary.json').read_text())['categories'].items():
 lines.append(f"| {stage} | {v['runs']} | {v['process_wall_seconds']:.1f} |")
lines+=['','此表包含被替代的旧SUBJ采集/训练和历史评估，避免只统计成功主模型。墙钟包含模型加载、CPU等待；并行任务时间重叠，不能等同于总历时或GPU kernel时间。未覆盖人工准备、旧版本研发、部分设置和审计。上面的单prefix部署构建成本与本表的实验验证成本分别统计。']
lines+=['','## SUBJ历史读取器与修正版本对照','','第二前缀25%预算，原历史口径与纠正数据并重新校准后的结果；同时涉及2条评估句子修复和3条校准/验证修复，不能解释成单纯推理消融。','','| 臂 | 历史正确/N | 修正正确/N |','|---|---:|---:|']
for arm,v in json.loads((r/'results/v26_subj_parser_sensitivity.json').read_text()).items():
 q=v['comparison'];lines.append(f"| {arm} | {q['baseline_correct']}/{q['n_unique_queries']} | {q['student_correct']}/{q['n_unique_queries']} |")
lines+=['','旧/新无共享在其余1001条未改动查询上的全部候选token分数完全一致；共享checkpoint重新校准后则改变。该观察支持数据修正与重新校准过程会显著影响结果，但没有单独隔离训练数值波动。完整1003条的修正前后表只比较描述性总数；配对统计另限于未改动的1001条。']
lines+=['','## 各单元实际共享结构','','G16固定的是完整来源数，整数块预算不兼容时来源图按预登记规则适配。因此跨预算比较同时包含选择预算与几何适配，不能称为只改变预算的单变量消融。层编号均从0开始。','','| 单元 | source→consumers | 参与层数 | 最大层间距离 |','|---|---|---:|---:|']
for x in rows:
 tag=f"v26_{x['prefix']}_{x['task']}_b{x['budget']}"+('_canonical' if x['task']=='subj' else '');groups=analysis['detail'][tag]['shared_groups'];mapping='; '.join(k+'→'+','.join(map(str,v)) for k,v in groups.items())
 lines.append(f"| {x['prefix']}/{x['task']}/{x['budget']}% | {mapping} | {x['participating_layers']} | {x['max_source_distance']} |")
lines+=['','## VizTracer主线程诊断','']
for p in sorted((r/'results').glob('v26_viz_*_summary.json')):
 v=json.loads(p.read_text());lines+=['### '+p.stem,'','| 函数 | self ms/首轮prefill |','|---|---:|']
 for row in v['main_thread_by_self'][:6]:lines.append('| '+row['function']+' | '+f"{row['self_ms_per_prefill']:.3f}"+' |')
lines+=['','self时间包含未追踪的C/库调用和阻塞等待，不等于CPU计算时间。不同线程并行，不能相加得到关键路径。']

lines+=['','## 结果边界与解释','']
low=[x for x in rows if x['net_selected_plus_reader_reduction_pct']<0]
if low:
 lines.append('读取器有固定常驻成本，以下单元虽然所选KV减少，但计入reader/packed后该子系统总字节数增加：'+ '；'.join(f"{x['prefix']}/{x['task']}/{x['budget']}%增加{-x['net_selected_plus_reader_reduction_pct']:.2f}%" for x in low)+'。不能用payload减少替代净内存收益。')
for condition in ['preserve','warm']:
 p=primary[condition+'/shared/reuse'];d=[x['delta_pct'] for x in p['per_block']]
 lines.append(f"TREC/25%主研究的{condition}条件下，{sum(v<0 for v in d)}/{len(d)}个块的共享TTFT低于无共享，各块变化范围为[{min(d):+.2f}%, {max(d):+.2f}%]；两种先后顺序均保留，不能推广为所有系统负载下的保证。")
certs=sum(t['timing']['preserve']['I_cold_certified']+t['timing']['preserve']['S_cold_certified'] for t in analysis['detail'].values())
lines.append(f"矩阵SSD-cold/preserve共{certs}个正式请求通过冷状态检查，四臂主研究另有512个；准确率来自完整划分，矩阵TTFT每单元来自128条请求，并非完整划分的冷计时。")
lines.append('复用预取做晋升已通过数值与生命周期审计，但本轮未显示普遍的速度优势；共享在同晋升策略下的收益与晋升改动本身的收益必须分开。跨任务主质量矩阵15/16格下降，当前支持恢复成本收益与质量折中，不支持通用无损或无需适配的结论。')


lines+=['','## 机制结果的解释','','固定G16/R4图下，单位读取器为16.80%，学习读取器为73.00%，说明当前结构依赖学习到的整体读侧补偿；A/B与偏置一起变化，不能分拆归因。source-proxy小块对照为76.80%，反而高于真实原生小块，因此本轮不支持“真实原生内容必不可少”的解释。先前重训R4优于R0还可能涉及训练轨迹或执行差异，此次干预未分别隔离。没有据此改选主模型，也不能外推到其他任务/预算。','','舍入检查只覆盖12个读取器各自首次调用的60-token小块。相同BF16输入和参数下，BF16结果相对FP32运算参考的K相对L2为0.001523～0.002996，V为0.002329～0.002542。没有进行全数据FP32评分干预，不能排除数值路径对分类变化的影响。proxy仍恢复真实原生块，只重建小控制区，不作为速度或资源收益方案。']

lines.append('\n\n## 恢复瓶颈与最终矩阵总结\n\n主矩阵16格冷缓存TTFT全部下降17.87%～42.01%，普通缓存有3格变慢（SST-2/5%、SUBJ/5%、TREC/10%）。矩阵计时每格只有一对独立进程，属于描述性结果；八块四臂主研究另行报告置信区间。准确率15/16格下降，不能把冷恢复收益等同于通用质量保持。\n\nNsight的reuse-prefetch同策略比较中，H2D从76.891 MB降到47.962 MB（约37.62%），累计拷贝时间从3.206 ms降到2.052 ms；GPU busy union从37.442 ms降到36.034 ms，GPU kernel累计时间则从29.508 ms略增到30.645 ms。主机host gather的NVTX包含时间约934.48→527.21 ms；VizTracer主线程wait的self时间约1065.23→640.85 ms。这些诊断与主要减少主机恢复/等待的解释一致，不能把重叠区间相加作因果分解。\n\n晋升复用在无共享臂把硬件H2D从135.103 MB降到76.891 MB，同时新增约58.212 MB的D2D拷贝；它确实改变了晋升数据来源，但注册、复制和等待成本仍在，正式四臂研究没有支持其普遍更快。每个profiler臂仅8请求，诊断延迟不替代正式计时。\n')
with (r/'EXPERIMENT_REPORT_V26_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('REPORT WRITTEN')
