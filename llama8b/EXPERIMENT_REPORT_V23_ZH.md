# v2.3：多消费者共享与完整 TREC 冷缓存实验

全部工作位于 /home/panzihang/src/prism_layerkv；原工程 /home/panzihang/src/prism_pans 未改动。

## 结论

source26同时供消费者25、27读取共享prefix；当前query和生成token的KV仍逐层独立。完整TREC500、25%预算：无共享68.60%，联合共享68.40%。
普通缓存平均变化-15.32%，但运行顺序敏感，未证明顺序无关的稳定加速；逐请求冷缓存：配对结果支持降低TTFT，变化-5.08%。
每请求唯一所选prefix KV从70,647,808 B降至65,617,920 B，减少7.12%，物理加载28→26。GPU reader参数528,384 B，packed额外266,240 B；计入两项后该部分合计净减少约5.99%。这不是模型总显存或持久磁盘文件的减少量。

## 完整500条普通缓存质量及单轮计时

| 配置 | 正确/500 | 准确率 | 单轮平均TTFT ms | P95 ms |
|---|---:|---:|---:|---:|
| 优化后无共享 | 343/500 | 68.60% | 244.56 | 327.57 |
| 独立层校准 | 337/500 | 67.40% | 221.79 | 299.42 |
| 联合，仅query | 341/500 | 68.20% | 208.35 | 280.72 |
| 联合，完整续写 | 342/500 | 68.40% | 215.67 | 272.32 |

联合共享相对基线：有害翻转4条、有益翻转3条；差-0.20个百分点，配对bootstrap95%区间[-1.2, 0.8]；McNemar p=1.0000；完整选择计划hash一致率100.00%。
保守1个百分点非劣判据通过：False。旧局部门限仍失败，所有checkpoint保留diagnostic_only，不能称为质量已认证。
本轮500条与此前175条的准确率分母不同，不能直接拿68.4%与旧72%比较；应使用本轮同500条的无共享对照。
分类沿用mean continuation log-probability。每条候选得分、gold margin和翻转UID均保留，没有拿首token贪心一致率替代任务质量。

## 配对TTFT

| 条件 | 设计 | 无共享ms | 联合共享ms | 变化 | block bootstrap95%区间 |
|---|---|---:|---:|---:|---|
| 普通缓存 | 8轮×同32条，ABBA | 269.15 | 227.92 | -15.32% | [-24.61%,-5.71%] |
| SSD-cold/native | 10批×50条，完整500，顺序平衡 | 2067.87 | 1962.81 | -5.08% | [-7.35%,-2.87%] |

冷顺序敏感性（各5批，仍可能混有query差异）：基线先运行时变化-6.84%，共享先运行时变化-3.28%。完整区间见JSON。
普通缓存顺序敏感性（事后诊断，各4轮）：基线先运行时变化-25.02%，共享先运行时变化-3.98%，后者95%区间[-10.02%, +2.22%]跨零。即使新增四轮和合计八轮均值都下降，仍不能排除顺序/缓存状态/主机调度干扰，不能宣称普通缓存稳定加速。所有轮次均保留。
冷测试每批50条互不重叠，两方案一一配对；不是将重复请求当作500个不同样本。普通缓存8轮仅对同32条重复，run-to-run统计能力有限。完整单轮500与配对32均值可以不同，不能混比。
前4轮变化-15.10%，新进程确认4轮变化-15.50%；额外确认因首轮baseline偏慢而增加，未删除该轮。
TTFT使用原first-logits-ready边界，不含冷准备、标签评分或profiler。GPU/CPU应用cache为55/131MiB，P8在线选择；没有selection replay、logits dumping或shadow load。
主推理使用独立GPU3（RTX3090），服务器GPU0/1存在其他任务，部分批次记录到100%利用率。主机CPU、内存与存储仍为共享资源，结果不是整机独占测量；每job的GPU状态保留于logs/v23_timing_telemetry.jsonl，平衡顺序和配对区间不能排除全部外部干扰。

## 完整冷缓存500条及验收

| 配置 | 准确率 | 准备时间（TTFT外）ms | first-decoder进程read_bytes MB | 零驻留页验收 |
|---|---:|---:|---:|---|
| identity | 68.60% | 4894.21 | 128.866 | 500/500通过 |
| joint | 68.40% | 4871.26 | 123.852 | 500/500通过 |

Pcache副本位于/dev/sda4 SSD，独立inode。每请求通过CKLFU实际驱逐清空应用payload，对17,304个目标文件DONTNEED，mincore验证69,160页全零；失败即中止。没有全局drop_caches或删除payload。
模型、reader、K4、metadata、allocator和kernel预热。“SSD-cold”不代表SSD控制器/NAND内部缓存为空。/proc/self/io仅包围first decoder，不是硬件H2D或NAND直接测量。

## 读取策略控制

vendor每次读时会DONTNEED，故page-hot必须用preserve，并同时跑SSD-cold/preserve。不能将不同策略的差值归因于纯冷热。

| 条件（各8条） | 配置 | TTFT ms | 进程读取MB |
|---|---|---:|---:|
| page-hot/preserve | identity | 960.91 | 0.000 |
| page-hot/preserve | joint | 873.09 | 0.000 |
| ssd-cold/preserve | identity | 1610.57 | 70.648 |
| ssd-cold/preserve | joint | 1481.96 | 65.618 |

## 独立Nsight/VizTracer

Nsight按TTFT/logits_ready精确NVTX范围，每配置8条；不包含加载、预热、标签评分。H2D包括payload、selector和metadata，不是payload-only。profiler耗时不混入配对计时。

| 冷缓存配置 | H2D MB/请求 | H2D累计ms | GPU busy union ms | profiled TTFT ms |
|---|---:|---:|---:|---:|
| nsys_cold_identity | 135.103 | 9.390 | 39.035 | 2785.31 |
| nsys_cold_joint | 130.034 | 9.167 | 39.579 | 2058.25 |

VizTracer cold：258153事件，overflow=False，8个first-prefill。主线程self前五：

- to (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:374)：1158.99 ms/请求
- wait (/usr/lib/python3.10/threading.py:288)：208.06 ms/请求
- uncache (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:109)：98.59 ms/请求
- to (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:829)：42.02 ms/请求
- decoder_logits (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v21/runtime.py:48)：34.83 ms/请求

VizTracer warm：57646事件，overflow=False，8个first-prefill。主线程self前五：

- decoder_logits (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v21/runtime.py:48)：47.56 ms/请求
- wait (/usr/lib/python3.10/threading.py:288)：45.25 ms/请求
- select_promixed_gqa_blocks (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/promixed.py:99)：21.58 ms/请求
- configure_impress_layer (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/flexgen_pcache.py:1331)：19.66 ms/请求
- _logical_segment (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/flexgen_pcache.py:1267)：18.81 ms/请求

Nsight冷缓存8条：H2D字节减少约3.75%，累计拷贝时间只减少约0.223ms，GPU busy约39ms且共享略增。因此，不能把正式测试约105ms的平均收益归因于PCIe拷贝本身。VizTracer共享冷路径中恢复to函数self约1159ms、wait约208ms，指向主机数据恢复与等待；这里只采集了共享路径的VizTracer，尚不能分解两方案的全部时间差。普通缓存则主要是decoder未展开调用、等待、选择器和配置元数据。
self包括未追踪的C/library调用及阻塞等待；不同线程及嵌套区间不能相加为关键路径。

## 校准

仅训练A/B/bk/bv，冻结真实四层attention、残差、norm、MLP；完整query和所有实际候选标签续写。三组同fit64/val64、200步、seed23；仅一个训练种子，小幅消融差异还需多种子复验。独立对照使用teacher层输入，不传播上游reader误差；checkpoint仍按group objective选取。

| 校准 | 完整续写objective | group update RMSE | suffix RMSE |
|---|---:|---:|---:|
| 独立层 | 0.046303 | 0.171750 | 0.103750 |
| 仅query训练，完整续写复评 | 0.051395 | 0.138863 | 0.178402 |
| 联合完整续写 | 0.045939 | 0.167046 | 0.103152 |

query-only复评0步，参数逐项不变，避免把不同验证对象的objective直接比较。原生group回放最大误差0.01249，小于预设0.02，没有放松阈值。
开发集128：无共享99、联合95、独立100、query-only98。因联合配置下降3.125个百分点，没有继续扩至14/7/4物理source。本轮完成的是一source、两消费者的多层组；没有依据test500挑checkpoint或扩拓扑。
128条fit/val实际source/consumer token IDs全部完全一致，没有隐式合并选择集合。

## 数据与回归检查

校准从官方训练集重新抽取，排除prefix示例及测试重复，实际tokenizer query hash无交集。完整官方500测试此前已被观察，是固定历史回归，不是全新泛化测试。旧strict495子集结果也保留。
全套测试：461 passed, 5 skipped, 1 warning, 87 subtests passed in 12.31s。物理alias/private-tail/selector/数据隔离全部通过。561个既有源码配置文件哈希未变，原工程在快照后无文件修改。
普通/冷缓存混合长suite后的首个冷请求另因15,364页残留被拒绝，失败结果保留并排除；冷阶段改在新进程运行，具体残留页持有者未完全定位。
Nsight初次启动因缺少NVTX要求的diagnostic标志被拒绝，没有生成采集报告；修正仅剖析参数后重跑，失败日志另存。
初次冷冒烟因常驻对象持久化标志缺失而中止，失败日志保留、不混入计时。服务器适配先逐字节比对现存文件和内存，完全相同才补齐元数据；不写新payload。另修复同进程launcher补丁恢复、加入明确pcache路径、减少TTFT之外的重复路径解析。

复现说明：README_LAYERKV_V23_SERVER.md；原始参数：configs/v23_main_jobs.json；实际冷阶段：configs/v23_cold_resume_jobs.json；新增普通缓存确认：configs/v23_time_confirm_jobs.json；分析：results/v23_final_analysis.json；审计：audit/v23_final_invariants.json与v23_data_isolation.json；原始profiler：profiles/v23；图：figures/v23_summary.png及.pdf。所有文件均在本服务器目录。

## 白话版

三层共用一份历史资料，两层各带自己的读法转换器，新问题的记忆仍各存各的。完整500题，联合方案比基线少对1题，节省约7.12%的这部分历史KV；仍不能保证永远不掉点。速度要看上面的配对测试与冷热条件，不能只根据省下字节数推断。
