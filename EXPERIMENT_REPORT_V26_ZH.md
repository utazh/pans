# v2.6 服务器实验报告

主质量矩阵16格中15格准确率下降，当前结果不支持跨任务/预算普遍无损。冻结TREC/p0/25%为68.60%→73.00%，其余完整结果见下表。
主配置在相同reuse-prefetch晋升下，SSD-cold/preserve TTFT变化-39.15%（块bootstrap95%区间[-40.03%, -38.17%]）；普通缓存变化-23.00%（[-27.99%, -16.42%]）。此处是TREC/25%四臂主研究，不能代替每个矩阵单元的独立重复进程验证。

本轮冻结 G16/R4、seed24，验证预取复用晋升，完成四任务×四KV预算及第二前缀复核。G16指16个完整prefix KV来源，12个消费者各保留最后4个已选原生块；28层的Q、当前query/生成KV、残差和MLP均保留。
全部产物位于服务器 /home/panzihang/src/prism_layerkv。原 prism_pans 与旧版本受保护文件未修改。

## 实验设计

- 主矩阵：SST-2、SUBJ、TREC、RTE，预算5%、10%、25%、50%；第二份独立示例前缀分别复核25%。
- 准确率：完整可评估划分 SST-2 dev 872、SUBJ修正后冻结自定义划分1003、TREC test 500、RTE dev 277。历史回归不能称为新最终留出集。
- 各任务/预算/前缀分别采集64 fit +64 validation无共享教师输出，冻结主干，200步读取器校准，seed24，每25步按预定validation目标选checkpoint。TREC/p0/25%复用冻结的v25主checkpoint。新采集的校准query限于240 tokens以内以适配256-token训练上限，完整评估不按长度裁剪；历史TREC初始化与新单元从prefix仿射拟合起步的差别保留。
- I/S采用相同在线P8选择政策，不强制两个模型实际选中相同token；source和consumer在组内必须位置一致。选择反馈属于本轮端到端结果，未单独隔离。
- 同P8窗口只有选择位置完全一致的层共享。原映射不兼容整数预算时，按预登记几何规则确定G16来源；没有按评估质量搜索图，没有修改选择器或原精确预算分配器。各单元映射见configs/v26_*_plan.json。
- 主计时：TREC/p0/25%，8个不重叠块×16请求，四臂位置平衡，每臂独立进程；SSD-cold/preserve与普通缓存各32臂。模型、读取器、K4索引预热，冷请求的目标payload应用缓存及文件页清空。
- 矩阵计时：每单元同一批128条请求，I-reuse对S-reuse，冷/普通各一对独立进程；跨单元交替臂顺序。这是描述性外推，不能称为每个单元均已取得顺序无关的稳定加速。
- 正式计时与本任务训练/质量评估隔离；GPU0/1外部负载未受控制。

## SUBJ数据修正说明

旧读取器用latin-1加str.splitlines()，把21个0x85字节误作换行。原始两文件按物理LF各5000行。补充修正将2条fit、1条validation、2条完整评估碎片接回唯一原始完整句子，保持原有UID、标签、前缀和划分成员，不按模型成绩筛样；全部角色重新核对无交叉。SUBJ五个读取器重新采集教师并从头校准，主表使用修正版本。自定义评估仍是1003个完整句子，不能称为标准1000条划分。原结果及原文件保留，详见audit/v26_subj_canonical_correction.json。

## 四任务×四预算结果

I/S均使用通过审计的reuse-prefetch晋升。准确率使用完整标签续写平均token log-probability，TTFT以首token logits ready为终点。

| 任务 | 预算 | N | I准确率 | S准确率 | 差值pp | 冷 I→S ms | 冷变化 | 普通 I→S ms | 普通变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sst2 | 5% | 872 | 92.78% | 90.94% | -1.83 | 351.33→256.13 | -27.10% | 100.83→103.83 | +2.97% |
| sst2 | 10% | 872 | 93.12% | 91.86% | -1.26 | 587.71→417.86 | -28.90% | 130.73→117.96 | -9.77% |
| sst2 | 25% | 872 | 93.81% | 92.43% | -1.38 | 1471.19→898.95 | -38.90% | 223.28→167.98 | -24.76% |
| sst2 | 50% | 872 | 93.81% | 92.78% | -1.03 | 2891.74→1700.59 | -41.19% | 396.42→302.47 | -23.70% |
| subj | 5% | 1003 | 54.54% | 50.95% | -3.59 | 391.43→305.31 | -22.00% | 116.55→117.87 | +1.13% |
| subj | 10% | 1003 | 52.94% | 49.55% | -3.39 | 695.70→489.43 | -29.65% | 144.58→137.38 | -4.98% |
| subj | 25% | 1003 | 55.03% | 51.35% | -3.69 | 1645.44→985.26 | -40.12% | 236.95→220.74 | -6.84% |
| subj | 50% | 1003 | 84.65% | 78.07% | -6.58 | 3113.00→1893.64 | -39.17% | 363.84→282.75 | -22.29% |
| trec | 5% | 500 | 42.80% | 38.20% | -4.60 | 384.18→315.53 | -17.87% | 127.98→114.85 | -10.26% |
| trec | 10% | 500 | 64.80% | 59.80% | -5.00 | 708.20→478.17 | -32.48% | 140.28→156.63 | +11.66% |
| trec | 25% | 500 | 68.60% | 73.00% | +4.40 | 1569.59→989.91 | -36.93% | 250.44→195.66 | -21.87% |
| trec | 50% | 500 | 62.60% | 59.80% | -2.80 | 3536.69→2076.07 | -41.30% | 356.89→294.14 | -17.58% |
| rte | 5% | 277 | 88.45% | 83.39% | -5.05 | 558.77→386.21 | -30.88% | 141.76→133.25 | -6.01% |
| rte | 10% | 277 | 85.56% | 81.59% | -3.97 | 933.26→629.47 | -32.55% | 196.38→179.75 | -8.47% |
| rte | 25% | 277 | 87.36% | 85.56% | -1.81 | 2280.27→1322.37 | -42.01% | 366.04→221.29 | -39.54% |
| rte | 50% | 277 | 88.81% | 87.36% | -1.44 | 4570.16→2677.27 | -41.42% | 792.09→469.02 | -40.79% |

## 第二份独立前缀，25%预算

| 任务 | I准确率 | S准确率 | 差值pp | 冷变化 | 普通变化 |
|---|---:|---:|---:|---:|---:|
| sst2 | 92.78% | 92.55% | -0.23 | -34.67% | -29.34% |
| subj | 63.91% | 64.11% | +0.20 | -33.91% | -10.99% |
| trec | 69.20% | 74.20% | +5.00 | -32.97% | -32.20% |
| rte | 87.00% | 85.20% | -1.81 | -39.36% | -31.58% |

两前缀示例独立且与校准/评估隔离。第二前缀没有用于修改第一前缀读取器。不同前缀长度可能触发几何适配，因此该比较不是纯内容单变量实验。所有有害/有益翻转、保守界见v26_matrix_analysis.json；多单元历史回归不能替代新的预登记最终认证，v25非劣失败结论保留。

## 缓存晋升四臂主计时

| 条件/对照 | 基准 ms | 比较 ms | 变化 | 块bootstrap95%区间 |
|---|---:|---:|---:|---|
| preserve/shared/original | 1729.20 | 1093.92 | -36.74% | [-37.83%, -35.94%] |
| preserve/shared/reuse | 1745.57 | 1062.24 | -39.15% | [-40.03%, -38.17%] |
| preserve/admission/I | 1729.20 | 1745.57 | +0.95% | [-1.55%, +3.07%] |
| preserve/admission/S | 1093.92 | 1062.24 | -2.90% | [-4.14%, -1.62%] |
| warm/shared/original | 249.21 | 190.83 | -23.43% | [-28.26%, -18.02%] |
| warm/shared/reuse | 261.93 | 201.68 | -23.00% | [-27.99%, -16.42%] |
| warm/admission/I | 249.21 | 261.93 | +5.10% | [-1.32%, +10.69%] |
| warm/admission/S | 190.83 | 201.68 | +5.68% | [-1.04%, +14.20%] |

shared/*比较同晋升策略下I→S；admission/I或S比较同模型原晋升→复用晋升。区间以8个运行块重采样，不把块内请求当成独立进程重复；时间块仍可能相关，因此区间属于探索性不确定性估计。晋升优化同时用于I和S，共享贡献比较同晋升策略下I/S。更完整晋升差值与顺序分层见JSON。

## 固定checkpoint机制消融

| 读取器/补丁 | 正确/N | 准确率 | 相对I差值pp |
|---|---:|---:|---:|
| learned_native | 365/500 | 73.00% | +4.40 |
| unit_native | 84/500 | 16.80% | -51.80 |
| learned_proxy | 384/500 | 76.80% | +8.20 |

unit_native为相同图和R4、单位A/B和零偏置的直接共享控制，不是完整KVSharer。learned_proxy仍加载原生补丁，仅把小块内容换成source仿射预测，不能用于速度结论；BF16与FP32参考差异见audit/v26_proxy_rounding.json，不能等同于重训R0。

## 资源与离线成本

| 前缀/任务/预算 | 选中payload减少 | 加reader与packed后净减少 | 构建秒 | 冷TTFT估算回本请求数 |
|---|---:|---:|---:|---:|
| p0/sst2/5% | 29.96% | -15.45% | 333.8 | 3507 |
| p0/sst2/10% | 36.80% | 14.49% | 361.3 | 2128 |
| p0/sst2/25% | 40.60% | 31.78% | 392.6 | 687 |
| p0/sst2/50% | 41.65% | 37.27% | 413.5 | 348 |
| p0/subj/5% | 31.02% | -7.23% | 438.8 | 5095 |
| p0/subj/10% | 36.92% | 17.94% | 458.2 | 2222 |
| p0/subj/25% | 40.52% | 32.96% | 474.2 | 719 |
| p0/subj/50% | 41.69% | 37.92% | 522.9 | 429 |
| p0/trec/5% | 31.69% | -2.46% | 937.0 | 13649 |
| p0/trec/10% | 37.76% | 20.80% | 869.2 | 3779 |
| p0/trec/25% | 40.63% | 33.88% | 历史复用 | 不适用 |
| p0/trec/50% | 41.77% | 38.40% | 999.3 | 685 |
| p0/rte/5% | 34.41% | 6.28% | 772.8 | 4479 |
| p0/rte/10% | 38.59% | 24.66% | 821.4 | 2704 |
| p0/rte/25% | 41.17% | 35.62% | 844.8 | 882 |
| p0/rte/50% | 42.02% | 39.25% | 939.1 | 497 |
| p1/sst2/25% | 40.60% | 31.78% | 387.8 | 995 |
| p1/subj/25% | 40.52% | 32.96% | 465.3 | 1161 |
| p1/trec/25% | 40.96% | 34.16% | 987.5 | 2115 |
| p1/rte/25% | 41.45% | 35.86% | 840.8 | 1250 |

上述比例是所选prefix KV与读取器子系统字节数，不是模型总显存或持久SSD空间减少；完整28层KV仍保留。回本数仅用本轮教师采集+拟合/训练/验证成本除以描述性冷TTFT节省，不含研发搜索，不是吞吐保证。公共prefix构建两方法均需要；p1实际构建与索引成本另有日志。

## Nsight

| 臂 | H2D MB/请求 | H2D ms/请求 | GPU busy ms/请求 |
|---|---:|---:|---:|
| v26_profile_nsys_I_original | 135.103 | 9.342 | 38.813 |
| v26_profile_nsys_I_reuse-prefetch | 76.891 | 3.206 | 37.442 |
| v26_profile_nsys_S_original | 89.905 | 6.476 | 37.116 |
| v26_profile_nsys_S_reuse-prefetch | 47.962 | 2.052 | 36.034 |

每臂8请求，统计同步完成的TTFT NVTX区间，排除加载、warmup、标签评分。H2D包含payload、索引、元数据。NVTX/Python/API/kernel会重叠，不能相加当关键路径。VizTracer轨迹保留于profiles/v26。

## 验证与复现

- 完整回归：601 passed、4 skipped、87 subtests；既有readonly NumPy warning保留，见logs/v26_full_gpu_tests.log。
- 晋升GPU审计80请求含warmup：完整首token词表logits、全部候选token分数、ID、驻留内容、device_map、heap顺序及容量完全一致；晋升块独立storage，28层private KV独立。32个正式冷请求认证通过。
- 4877个原有文件及v25归档保持不变；两个p0缓存副本逐文件校验一致且无硬链接，见audit/v26_final_invariants.json。
- 审计脚本的两次修复、新前缀完成标记补全及失败日志保留。标记补全时校验每块几何，索引构建再次逐层比较实际FP16源与BF16 prefix转换结果。
- 仅验证Qwen2.5-7B、单请求和当前四任务/预算；未验证14B，未实现完整外部KVSharer/xKV基线。
- 调度入口：scripts/v26_training_worker.py、v26_gpu3_sequence2.py、v26_quality_worker.py、v26_formal_sequence3.py、v26_diagnostics_sequence3.py。命令、SHA和拆分在configs、audit、data/v26，禁止覆盖已有输出。最终服务参数另记于*.effective_cli.json；继承的内层command.json仅是覆盖前模板。
- 汇总文件：results/v26_matrix.csv、v26_matrix_analysis.json、v26_primary_analysis.json、v26_mechanism_analysis.json、v26_offline_costs.json、v26_nsight_summary.json。

## 本轮可追踪实验运行成本

| 阶段 | 运行数 | 进程墙钟秒之和 |
|---|---:|---:|
| matrix_timing | 80 | 25319.2 |
| teacher_capture_including_superseded_SUBJ | 48 | 4331.1 |
| full_quality_including_legacy_SUBJ | 42 | 10812.4 |
| profilers_and_exports | 12 | 908.9 |
| reader_training_including_superseded_SUBJ | 24 | 10378.7 |
| mechanism_evaluation | 2 | 463.2 |
| primary_study_serial_total | 1 | 6103.7 |

此表包含被替代的旧SUBJ采集/训练和历史评估，避免只统计成功主模型。墙钟包含模型加载、CPU等待；并行任务时间重叠，不能等同于总历时或GPU kernel时间。未覆盖人工准备、旧版本研发、部分设置和审计。上面的单prefix部署构建成本与本表的实验验证成本分别统计。

## SUBJ历史读取器与修正版本对照

第二前缀25%预算，原历史口径与纠正数据并重新校准后的结果；同时涉及2条评估句子修复和3条校准/验证修复，不能解释成单纯推理消融。

| 臂 | 历史正确/N | 修正正确/N |
|---|---:|---:|
| I | 643/1003 | 641/1003 |
| S | 713/1003 | 643/1003 |

旧/新无共享在其余1001条未改动查询上的全部候选token分数完全一致；共享checkpoint重新校准后则改变。该观察支持数据修正与重新校准过程会显著影响结果，但没有单独隔离训练数值波动。完整1003条的修正前后表只比较描述性总数；配对统计另限于未改动的1001条。

## 各单元实际共享结构

G16固定的是完整来源数，整数块预算不兼容时来源图按预登记规则适配。因此跨预算比较同时包含选择预算与几何适配，不能称为只改变预算的单变量消融。层编号均从0开始。

| 单元 | source→consumers | 参与层数 | 最大层间距离 |
|---|---|---:|---:|
| p0/sst2/5% | 12→10,11,13,14; 20→17,18,19,21,22,23; 26→25,27 | 15 | 3 |
| p0/sst2/10% | 12→9,10,11,13,14; 20→17,18,19,21,22; 26→25,27 | 15 | 3 |
| p0/sst2/25% | 6→5; 12→9,10,13,14; 15→11; 20→17,18,21,22; 23→19; 26→25 | 18 | 4 |
| p0/sst2/50% | 7→5; 12→10,14; 13→9,11,15; 20→18,22; 21→17,19,23; 27→25 | 18 | 4 |
| p0/subj/5% | 6→5; 12→9,10,13,14; 15→11; 20→17,18,21,22; 23→19; 26→25 | 18 | 4 |
| p0/subj/10% | 6→4; 12→9,10,14; 13→11,15; 20→17,18,22; 21→19,23; 27→25 | 18 | 3 |
| p0/subj/25% | 6→5,7; 10→9,11; 14→13,15; 18→17,19; 22→21,23; 26→25,27 | 18 | 1 |
| p0/subj/50% | 6→5,7; 10→9,11; 14→13,15; 18→17,19; 22→21,23; 26→25,27 | 18 | 1 |
| p0/trec/5% | 7→5; 12→10; 13→9,11,14,15; 20→18,22; 21→17,19,23; 27→25 | 18 | 4 |
| p0/trec/10% | 12→9,10,11,13,14; 20→17,18,19,21,22; 26→25,27 | 15 | 3 |
| p0/trec/25% | 6→5,7; 10→9,11; 14→13,15; 18→17,19; 22→21,23; 26→25,27 | 18 | 1 |
| p0/trec/50% | 7→5; 12→10,14; 13→9,11,15; 20→18,22; 21→17,19,23; 27→25 | 18 | 4 |
| p0/rte/5% | 6→5; 12→9,10,13,14; 15→11; 20→17,18,21,22; 23→19; 26→25 | 18 | 4 |
| p0/rte/10% | 6→4; 12→9,10,14; 13→11,15; 20→17,18,22; 21→19,23; 27→25 | 18 | 3 |
| p0/rte/25% | 6→5,7; 10→9,11; 14→13,15; 18→17,19; 22→21,23; 26→25,27 | 18 | 1 |
| p0/rte/50% | 6→5,7; 10→9,11; 14→13,15; 18→17,19; 22→21,23; 26→25,27 | 18 | 1 |
| p1/sst2/25% | 6→5; 12→9,10,13,14; 15→11; 20→17,18,21,22; 23→19; 26→25 | 18 | 4 |
| p1/subj/25% | 6→5,7; 10→9,11; 14→13,15; 18→17,19; 22→21,23; 26→25,27 | 18 | 1 |
| p1/trec/25% | 6→5,7; 10→9,11; 14→13,15; 18→17,19; 22→21,23; 26→25,27 | 18 | 1 |
| p1/rte/25% | 6→5; 12→9,10,13,14; 15→11; 20→17,18,21,22; 23→19; 26→25 | 18 | 4 |

## VizTracer主线程诊断

### v26_viz_original_I_summary

| 函数 | self ms/首轮prefill |
|---|---:|
| wait (/usr/lib/python3.10/threading.py:288) | 764.576 |
| to (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:374) | 442.435 |
| to (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:829) | 46.393 |
| decoder_logits (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v25/runtime.py:80) | 32.599 |
| insert (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:2522) | 23.415 |
| abspath (/usr/lib/python3.10/posixpath.py:374) | 22.487 |
### v26_viz_original_S_summary

| 函数 | self ms/首轮prefill |
|---|---:|
| wait (/usr/lib/python3.10/threading.py:288) | 497.573 |
| to (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:374) | 253.070 |
| to (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:829) | 28.502 |
| decoder_logits (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v25/runtime.py:80) | 28.262 |
| select_promixed_gqa_blocks (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/promixed.py:99) | 21.399 |
| configure_impress_layer (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/flexgen_pcache.py:1331) | 20.913 |
### v26_viz_reuse-prefetch_I_summary

| 函数 | self ms/首轮prefill |
|---|---:|
| wait (/usr/lib/python3.10/threading.py:288) | 1065.231 |
| promote (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v26/admission.py:88) | 170.879 |
| register_resolved (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v26/admission.py:64) | 88.891 |
| register (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v26/admission.py:44) | 65.123 |
| to (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:374) | 64.473 |
| call (/home/panzihang/src/prism_layerkv/scripts/profile_layerkv_v26.py:22) | 39.661 |
### v26_viz_reuse-prefetch_S_summary

| 函数 | self ms/首轮prefill |
|---|---:|
| wait (/usr/lib/python3.10/threading.py:288) | 640.850 |
| promote (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v26/admission.py:88) | 103.045 |
| register_resolved (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v26/admission.py:64) | 38.587 |
| register (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v26/admission.py:44) | 35.116 |
| decoder_logits (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v25/runtime.py:80) | 27.684 |
| select_promixed_gqa_blocks (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/promixed.py:99) | 22.272 |

self时间包含未追踪的C/库调用和阻塞等待，不等于CPU计算时间。不同线程并行，不能相加得到关键路径。

## 结果边界与解释

读取器有固定常驻成本，以下单元虽然所选KV减少，但计入reader/packed后该子系统总字节数增加：p0/sst2/5%增加15.45%；p0/subj/5%增加7.23%；p0/trec/5%增加2.46%。不能用payload减少替代净内存收益。
TREC/25%主研究的preserve条件下，8/8个块的共享TTFT低于无共享，各块变化范围为[-41.05%, -36.38%]；两种先后顺序均保留，不能推广为所有系统负载下的保证。
TREC/25%主研究的warm条件下，8/8个块的共享TTFT低于无共享，各块变化范围为[-32.94%, -3.47%]；两种先后顺序均保留，不能推广为所有系统负载下的保证。
矩阵SSD-cold/preserve共5120个正式请求通过冷状态检查，四臂主研究另有512个；准确率来自完整划分，矩阵TTFT每单元来自128条请求，并非完整划分的冷计时。
复用预取做晋升已通过数值与生命周期审计，但本轮未显示普遍的速度优势；共享在同晋升策略下的收益与晋升改动本身的收益必须分开。跨任务主质量矩阵15/16格下降，当前支持恢复成本收益与质量折中，不支持通用无损或无需适配的结论。

## 机制结果的解释

固定G16/R4图下，单位读取器为16.80%，学习读取器为73.00%，说明当前结构依赖学习到的整体读侧补偿；A/B与偏置一起变化，不能分拆归因。source-proxy小块对照为76.80%，反而高于真实原生小块，因此本轮不支持“真实原生内容必不可少”的解释。先前重训R4优于R0还可能涉及训练轨迹或执行差异，此次干预未分别隔离。没有据此改选主模型，也不能外推到其他任务/预算。

舍入检查只覆盖12个读取器各自首次调用的60-token小块。相同BF16输入和参数下，BF16结果相对FP32运算参考的K相对L2为0.001523～0.002996，V为0.002329～0.002542。没有进行全数据FP32评分干预，不能排除数值路径对分类变化的影响。proxy仍恢复真实原生块，只重建小控制区，不作为速度或资源收益方案。


## 恢复瓶颈与最终矩阵总结

主矩阵16格冷缓存TTFT全部下降17.87%～42.01%，普通缓存有3格变慢（SST-2/5%、SUBJ/5%、TREC/10%）。矩阵计时每格只有一对独立进程，属于描述性结果；八块四臂主研究另行报告置信区间。准确率15/16格下降，不能把冷恢复收益等同于通用质量保持。

Nsight的reuse-prefetch同策略比较中，H2D从76.891 MB降到47.962 MB（约37.62%），累计拷贝时间从3.206 ms降到2.052 ms；GPU busy union从37.442 ms降到36.034 ms，GPU kernel累计时间则从29.508 ms略增到30.645 ms。主机host gather的NVTX包含时间约934.48→527.21 ms；VizTracer主线程wait的self时间约1065.23→640.85 ms。这些诊断与主要减少主机恢复/等待的解释一致，不能把重叠区间相加作因果分解。

晋升复用在无共享臂把硬件H2D从135.103 MB降到76.891 MB，同时新增约58.212 MB的D2D拷贝；它确实改变了晋升数据来源，但注册、复制和等待成本仍在，正式四臂研究没有支持其普遍更快。每个profiler臂仅8请求，诊断延迟不替代正式计时。
