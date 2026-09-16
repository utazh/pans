# 固定 P8：四档预算、四个完整数据集与 TTFT 诊断

状态：complete。正式结果完成 16/16 组，Nsight 分析 16/16 组。所有修改和产物位于服务器 /home/panzihang/src/prism_pans；原 prism_max 不作修改。上一轮 P1 结果完整保留，本轮纠正为用户要求的固定 P8。

## 主要发现

1. 25% 预算下，固定 P8 的正式平均 TTFT 分别为 SST2 324.60、SUBJ 332.53、TREC 380.12、RTE 483.09 ms；相较上一轮同预算 P1，均有明显下降。但准确率变化具有任务差异：SST2 93.66%、SUBJ 54.81%、TREC 68.48%、RTE 87.50%。不能仅按速度选择配置。
2. SUBJ 在 P8 的 5%/10%/25% 预算下准确率为 54.21%/52.61%/54.81%，50% 才恢复到 84.57%；5% 时 998 条中有 934 条预测为 subjective。TREC 在本次四档预算中以 25% 的 68.48% 最高，50% 反而降至 62.42%。SST2 从 25% 增至 50% 只提高约 0.23 个百分点，平均 TTFT 从 324.60 增至 634.79 ms。这些是单次完整评估的观测，不是显著性检验或独立留出集调参结论。
3. 瓶颈随预算变化：在 5%–10% 的诊断中，主线程 decoder 调用和 mask 生成占比较突出（这里是 CPU 侧经过时间，不是 GPU kernel 时间）；25%–50% 时 KV 等待与收集成为主项。25% 四任务该阶段约占诊断 TTFT 的 43%–61%，50% 为 63%–77%。
4. 以 TREC 为例，5%/25%/50% 的诊断 TTFT 为 209.13/569.55/1287.96 ms，KV 等待与收集为 40.52/283.25/816.37 ms。相同区间的 GPU kernel 并集仅为 30.17/34.33/39.40 ms，copy 并集为 1.81/7.66/15.64 ms。数据增大时，主机侧准备、提交和等待的增长远大于实际 GPU 活动增长。
5. 50% RTE 的诊断 TTFT 为 1813.34 ms，其中 KV 等待与收集 1394.28 ms（约 77%），全部已记录 GPU 活动并集为 67.69 ms，kernel 与 copy 交集仅 0.607 ms。它们是在同一墙钟区间内的不同观察维度，不能相加。25%–50% 各任务的 kernel/copy 交集仅约 0.60–0.80 ms，不能因为有后台线程就认为搬运已被计算充分隐藏。
6. 工作线程轨迹进一步支持优化分块复制和 staging 路径。TREC 25% 每请求记录约 4514 次 _copy_readonly_source_to_gpu，inclusive 耗时 209.24 ms；RTE 50% 该函数约 10893.5 次、740.67 ms，_new_cpu_buffer 约 7373.5 次、225.50 ms。调用数受 5 微秒过滤影响；函数嵌套、不同工具开销及缓存状态不同，不能跨表加总，也不能把缓冲区函数时间全部称为底层 pinned allocation 时间。

后续工程优化的优先顺序：大预算先合并分块复制、使用有界且由 CUDA event 保护生命周期的 staging 池、减少逐 token/逐块元数据循环；低预算还需要关注 decoder 的 CPU 调用路径与 mask 构造。本轮只测量和分析，没有实现这些优化，也没有修改缓存容量或 page-cache 条件来制造加速。

## 配置与校验

使用上一轮已修复的核心代码；本轮新增 scripts/run_round2.py、scripts/round2_workflow.py 和四个明确标注 uniform 的配置。未引入上一轮讨论的批量复制、staging 池或新算法模块。

固定 P8：28 层中第 0、8、16、24 层进行选择，共四次；每个周期沿用 leader 的评分与排序。各层预算整数分配可能相差一个 block，复用评分不意味着所有层 token 集合必然完全相同。known-period prefetch 开启，period/subperiod 调度参数为 8/4。

四档预算 5%、10%、25%、50% 均为精确全局 block 预算、均匀层比例，无敏感度 profile。coverage=0.5，adaptive coverage 关闭，自适应层复用关闭。保留不确定性等诊断量的计算和实际计时。末尾短 block 会使实际 token/byte 比例与名义 block 比例略有差异。

Qwen2.5-7B-Instruct；BF16 计算；FP16 KV payload；CPU 预加载 INT4 G32 selector index；GPU/CPU payload cache 55/131 MiB。额外 selector、pinned 和运行时内存不隐去，逐组记录进程峰值 RSS 与 CUDA 峰值。模型、prefix 和索引使用上一轮校验通过的相同资产；启动前再次执行模型/前缀身份检查。

GPU 2 串行运行，GPU 0/1 原有作业保留，不是整机独占。各组均新进程、相同严格完整数据集顺序和 32 条 warmup。每档 2632 条，四档合计 10528 条评估；准确率使用完整 label continuation 平均 token log-probability。每条请求生成一个 token。

CPU 测试 298 passed、87 subtests passed，见 logs/round2_preflight_tests.log。P8 GPU 小样本检查验证四次 selector/P8 决策及精确预算。final_validation.json 逐条核对全部 UID 顺序、唯一性、有限数值、均匀比例、层预算之和、P8 决策数及时间边界。

## 正式结果

以下均为无 profiler 的计时，单位 ms。每组完整运行一次，没有独立重复实验置信区间。

| 预算 | 数据集 | N | Accuracy | 平均 logits-ready | P95 logits-ready | 平均 response-ready | 平均 evaluation-ready |
|---|---|---:|---:|---:|---:|---:|---:|
| 5% | SST2 | 867 | 92.50% | 132.52 | 164.83 | 132.76 | 133.13 |
| 5% | SUBJ | 998 | 54.21% | 149.31 | 182.08 | 149.54 | 180.57 |
| 5% | TREC | 495 | 42.22% | 162.28 | 198.94 | 162.52 | 252.68 |
| 5% | RTE | 272 | 88.60% | 195.97 | 270.42 | 196.21 | 339.29 |
| 10% | SST2 | 867 | 93.31% | 155.06 | 194.92 | 155.29 | 155.63 |
| 10% | SUBJ | 998 | 52.61% | 181.50 | 205.80 | 181.73 | 210.75 |
| 10% | TREC | 495 | 66.06% | 213.92 | 261.21 | 214.17 | 305.51 |
| 10% | RTE | 272 | 85.29% | 237.77 | 290.67 | 238.00 | 372.11 |
| 25% | SST2 | 867 | 93.66% | 324.60 | 360.62 | 324.86 | 325.23 |
| 25% | SUBJ | 998 | 54.81% | 332.53 | 404.94 | 332.76 | 362.54 |
| 25% | TREC | 495 | 68.48% | 380.12 | 445.07 | 380.35 | 471.19 |
| 25% | RTE | 272 | 87.50% | 483.09 | 568.99 | 483.35 | 637.98 |
| 50% | SST2 | 867 | 93.89% | 634.79 | 717.96 | 635.06 | 635.44 |
| 50% | SUBJ | 998 | 84.57% | 732.48 | 900.56 | 732.82 | 768.20 |
| 50% | TREC | 495 | 62.42% | 774.49 | 993.62 | 774.77 | 874.79 |
| 50% | RTE | 272 | 89.34% | 1223.56 | 1640.77 | 1223.87 | 1392.88 |

完整指标见 results/round2_p8_grid/all_metrics.csv；逐请求记录在 k005、k010、k025、k050 下各任务的 scored_records.jsonl，汇总 summary.json，命令 command.json。保持同一历史数据包，未将排除校准 UID 的现有 bundle 宣称为新的独立留出集。

### 与上一轮 25% P1 的配置级对照

| 数据集 | P1 准确率 | P8 准确率 | 准确率变化 pp | P1 TTFT | P8 TTFT | TTFT 变化 |
|---|---:|---:|---:|---:|---:|---:|
| SST2 | 94.69% | 93.66% | -1.04 | 747.24 | 324.60 | -56.56% |
| SUBJ | 83.57% | 54.81% | -28.76 | 853.78 | 332.53 | -61.05% |
| TREC | 63.03% | 68.48% | +5.45 | 932.25 | 380.12 | -59.23% |
| RTE | 88.60% | 87.50% | -1.10 | 1136.82 | 483.09 | -57.51% |

这是先后两次运行的配置级对照，P8 同时启用已知周期预取；不能把全部差异归因于单独一个代码改动，也不构成独立随机重复的因果消融。

## TTFT 的计时边界

logits-ready 从 query tensor 准备和初始 CUDA 同步后开始，到首 token logits 完成为止；不包括排队、tokenize、模型加载、prefix 注册、索引预加载和身份校验。first-token-ready 另含 cache score 后处理与首 token ID 选择；response-ready 再含文本 decode；evaluation-ready 再含候选标签评分。正式 TTFT 不混入候选标签评分。

## Nsight：主线程互斥时间分解

每个预算与数据集分别采集 4 条请求、4 条 warmup 的诊断运行。其缓存历史不同于正式实验的 32 条 warmup，并有 profiler 开销；不能用诊断比例直接拆分正式平均 TTFT。下表每行各阶段相加等于该行诊断 TTFT；嵌套 NVTX 区间按最内层归属，避免重复计时。

| 预算 | 数据集 | 诊断 TTFT | KV 等待/收集 | selector 控制+评分+决策 | decoder CPU 调用 | mask | 其余 |
|---|---|---:|---:|---:|---:|---:|---:|
| 5% | SST2 | 166.36 | 24.66 | 25.96 | 55.06 | 26.83 | 33.85 |
| 5% | SUBJ | 204.07 | 45.18 | 29.97 | 54.59 | 41.03 | 33.30 |
| 5% | TREC | 209.13 | 40.52 | 39.57 | 55.28 | 36.11 | 37.65 |
| 5% | RTE | 291.20 | 45.10 | 64.33 | 67.80 | 61.31 | 52.66 |
| 10% | SST2 | 189.39 | 37.48 | 31.22 | 53.72 | 27.11 | 39.85 |
| 10% | SUBJ | 319.01 | 70.17 | 58.49 | 76.28 | 54.97 | 59.11 |
| 10% | TREC | 290.87 | 67.79 | 52.13 | 75.34 | 39.57 | 56.04 |
| 10% | RTE | 312.42 | 78.30 | 51.02 | 66.72 | 60.01 | 56.38 |
| 25% | SST2 | 410.70 | 177.52 | 50.25 | 73.08 | 32.64 | 77.21 |
| 25% | SUBJ | 474.60 | 251.93 | 41.24 | 60.81 | 41.93 | 78.69 |
| 25% | TREC | 569.55 | 283.25 | 73.26 | 73.30 | 43.61 | 96.12 |
| 25% | RTE | 660.07 | 400.99 | 50.92 | 69.26 | 53.21 | 85.69 |
| 50% | SST2 | 901.28 | 615.18 | 73.64 | 70.52 | 30.32 | 111.62 |
| 50% | SUBJ | 1249.54 | 826.07 | 107.89 | 87.92 | 53.72 | 173.93 |
| 50% | TREC | 1287.96 | 816.37 | 134.99 | 84.87 | 47.14 | 204.59 |
| 50% | RTE | 1813.34 | 1394.28 | 93.65 | 73.10 | 61.11 | 191.20 |

其余包括预取提交、cache score 维护、KV 布局转换、其余 resolve 和同步控制。完整细分类在 profiles/round2_p8_grid/nsys_components.csv。CPU 阶段为主线程经过时间，包含其间等待与同步，不等价于线程占用 CPU 的时间。

## Nsight：GPU 活动与重叠

| 预算 | 数据集 | kernel 并集 | copy 并集 | kernel∩copy | 全部 GPU 活动并集 | decoder∩payload/other |
|---|---|---:|---:|---:|---:|---:|
| 5% | SST2 | 28.56 | 1.45 | 0.231 | 29.78 | 0.221 |
| 5% | SUBJ | 32.43 | 1.66 | 0.257 | 33.84 | 0.246 |
| 5% | TREC | 30.17 | 1.81 | 0.269 | 31.71 | 0.257 |
| 5% | RTE | 41.38 | 2.08 | 0.228 | 43.27 | 0.204 |
| 10% | SST2 | 29.35 | 2.69 | 0.545 | 31.49 | 0.532 |
| 10% | SUBJ | 33.27 | 3.03 | 0.477 | 35.82 | 0.469 |
| 10% | TREC | 31.22 | 3.27 | 0.443 | 34.05 | 0.422 |
| 10% | RTE | 40.37 | 3.71 | 0.582 | 43.54 | 0.561 |
| 25% | SST2 | 31.47 | 5.87 | 0.675 | 36.67 | 0.669 |
| 25% | SUBJ | 35.89 | 6.98 | 0.686 | 42.18 | 0.674 |
| 25% | TREC | 34.33 | 7.66 | 0.797 | 41.19 | 0.779 |
| 25% | RTE | 44.24 | 8.79 | 0.708 | 52.37 | 0.690 |
| 50% | SST2 | 35.23 | 11.82 | 0.601 | 46.45 | 0.593 |
| 50% | SUBJ | 40.41 | 13.91 | 0.613 | 53.71 | 0.604 |
| 50% | TREC | 39.40 | 15.64 | 0.690 | 54.35 | 0.677 |
| 50% | RTE | 50.22 | 18.03 | 0.607 | 67.69 | 0.601 |

GPU 数据按区间并集与交集计算，包含在主线程墙钟范围内，不能再与上表相加。kernel 并集 + copy 并集 - kernel∩copy 为二者联合活动；全部 GPU 活动另外包含 memset。payload/other 为无法归入 selector、decoder、mask、layout 的相关 CUDA 活动，不能声称全部均为 payload。

每组 nsys_ttft_breakdown.json 还给出不同 stream 的活动并集、decoder 与 selector index 的重叠，以及 KV 等待期间的实际 GPU 活动。每组四个完整 TTFT NVTX 范围均检查四次 selector、28 次 decoder layer 调用。

## VizTracer：预取工作线程

每组 Python 轨迹从 executor 创建之前开始启用，测量前清除 setup 事件；排除 torch/transformers/numpy 内部调用，过滤小于 5 微秒的事件。分析包括主线程和工作线程；原始文件位于各组 viztracer_run/viztracer.json。

汇总 profiles/round2_p8_grid/viztracer_hotspots.csv 包含每函数 inclusive/exclusive traced 时间和记录调用数；父子函数不能相加，短调用过滤导致记录数可能小于实际数。viztracer_worker_breakdown.json 的 cpu_elapsed_overlap 分别报告工作线程预取与主线程 _resolve_entry（等待与收集）、selection/mask/layout 区间的交集。这表示时间上并行存在，并不表示主线程正在做有效计算或 GIL 已完全解除。

诊断与正式运行前四条 UID 的选块哈希、预测及标签首 token logits 对照见 profiles/round2_p8_grid/formal_prediction_parity.json。32 组、128 条诊断请求全部选块/预测相同，标签首 token logits 最大差值为 0；不同 warmup 状态仍会影响物理缓存来源及计时。

不能把不同 profiler 的时间直接相加；不能将 future 等待都解释为 SSD 带宽，应用记录的 SSD 来源字节也不是设备实际读取字节。本轮没有改变 uncache/page-cache 条件。

## 轨迹、复现与限制

Nsight 使用项目内部 2025.5.1，可解析当前 CUDA 13。没有使用上一轮不兼容的 2023.4 轨迹。系统 perf 权限限制了 OS 调度采集，若工具给出潜在事件缺失警告，仅将统计解释为已记录 CUDA 活动，不推断精确 CPU 利用率。

每组 profiles/round2_p8_grid/kXXX/task/ 包含 nsys.nsys-rep、nsys.sqlite、nsys_ttft_breakdown.json、ttft_timeline.svg、viztracer_run/viztracer.json、viztracer_worker_breakdown.json 和日志。图使用真实区间宽度，可展开查看。

单组复现（输出目录必须不存在）：

    cd /home/panzihang/src/prism_pans
    CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /home/panzihang/venvs/vllm-stable/bin/python scripts/run_round2.py --period 8 --budget 25 --task trec --output results/new_p8_k025_trec

budget 可选 5、10、25、50；task 可选 sst2、subj、trec、rte。完整 16 组及 32 次独立诊断：

    /home/panzihang/venvs/vllm-stable/bin/python scripts/round2_workflow.py --gpu 2 --run-root results/new_p8_grid --profile-root profiles/new_p8_grid

源代码快照见 results/round2_p8_grid/source_sha256.json，测试日志见 logs/round2_preflight_tests.log，最终补丁见 audit/round2_p8_final.patch，原目录校验见 audit/round2_original_unchanged_check.log。按真实标签与预测标签的计数见 results/round2_p8_grid/confusion_counts.csv。

## 完成清单

| 预算 | 数据集 | 正式 | Nsight | VizTracer |
|---|---|---|---|---|
| 5% | SST2 | complete | complete | complete |
| 5% | SUBJ | complete | complete | complete |
| 5% | TREC | complete | complete | complete |
| 5% | RTE | complete | complete | complete |
| 10% | SST2 | complete | complete | complete |
| 10% | SUBJ | complete | complete | complete |
| 10% | TREC | complete | complete | complete |
| 10% | RTE | complete | complete | complete |
| 25% | SST2 | complete | complete | complete |
| 25% | SUBJ | complete | complete | complete |
| 25% | TREC | complete | complete | complete |
| 25% | RTE | complete | complete | complete |
| 50% | SST2 | complete | complete | complete |
| 50% | SUBJ | complete | complete | complete |
| 50% | TREC | complete | complete | complete |
| 50% | RTE | complete | complete | complete |
