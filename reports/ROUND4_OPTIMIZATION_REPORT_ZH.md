# 固定 P8：host gather 与元数据优化（第四轮）

本轮有收益，但不是所有配置都变快。16 组完整运行按样本加权的 TTFT 为 235.91 → 226.84 ms（下降 3.85%）；9 组低于第三轮历史值，7 组高于历史值。50% 预算下四个数据集均低于历史值。历史完整运行与本轮同期对照属于不同证据，不能互相替代。

本轮以第三轮优化版为基线，只进一步修改 host gather 与元数据路径。CPU 源块保留为只读视图，合并成一次 NumPy 原生 concatenate，直接写入现有 pinned slab；BF16 路径使用 CPU torch.cat。逻辑映射改为请求内独立 NumPy 数组，复用已配置选择对应的物理位置；物理块统计只生成一次唯一块集合，来源统计在消费时读取当前 device_map。没有新增选择/缓存算法，没有改变缓存评分更新和淘汰顺序。

固定 P8（0/8/16/24 层选择）、coverage=0.5、精确全局块预算、GPU/CPU payload 缓存 55/131 MiB、FP16 KV/BF16 计算和 8 MiB staging 上限保持不变。原目录 prism_max 未修改；所有文件均在服务器 /home/panzihang/src/prism_pans。

## 完整评测

每组完整运行同一 strict UID 清单：SST2 867、SUBJ 998、TREC 495、RTE 272 条，每次独立进程并预热 32 条。该清单沿用前几轮，并非新建的独立 holdout。正式 TTFT 为无 profiler 的 logits-ready 时间；模型与索引预加载、tokenizer 和服务排队不包含在这个口径中。comparison.csv 同时记录 first-token-ready、response-ready 和包含标签续写评分的 evaluation-ready。

|预算|任务|准确率：旧→新 %|TTFT：旧→新 ms|下降 %|P95：旧→新 ms|
|---:|---|---:|---:|---:|---:|
|5%|SST2|92.50 → 92.50|106.30 → 107.41|-1.0|116.06 → 118.61|
|5%|SUBJ|54.21 → 54.21|135.35 → 128.81|4.8|166.86 → 152.87|
|5%|TREC|42.22 → 42.22|142.89 → 141.27|1.1|169.80 → 171.00|
|5%|RTE|88.60 → 88.60|151.45 → 174.53|-15.2|186.60 → 208.56|
|10%|SST2|93.31 → 93.31|142.15 → 157.35|-10.7|175.16 → 177.92|
|10%|SUBJ|52.61 → 52.61|151.94 → 166.93|-9.9|178.96 → 201.53|
|10%|TREC|66.06 → 66.06|191.38 → 176.84|7.6|218.31 → 206.40|
|10%|RTE|85.29 → 85.29|202.35 → 202.25|0.0|250.04 → 244.07|
|25%|SST2|93.66 → 93.66|251.64 → 238.71|5.1|287.60 → 269.41|
|25%|SUBJ|54.81 → 54.81|238.53 → 244.67|-2.6|305.01 → 284.61|
|25%|TREC|68.48 → 68.48|269.64 → 287.87|-6.8|337.17 → 341.17|
|25%|RTE|87.50 → 87.50|277.30 → 281.30|-1.4|314.27 → 353.27|
|50%|SST2|93.89 → 93.89|331.69 → 275.49|16.9|411.93 → 292.72|
|50%|SUBJ|84.57 → 84.57|375.05 → 317.14|15.4|463.87 → 371.28|
|50%|TREC|62.42 → 62.42|409.69 → 390.07|4.8|521.75 → 479.67|
|50%|RTE|89.34 → 89.34|698.86 → 679.74|2.7|909.26 → 891.87|

以本次 10528 条评测记录等权汇总，均值 235.91 → 226.84 ms，下降 3.8%。这个加权结果对应当前数据集和预算构成。

完整输出校验：{"formal_groups": 16, "formal_samples": 10528, "exact_all": true, "profiles_exact_all": true, "original_files": 258, "original_unchanged": true, "frozen_source_unchanged": true, "baseline_snapshot_unchanged": true, "staging_bound_all": true}

完整表格的旧值来自第三轮历史完整运行，并非本轮同期重跑；当前主机其他 GPU 上仍有负载。下面的交替顺序配对才是本轮同期代码对照，两个表格应结合阅读。

## 配对复跑

每组 64 条测量、32 条预热；两次顺序分别为旧→新、新→旧，保持相同 UID。这里的旧版本来自冻结的第三轮源码快照。

|任务|预算|重复|旧 ms|新 ms|下降 %|输出精确一致|
|---|---:|---:|---:|---:|---:|---|
|trec|25%|0|376.11|370.04|1.6|True|
|trec|25%|1|390.50|291.64|25.3|True|
|trec|50%|0|531.06|475.37|10.5|True|
|trec|50%|1|586.33|403.27|31.2|True|
|rte|50%|0|805.58|745.08|7.5|True|
|rte|50%|1|730.90|652.24|10.8|True|

## TTFT 组成与重叠

每组分别使用 Nsight Systems 2025.5.1 和 VizTracer，4 条预热后记录前 4 条请求。诊断绝对耗时受样本和 profiler 开销影响，不能代替正式表格。主线程分区为互斥墙钟区间；GPU kernel、Memcpy、memset 使用时间并集，不能与 CPU 区间再次相加。不同工具独立运行，不拼接时间戳。

|配置|诊断墙钟 ms|KV 等待/gather ms|缓存维护 ms|GPU 活跃并集 ms|kernel/copy 重叠 ms|
|---|---:|---:|---:|---:|---:|
|k005/trec|220.80|42.35|9.38|29.54|0.005|
|k025/trec|383.67|106.28|42.00|34.16|0.097|
|k050/trec|418.04|158.38|47.91|40.37|0.596|
|k050/rte|700.06|241.10|122.30|52.12|0.538|

新增批次标记将 native host gather 与外围准备分开；prepare/pack 的剩余区间仍含源视图、文件访问、描述符、CUDA 提交和调度等待，不等于纯 Python CPU 时间。

|配置|prepare/pack 并集 ms|native host gather ms|扣除 host gather 后 ms|来源统计 ms|host gather 与主线程 resolve 重叠 ms|
|---|---:|---:|---:|---:|---:|
|k005/trec|53.93|12.81|41.11|3.66|9.07|
|k025/trec|208.26|89.60|118.67|4.26|51.80|
|k050/trec|281.72|143.79|137.93|4.22|67.85|
|k050/rte|452.40|237.71|214.69|6.16|122.41|

worker 与主线程 resolve 重叠往往表示消费线程正在等待 worker，不能全部解释为隐藏在 decoder 计算下的有效重叠。详细数值在 cpu_worker_overlap_comparison.csv。原始 Nsight 数据的 Memcpy 字节不包括 cat/index_copy kernel 内部显存访问；comparison.csv 另外记录实际物理预取字节和读放大。

## 资源、验证与边界

307 项测试和 87 项子测试通过，包括已有异步 KV 混合来源/尾块/重排/生命周期检查，以及新增 native gather、动态驻留来源统计、映射输入隔离、重新选择和物理块去重统计。日志 audit/round4_preflight_full_tests.log。存在原有只读 NumPy memmap 转为 torch 视图的警告；该源只读，写入只发生于目标 staging。

staging 容量维持 8 MiB；请求内映射数组及 host 源视图属于额外工作空间，进程 RSS 和 CUDA 峰值均记录在 results/round4_optimized_grid/staging_memory.json。没有池化 GPU 输出，也没有扩大 KV 缓存容量。

GPU 2 顺序运行实验；其他 GPU 上有用户任务，主机非独占。本轮完整配置各运行一次，配对复跑用于检查方向，不提供统计置信区间。没有 OS 调度采样，无法精确拆分 GIL、I/O 阻塞和被调度离开 CPU 的时间。

## 文件与复现

- 报告：ROUND4_OPTIMIZATION_REPORT_ZH.md
- 全量对比：results/round4_optimized_grid/comparison.csv
- 精确一致性：results/round4_optimized_grid/baseline_parity.json、final_validation.json
- 轨迹：profiles/round4_optimized_grid/kXXX/任务/，包含 nsys.nsys-rep、nsys.sqlite、viztracer_run/viztracer.json、ttft_timeline.svg。
- 新增批次分解：profiles/round4_optimized_grid/payload_batch_summary.csv。
- 上一轮基线：audit/round4_baseline/；补丁针对该快照，不能直接套用于 prism_max。
- 补丁：audit/round4_optimization.patch；最终交付验证：audit/round4_delivery_verified.json。

单组完整复现命令（输出目录必须尚不存在）：

```bash
cd /home/panzihang/src/prism_pans
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /home/panzihang/venvs/vllm-stable/bin/python scripts/run_round4.py \
  --variant optimized --task trec --budget 25 --period 8 --warmup 32 \
  --output /home/panzihang/src/prism_pans/results/reproduce_round4_trec25
```

使用 --variant baseline 并指定另一输出目录可运行冻结的第三轮代码。完整网格与独立轨迹由 scripts/round4_workflow.py 的 --run-root、--profile-root 和 --gpu 指定新目录执行。

## 历史对比异常点的同期复测

RTE 5% 及至多一个其他历史降速配置，另做相同 UID 的 64 条测量、32 条预热、两次反向顺序配对。该检查用于区分当前实现差异与历史运行波动，是针对异常点的补充验证，不是独立的总体收益估计。

|任务|预算|重复|当前旧版 ms|当前新版 ms|下降 %|输出精确一致|
|---|---:|---:|---:|---:|---:|---|
|rte|5%|0|149.25|152.58|-2.2|True|
|rte|5%|1|151.57|148.82|1.8|True|
|sst2|10%|0|185.08|156.60|15.4|True|
|sst2|10%|1|161.50|156.26|3.2|True|

原始记录：results/round4_followup_paired/。

## 本轮结果解读与剩余瓶颈

同期对照按每个配置两次运行的均值汇总如下。每次使用相同的 64 条 UID；两次均值相等权重。这只能支持当前测试条件下的方向判断，不能外推为所有预算的确定收益。

|配置|同期旧版 ms|同期新版 ms|下降 %|
|---|---:|---:|---:|
|TREC 25%|383.31|330.84|13.69|
|TREC 50%|558.69|439.32|21.37|
|RTE 50%|768.24|698.66|9.06|
|RTE 5%|150.41|150.70|-0.19|
|SST2 10%|173.29|156.43|9.73|

异常配置同期复测：RTE 5% 两次方向相反（慢 2.23%、快 1.82%），合并均值慢 0.19%，没有显示稳定收益；SST2 10% 两次分别快 15.39% 和 3.24%，合并快 9.73%。历史完整运行中这两组约 15.24% 和 10.69% 的回退没有在同期对照中重现，但这不足以证明所有低预算配置均不会回退。


下面直接对同一份 Nsight 时间轴求交集，检查 host gather 到底与多少 GPU 活动重叠。GPU 活动包括 kernel、Memcpy 和 memset；交集仅表示同时发生，不能全部当成无代价隐藏。

|配置|host gather ms|与 GPU 活动重叠 ms|重叠占 gather %|外围 prepare/pack ms|来源统计 ms|
|---|---:|---:|---:|---:|---:|
|k005/trec|12.81|2.10|16.4|41.11|3.66|
|k025/trec|89.60|8.43|9.4|118.67|4.26|
|k050/trec|143.79|16.04|11.2|137.93|4.22|
|k050/rte|237.71|23.77|10.0|214.69|6.16|

TREC 50% 中 gather 与 GPU 活动只重叠约 11%，RTE 50% 约 10%。主线程 resolve 与 worker 的大量重叠主要说明取数时仍在等 worker，不能据此宣称拷贝已被 decoder 计算覆盖。这些百分比只来自本轮 4 条 Nsight 样本，不代表完整数据集的无 profiler 比例。

剩余工作应按实测开销排序：首先是 host gather 及外围源块准备，其次是层配置、缓存块计数和评分更新。TREC 50% 的 Nsight 主线程 KV 等待/gather 为 158.38 ms，缓存维护为 47.91 ms；RTE 50% 分别为 241.10 ms 和 122.30 ms。来源统计仅约 4.22/6.16 ms，继续单独优化这一小段的上限较低。GPU Memcpy 并集分别约 3.97/5.33 ms，且该统计不含 cat/index_copy kernel 内部搬运，不能把所有等待都解释为 PCIe 传输。

VizTracer 的 RTE 50% 热点包括 _ordered_chunk_counts（约 135.40 ms inclusive elapsed）、configure_impress_layer（约 131.83 ms）、update_score（约 122.33 ms），以及 worker 的 _chunk_source_readonly（约 88.73 ms）。这些函数可能嵌套或并发，必须结合时间线看，不能与上面的 Nsight 阶段时间相加。

若继续优化，先对 gather 和源块准备补充分线程 CPU 时间、页故障或调度证据，区分实际内存复制、I/O 和等待，再决定是否把更多块描述符准备与 gather 一起下沉到原生循环。随后可批量处理 _ordered_chunk_counts 和层配置，在严格保持评分更新顺序、淘汰顺序及选中块一致的前提下减少 Python 循环。这些是后续建议，本轮没有修改这些算法或加入新模块。

一致性边界：10,528 条正式记录的预测、选择 SHA、各层选择数、精确预算、缓存评分更新计数、首 token logits 和完整标签续写 logprobs 均逐项相同；但物理预取流量和来源占比并非逐项完全相同。各配置平均物理预取字节相对历史值的最大绝对变化为 0.443%。因此不把预测一致表述成运行调度或物理 I/O 轨迹完全相同；明细保留在 comparison.csv。

