# 固定 P8 工程优化结果（第三轮）

本轮只改变实现：向量化并按形状复用因果 mask；用 CUDA 完成事件保护的 2 × 4 MiB pinned staging 池；CPU/磁盘块按来源批量打包与 H2D，并用索引恢复输出顺序；批量计算 token/block 元数据并保持 CKLFU 更新顺序。GPU 驻留块通过 cat 批量复制，始终留在 GPU 内。原工程 prism_max 未修改。

固定 P8（选择层 0/8/16/24）、coverage=0.5、精确全局块预算、GPU/CPU payload 缓存 55/131 MiB、FP16 KV 与 BF16 计算均保持原设置。不增加创新或消融模块。

## 完整数据集结果

TTFT 为无 profiler 的 logits-ready 时间，不含模型加载、索引预加载、tokenizer 和服务排队；另保留 first-token-ready、response-ready 与包含标签评分的 evaluation-ready 时间。每组独立进程、预热 32 条；沿用上一轮 strict 评测 UID 清单和顺序（每档预算 SST2/SUBJ/TREC/RTE 为 867/998/495/272 条），不是新增独立 holdout。前一轮基线与本轮完整实验并非同时运行，配对复测见后表。

|预算|数据集|样本|旧准确率 %|新准确率 %|旧均值 ms|新均值 ms|降低 %|旧 P95 ms|新 P95 ms|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|5%|SST2|867|92.50|92.50|132.52|106.30|19.8|164.83|116.06|
|5%|SUBJ|998|54.21|54.21|149.31|135.35|9.4|182.08|166.86|
|5%|TREC|495|42.22|42.22|162.28|142.89|11.9|198.94|169.80|
|5%|RTE|272|88.60|88.60|195.97|151.45|22.7|270.42|186.60|
|10%|SST2|867|93.31|93.31|155.06|142.15|8.3|194.92|175.16|
|10%|SUBJ|998|52.61|52.61|181.50|151.94|16.3|205.80|178.96|
|10%|TREC|495|66.06|66.06|213.92|191.38|10.5|261.21|218.31|
|10%|RTE|272|85.29|85.29|237.77|202.35|14.9|290.67|250.04|
|25%|SST2|867|93.66|93.66|324.60|251.64|22.5|360.62|287.60|
|25%|SUBJ|998|54.81|54.81|332.53|238.53|28.3|404.94|305.01|
|25%|TREC|495|68.48|68.48|380.12|269.64|29.1|445.07|337.17|
|25%|RTE|272|87.50|87.50|483.09|277.30|42.6|568.99|314.27|
|50%|SST2|867|93.89|93.89|634.79|331.69|47.7|717.96|411.93|
|50%|SUBJ|998|84.57|84.57|732.48|375.05|48.8|900.56|463.87|
|50%|TREC|495|62.42|62.42|774.49|409.69|47.1|993.62|521.75|
|50%|RTE|272|89.34|89.34|1223.56|698.86|42.9|1640.77|909.26|

逐样本校验：{"formal_samples": 10528, "formal_groups": 16, "exact_all": true, "profiles_exact_all": true, "original_files": 258, "original_unchanged": true, "frozen_source_unchanged": true, "staging_bound_all": true}

## 同机交替配对复测

每组 64 条相同请求、32 条相同预热；两次运行顺序分别为旧→新、新→旧。配对包括逐样本 logits 和标签续写分数的精确核对。

|任务|预算|重复|旧 ms|新 ms|降低 %|精确一致|
|---|---:|---:|---:|---:|---:|---|
|trec|25%|0|371.31|329.92|11.1|True|
|trec|25%|1|511.59|350.13|31.6|True|
|trec|50%|0|892.64|410.63|54.0|True|
|trec|50%|1|876.26|423.66|51.7|True|
|rte|50%|0|1392.70|664.74|52.3|True|
|rte|50%|1|1281.10|773.57|39.6|True|
|trec|5%|0|145.95|130.62|10.5|True|
|trec|5%|1|173.17|136.32|21.3|True|

## TTFT 构成与重叠

每个配置另用 Nsight Systems 2025.5.1 和 VizTracer 分别采集 4 条测量请求（4 条预热）。不能把带 profiler 的绝对时间当成上表正式 TTFT。CPU 分区是墙钟区间，含等待和调度；GPU kernel/copy 使用区间并集，重叠只计算一次。GPU kernel 也包含 KV 重排等操作，不能全部当成模型计算。不同工具的区间不拼接相加。Nsight Memcpy 字节不包含 cat/index_copy 内核内部的访存；不能把 Memcpy 次数或字节下降直接解释为同等比例减少实际 KV 访问量。comparison.csv 同时保留物理 KV 字节与读放大。

详细 16 组前后分解见 profiles/round3_optimized_grid/baseline_optimized_components.csv；其中也列出 decoder 与 payload（含重排内核）、selector index 的 GPU 重叠。CPU worker 与主线程等待/选择的重叠见 cpu_worker_overlap_comparison.csv。下表为优化后 Nsight 的代表配置：

|配置|诊断墙钟 ms|KV 等待/整理 ms|缓存维护 ms|mask ms|GPU kernel ms|GPU copy ms|kernel/copy 重叠 ms|H2D 次数|mask 构造次数|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|k005/trec|202.29|35.83|9.05|0.33|28.85|0.35|0.01|105.0|2.0|
|k025/trec|412.21|130.01|42.11|0.36|33.46|1.17|0.07|276.0|2.0|
|k050/trec|604.14|277.46|64.90|0.28|36.20|3.99|0.15|208.0|2.0|
|k050/rte|666.42|319.41|100.17|0.14|47.19|5.26|0.35|200.0|1.0|

staging 池上限为整个进程 8 MiB，独立计入额外工作区，不算入 55/131 MiB payload 缓存额度；实测峰值及复用次数见 staging_memory.json。8 MiB 上限同时包含 KV 和索引暂存；GPU 输出及批量 cat 的临时张量没有池化，整体 GPU 峰值另行记录；mask 缓存最多 8 个形状、只在一次 decoder 调用内存活。

Nsight 每组提供 nsys.nsys-rep、nsys.sqlite、nsys_ttft_breakdown.json 和 ttft_timeline.svg；VizTracer 提供 viztracer_run/viztracer.json 与 viztracer_worker_breakdown.json。后者的 worker 嵌套耗时不可直接求和。

## 解释与边界

这是保持计算语义的工程优化，不应把 SUBJ 小预算的准确率问题解释成已解决；选择、复用和预算策略均未修改。也未移出计时范围来隐藏 cache-score 工作，未取消磁盘 DONTNEED 操作。GPU 0/1 的既有任务继续运行，正式实验使用 GPU 2；主机不是独占环境。每个完整配置只跑一遍，配对复测提供方向性验证，不宣称统计置信区间。

## 复现与位置

代码基线快照：audit/round3_baseline/。运行入口：scripts/run_round3.py；--variant baseline 切换快照源代码，但仍从本项目读取同一组只读模型/数据/存储。完整实验入口：scripts/round3_workflow.py。所有文件均留在服务器 /home/panzihang/src/prism_pans。

测试日志：audit/round3_preflight_full_tests.log、audit/round3_full_tests.log。结果：results/round3_optimized_grid/comparison.csv 与 final_validation.json。

## 工程改动与验证

1. sparse_qwen_reprefill.py 用向量化 triu 构造因果 mask；flexgen_qwen_reprefill.py 在一次 decoder 调用内按形状复用，最多保留 8 种形状。
2. vendor/flexgen/my_pcache_fast.py 在批量预取路径复用 pinned staging；CUDA event 决定可复用时机，GPU 源张量用 record_stream 防止缓存淘汰后过早复用存储。
3. CPU/磁盘来源跨交错 GPU 块一起打包；连续目标直接批量复制，分散目标用批量数据与索引恢复原顺序。GPU 来源在设备内批量 cat，避免逐块 D2D 提交。
4. flexgen_pcache.py 和预取入口批量生成 token/block 计数与索引，保持原有缓存评分更新及淘汰顺序。缓存维护和按需读取中的其他旧路径没有被替换为新的算法。

最终代码测试为 303 passed、87 subtests。新增检查覆盖 mask 逐元素一致性、随机重排/计数、staging 事件完成前不得复用、GPU 源淘汰时的异步生命周期，以及混合 CPU/GPU/磁盘、尾块、重排和多次异步预取的 KV 精确一致性。完整实验另外逐条核对预测、选中 token 的哈希与数量、预算、缓存更新计数、首 token logits 和标签续写 logprob。

补丁 audit/round3_optimization.patch 的基线是 audit/round3_baseline/ 中保存的上一轮固定 P8 代码；不要把它直接用于原始 prism_max。audit/round3_delivery_verified.json 保存补丁应用与交付校验结果。

以下命令在服务器项目目录运行；使用不存在的新结果目录。单组完整复现示例：

```bash
cd /home/panzihang/src/prism_pans
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /home/panzihang/venvs/vllm-stable/bin/python scripts/run_round3.py \
  --variant optimized --task trec --budget 25 --period 8 \
  --warmup 32 --output /home/panzihang/src/prism_pans/results/reproduce_trec25
```

将 --variant 改为 baseline 并更换输出目录可运行冻结的旧实现。全部 16 组及独立轨迹由 scripts/round3_workflow.py 的 --run-root、--profile-root、--gpu 参数指定新目录后顺序执行。

以这 10,528 条评测记录等权汇总，TTFT 均值为 362.28 → 235.91 ms，下降 34.9%。这是当前评测样本构成的加权描述，不代表其他任务分布。


## 剩余瓶颈的具体解释

以下 Nsight 数字来自各配置前 4 条诊断请求（另有 4 条预热），不能直接替代全量正式 TTFT。RTE、50% 的正式均值是 1223.56 → 698.86 ms，正式预取等待是 755.22 → 339.99 ms。

RTE、50% 的 Nsight 主线程互斥区间分解如下；这些 CPU 区间可以相加，GPU 活动则发生在同一墙钟窗口内，不能再加一次：

|主线程区间|优化后 ms|占诊断窗口|
|---|---:|---:|
|KV 等待与 gather|319.41|47.9%|
|缓存评分维护|100.17|15.0%|
|decoder 调用与提交|85.44|12.8%|
|选择器控制与同步|62.27|9.3%|
|其余控制、准备、调度与转换|99.13|14.9%|
|合计|666.42|100%|

同一窗口的 GPU kernel 并集为 47.19 ms，Memcpy 并集为 5.26 ms，二者重叠仅 0.35 ms；含 memset 的 GPU 活跃并集为 52.14 ms，约占墙钟 7.8%。decoder 与 payload/其他 GPU 操作重叠约 0.41 ms，与 selector index 重叠约 0.05 ms。因此本轮主要通过缩短准备、分块处理和提交路径取得收益，并未形成大量计算/传输重叠。没有记录到 GPU 活动的区间不等同于 CPU 在执行计算，仍可能包括线程等待、调度和 I/O。

同一 RTE 配置的 H2D 提交从 7284.5 → 200 次/请求，记录的 H2D 字节为 126.19 → 126.57 MB/请求；数据量基本相当，批量化主要减少了调用数量。mask 从 28 → 1 次，CPU 区间从 61.11 → 0.145 ms。旧的约 3520 次 D2D Memcpy 被设备内批量拼接/索引操作替代，其内部显存访问仍然存在。

独立 VizTracer 记录中，RTE、50% 的 worker 预取区间并集为 577.91 ms，主线程 resolve 区间为 406.27 ms，两者重叠 380.42 ms。这段重叠多数表示主线程正在等待 worker，不能算成隐藏在有效 decoder 计算下的加速。worker 与主线程选择/mask/layout 区间重叠约 7.03 ms。VizTracer 和 Nsight 是分开的运行，不按时间戳拼接两者。

剩余热点已具体落到：
- worker 的 `_batch_copy_chunk_sources`：约 448.59 ms inclusive、367.40 ms exclusive traced elapsed/请求；其中仍有逐块读取、NumPy/PyTorch 打包以及调用提交。
- `prefetch_by_chunk`：约 117.19 ms exclusive traced elapsed/请求，仍包含块枚举与时间预算处理。
- 主线程的 `prefetch_source_tensor_tokens`、`configure_impress_layer`、`_record_physical_access`、`update_impress_layer_scores` 等元数据及缓存维护路径仍可见。
- RTE、50% 正式运行中 staging pool 的容量等待计数为 0，当前证据不支持把继续扩大 pinned 池作为首要优化。

这些 exclusive traced elapsed 包含未被单独追踪的 C 扩展调用、等待和调度，不是纯 Python CPU 时间；本轮未启用 OS 调度采样，因此尚不能精确分离磁盘阻塞、GIL 等待和真正的 CPU 执行占比。后续如果继续优化，应先细分批量函数内部的逐块 host gather 与元数据循环，再决定如何实现更早的数据就绪。本轮没有继续改变缓存策略、P8 复用或添加模块。
