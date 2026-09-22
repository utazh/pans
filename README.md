# pans

## LayerKV v2.5：扩大共享范围与少量本层 KV 保留

基于 [v2.4](https://github.com/utazh/pans/tree/v2.4)（`a6be130c921024a4cd358ccf5841e4a46f547ae2`），发布服务器已经完成的代码、报告和验证记录。

**已测试 G=26、24、22、20、16、14 × R=0/4 的全部 12 个配置。** G 是完整 prefix KV 来源数；R4 只保留原选择集合中的最后4个本层块，其余由层专属 Q/输出变换读取共享 prefix KV。预算仍为25%，当前 query 和生成 token 的 KV 始终28层独立。G14 有14个共享消费者；预先按开发集规则选定的主配置 G16/R4 有12个消费者与16份完整来源，另有12份小型本层保留区。

- [完整实测报告](EXPERIMENT_REPORT_V25_ZH.md) · [发布说明与复现边界](docs/layerkv_v25/RELEASE_ZH.md)
- [统计分析](docs/layerkv_v25/results/v25_final_analysis.json) · [质量表](docs/layerkv_v25/results/v25_quality_table.csv)
- [校准报告](docs/layerkv_v25/calibration) · [审计](docs/layerkv_v25/audit) · [验证记录](docs/layerkv_v25/validation)
- [方法与运行文档](docs/sharedkv_v25_package/README_ZH.md) · [服务器运行说明](README_LAYERKV_V25_SERVER.md)

Qwen2.5-7B-Instruct / RTX 3090，25% 精确块预算，4940-token prefix；主计时固定种子24：

| 指标 | 无共享 | G16/R4 |
|---|---:|---:|
| 历史完整 TREC 500条，种子24 | 68.60% | 73.00% |
| 新留出1536条，种子24/25/26 | 73.24% | 74.22% / 73.05% / 74.48% |
| SSD-cold/native TTFT | 2045.01 ms | 1417.24 ms（−30.70%） |
| SSD-cold/preserve TTFT | 1802.65 ms | 1147.36 ms（−36.35%） |
| 普通缓存 TTFT | 254.52 ms | 194.33 ms（−23.65%） |

每种主冷测4轮重复同16题，普通缓存4轮重复同32题；各轮均下降，但属于有限重复请求实验，不是完整500条计时。覆盖率曲线另有两轮整网格16题对照。704个正式冷请求全部通过认证；认证范围是应用缓存和OS文件页，不含SSD设备内部缓存。模型/内核预热、冷态准备及标签评分不计入TTFT。

**严格三种子非劣验收仍未通过；所有读取器保持 diagnostic-only。** 新留出集三种子均值为73.91%，不能把同一1536题的三次训练结果合成4608个独立样本。历史500条与开发768条是已观察数据。

G16/R4 的实际 selected prefix KV 减少40.63%；计入 resident/packed reader 后，这部分净减少33.88%，不是模型总显存或持久磁盘占用降幅。Nsight 冷态 H2D 约135.10→89.90 MB/请求（各8条），直接拷贝只省约2.9 ms；恢复诊断与 VizTracer 支持主机恢复、文件访问和等待减少是主要时间收益来源。

![v2.5 覆盖率、质量与延迟](figures/v25_coverage_quality_latency.png)

服务器完整回归：**547 passed、4 skipped、87 subtests passed**。物理共享、本层保留区、28层私有KV、数据隔离及4025个旧文件保护审计通过。最终在线/回放逐token评分误差为0。公开副本的CPU验证记录单独保留；发布没有新增模型实验。

模型、读取器checkpoint、原始问题/teacher/KV、全量逐请求记录和原始profiler继续保留在服务器。新v25实现、入口、测试及完整报告按源文件哈希发布；历史路径兼容处理沿用v2.4。安装依赖及复现边界见发布说明。

以下保留历史版本说明；本分支以 v2.5 发布说明为准。

## LayerKV v2.4：最终评分校准与 SSD 冷恢复

本分支基于 [v2.3](https://github.com/utazh/pans/tree/v2.3)（`3868c6b64332f89cc0b301c75b53c35630302f0e`），发布服务器完成的 v2.4 代码、实测报告及轻量证据。

**本轮实际停在 G26：第 26 层的 prefix KV 服务第 25、26、27 层，独立来源从 28 份减到 26 份。** query 和生成 token 的 KV 仍 28 层独立；28 层都参与模型计算。虽然实现了更少来源数的拓扑规划，三个种子均未通过新开发集非劣门限，因此未执行 G24 及后续扩层。

- [发布说明与复现边界](docs/layerkv_v24/RELEASE_ZH.md)
- [完整实测报告](EXPERIMENT_REPORT_V24_ZH.md) · [服务器运行说明](README_LAYERKV_V24_SERVER.md)
- [统计结果](docs/layerkv_v24/results/v24_final_analysis.json) · [质量表](docs/layerkv_v24/results/v24_quality_table.csv)
- [校准报告](docs/layerkv_v24/calibration) · [验证记录](docs/layerkv_v24/validation) · [源码与数据审计](docs/layerkv_v24/audit)
- [真实模型回放修复](docs/sharedkv_v24/SERVER_REPLAY_REPAIR.md) · [冷缓存进程隔离](docs/sharedkv_v24/SERVER_COLD_ISOLATION.md)

新增最终候选完整续写得分校准、候选 KL、中心化得分/逐 token 误差及教师 margin 约束；主干冻结，仅训练读取器。
旧组目标、新评分目标、去 margin 目标各训练三个种子，共九个模型。新评分配方的三种子开发集平均候选 KL 从 0.015899 降至 0.006628（下降 58.3%）；新旧每步计算量不同，不能把全部变化归因于单一损失项。

Qwen2.5-7B-Instruct，25% 精确块预算，固定 4,940-token prefix，主计时 checkpoint 预先固定 seed24：

| 指标 | 优化后无共享 | 评分校准共享 |
|---|---:|---:|
| 历史完整 TREC 500 条准确率 | 68.60%（343/500） | 68.80%（344/500） |
| 新最终留出集 768 条准确率 | 76.04%（584/768） | 75.78%（582/768） |
| SSD 冷缓存 TTFT / 原读策略 | 2168.22 ms | 2098.87 ms（−3.20%） |
| SSD 冷缓存 TTFT / 读后保留文件页 | 1717.56 ms | 1582.05 ms（−7.89%） |

每种冷策略是 **4 轮 × 同一批 32 条配对请求**，共 512 次正式冷认证全部通过，不能称作完整 500 条计时。冷准备在 TTFT 外；认证覆盖应用 payload 和操作系统文件页，不声称清空 SSD 设备内部缓存。
原读策略和保留页策略的四轮块 bootstrap 95% 下降区间分别为 1.48%–5.30% 和 7.20%–8.57%，属于少量重复配对块的探索性区间。

普通缓存原同进程实验慢 14.77%，事后独立进程复核快 12.60%，**尚不能宣称不受运行条件影响的稳定加速**。两个结果全部保留。
所选 prefix KV 减少 7.12%；计入读取器及 packed 副本后，该局部缓冲净减少 5.99%，不是总显存或持久磁盘占用降幅。
新留出集来自 TREC 训练池排除历史问题后的划分；官方 500 条已被历史实验观察，不重新视为全新验收集。

![v2.4 实测汇总](docs/layerkv_v24/figures/v24_results.png)

服务器完整回归：**510 passed，4 skipped，87 subtests passed**。物理共享、28 层私有生成 KV、数据隔离、评分回放及原工程保护审计全部通过。
新 v24 算法、入口与测试文件与服务器交付逐字节一致；保留 v2.3 已发布的历史路径兼容处理。模型、读取器权重、原始数据、KV 缓存和原始 profiler 文件保留在服务器。
安装 v24 额外依赖可使用 `python -m pip install -r requirements-v24.txt`；真实 GPU 回放还依赖报告记录的 vLLM/CUDA 环境。
服务器调度脚本含实测路径，使用前须按发布说明准备输入并调整路径，不能在空克隆中直接重跑服务器编排。

以下保留历史版本说明；本分支以 v2.4 发布说明为准。

## LayerKV v2.3：多消费者联合校准与冷缓存实验

本分支基于 [v2.2](https://github.com/utazh/pans/tree/v2.2)（`fddabb7015c66c870a11810069bfbe20d5a34bbf`），发布服务器完成的 v2.3 代码与报告。
第 26 层的 prefix KV 同时供第 25、26、27 层使用；消费者 25、27 各自训练 Q/输出读取变换。当前 query 和生成 token 的 KV 继续逐层独立。
校准使用冻结的真实四层 attention、残差、norm 和 MLP，并包含所有候选标签的实际续写分支。

- [v2.3 发布说明与复现边界](docs/layerkv_v23/RELEASE_ZH.md)
- [完整实测报告](EXPERIMENT_REPORT_V23_ZH.md) · [服务器运行说明](README_LAYERKV_V23_SERVER.md)
- [结果与配对统计](docs/layerkv_v23/results/v23_final_analysis.json) · [校准报告](docs/layerkv_v23/calibration)
- [验证记录](docs/layerkv_v23/validation) · [源码与数据审计](docs/layerkv_v23/audit)

Qwen2.5-7B-Instruct，25% 精确块预算，TREC 完整官方测试 500 条：

| 指标 | 优化后无共享 | 联合共享 |
|---|---:|---:|
| 准确率 | 68.60%（343/500） | 68.40%（342/500） |
| SSD 冷缓存平均 TTFT，完整500条 | 2067.87 ms | 1962.81 ms |
| 普通缓存平均 TTFT，8轮×同32条 | 269.15 ms | 227.92 ms |

冷缓存 TTFT 平均下降 **5.08%**，配对批次 bootstrap 95% 区间为下降 2.87%～7.35%；1000个正式请求均通过逐请求冷状态验收。
所选 prefix KV 减少 **7.12%**，计入 reader 与 packed 参数后该部分净减少约5.99%。
普通缓存虽然平均下降15.32%，但顺序敏感，共享先运行的子组区间跨零，**尚未证明顺序无关的稳定加速**。
准确率下降0.2个百分点，未通过预设非劣验收；开发集下降3.125个百分点，因此没有继续扩大至14/7/4物理source。
所有 reader 仍为 diagnostic-only；500条是历史上已观察的回归数据，服务器也非整机独占环境。

![v2.3 实验汇总](docs/layerkv_v23/figures/v23_summary.png)

算法、入口和测试文件与服务器实验版本逐字节一致。模型、checkpoint、原始数据、KV cache 和大型 profiler 文件保留在服务器。
历史版本说明如下；本分支优先使用 v2.3 发布说明。

## LayerKV v2.2：GQA 组对齐

本分支 v2.2 基于 v2（b48c45d），发布最新头对齐代码与服务器实测报告。
新增按真实读取误差搜索 4×4 GQA 组匹配、24 种一对一排列、等条件校准及 200 步 reader 蒸馏。
运行时只路由短 Q、prefix 输出和 LSE，保持 source K/V 物理共享及当前 query/generated KV 逐层独立。

- [v2.2 发布说明与变更](docs/layerkv_v22/RELEASE_ZH.md)
- [完整实测报告](EXPERIMENT_REPORT_V22_ZH.md)
- [方法与运行说明](README_HEADALIGN_V22.md) · [服务器运行记录](README_LAYERKV_V22_SERVER.md)
- [结果摘要](docs/layerkv_v22/results) · [校准报告](docs/layerkv_v22/calibration) · [验证记录](docs/layerkv_v22/validation)

本轮局部平均误差下降，但未观察到分类准确率收益或头对齐带来的稳定加速。
18→19：同编号 68.00% / 249.0 ms，头对齐 66.86% / 273.3 ms；
26→27：同编号 72.00% / 301.7 ms，头对齐 71.43% / 278.5 ms。
准确率来自此前已观察过的 175 条 TREC 回归数据；TTFT 来自同一 32 条请求的四轮配对计时。
不同轮次的绝对 TTFT 不直接比较。所有 reader 仍是 diagnostic-only 候选。


## LayerKV v2（SharedKV v2.1 实验实现）

本分支 v2 基于已发布的 v1（59312f2），对应服务器最新完成的 SharedKV v2.1 实验。
当前层独立计算 Q 和当前 query 的 K/V；历史 prefix 使用物理共享 K/V，并通过层专属 Q/输出变换近似原层行为。

- [v2 发布说明与复现边界](docs/layerkv_v2/RELEASE_ZH.md)
- [运行与校准说明](README_LAYERKV_V21.md)
- [完整实验报告](EXPERIMENT_REPORT_V21_ZH.md)
- [轻量结果快照](docs/layerkv_v2/results) · [审计](docs/layerkv_v2/audit) · [验证日志](docs/layerkv_v2/validation)

Qwen2.5-7B / TREC：独立 175 条 holdout；TTFT 为四轮配对测量。
原系统为 70.29% / 274.0 ms，优化后无共享为 70.86% / 203.6 ms，
单对共享 26→27 为 72.00% / 206.9 ms，三对共享为 69.14% / 206.9 ms。
运行时优化带来主要延迟收益；尚未证明跨层共享本身稳定加速。
所有新 reader 仍为 diagnostic-only 候选，未通过严格局部误差门限。


## LayerKV v1

跨层共享 prefix K/V 的最新研究实现见 [运行说明](README_LAYERKV.md) 与 [实测报告](EXPERIMENT_REPORT_ZH.md)。当前 query/生成 token 的 KV 仍逐层独立。严格校准门限默认开启；报告中的共享配置为诊断实验，尚未证明稳定加速。


固定 P8 的 ProMixed KV 选择、缓存与 TTFT 实验代码。当前版本对应第四轮 host gather 与元数据优化。

- 模型：Qwen2.5-7B-Instruct；BF16 计算、FP16 KV。
- 选择层：0 / 8 / 16 / 24；固定 P8；coverage=0.5；关闭自适应 coverage。
- 精确全局块预算：5%、10%、25%、50%；GPU / CPU KV 缓存：55 / 131 MiB；staging 上限 8 MiB。
- 数据集：SST2、SUBJ、TREC、RTE。strict 评测集分别为 867、998、495、272 条，共 10,528 次四预算评测。
- 本轮改动：CPU 源块批量 gather 至 pinned slab；映射数组与选择元数据复用；来源统计在消费时读取最新驻留状态。

[安装与复现](docs/REPRODUCE_ZH.md) · [实验结果](reports/ROUND4_OPTIMIZATION_REPORT_ZH.md) · [上传版本说明](docs/PUBLICATION_ZH.md) · [第三方说明](THIRD_PARTY_NOTICES.md)

## 快速开始

Python 3.10、Linux、NVIDIA CUDA GPU。先安装适合驱动的 PyTorch；实测版本为 2.10.0+cu130。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

GPU 测试验证异步传输、尾块与缓存释放后的存储生命周期；没有 CUDA 时这些测试会跳过。发布前在服务器验证：307 tests、87 subtests 通过，保留一项既有只读 memmap 警告。

准备好模型、strict bundle、prefix store、chunk-16 Pcache 和 K4 索引后：

```bash
export PANS_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export PANS_BUNDLE_DIR=/path/to/paper_task_bundles_full_eval_strict
export PANS_STORE_ROOT=/path/to/prefix_store
export PANS_PCACHE_DIR=/path/to/pcache_c16
export PANS_SELECTOR_INDEX_DIR=/path/to/selector_index_k4_g32
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python scripts/run_round4.py \
  --task trec --budget 25 --period 8 --warmup 32 --output results/trec25
```

模型、数据集、缓存和原始 profiler 文件不随 Git 仓库分发，准备步骤见复现文档。所有输出目录须位于仓库内且首次运行时不存在。

## 结果与测量边界

第四轮与第三轮历史完整运行比较：按当前样本构成加权的平均 TTFT 为 **235.91 → 226.84 ms（下降 3.85%）**，9 组变快、7 组变慢；该加权数仅是工程汇总，不是跨任务准确率指标。

同期交替顺序对照的平均 TTFT 改善：TREC 25% 为 13.7%、TREC 50% 为 21.4%、RTE 50% 为 9.1%、SST2 10% 为 9.7%；RTE 5% 基本持平。完整 10,528 条预测、选择结果、预算、评分更新计数和标签 logits/logprobs 均逐项一致。

TTFT 使用 logits-ready 边界，不含模型/索引预载、tokenizer 和服务排队。正式计时不启用 profiler；32 份诊断采集与正式计时分开。服务器不是独占环境，不提供统计显著性结论。详见 [16 组完整对比](reports/round4/comparison.csv) 和 [时间线](reports/round4/timelines)。

## 目录

|目录|用途|
|---|---|
|src/contiguous_fuxian|选择、缓存桥接与 Qwen 运行实现|
|vendor/flexgen|既有 FlexGen / Pcache 相关实现及本地修改|
|scripts/run_round4.py|当前单配置入口|
|scripts/round4_workflow.py|16 组完整实验和 32 份 profiler 采集|
|scripts/profiling|Nsight、VizTracer 分析与时间线生成|
|configs|实验配置与 strict UID 排除规则|
|tests|单元测试与回归测试|
|audit/round4_baseline|同期对照所需第三轮源代码参考快照|
|reports|实验汇总、校验记录与轻量时间线|

其余历史脚本保留用于结果追溯，部分带有旧实验的路径或默认参数。复现当前方案请以 `run_round4.py` 和本文档为入口。
