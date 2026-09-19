# pans

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
