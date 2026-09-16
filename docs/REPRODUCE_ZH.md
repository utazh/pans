# 安装与复现

## 1. 环境

参考环境：Ubuntu 22.04、Python 3.10、RTX 3090 24 GiB、驱动 580.95.05、PyTorch 2.10.0+cu130、NumPy 2.2.6、Transformers 4.57.6。依赖文件记录本轮实际版本。Nsight Systems 2025.5.1 为可选外部工具，VizTracer 1.1.1 在开发依赖中。

从 README 安装依赖后执行 `python -m pytest -q`。测试不需要下载模型或真实数据集；CUDA 测试需要空闲 GPU。

## 2. 数据集与 strict UID

仓库不包含原始样本。以下命令使用项目已有下载/构建器，seed=42；先排除校准 UID 32–35，再排除每个数据集的 UID 0。

```bash
export PANS_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
python scripts/build_datasetwise_bundles.py \
  --data-root data/raw --output-dir data/paper_task_bundles_full_eval \
  --tokenizer "$PANS_MODEL_PATH" --seed 42 \
  --excluded-uids-manifest configs/calibration_exclusions.json

python scripts/build_strict_eval_bundle.py \
  --source-bundle data/paper_task_bundles_full_eval \
  --output-bundle data/paper_task_bundles_full_eval_strict \
  --exclusions configs/strict_eval_exclusions.json
```

校验生成的样本 UID 顺序、prefix SHA256 和 JSONL SHA256，分别参照 `reports/round4/task_uids.json` 和 `reports/round4/dataset_manifest.json`。上游数据或 tokenizer 版本变化可能影响字节哈希；不应把哈希不同的新构建称作原实验精确复现。四组行数应为 867 / 998 / 495 / 272。本次 strict 集沿用已有协议，不是新建独立 holdout。

## 3. Prefix store、Pcache 与索引

也可以直接指定已有且经过验证的缓存。若从头准备，使用新的空目录，避免覆盖已有实验。

```bash
export PANS_BUNDLE_DIR="$PWD/data/paper_task_bundles_full_eval_strict"
export PANS_STORE_ROOT="$PWD/runtime_data/prefix_store"
export PANS_PCACHE_DIR="$PWD/runtime_data/pcache_c16"
export PANS_SELECTOR_INDEX_DIR="$PWD/assets/selector_index_k4_g32"

CUDA_VISIBLE_DEVICES=0 python -m contiguous_fuxian.sparse_qwen_reprefill prepare \
  --model-path "$PANS_MODEL_PATH" --bundle-dir "$PANS_BUNDLE_DIR" \
  --store-root "$PANS_STORE_ROOT" --tasks sst2,subj,trec,rte \
  --samples-per-task 1 --dtype bfloat16 --device cuda

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m contiguous_fuxian.flexgen_qwen_reprefill \
  --model-path "$PANS_MODEL_PATH" --bundle-dir "$PANS_BUNDLE_DIR" \
  --store-root "$PANS_STORE_ROOT" --plan configs/qwen25_k005_contigkv.json \
  --output-dir results/build_pcache --flexgen-root vendor/flexgen \
  --flexgen-kv-dir "$PANS_PCACHE_DIR" --tasks sst2 --store-tasks sst2,subj,trec,rte \
  --samples-per-task 1 --max-tokens 1 --dtype bfloat16 --device cuda \
  --gpu-cache-mb 0 --cpu-cache-mb 0 --cache-type CKLFU --online-selection \
  --probe-query-heads 0,7,14,21 --selector-kv-head-ids 0,1,2,3 \
  --period-size 8 --subperiod-size 4 --expected-keep-ratio 0.05

python scripts/build_selector_index.py \
  --source-pcache-dir "$PANS_PCACHE_DIR" --store-root "$PANS_STORE_ROOT" \
  --output-dir "$PANS_SELECTOR_INDEX_DIR" --model-path "$PANS_MODEL_PATH" \
  --tasks sst2,subj,trec,rte --source-task-order sst2,subj,trec,rte \
  --selector-kv-head-ids 0,1,2,3 --group-size 32
```

Pcache 要求 chunk_size=16、在线选择、四个 KV head；不要替换为历史 chunk-64 IMPRESS 重排缓存。K4 索引必须从对应 Pcache 构建，任务注册顺序必须一致。

上述从头构建命令按项目实际 CLI 整理。发布检查使用已有实验缓存验证单配置入口，没有重新下载数据、重建全量缓存或重跑整套四预算实验。

## 4. 单配置与完整实验

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python scripts/run_round4.py \
  --task trec --budget 25 --period 8 --warmup 32 --output results/trec25

# 所有 16 组正式实验，以及分开执行的 32 份诊断采集。
export PANS_NSYS=/path/to/nsys
python scripts/round4_workflow.py --gpu 0 \
  --run-root results/full_grid --profile-root profiles/full_grid
```

单配置默认跑完全部 strict 样本；`--samples 64` 只用于小样本检查。工作流仅在目标 GPU 无计算进程时启动下一组，并校验 P8、精确预算、UID 与计时边界。中断后可在源文件和输出保持一致时用 `--resume`。

`--variant baseline` 使用 `audit/round4_baseline` 中第三轮参考实现。同期对照使用相同数据和缓存、相同 64 条 UID、32 条预热，运行顺序为旧/新/新/旧。不要同时运行两个测量进程。

## 5. 单独采集与分析

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python scripts/run_round4.py \
  --task trec --budget 50 --profile viztracer --samples 4 --warmup 4 \
  --output profiles/manual/trec50/viztracer_run

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$PANS_NSYS" profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o profiles/manual/trec50/nsys \
  python scripts/run_round4.py --task trec --budget 50 --profile nsys \
  --samples 4 --warmup 4 --output profiles/manual/trec50/nsys_run

"$PANS_NSYS" export --type=sqlite --output profiles/manual/trec50/nsys.sqlite \
  profiles/manual/trec50/nsys.nsys-rep
python scripts/profiling/analyze_round2_nsys.py profiles/manual/trec50
python scripts/profiling/analyze_round2_viztracer.py profiles/manual/trec50
python scripts/profiling/analyze_round4_payload.py profiles/manual/trec50
python scripts/profiling/plot_round4_ttft_svg.py profiles/manual/trec50
```

CPU 阶段互斥分区可合回 TTFT；worker、kernel 与 copy 用时间并集/交集分析，不能相加。host gather 的 NVTX elapsed 包含可能的 I/O、调度与 GIL 等待，不等于纯内存拷贝 CPU 时间。GPU Memcpy 统计不含 cat/index_copy kernel 内部搬运。
