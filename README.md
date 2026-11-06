# v3090：3090 服务器最终实验快照

本分支是 RTX 3090 服务器上 `~/src/prism_layerkv`（Qwen2.5-7B）和 `/data1/pans_l8/raps_l8`（Llama-3.1-8B）两棵工作树的原样快照，打包于 2026-11-06。代码最后一次修改是 2026-10-28 的 `scripts/v36_final_report.py`。快照覆盖 v29–v36 的全部实验：RAPS-P4 方法冻结、7B/Llama 的 4 个任务（25% 预算）、QuALITY 12 篇文章，以及 8 种方法（RAPS4、NS4、NS8、ContiguousKV、IMPRESS、AS+H2O+LRU、CAKE、LAVa）的精度、冷启动和热启动 TTFT。

- 父提交是 `v6`（旧 A100 快照）。本分支的工作树整体换成了 3090 上的文件，v6 独有的文件仍保留在 git 历史里。
- 文件逐字节保持不变：`.gitattributes` 设为 `* -text`，不做换行转换。仓库里的守护脚本（`src/contiguous_fuxian/sharedkv_v*/guard.py`）会校验文件哈希，所以不要用会改换行的编辑器保存受守护的文件。
- 完整性校验：下载后执行 `sha256sum -c env_3090/MANIFEST_3090.sha256`，应输出 21100 个 OK。这份清单是在 3090 上对原文件直接计算的。

## 目录

| 路径 | 内容 |
|---|---|
| `src/`、`scripts/`、`configs/`、`tests/`、`upstream/`、`vendor/flexgen/`、`runtime_deps/` | 7B 主树的全部代码（不含 `__pycache__`；只收 2 MB 以下的文件，被排除的大文件都是 FlexGen 自带的 profile 或数据样例） |
| `audit/` | 规则冻结、选择结果、守护哈希等审计文件（不含 `.pt`，只收 2 MB 以下的文件） |
| `data/` | 全部任务数据：`v26/p0`（fit/val/eval）、`v29/p0/dev`、`v29/quality`（QuALITY 12 篇，含合成校准问题）、`paper_task_bundles_*` 等 |
| `studies/v28_frozen/{bundles,manifest.json,gates,jobs}` | 冻结的评测 bundle（计时和基线都读这里） |
| `studies/v29*`、`studies/v3*` | 每次运行的 `summary.json`、`*.effective_cli.json`、`*.context.json`、`blocks.jsonl`、报告和链脚本 |
| `studies/v30/p4_readers/{sst2/G24,trec/G16,rte/G20}`、`studies/v29/subj_p4/readers/G20`、`studies/v30/qa_p4/*/readers/G24` | 7B 最终选中图的读取器（RAPS-P4），可直接复现精度、不必重新训练 |
| `results/` | 汇总结果（不收 logits 和 5 MB 以上的文件）；**最终表在 `results/v36/final.csv` 和 `final.json`** |
| `llama8b/` | Llama-3.1-8B 工作树：代码、运行汇总、`results/v30/p4_selection.json`，以及最终读取器 `studies/v30/p4_readers/{sst2/G24,trec/G23,rte/G27}`（SUBJ 由 dev 规则选了不共享） |
| `env_3090/` | `pip_freeze_3090.txt`、原始 requirements、模型权重 sha256、`runtime_data`/`assets` 目录清单、`MANIFEST_3090.sha256` |
| 根目录 `*.md`、`EXPERIMENT_REPORT_*` | 历次实验报告（V21–V28） |

**没有上传、需要重建的内容：**
- `runtime_data/`（53 GB）：prefix store、pcache、teacher 采集、IMPRESS/AS 缓存、QuALITY store。
- `assets/`（6.3 GB）：选择器索引、QuALITY 索引、v27 资产。
- 逐请求记录 `scored_records.jsonl`（3.5 GB）、各种 logits、未选中的读取器、模型权重。

## 环境（3090 上的实测环境）

- **硬件与系统：** Ubuntu 22.04.5，4×RTX 3090，驱动 580.95.05。
- **软件：** Python 3.10.12，`torch==2.10.0+cu130`、`vllm==0.19.0`、`transformers==4.57.6`，完整列表见 `env_3090/pip_freeze_3090.txt`。
- **系统设置：** 运行前执行 `ulimit -n 1048576`，否则 pcache 会报 "Too many open files"。

```bash
python3.10 -m venv /home/panzihang/venvs/vllm-stable
/home/panzihang/venvs/vllm-stable/bin/pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
/home/panzihang/venvs/vllm-stable/bin/pip install -r env_3090/pip_freeze_3090.txt
```

如果驱动不支持 CUDA 13，改用 cu128 版的 torch 2.10.0。在 A100 上验证过：PyPI 上的 vllm 0.19.0 链接的是 `libcudart.so.12`，搭配 cu128 的 torch 可以正常运行。

## 路径约定（必须照做）

代码和链脚本里写死了绝对路径（约 7000 处）。最简单的办法是按原来的位置放置，或者用软链接：

| 原路径 | 内容 |
|---|---|
| `/home/panzihang/src/prism_layerkv` | 本仓库根目录（clone 后建软链接即可） |
| `/home/panzihang/venvs/vllm-stable/bin/python` | Python 解释器 |
| `/data1/llm/Qwen/Qwen2.5-7B-Instruct` | 7B 权重（用 `env_3090/sha_Qwen2.5-7B-Instruct.txt` 校验） |
| `/data1/llm/meta/Meta-Llama-3.1-8B-Instruct` | Llama 权重（用 `env_3090/sha_Meta-Llama-3.1-8B-Instruct.txt` 校验） |
| `/data1/pans_l8/raps_l8` | 本仓库的 `llama8b/` |
| `/data1/pans_l8/work_l8` | Llama 的数据和缓存（`llama8b/data` 原来是指向 `work_l8/data` 的软链接） |
| `/data1/pans/pans_kv/pcache_{gpu2,gpu3,a2_gpu2,a2_gpu3}` | `runtime_data/pcache_c16` 的副本（多卡并行评测用） |

```bash
git clone -b v3090 https://github.com/utazh/pans.git ~/pans_v3090
mkdir -p /home/panzihang/src /data1/pans_l8
ln -sfn ~/pans_v3090 /home/panzihang/src/prism_layerkv
ln -sfn ~/pans_v3090/llama8b /data1/pans_l8/raps_l8
cd /home/panzihang/src/prism_layerkv && sha256sum -c env_3090/MANIFEST_3090.sha256 | grep -vc ': OK$'   # 应为 0
```

部分路径可以用环境变量覆盖：`PANS_MODEL_PATH`、`PANS_PROBE_QUERY_HEADS`、`PANS_SELECTOR_KV_HEAD_IDS`、`V30_ROOT`、`V30_MODEL`、`V30_PCACHE`、`V30_STORE`、`V30_BUNDLE`、`V30_STEPS`、`V30_TAG`、`V33_STUDY`。Llama 的用法见 `studies/l8pipeline.sh`。

## 重建运行时缓存

`env_3090/runtime_data_list.txt` 和 `assets_list.txt` 列出了 3090 上原有的条目。最终实验需要的缓存如下：

1. **prefix store、ContiguousKV pcache（chunk 16）和 K4 选择器索引：**
   - 命令模板见 `studies/l8build.sh`（3090 上 Llama 用的版本）和 v6 分支的 `server_a100/xfer/build_base.sh`。
   - 7B 的参数为 `--probe-query-heads 0,7,14,21 --selector-kv-head-ids 0,1,2,3`。
   - 生成三个目录：`runtime_data/prefix_store`、`runtime_data/pcache_c16`、`assets/selector_index_k4_g32`。
   - 7B 前缀长度应为 3811/4412/4930/5986 token，token 哈希见 `audit/v26_prefix_build.json`。
2. **计时用 pcache：** 把 `pcache_c16` 复制到 SSD 上的 `runtime_data/v29_pcache_gpu2_ssd`；多卡评测的副本放在 `runtime_data/v27_pcache_gpu{2,3}` 和 `/data1/pans/pans_kv/pcache_a2_gpu{2,3}`。
3. **IMPRESS：**
   - 用 `configs/v29_impress_reorder_p0_fit4.json` 运行 `scripts/run_round4_methods.py --method impress --build-kv`，生成 `runtime_data/v29_pcache_impress_c64_ssd`。
   - 用 `scripts/v29_fill_impress_head.py` 补齐缺失的选择器 key 文件。
   - 完整流程见 v6 的 `server_a100/xfer/build_impress.sh`。
4. **AS+H2O+LRU：** `scripts/v33_as_setup.py runner && scripts/v33_as_setup.py build 0`，生成 `runtime_data/v33_pcache_as_c64_ssd`。完整链见 `studies/v33_chain.sh`。
5. **CAKE/LAVa 选择：** `scripts/v30_kvscore.py prefix <task>` 和 `scripts/v30_select_build.py static ...`，结果在 `runtime_data/v30_kvscore`。见 `studies/v30/chainA.sh` 和 `chainB.sh`。
6. **QuALITY：** `scripts/v29_quality_pipeline.py <f0|f1|f2> build` 生成 `runtime_data/v29_quality_<set>_{store,pcache}` 和 `assets/v29_quality_<set>_index`；基线缓存见 `studies/v35_qa_baselines.sh`。
7. **Llama：** 按 `studies/l8prep.sh` 和 `studies/l8build.sh` 执行，再跑 `studies/l8pipeline.sh` 和 `l8pipeline2.sh`。Llama 的数据是由 7B 数据转换来的：只替换对话标记，使用 plain completion 格式；转换脚本是 v6 分支的 `server_a100/xfer/convert_llama.py`。

## 复现顺序（7B）

链脚本里是当时实际执行的命令，按下面的顺序运行：

1. **读取器：** `scripts/v30_p4_readers.py <task> <gpu> <cores>`；SUBJ 用 `scripts/v29_subj_p4.py`。如果直接用仓库里的读取器，可以跳过这一步。
2. **dev 选图：** `scripts/v30_quality.py ns_p4_dev,p4_dev <gpu>`，然后 `scripts/v30_report.py select-only`，结果写到 `results/v30/p4_selection.json`（应选出 SST-2 G24、SUBJ G20、TREC G16、RTE G20）。
3. **测试集：** `scripts/v30_quality.py ns_p4_test,p4_test <gpu>`；基线：
   - CKV 和 IMPRESS 用 v29 c_quality；
   - AS 用 `studies/v33_chain.sh`；
   - CAKE 和 LAVa 用 `v30_quality.py cake_static,lava_static`。
4. **计时：** `scripts/v33_timing.py`（8 种方法 × 冷/热 × 4 任务 × 6 块）；QuALITY 计时用 `scripts/v35_timing_qa.py`；Llama 计时用 `studies/v34_chainT*.sh`。
5. **最终表：** `scripts/v36_final_report.py`，生成 `results/v36/final.csv`。它要读逐请求记录，所以必须先跑完前面几步。

计时注意事项：
- 计时进程用 `taskset` 绑定到 28-31,60-63 号核，模型用 GPU 2。
- 冷启动用 `POSIX_FADV_DONTNEED` 清空页缓存，并逐请求校验。
- 计时期间磁盘必须空闲。3090 当时后台有持续的高 CPU 负载，所以应当比较同一块内的相对加速，不要比绝对 TTFT。

## 最终结果（`results/v36/final.csv`，25% 预算，测试集）

精度（%）：

| 模型 / 任务 | RAPS4（选中图） | NS4 | NS8 | CKV | IMPRESS | AS | CAKE | LAVa |
|---|---|---|---|---|---|---|---|---|
| Qwen7B SST-2 | 94.27 (G24) | 94.27 | 93.81 | 94.38 | 94.04 | 94.61 | 94.27 | 94.15 |
| Qwen7B SUBJ | 82.35 (G20) | 82.95 | 55.03 | 46.36 | 78.86 | 87.34 | 77.17 | 77.57 |
| Qwen7B TREC | 77.6 (G16) | 63.4 | 68.6 | 50.6 | 62.4 | 62.4 | 62.0 | 62.6 |
| Qwen7B RTE | 87.0 (G20) | 89.53 | 87.36 | 87.0 | 88.09 | 89.17 | 89.17 | 89.89 |
| Llama8B SST-2 | 94.61 (G24) | 94.84 | 95.3 | 95.3 | 94.5 | 95.53 | 94.72 | 95.18 |
| Llama8B SUBJ | 91.92 (I) | 91.92 | 91.33 | 91.33 | 89.13 | 93.12 | 87.44 | 84.95 |
| Llama8B TREC | 73.8 (G23) | 73.6 | 74.6 | 77.4 | 75.0 | 79.4 | 68.6 | 68.4 |
| Llama8B RTE | 86.28 (G27) | 88.09 | 88.09 | 87.36 | 87.0 | 88.45 | 88.81 | 89.17 |
| Qwen7B QuALITY | 71.88 | 74.55 | 68.75 | 65.62 | 75.89 | 78.57 | 61.16 | 61.61 |

QuALITY 的完整 KV 精度是 77.23。

冷启动 TTFT 是 RAPS4 的多少倍（同块几何平均，括号内为 RAPS4 的绝对 TTFT，单位 ms）：

| 模型 / 任务 | RAPS4 | NS4 | CKV | IMPRESS | AS | CAKE | LAVa |
|---|---|---|---|---|---|---|---|
| Qwen7B SST-2 | 1.00 (1093) | 1.15 | 1.21 | 2.77 | 2.79 | 1.11 | 1.11 |
| Qwen7B SUBJ | 1.00 (1029) | 1.35 | 1.44 | 3.43 | 3.42 | 1.28 | 1.27 |
| Qwen7B TREC | 1.00 (923) | 1.64 | 1.82 | 4.20 | 4.28 | 1.55 | 1.55 |
| Qwen7B RTE | 1.00 (1336) | 1.38 | 1.50 | 3.51 | 3.43 | 1.39 | 1.37 |
| Llama8B SST-2 | 1.00 (1244) | 1.27 | 1.47 | 3.68 | 4.95 | 1.25 | 1.24 |
| Llama8B SUBJ | 1.00 (1810) | 1.01 | 1.11 | 2.76 | 3.85 | 0.93 | 0.98 |
| Llama8B TREC | 1.00 (1522) | 1.35 | 1.68 | 3.60 | 5.17 | 1.25 | 1.32 |
| Llama8B RTE | 1.00 (2054) | 1.16 | 1.37 | 3.16 | 4.52 | 1.11 | 1.19 |
| Qwen7B QuALITY | 1.00 (1609) | 1.16 | 1.36 | 3.31 | 2.88 | 1.09 | 1.11 |

热启动 TTFT、每请求读取量（MB）和 GPU 命中率见 `final.csv`。热启动时，CAKE/LAVa 的静态选择在多数任务上比 RAPS 快（0.56–0.92 倍），只有 TREC 例外。
