# SharedKV v2.1 服务器实验

工作目录：`/home/panzihang/src/prism_layerkv`。原始 `/home/panzihang/src/prism_pans` 不作修改。上一版代码、readers 和结果保留。

完整结果见 `EXPERIMENT_REPORT_V21_ZH.md`（实验全部完成后生成）。原始 v2.1 包在 `upstream/pans_sharedkv_v21`，本地适配差异在 `audit/v21_package_fixes.patch`。

## 实现

- 当前层正常产生 Q 和当前 query 的 K/V；当前 query 的 cache 逐层独立。
- 共享 consumer 的历史 prefix 只读取 source 层的同一物理 K/V buffer。
- 本层 A/bK 改变 prefix 读侧 Q/归一化，本层 B/bV 变换 prefix 聚合结果；与本层 private tail 用联合归一化合并。
- 新增请求内 cu_seqlens 缓存、reader 预打包、普通层 hybrid attention、window/demand payload 调度、shadow-load 诊断。
- P8 选择器保持执行。固定选择仅替换最终 IDs，不删除选择器计算。
- teacher 为同一 vLLM FA2 后端的 identity。顺序校准按 consumer 2→19→27 进行，每步重采集实际 student 的 fit/val 状态，并检查 upstream reader fingerprint。

所有新训练 reader 均保留 `diagnostic_only=true`，未通过局部门限时不自动晋升。任务准确率较好也不等同于通过局部误差门限或统计非劣验收。

## 数据与运行

旧 128 条为回归/开发集。新划分是 fit=64、val=64、holdout=175，记录于 `data/v21_splits/manifest.json`。它们与旧 SharedKV 的 fit/val/test 请求去重；都来自同一 TREC 数据池、共用同一 prefix，不是跨任务泛化实验。

Python：

```bash
cd /home/panzihang/src/prism_layerkv
PY=/home/panzihang/venvs/vllm-stable/bin/python
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
```

运行单对新 reader：

```bash
$PY scripts/run_sharedkv_v21.py --mode shared --backend vllm2 \
  --adapters assets/v21_refit27_s200 --allow-candidate \
  --metadata-cache --packed-ops --layout hybrid \
  --stats results/my_new_shared.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir data/sharedkv_splits/test --output results/my_new_shared
```

公平的同执行路径无共享对照：改为 `--mode identity`，去掉 adapters/allow-candidate/packed-ops，保留 metadata-cache/hybrid，并使用新输出路径。强系统基线为 `--mode original`，去掉所有 v21 runtime 优化选项。

三对 reader 为 `assets/v21_seq3/trec.pt`；单独早层 reader 为 `assets/v21_refit2_s200/trec.pt`。以上均为未认证候选。

批量脚本的 JSON 参数清单：

- `scripts/v21_quality_jobs.json`：teacher 与旧 7 对 isolation。
- `scripts/v21_refit_quality_jobs.json`：闭式/蒸馏、顺序组合及 hybrid 回归。
- `scripts/v21_student2_jobs.json`、`v21_student19_jobs.json`：真实 student 状态重采集。
- `scripts/v21_holdout_gpu2_jobs.json`、`v21_holdout_gpu3_jobs.json`、`v21_hybrid_seq3_holdout_jobs.json`：独立评估。
- `scripts/v21_timing_jobs.json`：11 配置 × 4 轮 × 32 请求，每组 warmup=8，正序/倒序/倒序/正序。
- `scripts/v21_fixed_jobs.json`：同 IDs 的 shared/shadow/demand 对照。

`run_layerkv_v21_suite.py` 每个配置新建 Pcache store，只在权重文件 stat 身份一致时复用冻结模型和内容指纹。所有现有输出均禁止覆盖；复跑前复制 JSON 清单并改用新的结果路径。

## Profiling

Profiler 工具的独立副本在本工程 runtime_tools/nsight-2025.5.1、runtime_deps；v21 wrapper 后续不再从原工程导入工具。

`profiles/v21/nsys_*.nsys-rep` 为 Nsight 原始轨迹，`*.sqlite` 为导出数据。
`profiles/v21/viz_hybrid_shared_retry/viztracer.json` 为通过事件数检查的 VizTracer 轨迹。

`scripts/profile_layerkv_v21.py` 是单独的 profiler wrapper，warmup 后启动/清空采集。Nsight 的 `TTFT/logits_ready` 范围含最终 CUDA 同步；标签续写在范围外。VizTracer 分析取每个 greedy completion 的第一次 decoder 调用，排除续写。

`scripts/analyze_v21_nsight.py` 和 `analyze_v21_viz.py` 生成摘要。Nsight 硬件 H2D、host 来源计数、唯一 selected payload 是不同口径，不应混称。函数 inclusive 时间、跨线程工作、GPU 操作可能重叠，不能相加当关键路径。

正式速度实验不带 profiler 或 logits 转储，待质量与 profiler 任务退出后运行。微基准是合成的 GPU resident tensor，不能当模型 TTFT；真实 query 平均长度约 41。额外 `--private-tokens` 可模拟 query=1、private tail=42 的 decode 几何。

## 修正与审计

服务器适配修正了：
1. 标签诊断按平均 token log-probability 计算，与原 runner 实际分类规则一致。
2. shadow-load 的 BF16 buffer 替换 loader 中的旧 FP16 view，避免无意保留两份。
3. logits 转储运行不会标记为正式计时。
4. VizTracer 使用库排除规则，避免 include_files 跳过 runpy 整个子调用树，并检查有效事件数。
5. 微基准允许单独指定 private cache 长度。

`audit/v21_original_integrity.json` 检查原工程 276 个文件；`audit/v21_prior_experiment_integrity.json` 检查上一版实验代码未变。包内 manifest 和 8 个发布版接口 hash 均核验通过。
