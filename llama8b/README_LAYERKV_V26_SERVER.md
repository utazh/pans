# v2.6 服务器实验复现说明

本轮工作目录：`/home/panzihang/src/prism_layerkv`。原工程 `/home/panzihang/src/prism_pans` 只读保护。代码、数据、缓存和结果均留在服务器。

- Python：`/home/panzihang/venvs/vllm-stable/bin/python`
- 模型：`/data1/llm/Qwen/Qwen2.5-7B-Instruct`
- 环境：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH=src:runtime_deps_v24`
- 当前运行使用 GPU 2、3；正式计时在 GPU 3 串行执行。
- 输入补丁：`pans_sharedkv_v26.zip`；安装器仅增加文件。原有文件保护清单为 `audit/v26_before.json`。

最终报告为 `EXPERIMENT_REPORT_V26_ZH.md`，完整主矩阵为 `results/v26_matrix.csv`。报告只有在全部阶段成功后才生成；`audit/v26_report_ready.json` 表示待人工核验，`audit/v26_archive.json` 表示已校验归档。不能把正在增长的日志或 `audit/v26_live_quality_snapshot.json` 当成最终结果。

## 数据和冻结模型

SST-2 872、TREC 500、RTE 277、SUBJ 自定义1003条完整句子；另做第二独立前缀25%复核。主矩阵为四任务×5/10/25/50%。G16/R4、seed24；除复用历史 TREC/p0/25% checkpoint 外，每格64 fit、64 validation、200步校准，按预定验证目标保存。没有按完整评估结果选种子。

SUBJ 旧读取器将 `0x85` 当成换行。修正数据位于 `data/v26/p*/canonical_*`；主结果、模型及命令使用 `_canonical` 命名。原数据与原结果保留，详见 `audit/v26_subj_canonical_correction.json`。SUBJ 保持冻结成员，不能称为标准1000条划分。

每格 checkpoint SHA 在 `audit/v26_*_frozen.json`。实际来源图在 `configs/v26_*_plan.json`，历史主配置在 `configs/v25_explore/g16.json`。同一G16来源数不等于所有预算使用完全相同的来源图。

## 命令和执行阶段

每次运行的完整外层命令保存在 `configs/*_command.json`，最终实际服务参数在 `results/*.effective_cli.json`。继承的内层 `command.json` 是覆盖前模板。已有输出禁止覆盖；复跑须使用新输出名称和独立缓存副本。

1. `v26_training_worker.py` 与 `train_subj_canonical_v26.py`：教师采集和读取器校准。
2. `v26_quality_worker.py`：完整准确率评估。
3. `v26_formal_sequence3.py`：8×16不同query block的四臂主研究，以及20格×128配对请求的描述性矩阵。主研究每臂每块独立进程；矩阵每臂独立进程，无独立重复进程区间。
4. `v26_diagnostics_sequence3.py`：单位读取器、source-proxy原生块诊断和Nsight/VizTracer。
5. `v26_delivery_sequence2.py`：汇总、原工程保护审计、图表和报告。
6. `archive_v26.py`：核验后归档，包含文件SHA及ZIP CRC校验。

这些调度器为本轮一次性执行记录，包含已有状态文件与PID检查；恢复状态以 `audit/v26_continuation_state.json` 和实际进程为准，不要直接重启已有任务。可用 `scripts/status_results_v26.py` 读取当前已完成的主质量配对。

## 结果边界

完整分类准确率与正式TTFT分开。冷态为模型/索引已预热、目标payload应用缓存与OS页均为空；每个正式冷请求必须通过认证。原晋升与预取复用晋升同时用于无共享和共享，以区分系统优化和跨层共享贡献。source-proxy仅作质量诊断，不计入正式速度结果。

本轮未验证14B、自由生成质量，也没有完整复现KVSharer/xKV。所选payload和读取器字节数不是模型总显存，完整28层KV仍持久化。历史非劣认证失败不被本轮探索覆盖。

原始profiler轨迹、模型权重、物理KV缓存和teacher张量留在服务器，归档只收纳实现、配置、reader、数据划分和派生证据。运行时版本见 `audit/v26_runtime_versions.json`。
