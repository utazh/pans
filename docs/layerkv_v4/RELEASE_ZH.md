# v4 发布与复现范围

公开 v4 保留内部 v29 名称，以对应服务器脚本、冻结规则和原始运行记录。基于 v3 的历史实现与兼容路径处理，新增 v29 编排脚本、配置、冷页驱逐重试及既有结果。

## 文件

scripts/*v29* 为服务器实际脚本；src/contiguous_fuxian/sharedkv_v23/cold.py 是本轮修改的运行模块。source_manifest.json 记录本次复制文件哈希。历史脚本在 v3 中已有可移植性处理，本次没有用服务器旧绝对路径版本覆盖它们。

results 包含完整正负汇总、开发筛选、机制对照、主矩阵、行恢复/xKV、QuALITY、P4/P8 和成本摊销。
evidence/*.csv.gz 保留逐请求 run/UID/答案/预测/正确性/TTFT/两个字节字段，来自 1808 个 scored_records 文件，共 136027 条记录，包含开发、校准、测试、计时与失败尝试中的已生成记录，**不是 136027 个独立测试样本**。按目录、运行名和 protocols 的 valid block 过滤才能重建正式表。不应将校准/质量运行的 TTFT 当成正式计时。
evidence_manifest.json 为原始完整 JSONL 的 SHA256、大小和记录数；不是压缩 CSV 的哈希。per_run_summary.json 只是全量索引，不是自动筛选后的正式表。
protocols 保留原始 plan/blocks/done；失败及门控阻塞尝试不删除。row2_direct_recheck.json 单独从有效块重算，并检查测量请求驻留页。
审计报告保留生成日期；后续实验可超出早期审计覆盖范围。

## 服务器复现条件

这些编排脚本绑定 /home/panzihang/src/prism_layerkv、既有 v26/v28 数据/reader 和 vLLM 环境，不是在空克隆中直接运行即可得到原表。首先阅读 configs/v29_selection_rule.json、audit 中的规则冻结与最终选择，再准备相应的模型、数据、prefix store、索引、reader 和因子文件。运行涉及 GPU0–3 的脚本前应按当前空闲情况调整脚本，不自动占用这些卡。

主要入口：
- prepare_v29_dev.py、v29_a1_select.py、v29_freeze_final.py：开发和冻结选择。
- v29_timing.py / v29_timing_report.py：主计时矩阵。
- v29_rows_export.py、v29_xkv_export.py、v29_rowengine_entry.py、v29_timing_row2.py：统一行恢复。
- prepare_v29_quality.py、v29_quality_pipeline.py：QuALITY 准备、训练和评估。
- v29_subj_p4.py、v29_timing_subj_p4*.py：P4 对照。
- v29_p1c_report.py：同块配对统计，A_vs_B = A/B−1。

本次 CPU 验证在发布副本运行：
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:/home/panzihang/src/prism_layerkv/runtime_deps_v24 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/panzihang/venvs/vllm-stable/bin/python -m pytest -q tests

## 不进入 Git 的文件

模型、reader 权重、KV/teacher cache、低秩因子、原始数据正文、完整 logits、运行依赖目录、原始 profiler、大型生成排序配置留在服务器。configs/v29_impress_reorder_p0_fit4.json 为约 8.5 MB 的生成排序中间文件，不纳入此轻量发布；相应生成脚本保留。原始 JSONL 留在服务器，发布紧凑逐请求证据。未新增论文文件。

没有改动原 prism_pans，也没有移动或清理当前实验数据。本次 publication 目录为独立副本。当前代码哈希只能标识发布快照，不能证明它与每一个历史运行时源码完全相同。
