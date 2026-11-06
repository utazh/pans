# v2.7 服务器交付
主报告：EXPERIMENT_REPORT_V27_ZH.md
质量：results/v27_quality.csv 与 results/v27_quality_analysis.json
SSD 冷恢复：results/v27_cold_analysis.json
保护审计：audit/v27_final_protection.json
本轮只做验证与比较，没有写论文。TREC/25% 全部质量结果是完整500条已观察回归数据，不是新最终留出集。
原始用户包 SHA256：149353f039bc4caa64c381c3a10ead26eecfc014def5ed1aab10a61793443903

执行链：
1. v27_prepare_caches.py / v27_stage_a2.py：独立缓存、固定选择集合诊断。
2. v27_online_sequence.py：I、R4、proxy、R0 完整在线质量。
3. v27_compile_sequence3.py / v27_compile_full64.py：FP32与FP64因子验证。
4. v27_search_kvsharer.py：作者距离/remap规则的固定prefix搜索。
5. v27_baseline_quality_sequence.py / v27_bf16_sequence.py：外部质量桥与BF16作者尺度因子。
6. v27_cold_sequence.py：独立三块×32请求×三臂冷测。
7. v27_analyze_results.py / v27_write_report.py：汇总。
逐条真实命令见 configs/v27_*_command.json；有效运行覆盖参数见 results/*.effective_cli.json。
脚本使用独占输出创建；不要直接对已有结果重复执行。复现应在新的实验目录/输出名称下进行，保留旧结果。

外部质量桥尚未接入SSD因子I/O/作者SR landmark，因此它的原生日志TTFT不可用于性能比较。
KVSharer-Prefix 不等于原作者完整模型复现；xKV-CLF/P8 不等于官方xKV-SR系统。
R0保留训练好的原R4 reader并仅改推理策略；没有重训，也不能标成training-free方法。

失败日志保留：旧v26 cache副本已不存在的首次启动；默认SVD精度验证失败；FP32全秩模型分数门限失败。
这些均未改写成成功。后续新v27独立副本、gesvd、FP64数学桥分别处理相关问题。

大型 prefix/cache、模型、外部因子/source张量及原始logits均保留服务器，不纳入实现归档。
audit/v27_checkpoint_inventory.json 记录checkpoint/因子路径、大小与SHA。
