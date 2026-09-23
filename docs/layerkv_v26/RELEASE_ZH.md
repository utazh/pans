# v2.6 发布说明

本分支以 v2.5 提交 5e523ec7f48b99b54cf449ace7b8f95e94c7e8db 为父版本，只附加 v26 实现、实验脚本、配置、报告和证据。服务器原工程、运行目录和 v2.5 公开适配保持不变；历史 runner 的可移植测试修复继续保留。

## 实际发布内容

新增 src/contiguous_fuxian/sharedkv_v26、tests/test_sharedkv_v26.py、scripts/*v26*、configs/v26*、docs/v26 以及完整报告。结果、Nsight/VizTracer 汇总、冻结 checkpoint 哈希、数据修正记录、校准报告、测试日志位于本目录。audit/publication_source_sha256.json 记录从服务器复制的文件和 SHA256；完整报告保持原始字节。

不发布模型权重、reader checkpoint、原始问题/teacher 张量、物理 KV、全量逐请求输出、原始 profiler 或服务器交付 ZIP。相关文件仍在服务器，空克隆不能直接复现实模型结果。服务器归档为 pans_sharedkv_v26_server_impl.zip，282218693 字节，SHA256：223ac0a9e1574182aebb807db8ce14d762297e367be2ad294130fdbfa971673a。

报告中的 results/、audit/ 路径描述服务器原始布局；对应公开轻量证据位于 docs/layerkv_v26/results、audit、validation。配置保留实际服务器路径及旧过程记录，不能直接重启历史调度器；迁移时配置自有模型、数据和新输出目录。冻结来源图可能随整数预算而改变，TREC/p0/25% 复用历史 checkpoint，其余格重新校准，不能将跨格比较视为严格单变量实验。

## 验证与复现

沿用 requirements.txt、requirements-dev.txt、requirements-v24.txt。匹配的 CUDA/PyTorch/vLLM 环境版本见 audit/v26_runtime_versions.json。仅 CPU 验证命令：

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q tests -p no:cacheprovider

服务器模型实验已经完成，日志 validation/v26_full_gpu_tests.txt 为 601 passed、4 skipped、87 subtests passed。公开副本测试另存 publication_cpu_tests.txt，不替代真实模型质量和计时结果。关键继承接口的 Git blob guard 保持启用。

模型主干冻结，仅离线拟合读取器。每个新格使用 64 fit / 64 validation、200 步、seed24，按预定 validation 目标保存；数据/前缀/预算适配成本不是零，见 results/v26_offline_costs.json。没有验证 14B 或自由生成；历史非劣失败保留。15/16 主质量格下降，不能宣称跨任务无损。

原生块 source-proxy 结果只是质量诊断，没有重新选作正式速度模型。Nsight 与 VizTracer 为小样本带探针诊断，主机恢复/等待支持解释大部分收益，不能把 profiler 重叠区间相加或只用 DMA 时间解释全部延迟。

Git 文本规则可能将 CSV 的 CRLF 规范化为 LF；来源清单记录原始复制字节，Git blob 校验记录实际发布字节。
