# v2.7 发布与复现边界

在v2.6上附加v27实现、脚本、测试、配置、报告与轻量统计。新增实验源码与服务器交付逐字节一致，映射和SHA256见 audit/server_source_manifest.json。历史公开版本的路径兼容修改保持不变。

本轮不重训reader、不包含新的四数据集四预算实验，不作跨任务非劣或外部系统速度认证。完整结果和失败数值验证见根目录 EXPERIMENT_REPORT_V27_ZH.md。

## 文件与复现

报告中的 results/v27_* 与 audit/v27_* 是服务器原运行路径；GitHub轻量统计位于本目录 results/、audit/。原始逐请求记录、logits、模型、reader checkpoint、完整prefix/cache、外部因子/来源张量和实现ZIP留在服务器。路径与哈希见 audit/v27_checkpoint_inventory.json、audit/v27_archive.json。

复现需准备对应模型、校准/评估划分、prefix store、Pcache、K4索引和v25 seed24 R4 checkpoint。编排脚本保留服务器绝对路径，必须在独立输出目录调整后执行，不能覆盖已有实验；空克隆不能直接重跑全部模型实验。成功执行链见 README_LAYERKV_V27_SERVER.md，失败尝试一并保留。

冻结R0入口：scripts/export_frozen_r0_v27.py。基线编译/质量入口：scripts/compile_prefix_baseline_v27.py、scripts/run_baseline_quality_v27.py。实际KVSharer作者距离/remap搜索由 scripts/v27_search_kvsharer.py 执行；baseline_math.py通用参考搜索不是该实际脚本的等价替代。

## 基线来源与范围

- KVSharer: https://github.com/yangyifei729/KVSharer/tree/39e5c19176940f8d0089829fa8aab7992f85cb37
- xKV: https://github.com/abdelfattah-lab/xKV/tree/05d91ecb0d698279aa4220fda5d7a5108036d692

本发布提供项目适配代码及作者来源/哈希记录，不再分发服务器保存的第三方notebook/源码快照。原输入包中的论文草稿和论文计划不纳入发布。docs/sharedkv_v27是原方案材料，其未执行计划不能当作已完成实验。

服务器完整测试656 passed、4 skipped、87 subtests passed，原日志在 validation/server_full_tests.log。发布副本CPU验证禁用CUDA，结果在 validation/publication_cpu_tests.log。本次发布不重跑模型质量和冷恢复实验。

发布副本实测：628 passed、32 skipped、87 subtests passed；244个交付文件哈希一致，60个Python文件语法检查通过。
