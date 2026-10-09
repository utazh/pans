# LayerKV v6：Llama 适配、读取器来源对照与并发恢复

本次是 2026-10-10 对服务器现有实验的发布快照，基于公开 v5。没有重新训练模型、重跑 GPU 实验或改变实验算法。来源、未提交修改、文件 SHA-256 见 [snapshot_manifest.json](snapshot_manifest.json)。

## 新增代码

- **Qwen7B**：共享图与旧 G16 不完全一致时记录警告，仍逐请求验证 source/consumer 选择集合相容性。来自 bc36ce63f396ccb5072eac4e1041d6af04be07db。
- **Llama-3.1-8B**：允许 Llama/llama3 RoPE，使用模型自己的 cos/sin；相邻组无法满足整数块选择约束时尝试 stride-2 共享组；增加计时条件选择。提交 941420bb28062618355670712cd04a00d97bbb0d，并包含服务器未提交的 v29m_timing_sys.py --conds 改动。
- **读取器来源对照**：逐前缀 800 步训练、逐前缀闭式 ridge、WikiText 闭式、其他任务前缀闭式、留一任务多前缀训练。代码在 server_a100/xfer/mlr_build.py、mlr_wiki_kv.py、train_multi_v24.py。
- **并发恢复**：独立 worker、独立 Pcache 副本、固定核、1/2/4/8 worker，包含冷状态记录及吞吐统计。
- Qwen14B/32B 源码仍停留在 v5 的 2ac246a，不将这两者描述为本轮新模型实验。

根目录保存 Qwen7B 实际版本；Llama 的 12 个差异文件按原路径保存在 server_a100/overlays/llama31/。[应用说明](../../server_a100/V6_README_ZH.md)。没有把两个分支拼成未经验证的新实现。A/B 读侧补偿、实际共享 prefix KV、逐层独立 query/生成 KV 的基本结构保持原实验实现。

## Llama 逐前缀训练结果

以下来自 v29m_s800，预算 25%；共享档由开发集选择，未按测试准确率重新挑选。

| 任务 | 评估条数 | 选中档 | I 准确率 | 共享准确率 | 冷 I TTFT | 冷共享 TTFT |
|---|---:|---|---:|---:|---:|---:|
| SST2 | 872 | S3 | 95.18% | 94.50% | 1786.81 ms | 1026.46 ms |
| SUBJ | 1003 | S3 | 91.33% | 90.33% | 2114.82 ms | 1604.18 ms |
| TREC | 500 | S2 | 74.80% | 74.00% | 2398.13 ms | 1735.99 ms |
| RTE | 277 | S2 | 88.09% | 88.45% | 2938.15 ms | 2084.14 ms |


I 是同后端、同 P8 的无共享路径，与 CKV 独立区分。CKV/IMPRESS 质量结果在 studies/src_l8/v29m_sys/quality/，专用计时在 timing/；质量运行日志的 TTFT 不替代配对计时。

冷测每块每臂 16 条，SST-2/SUBJ/TREC 各 6 块；**RTE 有 2 块因外部核占用被标记且重试仍未解决，主统计只保留 4 块**。主口径 1408/1408 条记录满足报告的 completed/application-empty 统计。a100_sys_l8_preserve_all.json 是纳入标记块的敏感性分析，不替代主口径。普通缓存四任务各 6 块。原始冷记录摘要、逐请求 TTFT、块顺序与原文件哈希均收录，不能把这些样本计时称为完整测试集 TTFT。代码验证文件页与应用缓存状态，不声称 SSD 控制器内部缓存完全冷。

Llama 使用从相同 few-shot 内容转换的纯续写格式，标签前加空格。格式诊断用 fit 样本；早期 chat 格式结果标为无效，未并入当前表。不同模型 prompt 格式不同，不能直接把模型间绝对准确率差归因于 KV 方法。

## 读取器来源的结果边界

[reader_sources.txt](analysis/reader_sources.txt) 与 [JSON](analysis/reader_sources.json) 同时保留所有档位及开发选择。Llama 逐前缀训练选中档测试差为 -0.69/-1.00/-0.80/+0.36 pp（SST-2/SUBJ/TREC/RTE）；选档规则并非质量非劣认证。

Wiki/留一闭式及留一梯度训练的更广共享档有明显退化。多前缀训练 selected 在 SST-2 为 +0.11 pp、RTE 为 -4.33 pp，SUBJ/TREC 回退 I。**当前证据不支持模型级 reader 已普遍泛化**。不能把回退 I 的质量与未选共享档的速度拼接。

“留一”描述 reader 拟合与 checkpoint 选择的数据来源：代码仍读取目标任务共享图和 teacher header 身份信息；共享图/相容性准备并非完全与目标无关。部分目标层对可能没有梯度训练行，导出保留 target_pairs_without_training_rows。这些约束必须随泛化结果一起说明。

## 并发恢复

Qwen7B、25% 预算，SST-2/TREC，I/SEL/CKV，1/2/4/8 worker，2 次重复，每 worker 48 条；汇总 8640/8640 条冷记录通过报告检查。[结果](analysis/conc_main.json)，[编排与事件](concurrency)。

这是多进程、最多 4 张 GPU 的独立模型副本/独立 Pcache 压力实验，8 worker 时每卡两个进程。不是单 GPU 连续批处理，也不是同一前缀服务的共享批次。吞吐窗口以 0.25 秒轮询完成事件确定；磁盘读速率来自整块 sda 设备，不能当作进程独占流量。

## 发布验证与已知问题

- Qwen 根目录：**680 passed、33 skipped、87 subtests passed**。
- 原样 Llama overlay：**678 passed、2 failed、33 skipped、87 subtests passed**。失败是 test_bad_model_geometry[bad0/bad1]：Llama/RoPE 接受范围扩大后，不完整配置走到缺失几何字段并抛出 KeyError，旧测试期望 ValueError。
- Llama rope_type 缺省按 none 处理，因此未知字段形式的 rope_scaling 也可能进入后续检查；不将这两项失败宣称为通过，未为上传悄悄改动实验代码。错误消息仍有 Qwen-only 旧措辞。
- 42 个服务器/overlay Python 文件语法检查、51 个 shell 的 bash -n 通过；发布记录包含 guard 文件哈希核对。
- CUDA 被明确禁用，本次未重跑 GPU 精度/速度认证。测试 TMPDIR 使用独立目录，避免系统 /tmp 空间问题。

[验证日志](validation)。快照不含模型权重、训练 checkpoint、KV/cache、数据集原文、大型 profiler 文件或访问凭据；复现实验仍需服务器模型/数据和重建产物。原实验 worktree、模型和缓存未修改。历史操作脚本原样归档，不应不经检查直接运行其停止/清理/排队命令。
