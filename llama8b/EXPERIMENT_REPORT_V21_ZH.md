# SharedKV v2.1 实验报告

功能已经接入并完成真实 Qwen 实验。新校准明显缓解了旧三对共享的质量下降，但所有训练候选仍未通过预先设定的局部门限；以下结果不等同于部署认证或 1pp 非劣证明。

工作目录：/home/panzihang/src/prism_layerkv。全部代码、训练、实验和 profiler 输出在服务器进行。原 prism_pans 的源码和原有文件内容未修改；Nsight 帮助命令自动生成的两个字节码缓存已清理。上一版实验代码未变。

## 实验条件

- Qwen2.5-7B-Instruct；28 层、28 Q heads、4 KV heads、head dim=128；RTX 3090；BF16 计算、FP16 payload；batch=1。
- TREC 固定 prefix=4940 tokens，P8 selector，约 25% 保留预算，GPU/CPU payload cache=55/131 MiB；保留原选择器和异步预取。
- 旧 128 条只用于回归/开发。新 fit=64、val=64、holdout=175，与旧 SharedKV 使用过的请求按 UID 和 query_text 排除；manifest 在 data/v21_splits。全部仍来自同一 TREC 数据池和同一 prefix。
- 最终 10 个主要质量对照在观察 holdout 前列明；额外 hybrid 三对用于执行路径数值核对。没有根据 holdout 调整 reader 参数或误差门限。

## 独立评估：175 条

标签按完整续写的平均 token log-probability 判定。下表 KL 为首 token 全词表 KL，相对原系统；它与分类正确率是不同指标。字节为实际恢复的 payload，原系统还含推测预取。

| 配置 | 正确数/175 | 准确率 | 相对原系统 | 首 token KL | payload MB |
|---|---:|---:|---:|---:|---:|
| 原系统 | 123 | 70.29% | +0.00 pp | 0.00000 | 73.45 |
| vLLM split，无共享 | 125 | 71.43% | +1.14 pp | 0.00117 | 70.65 |
| 旧单对 26→27 | 124 | 70.86% | +0.57 pp | 0.00634 | 68.13 |
| 旧三对 | 99 | 56.57% | -13.71 pp | 0.09340 | 63.10 |
| 新单对 3→2 | 126 | 72.00% | +1.71 pp | 0.02508 | 68.13 |
| 新单对 18→19 | 118 | 67.43% | -2.86 pp | 0.01583 | 68.13 |
| 新单对 26→27 | 126 | 72.00% | +1.71 pp | 0.00204 | 68.13 |
| 新顺序三对 | 121 | 69.14% | -1.14 pp | 0.03440 | 63.10 |
| hybrid，无共享 | 124 | 70.86% | +0.57 pp | 0.00128 | 70.65 |
| hybrid，新单对 26→27 | 126 | 72.00% | +1.71 pp | 0.00254 | 68.13 |
| hybrid，新三对 | 121 | 69.14% | -1.14 pp | 0.03201 | 63.10 |

新三对相对旧三对提升 12.57 pp。回归集旧三对为 53.91%，新三对为 64.84%；独立集恢复没有完全保持到无共享水平。

同后端对照更能隔离共享本身的影响。下表 harmful 表示 baseline 对、student 错，beneficial 为反向翻转。

| 配对 | 净变化 pp | harmful/beneficial | 配对 bootstrap 95% pp | 有害翻转率单侧 95% 上界 |
|---|---:|---:|---:|---:|
| identity → refit2_s200 | +0.57 | 5/6 | [-2.86, 4.00] | 5.91% |
| identity → refit27_s200 | +0.57 | 0/1 | [0.00, 1.71] | 1.70% |
| identity → seq3 | -2.29 | 7/3 | [-5.71, 1.14] | 7.38% |
| old_dense3 → seq3 | +12.57 | 8/30 | [6.29, 19.43] | 8.10% |
| hybrid_identity → hybrid_refit27_s200 | +1.14 | 0/2 | [0.00, 2.86] | 1.70% |
| hybrid_identity → hybrid_seq3 | -1.71 | 9/6 | [-6.29, 2.29] | 8.80% |

这些区间以固定 checkpoint 和独立 query 假设为条件，没有多重选择/数据簇校正。175 条即使零 harmful，也不能用退化 bootstrap 宣称总体风险为零。完整逐层 token Jaccard、标签 margin、harmful UID 在 results/v21_hold_compare.json。

## 旧七对隔离与校准

| 共享对 | 旧 reader 回归准确率 | 新闭式校准 | 新 200 步蒸馏 |
|---|---:|---:|---:|
| 3→2 | 54.69% | 63.28% | 64.06% |
| 6→7 | 63.28% | — | — |
| 10→11 | 65.62% | — | — |
| 15→14 | 64.84% | — | — |
| 18→19 | 57.81% | 57.81% | 59.38% |
| 22→23 | 64.06% | — | — |
| 26→27 | 64.84% | 64.06% | 63.28% |

teacher 为同一 vLLM FA2 后端的 identity。闭式校准固定 A，使用 student 的 prefix 权重、private tail 和 teacher 输出拟合 B/bV。蒸馏只更新 A/B/bK/bV，主干冻结，200 步、lr=1e-4、BF16 STE，使用独立 64 条 val 选择 checkpoint。

顺序诊断采用 consumer 2→19→27，每步重跑真实 student fit/val，回放 teacher IDs，但仍执行 selector。上游 reader fingerprint 检查启用；未把 teacher 状态冒充已共享 student。所有 carry 都明确是未认证诊断候选。

局部门限保持 output RMSE≤0.03、单请求最大 RMSE≤0.08、prefix mass P95≤0.03。

| checkpoint | 输出相对 RMSE | 最大请求 RMSE | prefix mass P95 | local pass |
|---|---:|---:|---:|---|
| v21_refit2_s0 | 0.1749 | 0.2169 | 0.1818 | False |
| v21_refit2_s200 | 0.1045 | 0.1516 | 0.0652 | False |
| v21_refit19_s0 | 0.2837 | 0.3640 | 0.5544 | False |
| v21_refit19_s200 | 0.2131 | 0.3357 | 0.2397 | False |
| v21_refit27_s0 | 0.1160 | 0.1886 | 0.1396 | False |
| v21_refit27_s200 | 0.1228 | 0.1909 | 0.0940 | False |
| v21_seq2 | 0.3368 | 0.4470 | 0.2701 | False |
| v21_seq3 | 0.2299 | 0.2788 | 0.1469 | False |

局部 surrogate 误差下降不保证分类准确率单调上升。顺序 trace 的层间漂移、K-only/V-only 条件替换和非加权 logit 子空间诊断在 results/v21_operator_*.json；这些诊断不构成任务准确率或 KL 的理论下界。

## 正式计时

11 配置 × 4 轮，每组相同 32 条回归 query、8 条 warmup。顺序为正/倒/倒/正；每组重建 Pcache store，只复用冻结模型。所有 profiler/质量 GPU 任务退出后单独在 GPU2 运行；其他用户的 GPU0/1 负载仍存在，telemetry 已保存。正式组不带 profiler 或 logits 转储。

TTFT 截止首 token logits ready，含最终 CUDA 同步；不含 benchmark 标签续写评分。P95 为每轮 P95 的四轮平均。

| 配置 | 平均 TTFT ms | 平均 P95 ms |
|---|---:|---:|
| original | 274.03 | 324.57 |
| split_identity | 220.00 | 254.45 |
| split_shared | 211.46 | 234.89 |
| meta_identity | 211.77 | 238.79 |
| meta_shared | 214.02 | 226.98 |
| packed_shared | 206.89 | 232.34 |
| hybrid_identity | 203.60 | 229.92 |
| hybrid_shared | 206.89 | 242.42 |
| hybrid_seq3 | 206.95 | 239.18 |
| demand_identity | 237.98 | 283.96 |
| demand_shared | 255.42 | 296.40 |

| baseline → student | 平均延迟变化 | 四轮 block-bootstrap 95% 差值 ms |
|---|---:|---:|
| original → hybrid_seq3 | -24.48% | [-100.11, -34.06] |
| hybrid_identity → hybrid_seq3 | +1.65% | [-14.83, 29.89] |
| original → hybrid_shared | -24.50% | [-94.79, -20.61] |
| original → hybrid_identity | -25.70% | [-89.08, -46.13] |
| hybrid_identity → hybrid_shared | +1.62% | [-11.50, 25.52] |
| split_identity → split_shared | -3.88% | [-22.48, 3.29] |
| meta_identity → meta_shared | +1.07% | [-10.91, 25.63] |
| meta_shared → packed_shared | -3.33% | [-29.18, 5.31] |
| packed_shared → hybrid_shared | +0.00% | [-11.25, 17.07] |
| split_shared → meta_shared | +1.21% | [-20.22, 27.84] |
| meta_identity → demand_identity | +12.38% | [12.24, 45.55] |
| packed_shared → demand_shared | +23.46% | [29.42, 78.78] |
| demand_identity → demand_shared | +7.33% | [-10.77, 57.31] |

原系统到 identity/hybrid 已经改变 attention 后端和执行组织；只有相同执行配置的 identity/shared 配对，才适合讨论跨层共享的额外收益。四轮重复只能提供有限的运行间证据，不能将同一 32 条 query 的重复当作新的独立质量样本。全部原始轮次和长尾保留。

## 物理共享与固定计划

单对唯一 selected prefix payload 从 70,647,808 降为 68,132,864 bytes（-3.5598%）；三对为 63,102,976 bytes（-10.6795%）。这是所选 prefix payload，不是模型总显存，也不是 SSD 文件压缩率。每个 dense reader 另占 264,192 bytes，预打包额外 133,120 bytes；峰值还包含临时 workspace 和标签评分。

hybrid 的普通层会临时拼接 prefix/tail；拼接时间和分配均在运行中计入。去重的是持久 source bank，consumer 不重建自己的 prefix KV。

physical alias、唯一 source 数、private tail 长度、selector 调用数，以及 fixed shared/shadow/demand 的逐 query logits/计划一致性，见 results/v21_invariants.json。shadow-load 恢复 consumer 原生 KV 但不参与 attention；它会改变缓存驻留和队列，是诊断对照。

## Nsight 与 VizTracer 瓶颈

Nsight 统计只取 8 个正式请求的 TTFT/logits_ready 范围，排除加载、warmup 和标签评分。H2D 包含 payload、selector index 与 metadata，是 CUPTI 硬件拷贝记录，和 host 来源 payload 计数不同。下列时间来自 profiler，不作正式加速结论。

| 配置 | profiled TTFT ms | kernel 数/请求 | GPU kernel sum ms | GPU busy union ms | H2D MB | H2D sum ms |
|---|---:|---:|---:|---:|---:|---:|
| nsys_hybrid_identity | 269.50 | 1767 | 30.33 | 31.09 | 20.09 | 0.989 |
| nsys_hybrid_shared | 295.55 | 1782 | 30.43 | 31.09 | 18.62 | 0.963 |
| nsys_original | 290.07 | 1964 | 31.95 | 32.74 | 22.29 | 1.096 |
| nsys_split_shared | 263.64 | 2160 | 31.90 | 32.66 | 18.19 | 0.925 |

VizTracer 重采集获得 63,250 个函数事件，overflow=False；选取每个请求的第一次 decoder，排除标签 decode。self 时间包含未被追踪的库调用、C 调用及阻塞等待，不等于纯 Python 指令时间。

| 主线程函数 | self ms/prefill |
|---|---:|
| decoder_logits | 48.38 |
| wait | 23.43 |
| select_promixed_gqa_blocks | 21.29 |
| configure_impress_layer | 19.59 |
| update_impress_layer_scores | 14.16 |
| _logical_segment | 13.43 |
| update_score | 9.28 |
| to | 8.44 |
| apply_rope | 7.44 |
| _ordered_chunk_counts | 6.94 |

硬件 H2D 约 1 ms，GPU kernel 总时间约 30–32 ms，而 profiled TTFT 约 260–295 ms。结合主线程和 NVTX 记录，更值得优先处理的是 selector、缓存分数更新、分段/块元数据管理、KV 就绪等待和小算子提交。继续只压缩一份 prefix payload，不能直接消除这部分开销。跨线程 inclusive 时间存在重叠，不应相加估算关键路径。

## 复现与完整性

- 运行入口和参数说明：README_LAYERKV_V21.md。
- 新读者：assets/v21_refit*_s*/trec.pt、assets/v21_seq2/trec.pt、assets/v21_seq3/trec.pt；均为 diagnostic_only。
- 全部单测：380 passed、4 skipped、87 subtests passed，日志 logs/v21_all_tests.log；真实 vLLM GQA/hybrid/packed GPU 测试通过。
- 原工程 276 个文件 SHA256 未变，上一版代码未变；原包 manifest 和 8 个接口 hash 通过。见 audit/v21_*integrity.json、audit/v21_provenance.json。
- 适配修正和补充工具见 audit/v21_package_fixes.patch、README_LAYERKV_V21.md。
- 原始 Nsight：profiles/v21/nsys_*.nsys-rep；有效 VizTracer：profiles/v21/viz_hybrid_shared_retry/viztracer.json。
- 首次 profiler 因前一进程未退出发生加载 OOM，失败记录归档于 profiles/v21/failed_startup_overlap；首次空 VizTracer 文件仅保留作调试记录，未进入分析。
- 结果 CSV/JSON、原始逐请求记录、logits、trace、训练 history 和命令清单均保留在服务器。

## Resident 微基准

合成 GPU resident tensors，prefix=1228；prefill 的 Q/private=41/41，decode 几何为 1/42。每项 warmup=20、交替顺序采样 80 次。下面为同步 host wall 中位数；不含真实 KV 加载、selector、完整模型或服务排队，不能相加预测 TTFT。CUDA-event 间隔也保存在原始 JSON 中，可能包含主机提交间隙。

| 算子路径 | Q=41/private=41 ms | Q=1/private=42 ms |
|---|---:|---:|
| native_cat_one_attention | 0.1869 | 0.1818 |
| identity_two_attention | 0.3177 | 0.3500 |
| v1_reader_two_attention | 0.6985 | 0.6559 |
| v21_reader_two_attention | 0.6177 | 0.5830 |
| v21_reader_cached_metadata | 0.5687 | 0.5422 |
| q_transform_offset_only | 0.1956 | 0.1591 |
| v_transform_merge_only | 0.2666 | 0.2392 |

Profiler 工具已独立复制到 runtime_tools/nsight-2025.5.1 和 runtime_deps；后续 v21 wrapper 只从新目录调用。清理记录见 audit/v21_nsight_side_effect_cleanup.json。
