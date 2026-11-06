# SharedKV v2.2 服务器实验报告

本轮已按最新 GPT 方案完成：16 条候选边、4! 组匹配、等条件同编号控制、200 步蒸馏、完整模型回归、固定选择集合对照、私有 decode cache 审计、配对计时、Nsight 与 VizTracer。

**结论：GQA 组对齐降低了局部平均读取误差，但在本轮 175 条 TREC 回归请求上没有带来准确率改善；所有候选仍未通过原局部门限。当前不能据此扩大共享范围或声称方法整体更强。**

## 实施与保护边界

- 全部安装、拟合、执行、结果与 profiler 文件位于服务器 /home/panzihang/src/prism_layerkv；没有在本地创建实验代码或结果。
- 上传包 pans_sharedkv_v22.zip 的 21 项 manifest 校验通过；8 个关键 v2 接口 Git blob 哈希与包内预期完全一致。服务器是无 .git 的独立目录，因此按内容身份审计后只增加文件，未伪造 Git 提交或删除运行 guard。
- 原 prism_pans 及此前实验源文件哈希检查通过，原目录本轮没有新增修改文件。具体范围见 audit/v22_before.json 和 results/v22_invariants.json。
- 主干冻结；当前层原生 Q 与 private query/generated KV 保留。只对短 Q、prefix 输出和 LSE 路由，长 source K/V 不重排。未修改 selector、P8 或 token 保留预算。
- 本次未修改包内算法源文件；新增服务器运行、分析、审计脚本与真实 Qwen 几何 CUDA 测试。

## 数据与公平性

模型 Qwen2.5-7B-Instruct，RTX 3090，28 层 / 28 Q heads / 4 KV heads / head_dim 128；TREC prefix 4940 tokens，P8，25% exact block budget。
64 fit 按 query 分成 48 edge-fit + 16 assignment-selection，64 val 用于局部报告和训练 checkpoint 选择。两种映射均测量完整 4×4 表、相同参数量、相同 seed / ridge / 学习率 / 训练步数，未使用 initial-reader。K/V 共用一对一映射。
**175 条端到端数据已在 v2.1 被观察，本轮只作回归/开发证据，不是新最终测试。** 它们与本轮 64 fit / 64 val 的 tokenized query hash 无交集。当前实验没有新任务或新 prefix 的泛化证明。
局部门限保持 output RMSE ≤0.03、max request ≤0.08、prefix mass P95 ≤0.03。所有产物 diagnostic_only=True，没有部署认证。

## 局部验证：均值改善，但最坏请求并未一同改善

| source | target | steps | mapping | π(target→source) | output RMSE | max request | mass P95 | attention KL | local pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 19 | 0 | diag | [0, 1, 2, 3] | 0.284512 | 0.364815 | 0.554426 | 0.666257 | False |
| 18 | 19 | 0 | map | [1, 0, 3, 2] | 0.221209 | 0.372727 | 0.254061 | 0.459494 | False |
| 18 | 19 | 200 | diag | [0, 1, 2, 3] | 0.213450 | 0.340023 | 0.244510 | 0.304005 | False |
| 18 | 19 | 200 | map | [1, 0, 3, 2] | 0.197090 | 0.375150 | 0.155095 | 0.243284 | False |
| 26 | 27 | 0 | diag | [0, 1, 2, 3] | 0.115976 | 0.188761 | 0.139596 | 0.238120 | False |
| 26 | 27 | 0 | map | [0, 3, 2, 1] | 0.105900 | 0.186737 | 0.125849 | 0.183417 | False |
| 26 | 27 | 200 | diag | [0, 1, 2, 3] | 0.124503 | 0.198071 | 0.092627 | 0.127889 | False |
| 26 | 27 | 200 | map | [0, 3, 2, 1] | 0.105417 | 0.208173 | 0.082402 | 0.095378 | False |

diag 为同编号重新校准；map 为搜索组对应。200 步的 18→19 平均误差下降约 7.7%，26→27 下降约 15.3%；两者最坏请求误差均高于各自同编号控制。不能把均值下降写成所有请求都改善。

### target 19 的 4×4 赋值代价

| target组/source组 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 0 | 0.838833 | 0.334293 | 0.712661 | 0.788166 |
| 1 | 0.394244 | 0.956735 | 1.761593 | 1.472695 |
| 2 | 1.214666 | 0.802556 | 0.686299 | 0.849042 |
| 3 | 0.935722 | 0.489592 | 0.439899 | 0.678115 |

选择 π=[1, 0, 3, 2]，内部匹配代价下降 36.16%。该百分比只属于 16 条内部匹配请求，不是任务质量提升。

### target 27 的 4×4 赋值代价

| target组/source组 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 0 | 0.245082 | 0.366520 | 0.200373 | 0.572990 |
| 1 | 0.070736 | 0.091375 | 0.081111 | 0.073557 |
| 2 | 0.176415 | 0.650517 | 0.127572 | 0.511886 |
| 3 | 0.251938 | 0.341826 | 0.210351 | 0.561666 |

选择 π=[0, 3, 2, 1]，内部匹配代价下降 23.17%。该百分比只属于 16 条内部匹配请求，不是任务质量提升。

## 完整模型：175 条在线选择回归

| 配置 | 准确率 | 正确数/175 | 首token KL vs identity | 标签 KL vs identity | identity对→错 | identity错→对 |
| --- | --- | --- | --- | --- | --- | --- |
| identity | 70.86% | 124 | 0.000000 | 0.000000 | 0 | 0 |
| diag19_s0 | 68.00% | 119 | 0.034669 | 0.116833 | 12 | 7 |
| map19_s0 | 66.29% | 116 | 0.034282 | 0.081434 | 11 | 3 |
| diag19_s200 | 68.00% | 119 | 0.016147 | 0.075811 | 10 | 5 |
| map19_s200 | 66.86% | 117 | 0.025202 | 0.053066 | 9 | 2 |
| diag27_s0 | 70.29% | 123 | 0.004121 | 0.003378 | 1 | 0 |
| map27_s0 | 70.29% | 123 | 0.001222 | 0.002578 | 1 | 0 |
| diag27_s200 | 72.00% | 126 | 0.001626 | 0.002174 | 0 | 2 |
| map27_s200 | 71.43% | 125 | 0.000685 | 0.001810 | 0 | 1 |

准确率按实际 runner 的标签续写**每 token 平均对数似然**计算；首 token 全词表 KL 与该指标不同。对照使用同一 hybrid / metadata-cache 后端，避免把执行优化收益归于组对齐。

| 对照 | diag准确率 | map准确率 | 有害翻转 | 有益翻转 | 准确率差95%区间（百分点） |
| --- | --- | --- | --- | --- | --- |
| 19_s0 | 68.00% | 66.29% | 7 | 4 | [-5.714285714285714, 2.2857142857142856] |
| 19_s200 | 68.00% | 66.86% | 6 | 4 | [-4.571428571428571, 2.2857142857142856] |
| 27_s0 | 70.29% | 70.29% | 0 | 0 | [0.0, 0.0] |
| 27_s200 | 72.00% | 71.43% | 1 | 0 | [-1.7142857142857144, 0.0] |

18→19 零步少对 3 题，200 步少对 2 题；26→27 零步正确数持平，200 步少对 1 题。这些小差异不能证明稳定退化，但本轮没有准确率增益证据。局部 surrogate 改善不能替代完整模型质量验收。26→27 的首 token KL 从同编号约 0.00163 进一步降到约 0.00068，这是分布接近度的正向证据；18→19 则从约 0.01615 上升至 0.02520。不能只保留有利指标。

## 固定选择集合与 decode

固定选择集合使用同一 identity 的实际逐层 token IDs，selector 计算仍计入；32 条单独比较，不能作为正式速度证据。
| 配置 | 准确率 | 首token KL | 标签 KL |
| --- | --- | --- | --- |
| v22_fixed_diag19_s200 | 68.75% | 0.015707 | 0.058153 |
| v22_fixed_identity | 75.00% | 0.000000 | 0.000000 |
| v22_fixed_map19_s200 | 71.88% | 0.019122 | 0.021146 |

固定32条只用于控制诊断，其结果与175条整体方向不完全相同，不能只取小样本中的有利结果。另跑 8 条、最多 16 token 生成的 identity 与两个映射配置，仅验证运行和私有 cache 边界，不据此宣称长生成质量。结果：{"identity": {"requests": 8, "requested_max_tokens": [16], "private_extra_tokens_range": [3, 15], "private_prefix_excluded": "see invariant check"}, "map19_s200": {"requests": 8, "requested_max_tokens": [16], "private_extra_tokens_range": [3, 15], "private_prefix_excluded": "see invariant check"}, "map27_s200": {"requests": 8, "requested_max_tokens": [16], "private_extra_tokens_range": [3, 15], "private_prefix_excluded": "see invariant check"}}

## 资源与开销

| 配置 | 唯一 prefix bytes | reader bytes | packed副本 bytes | 路由 bytes | 净节省%（prefix+reader） |
| --- | --- | --- | --- | --- | --- |
| v22_reg_diag19_s0 | 68132864 | 264192 | 133120 | 448 | 2.9968 |
| v22_reg_diag19_s200 | 68132864 | 264192 | 133120 | 448 | 2.9968 |
| v22_reg_diag27_s0 | 68132864 | 264192 | 133120 | 448 | 2.9968 |
| v22_reg_diag27_s200 | 68132864 | 264192 | 133120 | 448 | 2.9968 |
| v22_reg_identity | 70647808 | 0 | 0 | 0 | 0.0000 |
| v22_reg_map19_s0 | 68132864 | 264192 | 133120 | 448 | 2.9968 |
| v22_reg_map19_s200 | 68132864 | 264192 | 133120 | 448 | 2.9968 |
| v22_reg_map27_s0 | 68132864 | 264192 | 133120 | 448 | 2.9968 |
| v22_reg_map27_s200 | 68132864 | 264192 | 133120 | 448 | 2.9968 |

组映射本身不进一步减少相同层对的 KV；其潜在价值应是让更多层通过质量验收，本轮尚未实现。请求 prefix buffer + reader 的净节省不是总模型显存节省，hybrid 原生层还有短期 prefix/tail 拼接 workspace。计算器没有改变保留率，也不是容量实验。

## 正式 TTFT

6 配置 ×4 轮 ×32 条相同回归请求，每配置 8 次 warmup、fresh Pcache；前/反/反/前顺序。复用冻结模型加载，设备2；GPU3 本任务空闲，其他用户 GPU0/1 作业继续，主机竞争无法完全消除。无 profiler、无 logits/trace 输出，新增 Q/output/LSE 路由在 TTFT 内。时间异常值全部保留。
| 配置 | 均值 ms | 4轮均值 ms |
| --- | --- | --- |
| original | 341.061 | 393.01, 325.56, 317.58, 328.09 |
| identity | 276.946 | 300.91, 282.72, 268.92, 255.24 |
| diag19_s200 | 249.027 | 253.97, 246.73, 259.70, 235.70 |
| map19_s200 | 273.319 | 287.05, 291.67, 272.87, 241.69 |
| diag27_s200 | 301.719 | 365.95, 293.91, 309.60, 237.41 |
| map27_s200 | 278.474 | 290.29, 285.17, 270.20, 268.24 |

| baseline | student | 变化%（负为更快） | 配对差值95%区间 ms |
| --- | --- | --- | --- |
| original | identity | -18.80 | [-82.47618465611595, -45.753463011351414] |
| identity | diag19_s200 | -10.08 | [-41.46007206873037, -14.378260442754254] |
| identity | map19_s200 | -1.31 | [-13.704981985938502, 6.450940825743601] |
| identity | diag27_s200 | 8.95 | [-3.3169493872264866, 52.8637278075621] |
| identity | map27_s200 | 0.55 | [-7.354883693551528, 10.071018852613633] |
| diag19_s200 | map19_s200 | 9.75 | [9.578214587236289, 39.00607676405343] |
| diag27_s200 | map27_s200 | -7.70 | [-58.9375717954681, 13.2713319544564] |

本轮无共享均值276.946 ms、头对齐18→19为273.319 ms、头对齐26→27为278.474 ms；两种头对齐相对无共享的差值区间均跨零，未证明稳定加速。18→19头对齐比同编号控制慢9.75%，但此差值包含选择/缓存/调度变化，不能全部归因于几个路由算子。不同日期/运行的绝对TTFT不宜直接与v2.1比较。四轮 bootstrap 只能提供有限的轮次不确定性估计，不能用单个最佳点判断稳定加速。GPU常驻算子微基准中新增路由约 0.08–0.09 ms/consumer，该结果不是 TTFT，也不是硬件 copy 时间。

## Nsight / VizTracer

独立 profiler 请求，仅诊断。Nsight 统计严格限定 TTFT/logits_ready NVTX 范围，排除模型加载、warmup 和标签评分；H2D 含 payload/index/metadata，不能称为纯 KV 流量。重叠/嵌套事件时间不可相加当作关键路径。两条路径的 H2D 均约0.91ms；VizTracer中较大项仍包括线程等待、selector和缓存元数据维护。头对齐的 output_route NVTX 包含时间约0.182ms/请求，Q变换+路由约0.635ms；这些包含profiler开销，不能解释成全部端到端差值。
| 配置 | H2D MB | H2D累计 ms | kernel数 | GPU busy union ms | profiled TTFT ms |
| --- | --- | --- | --- | --- | --- |
| nsys_diag19_s200 | 17.8882 | 0.9117 | 1781.5 | 30.194 | 405.512 |
| nsys_map19_s200 | 17.8252 | 0.9093 | 1785.0 | 30.757 | 309.329 |

VizTracer 有 96481 个函数事件，overflow=False，识别 8 个首次 prefill。self 时间包含未跟踪库调用和阻塞，不等于纯 Python 计算。
| 函数 | self ms/prefill |
| --- | --- |
| wait (/usr/lib/python3.10/threading.py:288) | 81.614 |
| decoder_logits (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/sharedkv_v21/runtime.py:48) | 57.856 |
| select_promixed_gqa_blocks (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/promixed.py:99) | 36.487 |
| configure_impress_layer (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/flexgen_pcache.py:1331) | 33.725 |
| _logical_segment (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/flexgen_pcache.py:1267) | 28.139 |
| update_score (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:2492) | 19.086 |
| _ordered_chunk_counts (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/flexgen_pcache.py:401) | 12.629 |
| update_impress_layer_scores (/home/panzihang/src/prism_layerkv/src/contiguous_fuxian/flexgen_pcache.py:739) | 11.714 |
| _siftup (/home/panzihang/src/prism_layerkv/vendor/flexgen/myheap.py:13) | 11.309 |
| to (/home/panzihang/src/prism_layerkv/vendor/flexgen/my_pcache_fast.py:374) | 9.522 |

## 验证与后续决定

- 416 项测试通过、5 项可选依赖测试跳过、87 个 subtests 通过；包括真实 CUDA / GQA 28:4 / d128 / prefix1228 / prefill与decode/private-tail数值和物理输入身份验证。
- 旧 reader 经 v21 / v22 新入口的 8 条固定计划桥接对照：logits bitwise相等、标签分数相等、逐层 token IDs 相等。
- 物理 alias、源去重、private KV 边界、每请求4次 P8 selector、128校准与175回归 query隔离全部通过：True；此前文件 472 项哈希未变。
- 局部平均误差改善是真实信号，但完整分类未改善，且所有局部门限仍失败。按本轮停止条件，不继续组合更多共享层，不放宽门限，不开展基于“更多合格共享层”的资源曲线。
- 目前只能说本模型的非同编号 head 有更低局部拟合代价；无法证明组错配是任务质量问题的主因。校准只观察少量 prefill 位置，最终评价涉及多 token 标签续写，目标不匹配是待验证解释，不能由本轮结果直接认定。

## 文件与复现

代码：src/contiguous_fuxian/sharedkv_v22/；运行清单：scripts/v22_*_jobs.json；拟合入口：scripts/screen_headalign_v22.py、scripts/train_headalign_v22.py；完整命令保存在各结果目录 command.json 和日志中。
主要汇总：results/v22_analysis.json、v22_quality_compare.json、v22_fixed_compare.json、v22_timing_summary.json、v22_nsight_summary.json、v22_viz_summary.json、v22_invariants.json。
原始 traces/readers/results 保留在服务器。profiles/v22/ 包含 .nsys-rep、SQLite 与 VizTracer JSON。详细包使用说明见 README_HEADALIGN_V22.md。

## 白话版

新方案确实给四组内容找到了更好读的对应关系，小范围检查也更接近原模型；但完整答题没有更准。翻译某一小段更准确，不代表整份答案更准确。当前不能据此放心共享更多层，所以先保留诊断结果，主流程继续保留质量更稳的配置。
