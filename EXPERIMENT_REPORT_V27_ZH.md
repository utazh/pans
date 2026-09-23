# v2.7 服务器验证与层间共享比较

本轮按用户要求先验证，再比较层间共享；没有编写论文。全部运行与产物保存在 prism_layerkv，原 prism_pans 与此前代码保持不变。

## 实验范围

Qwen2.5-7B-Instruct，TREC 固定 p0 prefix，25% token 预算，完整 500 条已观察过的回归数据；不是新的最终留出集。所有方法使用同一在线 P8 选择规则与原标签完整续写平均对数似然评分。不同方法的 hidden state 会影响后续实际选择集合。

主方法 checkpoint 仍是 v25 冻结 seed24，v27 没有重训 reader；R0 只移除 native_policy.blocks，A/B/bK/bV、来源图及查询禁用名单保留。G16 有 16 个完整来源、12 个消费者，query/生成 KV 仍逐层独立。

## 正确性与数值验证

- 完整回归：656 passed、4 skipped，87 subtests passed；日志 logs/v27_full_tests.log。
- 7 个旧接口 Git blob 哈希与补丁要求一致；安装不改旧训练、推理内核、晋升或冷认证代码。
- 16 条固定 identity 选择计划：原 R4/proxy/R0 均使用同一逐层 token IDs，logits 全部有限；R0 每次仅 16 次完整 source resolve，无 native repair resolve；物理存储别名与 28 层 private tail 检查通过。
- proxy 与 frozen-R0 的平均首 token 完整词表 KL 为 0.000152739；BF16 数值路径不逐位相同。完整 500 条中 495 条预测相同，正确数相同、各有 2 条相对对方由错变对。
- 外部 direct-map identity 接入：16 条全部候选标签分数与原 identity 完全相同。
- xKV 最初默认 GPU SVD 的 96-token 全秩误差约 2.3e-5；使用 gesvd 后约 5e-7，失败尝试和日志保留。
- 真实完整 prefix 的 FP32 全秩重建误差约 1.5e-6 至 5.4e-6，但真实模型 16 条最大标签分数差仍为 0.499757，未通过原 0.05 门限。
- FP64 全秩分解/重建在 GPU2、GPU3 两次真实接入均得到全部标签分数差 0。证明接入/位置/评分路径可正确复现；FP32 数值敏感性保留，不将其失败改记为通过。

## 完整质量结果

| 配置 | 正确数 / 500 | 准确率 | 相对 identity | 有害 / 有益翻转 |
|---|---:|---:|---:|---:|
| 同后端 P8 identity | 343 | 68.60% | +0.00 pp | 0 / 0 |
| 冻结 G16/R4 native | 365 | 73.00% | +4.40 pp | 17 / 39 |
| 冻结 G16/R4 source-proxy（质量诊断） | 384 | 76.80% | +8.20 pp | 16 / 57 |
| 冻结 reader → R0 | 384 | 76.80% | +8.20 pp | 16 / 57 |
| KVSharer-Prefix G24 | 278 | 55.60% | -13.00 pp | 118 / 53 |
| KVSharer-Prefix G20 | 240 | 48.00% | -20.60 pp | 161 / 58 |
| KVSharer-Prefix G16 | 186 | 37.20% | -31.40 pp | 205 / 48 |
| xKV-CLF/P8 FP32 256/384 | 328 | 65.60% | -3.00 pp | 39 / 24 |
| xKV-CLF/P8 FP32 384/576 | 349 | 69.80% | +1.20 pp | 25 / 31 |
| xKV-CLF/P8 BF16 256/384 | 323 | 64.60% | -4.00 pp | 44 / 24 |
| xKV-CLF/P8 BF16 384/576 | 347 | 69.40% | +0.80 pp | 28 / 32 |

配对记录、保守质量下界在 results/v27_quality_analysis.json。所有行 certified=false；这里的单种子、已观察 TREC 结果不替代此前非劣失败，也不证明跨任务泛化。

## 外部基线的适配边界

KVSharer-Prefix 使用作者 notebook 的分别计算 K/V token/head 向量距离后取均值、距离降序、来源重映射以及最终隐藏状态 cosine > 0.5 的搜索规则；在 30 条按 hash 固定排序的独立 FIT 查询上搜索 G24/G20/G16，不用评估标签选图。原补丁 dissimilar_edges 与作者距离公式不完全相同，本轮单独补齐作者计算与 remap；没有把我们的 G16 图或单位 reader 当作 KVSharer。原版按多个 calibration prompt 平均 KV，这里改为已缓存的固定 prefix；原例子截断 64 token，这里保留完整 fit query。没有声称完成作者 Llama/InternLM 原模型复现。

搜索使用相同 FA2/hybrid 的冻结模型回放，30 条 identity 首 token 标签 logprob 与真实 teacher 的最大差为 0。最终隐藏状态仅用于无标签搜索，不是分类质量保证。来源可跨 P8 窗口，消费者读取其自身选中位置的 source KV；所有 current-query/private KV 原生。图、每次尝试的 cosine、拟合集 hashes 在 configs/v27_kvsharer_g*.json 与 audit/v27_kvsharer_search.json。

xKV-CLF/P8 是共享 token basis 的 prefix 适配：4 层一组，固定缓存 K 逆 RoPE 后联合分解，V 直接分解；按 P8 实际选中行重建，再加回原绝对位置 RoPE。它没有作者 SR landmark 选择器，也没有 SSD-native 因子 I/O。FP32 两档沿用包内随机 SVD（seed27，4 次迭代，oversample8）；BF16 两档使用精确 FP64 SVD 后按作者 U、SVh 的尺度分配量化。因此两条系列同时改变了求解器/精度，不是单变量量化消融。

外部质量桥仍恢复原生 payload，再替换 attention 输入；因子或 source 张量常驻，故原 runner 的 TTFT、硬件流量均无外部基线性能意义。有效外部输出明确标注 quality_only，不能将本表宣称为正式 KVSharer/xKV-SR 系统速度比较。

作者代码来源：[KVSharer](https://github.com/yangyifei729/KVSharer/tree/39e5c19176940f8d0089829fa8aab7992f85cb37)，[xKV](https://github.com/abdelfattah-lab/xKV/tree/05d91ecb0d698279aa4220fda5d7a5108036d692)。源码快照与 SHA 在 upstream/v27_baseline_sources、audit/v27_source_provenance.json。

## 独立 SSD 冷恢复计时

仅对真实可部署恢复路径 I/R4/R0 测量；proxy 与外部质量桥不参与。模型/reader/K4 index/metadata 预热，应用 payload 和目标文件页驱逐后逐请求认证，read-advice=preserve、admission=reuse-prefetch。三个互不重叠的随机块，每块 32 条，各臂独立进程，Latin-square 顺序使每臂各处于第一/第二/第三位一次。计时与质量任务不重叠。

| 配置 | 平均 TTFT | 相对 I | 三块均值 ms |
|---|---:|---:|---|
| identity | 1880.14 ms | +0.00% | 1948.56, 1876.14, 1815.72 |
| G16/R4 | 1124.85 ms | -40.17% | 1135.80, 1118.58, 1120.17 |
| frozen-R0 | 1115.93 ms | -40.65% | 1148.99, 1132.03, 1066.76 |

共 288 次正式冷请求全部通过应用缓存为空和目标文件页 resident=0 检查；96 条不同查询，每个配置均测一次。设置冷态的开销不计入 TTFT；不声称 SSD 控制器物理冷态。只有三个时间块，是本次有限验证，不是每任务/预算的稳定加速认证。

R0 相对 R4 的本轮平均 TTFT 变化为 -0.79%；平均额外降幅仅0.79%，逐块方向不一致（前两块R0略慢、第三块较快），不能认定稳定的额外加速。其增量收益应表述为当前回归集质量与更简单、更少恢复对象的路径，而非额外的速度突破。

## 资源账与质量边界

R4 的所选 prefix payload 为 41,943,040 B；R0 为 40,468,480 B；identity 为 70,647,808 B。R0 所选 payload 减少 42.72%；加上与 R4 相同的 reader/packed 唯一存储 4,767,744 B 后，对该子系统净减少约 35.97%。不等于模型总显存或已有 SSD 文件存储减少；原逐层 prefix 文件保留。

外部参考表示的全部常驻因子、理论按选中行恢复的字节量和 source union 账在 results/v27_representation_accounting_final.json。理论 selective I/O 尚未落地，不能与我们实测工作集混为实际压缩；重建临时张量、索引、private cache 也不能忽略。

拟合集限定层距≤1、每 source≤2 consumers、前4层/selector leader原生、选择 IDs 全相同的几何检查：SST2/25% 的 G16 不可行；SST2/50%、TREC/50% 仅 identity 可行；SUBJ/RTE 的25/50%以及TREC/25%可构建 G16/G20/G24。这里只生成计划，未把未训练图当成质量结果。

v26 的16格按原 teacher margin、query 长度、类别做了事后描述性分层，结果在 results/v27_v26_error_strata_with_lengths.json。重复任务/预算的请求不能合并为独立样本；这些分层不用于反向修改本轮 checkpoint 或评估选择。

## 判断

固定 reader 去掉原生小块在当前 TREC/25% 上保留了 proxy 的总准确率，并确实省去原生块恢复，支持继续研究更简洁的读侧共享。R0 与 proxy 不逐位相同，也没有新的跨任务质量保持认证。外部对照说明了这组 prefix 适配的相对质量，但还需要独立数据与外部真实因子恢复实现才能形成通用质量/速度结论。本轮未新增训练损失或推理 attention 内核，未写论文。

## 文件

- configs/v27_server_protocol.json、configs/v27_cold_design.json：协议与命令。
- results/v27_quality.csv、results/v27_quality_analysis.json：完整质量与配对翻转。
- results/v27_cold_analysis.json：计时汇总；原始请求及认证在 results/v27_cold_*。
- logs/v27_*、audit/v27_*：全部失败/成功过程、来源、保护与验证记录。
- scripts/v27_*：服务器执行、分析、复现脚本。
