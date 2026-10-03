# LayerKV 独立研究审计（只读）

审计日期：2026-10-02（Asia/Tokyo）。审计对象：服务器 10.147.20.12，项目根目录 `/home/panzihang/src/prism_layerkv`（下文 R），论文目录 `/home/panzihang/src/lunwenkv`（下文 P）。行号对应本次读取的服务器文件。报告是本次唯一主动新建的文件；未修改项目代码、模型、数据、缓存、结果或实验配置，未运行测试、模型前向、训练或 GPU 作业，未终止服务器进程。标准库分析以只读、禁止生成 pyc 的方式运行。

## 1. 结论摘要

**实际跨层物理共享和 prefix 冷恢复收益有较强证据；“跨任务保持质量”和“READ 本身显著快于 EXPLICIT”尚未获得同等支持。** 没有发现足以判定整个机制失效的 private KV、LSE 或偏置处理错误。

- 最新冻结 v29 质量矩阵中，按开发集选择的配置相对同后端无共享 I，16 格准确率宏平均提高 **1.55685 个百分点**；去掉 TREC 后为 **−0.44087 个百分点**。4 格下降超过 1 个百分点，最差 RTE/50% 为 −2.52708。不能用总体正均值替代逐格或跨任务非劣证明。
- v29 主冷计时的 selected 配置宏平均延迟变化为 **−23.5110%**（含 SUBJ/25% 回退 I 的零变化）；固定 G16 为 −37.0273%。普通缓存只覆盖 25%、50% 两档，selected 为 −22.2745%。这是各格块级配对比率的平均，不是完整数据集逐条冷计时，也不是所有预算普通缓存都已验证。
- 在 25% 预算的同块 DIRECT / EXPLICIT / READ 比较中，四个任务的 **READ 相对 EXPLICIT 冷延迟区间均跨零**。普通缓存 TREC 中 READ 反而慢 **2.8408%**，块级 bootstrap 95% 区间为 [1.4479%, 4.4507%]。这不否定共享减少恢复的价值，但不支持额外宣称读侧改写在当前负载下带来明确速度优势。
- 已逐条检查 **11,712 条冷认证**，均满足 completed、应用 payload 为空、目标文件页驻留数为零；已检查 **9,605 个共享质量请求**，来源恢复次数、共享指针和 private 长度符合设计。它们分别是缓存状态证据和运行路径证据，不等于硬件 SSD/H2D 字节测量，也不等于所有张量内容均已重新数值验算。
- KVSharer-Prefix、xKV-CLF/P8 的质量记录可以复核，但其接入明确是质量桥接：仍恢复原生 payload，再替换 attention 输入。**这些运行的 TTFT 或 loader 字节不能作为外部方法的系统性能。**
- 最新 QuALITY pilot 也已从原始记录重算：4 篇文章共 76 题，I 为 46/76，G16 synthetic 200/400 步为 36/76、38/76；这批新增探索仍未证明广泛共享的质量保持。
- 当前服务器有两类论文：较早的实测论文，以及更新的目标/情景分析稿。后者的 25% 冷加速、10% 普通缓存加速、≤1 pp 质量损失是设计目标，文档已有声明；不能把目标表当作新实测，也不应因其存在而指控数据造假。

最稳妥的当前研究表述是：**在冻结 Qwen、prefix 命中后的恢复场景中，层专属仿射 reader 允许多个层读取同一实际 prefix KV，能够减少恢复数据和该服务器上的冷恢复延迟；收益随质量、预算、来源图和任务变化。READ 避免了逐消费者显式重建 KV，但尚未在现有短 query 实验中证明相对 EXPLICIT 的额外速度优势。**

## 2. 定位到的版本、论文和证据

| 对象 | 定位与状态 |
|---|---|
| 实测论文 | P/main.tex，SHA256 `79bf2671caf13089438127d22ffe48acbc7dc9fe89de3bb98b665212812da50d`；服务器时间 2026-09-29。主要证据来自 v27/v28，不是 v29。 |
| 最新情景稿 | P/visual_scenario_v3/main.tex，SHA256 `cf6e8b8f1c8865e45226bb20ee4828cf6f5bebc5d8f8c84f0604639c17ef4e2c`；服务器时间 2026-09-30。README 和正文说明是目标/分析稿。未访问线上 Overleaf，不能保证它与服务器相同。 |
| 当前项目 | R 不是 Git checkout，不能将某个历史 GitHub commit 自动等同于当前运行代码；本报告末尾记录关键文件的审计时 SHA256。publication/v3_github 也不是最新 v29 工作目录。 |
| 冻结配置选择 | R/audit/v29_final_selection.json，选择规则 revision 2、SHA `9151d971dbd982fb46993b64d24909b7ab53a0213f8087d4b87e4d6a4e166cbd`。15 个共享 checkpoint 的当前文件 SHA 与冻结清单匹配，SUBJ/25% 为 identity，无 checkpoint。 |
| 计时规则 | R/audit/v29_rule_freeze.json 与 configs/v29_selection_rule.json，revision 3，SHA `56faa2f328ea71c64836335d9598a06853f5f3b107ad29a6e0746fb7f4418a27`。JSON 逐字段比较确认 rev1→rev2 只改 A2 sampling/teacher 与 revision；rev2→rev3 只改计时环境/遥测与 revision，没有改 A1/A2 候选选择。 |
| 原始质量记录 | R/studies/v29/test_quality/runs/test_{I或G}_{task}_b{budget}/scored_records.jsonl；附带 .extra.jsonl、.extra.jsonl.v25.jsonl、.extra.jsonl.v21.jsonl 等。 |
| 原始计时记录 | R/studies/v29/{timing,timing_c,timing_p1ab,timing_p1c}/blocks.jsonl，及各自 runs 下的逐请求 scored_records.jsonl 与旁侧 .cold.jsonl。 |
| 外部质量记录索引 | R/results/v29_crosslayer/table.csv 的 run 列；回溯原始运行，而非仅采用 CSV 精度值。 |
| 最新新增研究 | scripts/v29_quality_pipeline.py 的 QuALITY pilot。脚本中的 sst2/subj/trec/rte 是文章存储槽位别名，不是四分类数据集。审计开始时运行的 400 步训练在期间自然完成；随后从稳定的已生成原始评估文件重算，详见第 9 节。 |

**版本追溯限制：** 冻结 checkpoint、数据清单及部分 job/result hash 提供了实质追溯，但未找到能把当前所有 source 文件一一绑定到每次 v29 运行的完整执行时源码快照。当前静态代码与历史执行完全逐字相同，不能仅凭当前文件保证。内层 command.json 还可能保留旧缓存目录；本次优先核对外层编排与 `.v26.jsonl.effective_cli.json`，后者确认实际覆盖后的 SSD 路径。

## 3. “论文主张对应证据”表

“已复核”指本次从可定位的原始记录重新计算或直接检查了相关代码；“只有汇总记录”指找到既有结论/汇总，但本次没有独立重建底层测量；“无法核实”表示没有足够材料支持该主张。分析目标不视作实验结果。

| 主张 | 分级 | 本次证据与边界 |
|---|---|---|
| 实测论文 TREC/25%：I/R4/R0 为 68.6/73.0/76.8% | 已复核 | v28 完整 500 条的正确数为 343/365/384；与 P/main.tex:219–228 及主质量表吻合。是历史已观察数据。 |
| 同表冷 TTFT：1880.14/1124.85/1115.93 ms | 已复核 | v27 三个不相交块、每块 32 条、三臂，共 288 条记录；重算均值吻合。不是完整 500 条冷计时。 |
| 实测论文四任务×四预算 R4/R0 准确率 | 已复核 | 48 个原始主运行重算，吻合 P/main.tex:237–264。R4 15/16 格下降，frozen-R0 13/16 格下降。 |
| TREC/25% R0 所选 payload 减少 42.72% | 已复核 | I 70,647,808 B，R0 40,468,480 B；R4 加 1,474,560 B 原生小块成为 41,943,040 B。只描述选中 payload，不能改写为总显存或持久磁盘缩小。 |
| 计入 reader/packed 后资源减少 | 已复核（定义内） | reader 3,170,304 B，加 packed 1,597,440 B；合计 4.546875 MiB。R4/R0 定义内净减约 33.88/35.97%；没有包含所有 cache 副本、索引、模型/private KV 或 allocator 峰值。 |
| 最新 v29 selected 宏平均准确率 +1.56 pp | 已复核 | 10,608 对 I/selected 记录，UID、答案、正确数一致；宏平均 +1.55685 pp，非 TREC −0.44087 pp。 |
| selected 所有任务/预算保持 ≤1 pp 损失 | 无法作为成立结论核实，且有反例 | SST2/10%、RTE/5%、10%、50% 点估计低于 −1 pp；开发筛选是点估计条件，并非非劣检验。 |
| v29 冷/普通缓存延迟下降 | 已复核（指定范围） | 冷 16 格，普通缓存仅 8 格（25/50%）；每格 6 个不相交的 16 请求块，共 96 唯一请求。宏均值须使用本文说明的块比率口径。 |
| 冷缓存状态确实成立 | 已复核（payload 冷） | 11,712 条记录均过应用缓存/文件页检查；模型、reader、K4 索引及元数据预热，SSD 控制器冷状态明确未宣称。 |
| READ 在冷恢复下明显快于 EXPLICIT | 无法支持 | 四任务同块比较的 READ/EXPLICIT 区间均跨零；普通缓存 TREC 明显慢约 2.84%。 |
| 没有重复按消费者恢复整份共享 prefix | 已复核（所检查请求） | 9,605 个共享请求的 unique source/resolve/pointer 审计，以及 RequestState 的按 owner 去重。并不意味着全系统零复制。 |
| private KV、LSE/bias 的正确性 | 已复核代码与运行结构；完整数值认证仍有限 | 原生 Q/K/V、逐层 cache.update、两路 LSE 和 key/value bias 均存在；private 长度符合请求/标签续写；未在本次重跑 GPU 内容等价测试。 |
| KVSharer-Prefix / xKV-CLF/P8 质量比较 | 已复核 | 176 个方法-格条目回溯逐请求数据，逐一匹配 I 的 UID/答案并重算准确率。不是官方全系统速度复现。 |
| 16 格 selected 相对 CKV/IMPRESS 的直接配对速度 | 只有估计口径，不能当直接测量 | Stage C 用两批实验的 I 作桥接；25% 才另有 P1c 同块直接比较。Stage C 的中间 I 漂移不能被省略。 |
| 最新方法的 Nsight kernel/H2D 归因 | 无法核实 v29 归因 | 找到较早 v21–v26 profiles；未找到对应 v29 DIRECT/EXPLICIT/READ 的新 profile。旧 profiler 汇总不能为新算子差异提供直接证据。 |
| QuALITY pilot 新文章质量 | 已复核（探索） | 36 个运行、684 条记录；9 臂各 76 题，UID/答案匹配；200/400 步训练已完成。正式 f0/f1/f2 结果未发现，不能作确认实验。 |
| QuALITY no-document 对照 | 只有汇总记录 | pilot/nodoc.json 仅有每文章 n/acc，没有逐问题预测；不与逐条复核证据混同。 |
| 既有测试通过数、旧硬件 H2D 减量 | 只有既有日志/汇总层面的证据 | 本次未重跑测试，也未独立重解析旧 Nsight 二进制轨迹。不能将历史测试数当成本次重新测试结果。 |
| 情景稿“冷 25%、普通 10%、损失≤1 pp” | 分析目标，非实测 | P/visual_scenario_v3/main.tex:276–294、333–339；相应 README 有披露。理想 G/L 和恢复占比模型也不是独立实验。 |


## 4. 原始记录重算：准确率、预算与聚合

模型为 `/data1/llm/Qwen/Qwen2.5-7B-Instruct`，28 层，batch 1；主共享路径为 BF16 计算、FP16 payload 存储、vLLM FA2 后端、hybrid/packed 读路径和 P8 选择。预算是分块选择预算，存在整数块分配；不能把标称 25% 自动当成每层都恰好相同 token 数或全机显存预算。R0 为 R4 训练 checkpoint 冻结后的推理策略转换，并非整个方法免训练。各任务、各预算的 adapter/source graph 不相同。

质量指标按完整标签续写的平均 log probability 取最大标签；不是只比较首 token 贪心结果。独立重算使用 prediction == answer，并检查原记录 correct 字段。N 分别为 SST-2 872、SUBJ 1003、TREC 500、RTE 277；SUBJ 是项目当前规范化评估划分，不能称为标准 1000 条测试集。每格 I/selected 的 UID 集合、答案和唯一性核对通过。

| 任务/预算 | 配置 | N | I 正确数（%） | selected 正确数（%） | 差值 pp | 有害/有益翻转 |
|---|---:|---:|---:|---:|---:|---:|
| SST2/5% | G20 | 872 | 809（92.78） | 808（92.66） | -0.1147 | 17/16 |
| SST2/10% | G16 | 872 | 812（93.12） | 796（91.28） | -1.8349 | 32/16 |
| SST2/25% | G20 | 872 | 818（93.81） | 812（93.12） | -0.6881 | 9/3 |
| SST2/50% | G16 | 872 | 818（93.81） | 811（93.00） | -0.8028 | 18/11 |
| SUBJ/5% | G20 | 1003 | 547（54.54） | 592（59.02） | 4.4865 | 62/107 |
| SUBJ/10% | G24 | 1003 | 531（52.94） | 533（53.14） | 0.1994 | 35/37 |
| SUBJ/25% | I | 1003 | 552（55.03） | 552（55.03） | 0.0000 | 0/0 |
| SUBJ/50% | G24 | 1003 | 849（84.65） | 845（84.25） | -0.3988 | 22/18 |
| TREC/5% | G20 | 500 | 214（42.80） | 259（51.80） | 9.0000 | 29/74 |
| TREC/10% | G16 | 500 | 324（64.80） | 341（68.20） | 3.4000 | 40/57 |
| TREC/25% | G16 | 500 | 343（68.60） | 384（76.80） | 8.2000 | 16/57 |
| TREC/50% | G16 | 500 | 313（62.60） | 361（72.20） | 9.6000 | 20/68 |
| RTE/5% | G24 | 277 | 245（88.45） | 240（86.64） | -1.8051 | 9/4 |
| RTE/10% | G24 | 277 | 237（85.56） | 232（83.75） | -1.8051 | 12/7 |
| RTE/25% | G24 | 277 | 242（87.36） | 242（87.36） | 0.0000 | 1/1 |
| RTE/50% | G20 | 277 | 246（88.81） | 239（86.28） | -2.5271 | 14/7 |

16 格等权宏平均：I **75.60301%**，selected **77.15986%**，差 **+1.55685 pp**。把所有 10,608 个“任务-预算-请求”观测池化，差为 **+1.38575 pp**；同一问题在多个预算重复出现，因此这个池化数不是 10,608 个独立问题的泛化估计。非 TREC 12 格等权差 **−0.44087 pp**。selected payload 的格间宏平均变化为 **−26.78343%**，其中 SUBJ/25% 回退 I。

开发规则 `scripts/v29_a1_select.py:41–64` 对 500 条只要求净正确数下降不超过 5，再优先选低物理字节配置。这是预先规定的开发集点估计筛选，并非统计非劣检验。测试点估计 12/16 格在 −1 pp 内，也不能写成“12 格通过非劣认证”。

实测论文旧表的完整重算正确数如下，预算顺序均为 5/10/25/50%，每项为 I/R4/frozen-R0：

| 数据集 | 四档预算的正确数 |
|---|---|
| SST2，N=872 | 809/793/771；812/801/796；818/806/799；818/809/811 |
| SUBJ，N=1003 | 547/511/510；531/497/510；552/515/543；849/783/766 |
| TREC，N=500 | 214/191/201；324/299/341；343/365/384；313/299/361 |
| RTE，N=277 | 245/231/228；237/226/229；242/237/238；246/242/240 |

原始路径是 `studies/v28_frozen/runs/{cell}_quality_{I,R4,R0}/scored_records.jsonl`。旧表的 G16 主方法与 v29 开发选择 G24/G20/G16/I 是两种协议，不能只更新较好的数字而沿用旧方法表头。

## 5. 计时重算与边界

### 5.1 主矩阵

以下为 v29 主 `timing` 实验中的每臂总体均值（ms）。每格 6 个不相交块，每块 16 条配对请求，即每臂 96 个唯一 UID；不是六遍同一批 16 条，也不是完整评估集。普通缓存只测了 25% 与 50%。

| 任务/预算 | I 冷均值 | selected 冷均值 | 固定 G16 冷均值 |
|---|---:|---:|---:|
| SST2/5% | 267.393 | 206.954 | 182.097 |
| SST2/10% | 457.954 | 289.487 | 289.487 |
| SST2/25% | 1021.062 | 757.846 | 632.397 |
| SST2/50% | 1863.306 | 1120.570 | 1120.570 |
| SUBJ/5% | 323.478 | 248.233 | 219.486 |
| SUBJ/10% | 539.076 | 476.206 | 345.230 |
| SUBJ/25% | 1105.739 | 1105.739 | 692.039 |
| SUBJ/50% | 2017.623 | 1742.263 | 1209.933 |
| TREC/5% | 326.820 | 258.021 | 225.213 |
| TREC/10% | 587.596 | 366.376 | 366.376 |
| TREC/25% | 1277.807 | 773.066 | 773.066 |
| TREC/50% | 2319.662 | 1375.728 | 1375.728 |
| RTE/5% | 395.335 | 348.575 | 262.970 |
| RTE/10% | 694.890 | 610.761 | 442.087 |
| RTE/25% | 1488.841 | 1284.411 | 886.732 |
| RTE/50% | 2810.932 | 2077.872 | 1656.447 |

`scripts/v29_timing_report.py:13–47` 的百分比计算是先在块内取 16 条均值，再算 `arm/I−1`，最后平均 6 个比率；区间对 6 个块比率做 4,000 次 bootstrap，seed=0。已按这个口径重算。宏平均 selected 冷 −23.5110%、普通 −22.2745%；固定 G16 冷 −37.0273%、普通 −33.5010%。15 个实际共享 selected 冷格的该区间支持下降，SUBJ/25% 为零；这仍是指定请求、机器和运行环境下的条件性证据。

旧论文 TREC 冷主表采用三块的总体均值：I **1880.140691**、R4 **1124.849488**、R0 **1115.927970** ms。对应总体均值比率 −40.1721%、−40.6466%；若改算平均块比率，分别为 −40.1322%、−40.6481%。两种口径接近但不相同，论文应明确使用哪一种，不能在不同版本表中静默切换。

### 5.2 冷认证与时间定义

本次核对的正式记录规模：

| 子研究 | 块数 | 成功运行数 | 冷认证条数 |
|---|---:|---:|---:|
| timing | 144 | 372 | 4032 |
| timing_c | 144 | 432 | 4608 |
| timing_p1ab | 48 | 192 | 1536 |
| timing_p1c | 48 | 192 | 1536 |
| 合计 | 384 | 1188 | 11712 |

合计 19,008 条计时请求记录，均为每运行 16 条；块内 UID 配对通过，未发现这些正式块有 flagged/failed 标记。所有 11,712 条冷记录的 status=completed、application_payload_empty=true、page_after.resident=0。本次检查的是记录与实现的相互一致性，不能从记录反推重新测量当时的物理硬盘状态。

实际计时缓存位于 R/runtime_data/v29_pcache_gpu2_ssd 和 v29_pcache_impress_c64_ssd，解析后仍在系统 ext4 SSD 分区 /dev/sda4（ROTA=0）。不能因模型放在 /data1 就认为该次 KV 冷测发生在 /data1 的 HDD。模式为 payload SSD-cold，读策略为 preserve；模型、reader、K4 索引和元数据已预热。冷准备本身在计时外，样本中 setup 约 2.27 s，不能与 TTFT 混算。

`sharedkv_v23/cold.py:83–114,284–315` 使用应用层驱逐、POSIX 文件页驱逐及 resident 检查，失败中止；有限重试没有放宽最终零驻留条件。它明确不宣称 SSD controller 冷。记录中 `io_first_decoder` 是进程 I/O accounting，代码 314–315 明确排除“硬件 H2D/SSD 字节”的解释。

计时边界在 `src/contiguous_fuxian/flexgen_qwen_reprefill.py`：

- 1307–1313：query tensor、DynamicCache 创建和前置 synchronize 后启动计时。
- 1315–1329：包含 decoder、选择、KV 恢复等待、attention 等；最终 synchronize 后记录 logits ready。
- 1333–1345：之后才进行部分 cache-score 维护和 argmax，记录 first_token_ready_ms。
- 1388–1448：标签续写评分更晚发生，计入 accuracy_scores_ready/evaluation_ready，不计入主 TTFT。
- 2634–2641：输出元数据已有以上说明。

所以主 `ttft_ms` 是 **同步后的首 token logits 就绪时间**，不是从服务收到请求到 token 发出，也不是分类答案就绪时间。P1AB 中 first_token_ready 与它只差约 0.10–0.14 ms，不足以解释 READ/EXPLICIT 差异，但定义仍须保留。reader 打包/请求准备也有单独的计时外字段；已预热服务场景可以这样定义，不能据此推断首次加载模型/reader 的冷启动收益。

计时环境不是整机完全空闲：规则 rev3 明确保留后台 CPU 负载并将任务绑定到选定核心。各臂一致，因此不是已发现的单边偏袒；但 `scripts/v29_timing_lib.py:12–18,53–71` 主要使用进程累计 CPU 与 last_cpu 来判定干扰，并非逐线程、完整 cpumask 的资源隔离证明。脚本记录磁盘情况也不等于逐进程 I/O 完整归因。没有证据证明该不足已经造成某个结果失真，区间应保留环境适用范围。


## 6. 真实读取路径与逐项代码审查

以下路径均相对 R。表内省略目录的 flexgen_pcache.py、flexgen_qwen_reprefill.py 位于 src/contiguous_fuxian；sharedkv* 短路径也位于该目录；ops.py、backend.py 特指 sharedkv_v21 内的同名文件。这里区分“存在额外分配”和“错误地按消费者恢复完整 KV”；前者不自动意味着后者。

| 检查项 | 文件、行号与实际行为 | 审计判断 |
|---|---|---|
| source 所有权与去重 | src/contiguous_fuxian/sharedkv/runtime.py:32–58；schedule 按 source 去重，resolve 以 owner bank 缓存，已有来源直接复用。sharedkv_v25/runtime.py:97–109 接入实际选择与加载。 | 9,605 个请求中 physical/full source resolve 与 unique source 数相符，consumer 指针与 source 一致。未发现逐 consumer 重新恢复完整 source 的证据。 |
| 异步预取重复保护 | flexgen_pcache.py:1681–1701 检查 pending/speculative/resolved/loaded；1731–1743 调 Pcache prefetch_async；2216–2227 等待与恢复逻辑序。 | 去重保护存在。prefetch_wait 是 host wall 等待时间，不能当 CUDA memcpy 累计时间。 |
| loader 的隐式 gather | flexgen_pcache.py:1267–1294 即使无 physical_to_logical 映射，也 argsort 后对 K、V 执行 index_select。 | 会分配重排后的 tensor；不是纯 view。按 source 少恢复也可能少做 gather，但尚无 v29 profiler 分离该收益。 |
| dtype/连续布局副本 | sharedkv/runtime.py:49–58 对来源执行 to(device,dtype).contiguous；sharedkv_v25/runtime.py:160–170 要求 BF16 并将 reader 转成 BF16。 | FP16 payload→BF16 计算可产生整份来源副本，发生在 source bank 建立时；不是每个 consumer 重建目标层 KV。 |
| cache 晋升副本 | sharedkv_v26/admission.py:88–117，尤其 105 的 entry.source.to(target).clone(contiguous_format)。 | 为 cache 独立所有权进行实际 clone，可能有 D2D 和重复驻留。消费者共享一份 bank 不等于各缓存层级只保留一份物理副本。 |
| private KV 隔离 | sharedkv_v25/runtime.py:111–117 每层用原生投影算 Q/K/V、RoPE，并用本层 lid 更新 cache；123–124 保留 o_proj、残差、MLP。 | 当前 query/续写 KV 未被跨层 prefix bank 替代。逐请求 private_tail_lengths 都符合 query_tokens+最长标签 token 数−1；这验证结构/长度，不代替完整张量内容 parity。 |
| R0/R4 原生小块 | sharedkv_v25/runtime.py:29–60 的 RepairManager；sharedkv_v25/attention.py:20–34。R0 blocks=0，不恢复 repair；R4 从已选集合末尾保留块，source 剩余段用切片 view。 | 当前 selected 全为 frozen-R0（或 I），没有暗中恢复 R4 小块。R4 拼接 native repair+private 会分配，其资源必须另计。 |
| READ 的两个 attention | sharedkv_v21/ops.py:60–68：共享 prefix 非因果 attention + private 因果 attention，再合并；prepare_query 在 32–40，finish 在 43–53。 | READ 并不是“同一个 attention kernel 加两个免费变换”。短 query 下新增 launch、LSE 混合和 query/output 小矩阵运算可能抵消省掉的显式物化。这里是代码结构证据，耗时份额尚未实测。 |
| LSE 与 key bias | ops.py:37–39 使用原生 q 计算 q·bk/√d 的 prefix logit 偏移；50–53 用 logaddexp 和 beta 合并两分区。 | key bias 虽不改变 prefix 内部 softmax，却改变 prefix/private 的概率质量，代码有处理。未发现漏项。 |
| value bias | ops.py:47–53 先 O_prefix B+bv，再乘 prefix 权重 beta；private 输出独立加权。 | 没有把 bv 错加到 private 分区，也没有无条件把 prefix 偏置加到总输出。 |
| EXPLICIT 的完整物化 | scripts/v29_exec_variant.py:23–37 重建每个 consumer 的目标 K/V，再走 ops.py:71–77 的 prefix+private cat 单次 attention。 | 相比 READ 增加目标层 KV 物化，但减少了分区 attention/LSE 合并路径；理论 FLOP 和实际 wall time不能简单等同。 |
| 原生无 reader 路径 | ops.py:71–77 cat prefix/private 后一次 attention；backend.py:33–55 reshape/FA2/LSE 接口。 | 这里也有 cat 分配。没有看到通过重复 KV heads 扩成 Q heads 的实现；GQA 布局按组处理。 |

数学上，在固定选择、同一表示且不考虑有限精度时，若目标表示为 K~=K_s A+bk、V~=V_s B+bv，则可以把 A 移到 Q 侧，把 B/bv 移到 prefix 聚合后。与 private 分区合并时必须使用加入 key-bias 偏移后的 prefix LSE，代码确实如此。**精确性指对这份近似表示等价，不指恢复原模型真实目标层 KV。**

资源会计也有明确范围：TREC/25% 的 selected payload 缩小 42.72%，并不说明模型总显存、峰值缓存占用或持久化磁盘减少同样比例。原生 prefix store 仍保留全部层，情景稿 P/visual_scenario_v3/main.tex:72 也有说明。读取器、packed、FP16/BF16 bank、晋升 cache clone、索引及 private KV 会占额外空间。本次没有将这些正常副本误判成共享失效。

## 7. DIRECT、EXPLICIT、READ：什么已公平隔离，什么没有

三者使用相同任务、预算、checkpoint 来源图、恢复后端和 private KV 规则；DIRECT 关闭 reader，EXPLICIT 显式重建 affine prefix KV，READ 在 Q/聚合输出侧计算。DIRECT 是固定来源图上的 reader 消融，不能当作经过专门搜索的 KVSharer 官方方法。DIRECT 仍会加载/打包 adapter 的准备状态，因此其内存不是“完全没有 reader 参数的最小实现”。

### 7.1 原始同块计时（25%）

| 冷缓存 | I ms | READ ms | EXPLICIT ms | DIRECT ms | READ/EXPLICIT 块均差（95% CI） |
|---|---:|---:|---:|---:|---|
| SST2/G20 | 1131.927 | 840.543 | 843.073 | 835.716 | −0.2997% [−0.6626, +0.0570] |
| SUBJ/G16 reference | 1236.616 | 770.071 | 767.211 | 737.349 | +0.3763% [−0.2315, +1.0013] |
| TREC/G16 | 1399.035 | 839.975 | 839.608 | 852.932 | +0.0455% [−0.2172, +0.3405] |
| RTE/G24 | 1626.199 | 1404.663 | 1403.599 | 1400.535 | +0.0770% [−0.2796, +0.3899] |

| 普通缓存 | READ ms | EXPLICIT ms | DIRECT ms | READ/EXPLICIT 块均差（95% CI） |
|---|---:|---:|---:|---|
| SST2/G20 | 170.606 | 170.440 | 173.555 | +0.1321% [−1.2818, +2.0229] |
| SUBJ/G16 reference | 179.410 | 177.654 | 153.064 | +1.0406% [−2.5959, +4.6148] |
| TREC/G16 | 163.087 | 158.607 | 152.938 | +2.8408% [+1.4479, +4.4507] |
| RTE/G24 | 283.875 | 283.587 | 280.395 | +0.1605% [−1.6800, +1.8345] |

均为 6×16 同块请求，数字来自 `timing_p1ab` 原始记录。**SUBJ/25% 正式 selected 是 I，此处刻意用 G16 reference 做机制研究**（scripts/v29_p1ab.py:13–15）；不能把这里的 G16 速度和 selected/I 的质量拼为同一配置的点。

完整评估的机制正确数（DIRECT / EXPLICIT / READ）为：

| 任务/图 | N | DIRECT | EXPLICIT | READ |
|---|---:|---:|---:|---:|
| SST2/G20 | 872 | 685 | 811 | 812 |
| SUBJ/G16 | 1003 | 509 | 549 | 543 |
| TREC/G16 | 500 | 65 | 383 | 384 |
| RTE/G24 | 277 | 211 | 242 | 242 |

reader 补偿相对 DIRECT 的质量价值明显；没有证据表明 READ 在数学表达能力上比 EXPLICIT 更强。两者整体预测不同数分别为 3、34、6、0，属于数值/选择反馈路径需要解释的差异，不应把几题净改善归为新建模能力。

### 7.2 READ 没有明显更快的解释分级

**已经有证据的解释：**

1. **两条共享路径都省了相同 source 恢复，READ 额外省的是 consumer 物化。** 冷 READ 的 prefetch_wait 均值约为 SST2 584.1/840.5、SUBJ 529.3/770.1、TREC 573.9/840.0、RTE 953.8/1404.7 ms（等待/TTFT）。EXPLICIT 的等待量相近。共同恢复等待较大，足以让小的算子差异在端到端中难以看见；这些等待字段不等于可相加的 GPU critical path 分解。
2. **READ 与 EXPLICIT 的计算组织不同。** READ 两次 attention、LSE 合并和 Q/输出变换；EXPLICIT 重建 K/V 后 cat，单次 attention。额外小 kernel 和 launch 是代码事实；哪项占几毫秒尚无本轮 trace。
3. **正式计时并未固定 token IDs。** EXPLICIT 与 READ 的完整逐层选择计划 hash 相同数为 SST2 2/96、SUBJ 6/96、TREC 0/96、RTE 69/96，冷/普通相同；但 payload 字节相同。完整 hash 不同不代表 token 重合率低，仅说明不是严格同 IDs 的算子对照，缓存局部性和后续 Q/selector 反馈仍混在其中。
4. **实际精度不是注释所写的 FP32 显式重建。** scripts/v29_exec_variant.py:5–11 写 FP32 then cast，但 31–36 使用已转 BF16 的 A/B 及同 dtype einsum、bias；sharedkv_v25/runtime.py:170、ops.py:24–29 可追溯精度。底层 GEMM 可能内部 FP32 累加，不等于整条重建以 FP32 表示。这是明确的注释/协议不一致，不能据此直接宣布比较无效。

**有局部验证，但不能取代正式隔离：**

`scripts/v29_p1ab_validate_fixed.sh:9,17–28` 只为 TREC/SST2 各 32 条做固定选择诊断，没有形成固定选择计时研究。本次从记录重算：两臂均 32/32 选择计划相同、预测相同；候选分布 KL(READ||EXPLICIT) 均值为 TREC 0.000801086、SST2 0.00000837837，最大分别 0.0059940、0.000143697；标签平均 logprob 最大差为 0.49972、0.26073。此 KL 是对标签得分 softmax 的分布，不是完整词表 KL。小 KL 与有限精度不逐值相等可以同时成立。

**仍需测量，不能当成结论：** 各 attention/GEMM/LSE kernel 的时间与 launch 数，重建 tensor 的峰值生命周期、D2D/allocator 成本，selector token Jaccard，固定 IDs 后差异是否消失，query/prefix 长度变化后的转折点。普通缓存 TREC 中 READ 慢约 4.48 ms，而两者 selector_compute 与 prefetch_wait 接近，不能笼统说“只是 SSD 太慢”。本次未找到可用于这些 v29 归因的新 Nsight/VizTracer 轨迹。

## 8. 外部基线与系统比较的公平性

### 8.1 层间质量基线

176 个方法-格条目合计引用 116,688 条记录（包含多种方法标签重用相同运行，不能当作同数独立样本），独立核对全部 UID 集合、答案、唯一性与重算准确率，零不一致。等权宏平均如下：

| 方法 | 准确率 % |
|---|---:|
| 同后端 I | 75.6030 |
| ours selected（开发选择 G/I） | 77.1599 |
| ours G24 | 75.4817 |
| ours G20 | 74.9809 |
| ours G16 | 74.4677 |
| KVSharer-Prefix G24 | 67.1323 |
| KVSharer-Prefix G20 | 63.9194 |
| KVSharer-Prefix G16 | 51.8252 |
| xKV-CLF/P8 256/384 FP32 | 75.4568 |
| xKV-CLF/P8 256/384 BF16 | 75.1734 |
| xKV-CLF/P8 384/576 FP32 | 70.9458 |
| xKV-CLF/P8 384/576 BF16 | 70.5404 |

同 G 的 ours 对 KVSharer 在 G24/G20/G16 分别 16/16、14/16、14/16 格更高。selected 对 xKV 256/384 BF16 为 8 胜、1 平、7 负，平均 +1.98649 pp；不能写成逐配置全面优于 xKV。固定 G16 在 11/16 格低于该 xKV 配置，说明质量与资源折中必须呈现。

代码事实：

- `sharedkv_v28/kvsharer.py:13–30,39–40,79–101`：按 K/V 距离排序、来源映射重写，用 30 条 sorted FIT query 的最终 hidden cosine >0.5 筛选；只从 fit 取设计问题，达不到目标图不强行填充。它明确标记 Qwen prefix adaptation，不是官方 Llama/InternLM 全复现。G 相同也不代表 token IDs、恢复字节或参数训练成本相同。
- `sharedkv_v28/factors.py:12–28,39–54,69–75`：共同 exact FP64 SVD master；K 去 RoPE 后分解；U 与 S·Vh 的同一缩放分别截秩/转 FP32、BF16；运行再加 RoPE。没有因两种导出精度使用不同分解而额外引入不公平。该实现不是 xKV-SR 的原生 SSD/landmark 服务引擎。
- `sharedkv_v27/baseline_quality.py:3–9,33–51,54–65`：先 original resolve 原生 KV，再从驻留的 source/factor 生成所选 K/V，替换 state.bank。原生载入与重建额外开销真实存在，因此 latency_valid=false 是合理限制；不能摘取 scored_records 的 ttft 来宣称比外部方法快。
- `sharedkv_v28/execute.py:86–109`：全秩/identity 固定计划 parity gate 先行，再跑近似基线；本次检查其实现与既有记录，没有重跑该测试。对官方算法各细节的外部源码逐版本认证，现有材料仍不足。

### 8.2 ContiguousKV/IMPRESS 与 I 的关系

I 是同 P8、K4 索引、hybrid、packed 和 reuse-prefetch 的优化无共享控制，适合隔离共享方法。它不能改名为“完整官方 ContiguousKV”。

本地 CKV/IMPRESS wrapper 定义在 `scripts/v29_baseline_entry.py:10–19` 和 `scripts/run_round4_methods.py:32–47`。CKV 采用完整精度 probe/selector；IMPRESS 使用 chunk64、reorder、显式关闭异步预取；I/共享是 chunk16、K4 index、reuse-prefetch。精确参数在 `scripts/v29_timing_p1c.py:22–35`。这些是实际实现条件，不足以单凭“异步关闭”认定故意削弱基线，但大幅端到端差距不能全部归因于跨层表示。

各自完整四预算质量宏均值：本地 CKV 70.3268%，IMPRESS 77.2376%，I 75.6030%，selected 77.1599%。更快和质量相同不是同一个命题。

Stage C（`scripts/v29_c_report.py:1–4,37–49`）使用：

`selected/base 估计 = (selected/I)_B × (I/base)_C`

这不是 16 格同一块四臂实测。B/C 间 I 的冷均值漂移平均约 +5.31%，最大绝对约 10.01%。Stage C 宏均值 selected 对 CKV −37.12%、对 IMPRESS −77.85% 应明确标为桥接估计。

另有 25% 的真实同块 P1c，冷 READ 对 I / 本地 CKV / 本地 IMPRESS 的块平均变化分别为：

| 任务/配置 | 对 I | 对 CKV | 对 IMPRESS |
|---|---:|---:|---:|
| SST2/G20 | −25.74% | −32.39% | −72.82% |
| SUBJ/G16 reference | −38.05% | −45.02% | −78.15% |
| TREC/G16 | −39.82% | −47.10% | −78.27% |
| RTE/G24 | −13.34% | −26.07% | −70.16% |

这能支持这些指定本地系统的端到端对照，不能升级成所有预算直接比较或官方复现。六块包含四个 Williams 顺序加两个额外顺序，覆盖了位置但不是每个位置完全均衡；没有发现足以认定结果被顺序偏差解释的证据，仍应保留分块结果。


## 9. 校准、开发、评估与历史反馈

当前 reader 是离线训练/校准产物，主干冻结。训练与部署身份绑定包含 task、prefix token hash、模型身份、层/头几何、period、keep_ratio（sharedkv_v25/runtime.py:162–170）。这支持当前 prefix 与预算下的可追溯部署，但不是换文档、换预算或换到 14B 就自动可用的通用 reader 证明。

数据与选择链条如下：

1. 原 R4 checkpoint 在既定 fit/val teacher 记录上训练；`sharedkv_v24/train.py:106,119–123` 检查 fit/val 与先前角色，177–183 根据 validation objective 选择 best checkpoint；82 明确不以 gold accuracy 作为该训练的选择指标。模型标签续写 teacher 信号参与训练，不应把它描述成纯粹不学习参数的缓存压缩。
2. frozen-R0 保留 reader 权重，取消 native blocks。v29 G20/G24 再按预定、无标签的层距离顺序撤销一部分 consumer（scripts/v29_build_candidates.py:25–38），未重新训练这些撤销图。
3. 新开发集由 scripts/prepare_v29_dev.py:10–23,43–48 构造，每任务 500 条，seed 2901；来自训练池并排除扫描到的历史 query/prefix 文本。A1 在开发集按“净正确数不低于 I−5、尽量小 bytes”的规则选择。
4. A2 只对 SST2/RTE 5% 比较 fit64 与 fit512（scripts/prepare_v29_fit512.py）。开发净变化分别为 −9/−25 与 −18/−21，均不合格，没有触发切换。不能把主 selected 的结果归功于这一轮 A2 重训。
5. final_selection 以 revision 2 冻结，再运行既有完整评估集。rev3 为适应用户允许保留的后台负载调整计时条件，不改候选选择；本次已经独立比较 JSON 字段确认。
6. 旧 TREC 500 条及四任务测试结果已多轮被研究者观察，并影响 R4→R0、扩大/收缩图和下一轮协议设计。它们是研究回归基准，不再是从未观察的最终确认集。

独立去重检查扫描 **256 个历史数据文件**，对 query_text 做 Unicode casefold 与空白规范化，当前四个开发集各 500 条均内部唯一、与历史 query 精确交集为零；SST2/RTE fit512 与对应开发集、历史 query 的精确交集也为零。源脚本还有子串排除，但本次独立结果只担保上述规范化全文精确匹配，不担保语义改写、潜在同源题目或预训练数据完全无交集。

因此，本次**没有找到直接的 fit/dev/test 问题文本重叠或最终标签进入 reader 梯度训练的证据**；不能把“多次在测试集上汇报”直接判成训练泄漏。同时，历史反馈形成的适应性与当前单 prefix/任务校准限制，确实使“跨任务质量已认证”的表述过强。

### 最新 QuALITY pilot：审计期间自然完成的新增证据

初次定位时 400 步训练正在运行；后续只读复查发现完成标记 `readers_done_s400.json` 为 errors=[]，对应评估文件也已生成。本次未启动或催动任何这些作业。正式 f0/f1/f2 的 studies 结果目录在检查时未出现。

`scripts/prepare_v29_quality.py:1–8,39–47,81–82,131–135` 的设计是按文章构造 prefix，同文章真实问题用于评估，文档条件合成问题用于 fit/val；另设来自其他未选文章的控制校准。四个 slot 名为 sst2/subj/trec/rte，只用于复用存储接口，真实任务均为四选一 QuALITY（记录标签 A/B/C/D）。

从 `studies/v29/quality/pilot/eval/*/scored_records.jsonl` 重算 36 个运行的 684 条记录，读取前后文件大小/mtime 稳定；各臂 UID/答案匹配。以下列为同四篇文章，不是四个分类数据集：

| 方法（25%，FullKV 为 100%） | slot sst2 N20 | slot subj N19 | slot trec N19 | slot rte N18 | 合计正确/76 |
|---|---:|---:|---:|---:|---:|
| FullKV | 17 | 11 | 11 | 15 | 54（71.05%） |
| I | 15 | 8 | 11 | 12 | 46（60.53%） |
| control-G16，200步 | 12 | 12 | 9 | 7 | 40（52.63%） |
| synthetic-G16，200步 | 11 | 8 | 8 | 9 | 36（47.37%） |
| synthetic-G20，200步 | 14 | 5 | 10 | 10 | 39（51.32%） |
| synthetic-G24，200步 | 15 | 10 | 11 | 10 | 46（60.53%） |
| synthetic-G16，400步 | 12 | 9 | 8 | 9 | 38（50.00%） |
| synthetic-G20，400步 | 14 | 6 | 11 | 10 | 41（53.95%） |
| synthetic-G24，400步 | 15 | 8 | 11 | 10 | 44（57.89%） |

这些是 **pilot 探索结果**，不是正式多文档非劣认证。增加训练步数只带来部分恢复，没有证明解决 G16 的掉点；G24 的结果也不单调。只有四篇文章，按 76 题当完全独立样本会低估文章间变异。

`pilot/nodoc.json` 仅保存 n/acc 汇总，可推出合计 25/76；本次未找到该文件内的逐问题预测，所以 **no-document 属于“只有汇总记录”**，不与上表逐条复核的结果混为一类。合成问题由文档生成而不输入真实问题，是有意义的隔离设计，但本次未完成对合成题/真实题全部选项、答案与语义近似的独立核查；不能仅凭生成脚本就宣布语义泄漏为零。

每个 reader 的 `*_compute.json` 记录 200 步 353.4–510.4 s、400 步 618.3–899.3 s，包含该训练进程加载，但不等于完整 prefix 构建、teacher 采集、合成生成的总成本。新文档需要离线处理时，这一成本需要按未来命中次数摊销；当前 TTFT 排除了它。不能直接拿四分类任务的延迟节约去推算 QuALITY 的实际回本次数。

## 10. 按严重程度排序的问题清单

这里的“高”表示会改变论文可支持的主张，不等于已发现程序崩溃或数值错误。

| 优先级 | 具体问题/证据 | 影响与应采用的表述 |
|---|---|---|
| 高 1 | 质量筛选被误当成非劣保证的风险。scripts/v29_a1_select.py:41–64；v29 主表 4/16 格低于 −1 pp，QuALITY pilot G16 为 36/76 或 38/76，对照 46/76。 | 应报告质量—资源折中与失败格；总体 +1.56 pp、开发点估计合格不能认证跨任务无损。 |
| 高 2 | READ 额外速度优势缺少干净归因。ops.py:60–77；scripts/v29_p1ab.py:17–28 未固定选择；原始计划 hash 多数不同；冷区间跨零，普通 TREC 变慢。 | 可以主张真实共享降低恢复成本，不能主张已证明读侧改写比显式重建更快。 |
| 高 3 | 外部速度主张容易混用。baseline_quality.py:3–9,54–65 为质量桥接；v29_c_report.py:37–49 为跨批比率估计；P1c SUBJ 使用 G16 reference 而非 selected I。 | 标清 prefix adaptation、估计/实测、reference/selected；不得把外部桥接 TTFT 当官方结果或将不同配置的质量和速度拼接。 |
| 中 1 | 论文/代码/结果版本未形成单一快照。P/main.tex:219–264 是 v27/v28 实测；最新情景稿:276–294 是目标；R 非 Git。 | 两份稿件本身已有不同定位，不能混读。发表时需要对应版本表和源码/配置身份；本次只保存当前 hash，不改既有文件。 |
| 中 2 | EXPLICIT 注释与 dtype 实际不一致。scripts/v29_exec_variant.py:5–11 对照 31–36；sharedkv_v25/runtime.py:170。 | “FP32 then cast”当前不准确。BF16 执行可作同精度对照，但不能据其结果声称 FP32 重建已被验证。 |
| 中 3 | 全系统零复制/总显存节约的表述超出证据。flexgen_pcache.py:1267–1294、sharedkv/runtime.py:49–58、admission.py:105。 | 明确 selected payload、reader/packed、cache tier 副本与 peak VRAM 四种口径；这些额外复制存在但未发现破坏共享所有权。 |
| 中 4 | 离线训练成本与 prefix 泛化未纳入完整系统收益。sharedkv_v25/runtime.py:162–170 绑定 prefix/预算；QuALITY *_compute.json 显示分钟级逐 reader 训练。 | 适用于可摊销的重复 prefix 场景，当前尚未证明少量命中文档也获端到端总成本收益；这不是缓存读取 TTFT 造假。 |
| 中 5 | 计时定义/环境限制易被省略。flexgen_qwen_reprefill.py:1313–1345,1388–1448；v29_timing_lib.py:12–18,53–71。 | TTFT 是 logits ready；不含初始化、冷准备、完整标签决策。后台负载条件下的六块区间不能代表整机完全空闲的普遍速度。 |
| 中 6 | 评估历史反馈。prepare_v29_dev.py:14–23,43–48；train.py:177–183；历史 v27/v28 结果已经用于设计。 | 当前没有发现直接训练泄漏；应把主矩阵称为已观察基准上的冻结协议结果，新文档 pilot 也已观察，正式文章应独立保留。 |
| 低 | 日志入口的溯源容易误读。外层 v26 effective_cli 与内层 command.json 缓存路径不一致；QuALITY slot 仍沿用任务名。 | 解释以 effective CLI/外层 variant/source map 为准，报告标出文章 slot。没有证据表明实际用了错误模型或错误数据。 |

正向审查结论也应保留：没有发现 owner 链式/循环恢复、逐 consumer 重复整份恢复、用共享 prefix 覆盖 private KV、遗漏 key bias 或 value bias、错误省略 LSE 合并的代码证据；不应为了罗列问题而虚构这些错误。

## 11. 仅建议的三个后续实验（本次未执行）

### 实验一：固定选择后的 READ/EXPLICIT 成本归因

**要区分的假设：**（a）READ 算子确实省时，但冷恢复掩盖收益；（b）两个 attention、LSE/launch 开销抵消重建节省；（c）主要差异来自在线选择反馈与 cache 局部性。

**最小设置：** 复用 TREC/25% G16 和 SST2/25% G20 的冻结 checkpoint，各 32 条诊断 query；先冻结逐层 token IDs 与输入，READ/EXPLICIT 使用同一 source/private buffers、同 BF16 和同 cache 状态。保留在线选择对照但不混入固定 IDs 结果。按 8 个均衡顺序块、每块 16 条，分别测运行时预热后的普通缓存与 payload-cold；主 TTFT 无 profiler，另取 1 个代表请求做 NVTX/Nsight，CUDA events 仅用于算子区间。FP32 重建仅作数值参考，单独标记，不与 BF16 主速度混为一臂。

**判断标准：** 固定计划 UID/token IDs 必须完全一致，报告标签分布/最大差及预测；分别报告 host 恢复等待、重建、两路 attention/LSE、D2D 与峰值临时字节。预先以块级 95% 区间判断 READ 是否更快，并单列 ≥1% 的实用收益门限。若固定 IDs 后差异消失，支持选择反馈解释；若 READ 的 GPU 区间更慢，支持 split/launch 成本解释；若 GPU 更快而冷端到端不变，支持被恢复等待掩盖。

**预期成本：** 不重训；约 0.5–2 GPU 小时与半天内的测量接线/分析，取决于模型加载和 profiler 开销。这里是排程估计，未占用或预约 GPU。

### 实验二：在未观察文章上验证质量边界和校准分布

**要区分的假设：**（a）历史四分类收益来自特定 prefix/标签分布，文档条件校准能改善新文章；（b）主要瓶颈是共享强度，本地合成问题或加倍步数仍不能保住质量。

**最小设置：** 利用已准备但本次未发现正式结果的 f0/f1/f2 共 12 篇文章；pilot 四篇只用于确定一次训练步数和预先选择规则，不进入正式确认。冻结 200 或 400 步之一后，固定预算 25%，保留 I、FullKV、no-document、control-G16、synthetic-G16、synthetic-G24；所有问题和答案仅用于最终评分，任何按文档选择 G 的规则只能看独立校准验证题。先审计合成题与真实题的正文/选项/答案重复和近似匹配，再封存正式真实题；不新增模型或损失函数。

**判断标准：** 同时报告每文章正确数、文章等权质量差、按文章 cluster 的配对区间、native/source payload 与不利翻转。固定 −1 pp 非劣目标；只有预先指定主臂的区间下界高于 −1 pp 才作该确认，否则明确结果为未通过/样本不足，不能靠选择 pilot 或正式集最优 G/步数补救。12 篇仅是最低验证规模，可能不足以认证 1 pp 界限。区分比 I 的损失与 FullKV→I 的稀疏预算损失。

**预期成本：** 利用 pilot 实际记录，单 reader 200 步约 6–8.5 分钟、400 步约 10–15 分钟。12 篇×2 校准源约 2.4–6 GPU 小时训练；另加合成生成、teacher 采集和评估，暂估总 6–12 GPU 小时，不能仅用训练 wall time 当全部成本。已有 pilot 工作不重复；当前进行中的任何其他作业不因本建议而改动。

### 实验三：等资源的层间基线与离线成本摊销

**要区分的假设：**（a）LayerKV 在相近真实资源下仍形成更好的质量—恢复代价折中；（b）优势主要来自不等资源/非原生基线接入，或需要过高 prefix 命中数才能摊销训练成本。

**最小设置：** 只选既有 TREC/25%（收益点）和 RTE/25%（质量接近点），使用已冻结的 READ、KVSharer-Prefix 与 xKV-CLF/P8 表示；每种方法仅保留两个预先规定的资源点。先按实际驻留/恢复 source 或 factor 字节匹配（目标 ±5%，无法精确匹配就画相邻两点），同时报告 reader/factor 与峰值物化内存。KVSharer 必须按唯一来源恢复，xKV 必须真实只加载所需 factors；现有“先读原生再替换”的质量桥接不得进入速度表。与 I 同 selector、同 cache/admission/时间边界，采用 4 个均衡块×16 条，仅测小规模冷/普通对照。记录每 prefix 的合成、teacher、训练/搜索/SVD 一次性耗时与资源。

**判断标准：** 报告完整现有质量划分与小规模配对 TTFT，判断是否存在资源相近且质量/时间被另一方法同时支配的点；再列总成本 T(N)=C_offline+N·t_hit，在 N=1、16、128、1024 次命中下比较并计算回本 N。离线未计量的部分必须列缺失，不能填零；rank/G 不直接等同内存。若未实现外部原生恢复，就只交付等资源质量与离线成本，不宣称官方基线速度胜出。

**预期成本：** 最小原生 source/factor 读取接入约 1–3 工程日，复用现有因子/权重后的计量约 2–4 GPU 小时；如果必须重做 SVD/搜索或扩展质量样本，额外成本另计。只有用户批准后才编写、运行或修改任何这些实现。

以上三项覆盖算子归因、质量泛化、近邻方法与系统摊销三个缺口。文档中的精度、版本、资源定义修正不需要冒充第四个实验；本次也没有替用户修改论文或代码。

## Appendix A: Frozen checkpoint and observed source-map identity

The sources array is target-layer indexed (0-based). It is read from each completed selected run, not inferred from G alone. Checkpoint hashes below are the frozen manifest values; all 15 were matched to current checkpoint bytes earlier in this audit.

### sst2_b5 / G20

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/sst2_b5_G20`

Frozen checkpoint SHA256: `6aa9a6df0694921ae120bb8a6b934d754b9d53539de146b8d2a7baeaa07f5877`

Observed sources: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 12, 12, 14, 15, 16, 17, 20, 20, 20, 20, 20, 23, 24, 26, 26, 26]`

### sst2_b10 / G16

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v28_frozen/readers/sst2_b10`

Frozen checkpoint SHA256: `06a843cfe90e857c5a1343a5c1e0b6f9be0d2d8fa3d005119e3d55e7a02a6f53`

Observed sources: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 12, 12, 12, 12, 12, 15, 16, 20, 20, 20, 20, 20, 20, 23, 24, 26, 26, 26]`

### sst2_b25 / G20

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/sst2_b25_G20`

Frozen checkpoint SHA256: `482b001f6fb53aebe9a46c20562126dd5a147afe776ed384663c766e5e6bd499`

Observed sources: `[0, 1, 2, 3, 4, 6, 6, 7, 8, 9, 12, 11, 12, 12, 12, 15, 16, 17, 20, 19, 20, 20, 20, 23, 24, 26, 26, 27]`

### sst2_b50 / G16

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v28_frozen/readers/sst2_b50`

Frozen checkpoint SHA256: `050f79cd817d86a4f37d66f8461451b6ab5e259e635e3df058251c43eaa6355b`

Observed sources: `[0, 1, 2, 3, 4, 7, 6, 7, 8, 13, 12, 13, 12, 13, 12, 13, 16, 21, 20, 21, 20, 21, 20, 21, 24, 27, 26, 27]`

### subj_b5 / G20

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/subj_b5_G20`

Frozen checkpoint SHA256: `3a8426097b674b97da1a7c157b40633426efd28bf14288f2181ebd6d9bac7dfd`

Observed sources: `[0, 1, 2, 3, 4, 6, 6, 7, 8, 9, 12, 11, 12, 12, 12, 15, 16, 17, 20, 19, 20, 20, 20, 23, 24, 26, 26, 27]`

### subj_b10 / G24

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/subj_b10_G24`

Frozen checkpoint SHA256: `fbe1b2d5655ec800fce033e5c747e348ec5949c3e21640019a1d724782f1bb3c`

Observed sources: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 20, 21, 20, 21, 24, 27, 26, 27]`

### subj_b25 / I

Identity fallback; no adapter. sources = list(range(28)).

### subj_b50 / G24

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/subj_b50_G24`

Frozen checkpoint SHA256: `ed666f8dc5fe49c0326ec2433cfa72498a20e7d0381a80744c419535cdce57d0`

Observed sources: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 22, 22, 24, 26, 26, 26]`

### trec_b5 / G20

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/trec_b5_G20`

Frozen checkpoint SHA256: `430e8aafd04a6a1cd3aee4329e59ab9fa20b49f261cd6c700f7e37f7c5960c55`

Observed sources: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 12, 13, 13, 13, 16, 17, 20, 21, 20, 21, 20, 21, 24, 27, 26, 27]`

### trec_b10 / G16

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v28_frozen/readers/trec_b10`

Frozen checkpoint SHA256: `3a00b1811cb9c3645e143b3ff169622ba70a0eb0dcadcc71926b84fbab15a572`

Observed sources: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 12, 12, 12, 12, 12, 15, 16, 20, 20, 20, 20, 20, 20, 23, 24, 26, 26, 26]`

### trec_b25 / G16

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v28_frozen/readers/trec_b25`

Frozen checkpoint SHA256: `5e5912c766168ca122a8dd9064930017e36cc7f91341bb99bc2e18de5062228e`

Observed sources: `[0, 1, 2, 3, 4, 6, 6, 6, 8, 10, 10, 10, 12, 14, 14, 14, 16, 18, 18, 18, 20, 22, 22, 22, 24, 26, 26, 26]`

### trec_b50 / G16

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v28_frozen/readers/trec_b50`

Frozen checkpoint SHA256: `26d9b8a801074a6e42290f79b69743964b7545aeb7e64a0b7059bfe25b11ce91`

Observed sources: `[0, 1, 2, 3, 4, 7, 6, 7, 8, 13, 12, 13, 12, 13, 12, 13, 16, 21, 20, 21, 20, 21, 20, 21, 24, 27, 26, 27]`

### rte_b5 / G24

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/rte_b5_G24`

Frozen checkpoint SHA256: `61368381c1e69bdcc27baa6f049cbe0a8e40fedae137f6192671a43f17a9f886`

Observed sources: `[0, 1, 2, 3, 4, 6, 6, 7, 8, 9, 10, 11, 12, 12, 14, 15, 16, 17, 18, 19, 20, 20, 22, 23, 24, 26, 26, 27]`

### rte_b10 / G24

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/rte_b10_G24`

Frozen checkpoint SHA256: `89f41470003f9777b515d3fc021b62636ba1aa1cd0a6f1a32be5f22a7eb9796b`

Observed sources: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 20, 21, 20, 21, 24, 27, 26, 27]`

### rte_b25 / G24

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/rte_b25_G24`

Frozen checkpoint SHA256: `88c9962725b632aa888630fd4a486aae098191efaa0d6f3b15adcb1cf77d2b71`

Observed sources: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 22, 22, 24, 26, 26, 26]`

### rte_b50 / G20

Adapter directory: `/home/panzihang/src/prism_layerkv/studies/v29/readers/rte_b50_G20`

Frozen checkpoint SHA256: `a9453f798b42eb9de5a1516823691f6974615aa0ba6859f54359a41cc7f73bf8`

Observed sources: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 14, 14, 16, 18, 18, 18, 20, 22, 22, 22, 24, 26, 26, 26]`

## Appendix B: Code hashes at report save time

These identify the inspected server snapshot, not a claim that the files were identical at every historical run.

| File relative to R | SHA256 |
|---|---|
| `scripts/v29_exec_variant.py` | `eb6cf09201e531efe9a8bcd2318faef26ea6a70c36174457472d385953c52c3a` |
| `scripts/v29_p1ab.py` | `a7ad89994e3c317c1e39b30c1a2ac5c4e5dd5c9b111eca6d7da15c7c26b20b5e` |
| `scripts/v29_timing_p1c.py` | `3873c1b73d82152d440af291c3b9d5336964bc83532e60cb68d7256004faec1c` |
| `scripts/v29_timing_lib.py` | `8b719636df30090b519d0976bc7ab5cb104404d05b377ad8f82fdf0c39f8c0aa` |
| `scripts/v29_a1_select.py` | `376348e38dae54e9409f128d06da17167652f962397843da36e044304f9391d6` |
| `scripts/v29_test_report.py` | `4c3b439e52cbc1ce30626d31aaf900ecd16efe3868b38360dc4facef4de63e50` |
| `scripts/v29_c_report.py` | `a96ead29ade4de91ec2f350816ba353a15cf16b57c9acc19c4d928c2ea98d8a6` |
| `scripts/v29_quality_pipeline.py` | `afd32e170f0b69ae126cc2b14f54a0561d0bd36c7d52629aad06fce101cb7007` |
| `scripts/prepare_v29_quality.py` | `60a5519c8114065d5b789bfc426afcfe286f9c96566781ea0a0658b712759b65` |
| `src/contiguous_fuxian/sharedkv/runtime.py` | `03b4dd239a98472db7bba44376fd5e3361fc43377700d157acadc7155c65eb50` |
| `src/contiguous_fuxian/sharedkv_v25/runtime.py` | `f456b56b4afd9092c6308722c556f10455a38fa22b14fdbc0cdfd1173ff16c5f` |
| `src/contiguous_fuxian/sharedkv_v25/attention.py` | `203484c2225576f5b2563dc58aa3faee70a83ddd1ebd846dd21ab0acbaa555a3` |
| `src/contiguous_fuxian/sharedkv_v21/ops.py` | `b3b088682d4b06083ba3fd2792420edcc9b745ff639aa8723728d4f357b55583` |
| `src/contiguous_fuxian/sharedkv_v26/admission.py` | `f93b3ec001933fffc8217dce02f7bb65f774892ad7b01d64b92d5a3f02e16efd` |
| `src/contiguous_fuxian/sharedkv_v23/cold.py` | `d68bb8d235936542176e544b6d0841cd723db1e02adfa4bde082d5a27109d88a` |
| `src/contiguous_fuxian/flexgen_qwen_reprefill.py` | `6c6ed95f5123920181843df0efe04bcbbd07b31af9e2ad9c5e1a1e04feb232d9` |
| `src/contiguous_fuxian/flexgen_pcache.py` | `fc7a76b240333656c4e6f25f3fe0b47ffb7fe953562a08dbca4d08a5b488ecab` |
| `src/contiguous_fuxian/sharedkv_v27/baseline_quality.py` | `b30937016bde1190cbe4003d03d50c094ae6752e56c53e0270cfca75960db438` |

Saved at 2026-10-01T16:20:23.818403+00:00. All experiment suggestions remain unexecuted by this audit.
