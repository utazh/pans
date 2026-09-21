# SharedKV v2.3：多消费者组级校准 + 可验证的 SSD payload 冷恢复

基于 `utazh/pans` 的 **v2.2 / fddabb7015c66c870a11810069bfbe20d5a34bbf**。
这是新增的研究原型，不是已获质量认证或实测加速的版本。主干参数冻结，但训练时梯度穿过真实 attention、输出投影、残差和 MLP，只更新组内 A/B/bK/bV。

本轮不继续搜索 head permutation。v2.2 的同编号 reader 可以显式转换为初始化；非同编号映射不能悄悄丢弃。新训练结果继续标记 diagnostic/candidate。

## 1. 本轮改变什么

1. 一份原生 source prefix KV 服务多个 consumer，而不是独立地拼接更多两层对。训练组可以保留少量原生例外层。
2. 联合优化组末 hidden state、query 最后一位与候选标签前缀位置，保留中间层目标；最后一个网络组可额外匹配冻结 lm_head 的分布。
3. 重新采集完整 query 与实际标签续写前缀，不复用旧版抽样 8 个 Q 的 traces，不以固定的 teacher tail 输出替代 student 的真实 private KV。
4. 模型、reader、K4 索引和算子已热，每条测量请求前重新建立目标 payload 的应用缓存冷状态；可进一步验证其 OS page cache 为冷。
5. P8、token budget、在线 selector、现有 Pcache 加载路径、private KV、label scoring 和 logits-ready 计时边界保持。

没有实现：任意选择集合 union+mask、跨 P8 共享、完整模型端到端训练、长自由生成蒸馏、SSD 文件压缩、reader 冷加载到计时内、O_DIRECT/GDS、新缓存淘汰算法。组配置不等于已验证可部署配置。

## 2. 安装和已有资源

```bash
git fetch origin v2.2
git worktree add ../pans-group-v23 fddabb7015c66c870a11810069bfbe20d5a34bbf
python /absolute/path/to/pans_sharedkv_v23/scripts/install_sharedkv_v23.py \
  --repo ../pans-group-v23
cd ../pans-group-v23
python -m pytest -q tests/test_sharedkv_v23.py
# 之后还要运行原仓库完整测试，以及真实 Qwen/CUDA 数值、alias、private cache 测试。
```

安装器只新增 src/scripts/tests/configs 文件；核验包 manifest 和七个关键接口 hash，不覆盖原文件。不要同时运行不同版本的模块安装脚本覆盖同一个目录。

模型、数据集、prefix store、Pcache、K4 index、旧 reader 不在包内。新 worktree 需要按原仓库设置这些资源。注意原 `run_round4.py` 使用该工作树的 `runtime_data/pcache_c16` 等配置；`--owned-cache-root` 不是修改加载路径的参数。它必须指向**实际 loader 正在使用的**独立 Pcache 目录。

冷实验必须用自己拥有的独立文件副本，不要 hardlink 到其他实验目录，也不要以一个未被 loader 使用的空目录冒充冷缓存路径。常规模型、BF16 prefix store 可以复用只读资源；page eviction 只针对真实 Pcache payload 文件。

## 3. 先审计一个容易忽略的预算限制

TREC 的 4940 tokens、chunk16 共 309 blocks；28 层全局 25% 预算为 2163 blocks，不等于各层完全相同的整数预算。初始 uniform allocator 会分配 77/78 blocks（平分余数的代码采用 bit-reverse tie order）。相邻层可能选择不同数量的 blocks。

因此，不能直接假定 `24,25,26,27 -> source24` 满足现有 same-selected-IDs 合约。本版绝不取隐式 union、不丢一个 block、不改名义预算来跑通。

首个建议 pilot：

```text
完整计算/校准组 [24,28)
层24：保留原生 KV（P8 leader）
层25：读取 source26，用 reader25
层26：原生 source
层27：读取 source26，用 reader27
```

这是三层共享一份实际 KV、两个 consumer，共同穿过真实四层计算；全模型独立来源由28降到26。它不是全模型14份KV的实现结果。

`configs/v23_groups_g14_budget_compatible_candidate.json` 描述另一个开发目标：每四层 `[i,i+4)` 中保留 i 原生，让 i+1 服务 i+1/i+2/i+3，共14个来源。实际是否等 IDs 和是否质量可用必须验证。

`g7/g4` 以及朴素 `g14` 仅是结构模板；可能需要100%保留或今后明确的新选择协议。它们不是一键验收配置。配置 JSON 当前用于审计/规划，训练通过显式 start/end/source/native 参数逐组执行。

## 4. 重新采集 full-query + label-prefix 组 traces

以下约定：

```bash
export MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
export STORE=/absolute/path/to/prefix_store
export FIT_BUNDLE=/absolute/path/to/v21_fit_bundle
export VAL_BUNDLE=/absolute/path/to/v21_val_bundle
export DEV_BUNDLE=/absolute/path/to/regression_or_development_bundle
```

这些变量供下列显式参数使用，不替换原 runner 未实现的环境变量接口。旧的175条已观察数据只作回归/开发，最后的泛化验收要另定数据。

launcher 有 **两次 `--`**：第一次前是 v23 选项，中间是既有 v21 runtime 选项，第二次后是 run_round4 选项。

```bash
python scripts/run_sharedkv_v23.py \
  --capture-dir runtime_data/v23_tf_g24 --start 24 --end 28 --max-suffix 4 -- \
  --mode identity --backend vllm2 --diagnostic --metadata-cache --layout hybrid -- \
  --task trec --budget 25 --period 8 --samples 64 --warmup 0 --max-tokens 1 \
  --bundle-dir "$FIT_BUNDLE" --output results/v23_capture_tf_g24

python scripts/run_sharedkv_v23.py \
  --capture-dir runtime_data/v23_tv_g24 --start 24 --end 28 --max-suffix 4 -- \
  --mode identity --backend vllm2 --diagnostic --metadata-cache --layout hybrid -- \
  --task trec --budget 25 --period 8 --samples 64 --warmup 0 --max-tokens 1 \
  --bundle-dir "$VAL_BUNDLE" --output results/v23_capture_tv_g24
```

采集限制 `max_tokens=1`，使之后的 decoder 调用来自原评分器对所有候选标签的 teacher-forced prefix，而不是发散的自由生成。代码不读取 gold label/正确性。`max-suffix=4` 是捕获的标签前缀长度上限，不是完整长生成训练；需要更长候选时显式提高，并记录内存成本。

输出是 `ROOT/trec/<query_hash>.pt`，同时保存 `ROOT/selection/<query_hash>.pt`。每个 trace 包含完整的组输入、组内各层输出与 attention 输出、完整 query IDs 和所有捕获的候选分支。捕获旧 fit query 仅在这一显式 capture 模式解除运行时 ban，正式评估不会解除。

检查 fit 和 val 的真实 IDs：

```bash
python scripts/audit_group_ids_v23.py \
  --plans runtime_data/v23_tf_g24/selection \
  --start 24 --end 28 --source 26 --native-layers 24 \
  --output results/v23_fit_ids.json
python scripts/audit_group_ids_v23.py \
  --plans runtime_data/v23_tv_g24/selection \
  --start 24 --end 28 --source 26 --native-layers 24 \
  --output results/v23_val_ids.json
```

两份 `all_compatible` 必须为 true。否则先修改**公开的候选结构**保留原生例外，不绕过 equal-ID guard。代码不会基于测试 query 动态选择共享结构。

## 5. 联合校准首个多消费者组

孤立的最后一组没有上游共享，因此 student traces 可以直接使用上述 clean teacher traces。后续顺序扩组不是这样，见第7节。

```bash
python scripts/train_group_sharedkv_v23.py \
  --model-path "$MODEL" --store-root "$STORE" --task trec \
  --teacher-fit runtime_data/v23_tf_g24 --teacher-val runtime_data/v23_tv_g24 \
  --student-fit runtime_data/v23_tf_g24 --student-val runtime_data/v23_tv_g24 \
  --start 24 --end 28 --source 26 --native-layers 24 \
  --steps 300 --eval-every 50 --lr 0.0001 --device cuda \
  --logit-weight 0.05 --output assets/v23_pilot26/trec.pt
```

学习率、步数和损失权重是起始配置，不是已实测最优。建议也运行 `--steps 200` 作为等预算对照，并保留200/300步全部报告，不仅选最有利 checkpoint 的测试结果。

训练对每个 consumer 做原始同编号 affine ridge 初始化，也可用旧同编号 reader 初始化：

```bash
python scripts/convert_identity_v22_for_v23.py \
  --input assets/YOUR_V22_DIAG_BUNDLE/trec.pt \
  --output assets/v23_compat_diag/trec.pt
# 在训练命令加 --initial-bundle assets/v23_compat_diag/trec.pt
```

转换器验证所有 head maps 都是 identity 才转换；非同编号 reader 被拒绝，不能只删映射字段。旧 v21/v1 格式的同编号 reader 不需要该转换。

### 目标与实现边界

- 只更新多个消费者的 A/B/bK/bV；梯度穿过实际冻结 Q/K/V/O、归一化和 MLP。冻结参数不等于整个前向放进 no_grad。
- 训练后端使用可微 PyTorch attention，完整重算当前序列各层 private KV，保留 RoPE、GQA 与 prefix/private 联合 softmax；不是用推理内核的不可微结果假装训练。
- 不重建 consumer 的长 prefix KV；长 source KV 按引用复用。
- 主要目标是组末 hidden state，归一化尺度用 teacher 组更新量；另外加中间层输出、最后 query 位置、候选标签前缀位置和 reader trust regularizer。
- `--logit-weight` 只允许组末就是模型最后一层；这时通过冻结 final norm/lm_head 匹配输出分布。早层组不使用一个伪造的直接 lm_head 目标。
- `--query-only` 是去除后续候选前缀的消融。不是移除完整query。
- 导出保留 BF16 精度。最开始先检测原生组重放与采集后端数值是否一致；超过 `--max-identity-replay-error`（默认0.02、相对组更新量）会中止，不拿实现偏差去训练。
- 保留原局部 RMSE/mass 门限结果，并记录 group boundary 指标。所有输出继续 `diagnostic_only`，不自动晋升，不声称分类非劣。

输出格式兼容既有 v21 运行时，不增加在线读侧操作或每消费者参数尺寸。组训练更贵，离线时间/峰值要另报。`--max-sequence` 默认1024用于显式保护训练内存；不是支持最大上下文长度声明。

## 6. 完整模型回归/开发验收

```bash
python scripts/run_sharedkv_v21.py --mode shared --backend vllm2 \
  --adapters assets/v23_pilot26 --allow-candidate \
  --metadata-cache --packed-ops --layout hybrid \
  --stats results/v23_pilot_dev.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 175 --warmup 8 \
  --bundle-dir "$DEV_BUNDLE" --output results/v23_pilot_dev
```

同执行路径 identity 必须保留；质量采集可用原 v21 logits/label score 工具，但质量采集、profiler 和正式计时分开。完整候选标签的每-token平均log-probability才是当前分类协议，不只看首token KL。

比较至少包括：现有26→27单对、相同新增consumer的独立/顺序校准、多消费者组级校准、去除suffix目标的组级消融。共享数量、source、数据和步数尽量匹配。当前多消费者结构如果失败，记录失败，不用扩大组数掩盖它。

## 7. 全模型14来源的推进路径（研发目标，不是已有结果）

后续开发结构按消费者出现顺序执行：0–3、4–7、8–11……，每组使用 `native=i, source=i+1` 候选，检查实际sameIDs。每组通过独立开发集后固定，再捕获下一组的真实student输入。

teacher始终是无共享identity。student采集命令使用上一组的bundle，并回放**同一teacher的选择计划**：

```text
--capture-dir NEW_STUDENT_DIR --start NEXT_START --end NEXT_END --
--mode shared --adapters PREVIOUS_BUNDLE_DIR --allow-candidate --diagnostic
--selection-plan TEACHER_TRACE_ROOT/selection ... -- <相同fit或val bundle>
```

selector仍然执行，回放只覆盖最终IDs。然后训练加 `--carry PREVIOUS_BUNDLE_DIR/trec.pt`。代码要求 carry 消费者都在新组之前，检查模型/预算/位置/query身份与reader fingerprint；没有重采student不能沿用旧traces。不要把先做的最后组直接carry到前面的组。

最终7来源/4来源只有在sameIDs及质量等约束可实现时才推进。所有组不能用一条“for循环训练后自动认证”的脚本代替完整模型验收。

## 8. SSD冷恢复：先做已有同编号reader，不必等组训练结束

优先阅读 `docs/SSD_PROTOCOL_ZH.md`。本版的冷条件是：

**模型/reader/K4索引/算子/元数据已热；应用payload为空；目标Pcache payload文件的OS resident pages为0。**

不是整个进程冷启动，不是新prefix尚未计算，也不包括reader首次加载，不保证SSD控制器内部缓存冷。

设置 `OWNED_PCACHE` 为 loader 实际使用的独立目录。该目录必须已准备好持久化、干净的raw FP16 chunk文件；程序不会新建/覆盖它们。OS page eviction只作用在当前任务的所有层payload文件上，identity/shared使用相同文件集合。

```bash
export OWNED_PCACHE=/absolute/path/to/current_worktree/runtime_data/pcache_c16

python scripts/run_sharedkv_v23.py \
  --cold ssd-cold --owned-cache-root "$OWNED_PCACHE" \
  --cold-log results/v23_cold_identity.r0.jsonl -- \
  --mode identity --backend vllm2 --metadata-cache --layout hybrid \
  --stats results/v23_cold_identity.r0.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir "$DEV_BUNDLE" --output results/v23_cold_identity_r0

python scripts/run_sharedkv_v23.py \
  --cold ssd-cold --owned-cache-root "$OWNED_PCACHE" \
  --cold-log results/v23_cold_group.r0.jsonl -- \
  --mode shared --backend vllm2 --adapters assets/v23_pilot26 --allow-candidate \
  --metadata-cache --packed-ops --layout hybrid \
  --stats results/v23_cold_group.r0.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir "$DEV_BUNDLE" --output results/v23_cold_group_r0
```

上述默认 `--read-advice native` 保留原 vendor 每次磁盘读取时调用 uncache/DONTNEED 的行为，用作“未改读取策略”的冷恢复对照。

没有new group时先将adapters换成已验证桥接的v21同编号单对/三对。新格式v22同编号包先显式转换。所有reader均未认证时保留allow-candidate标记。

预热8条之后，**每一条**正式请求前都清空应用payload并验证page冷，32条都是独立冷恢复；不是只让第一条冷。清理准备时间单独记在 `setup_ms_outside_ttft`，不算入原logits-ready计时。这模拟“请求到来时缓存已不驻留”，并非生产中逐请求清空缓存的调度建议。

**重要审计发现：原 vendor 的 `uncache(path)` 会在每次磁盘读取前调用 DONTNEED。** 因此，仅在请求前把page读热、随后继续原read-advice，会马上又把page驱逐，不能声称这是有效page-hot对照。

本版提供显式 `--read-advice preserve`：只在这条正式请求期间，对已经核验的目标payload文件停止重复的读取时DONTNEED；退出前等待worker，恢复原函数。两个方法必须使用相同策略。它不是跨层共享的收益，要与native策略分开报告。`page-hot`配合native策略直接拒绝执行。

必做相同identity/shared的对照：

- warm：不用v23冷包装，原v21路径。
- `--cold ssd-cold --read-advice native`：应用payload/page冷，保留原读取时DONTNEED，原系统冷恢复对照。
- `--cold page-hot --read-advice preserve`：应用GPU/CPU payload冷，但文件page cache热；读取时不主动反复驱逐。
- `--cold ssd-cold --read-advice preserve`：应用payload/page冷，读取时不主动反复驱逐；与上一条隔离OS page状态。

`app-cold`仅应用冷、OS状态不限定，属于诊断，不应标成SSD冷。四轮至少ABBA交替，更稳妥增加轮次；每一轮/每个配置用新输出路径。既有配对计时分析工具仍可读取scored_records；`summarize_cold_v23.py`只做口径核查与描述统计，不伪造置信区间。

```bash
python scripts/summarize_cold_v23.py \
  results/v23_cold_identity.r0.jsonl \
  --output results/v23_cold_identity.r0.summary.json
```

`/proc/self/io`差值只包住第一次decoder，不包含候选标签后续评分；它是进程storage accounting，不是CUPTI H2D或设备实际NAND流量。正式TTFT仍使用原runner的最终CUDA同步边界。硬件H2D与块设备I/O需要独立Nsight/存储监测；所有计时、setup成本、失败冷前提都要保留。

## 9. 最小决策标准

一对多质量、冷缓存资源收益分开验收。冷实验可以先用现有reader跑；组训练不能以“cold更快”掩盖分类下降。至少同时报告独立来源数、consumer数、原生例外、真实质量、净reader+prefix工作集、SSD读取accounting、硬件H2D、P50/P95 TTFT、组校准成本与复用摊销。

失败的SSD page-cold检查直接停止，不能改名为成功；所有局部失败原样报告；当前175条不能重新当新holdout。若多消费者组仍有较大端到端下降，应减少该组消费者/保留另一个source，而不是盲目放松阈值或换测试子集。

测试的实际范围见 `TEST_RESULTS.md`；执行前先跑服务器完整回归和一个小型真实GPU闭环。
