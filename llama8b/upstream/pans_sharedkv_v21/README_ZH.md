# SharedKV v2.1：针对已发布 v1 的重新审核与可执行补丁

目标分支：`utazh/pans` 的 `v1`，提交 `59312f2bd2504079fed03b53f283d8ade931fee2`。
不是 `main`，也不是上一轮基于上传原型的旧 v2。新增模块全部放在 `sharedkv_v21`，不覆盖原 runner、Pcache、sharedkv 或旧 v2。

## 已实现与边界

本补丁实现：vLLM FA2 接口适配；固定 IDs 时仍执行选择器；请求内 cu_seqlens 缓存；稠密 reader 预打包；普通层 single-attention 对照；窗口预取/按需加载对照；相同 reader 下冗余 consumer 加载的诊断；逐层 teacher/student traces；顺序非对称闭式 V 校准；可选仅训练 A/B/bias 的蒸馏；按层 token Jaccard 与标签续写分数审核；单对 bundle 提取。

没有实现或证明：真实 Qwen 质量恢复、RTX3090 加速、硬件 H2D 认证、融合 CUDA/Triton kernel、MLP/block 联合蒸馏、跨请求 alias-aware 缓存评分、并发 serving、磁盘物理压缩、多个源头的共享子空间。

CPU 测试和 GPU 跳过情况见 `TEST_RESULTS.md`。所有候选始终导出 `diagnostic_only=true`，不自动晋升。仅显式 trace 校准可以重新使用拟合 query；普通性能诊断仍保留数据泄漏拦截。

## 1. 安装

先在自己的机器获取确切 v1 代码：

```bash
git fetch origin v1
git worktree add ../pans-v21 59312f2bd2504079fed03b53f283d8ade931fee2
python /path/to/pans_sharedkv_v21/scripts/install_sharedkv_v21.py --repo ../pans-v21
cd ../pans-v21
python -m pytest -q tests/test_sharedkv_v21.py tests/test_v21_published_bridge.py
```

安装器核对 8 个接口的 Git blob hash。不通过说明工作树与本次审阅不同，应该审查差异，不应直接关闭 guard。此包不分发模型、缓存、数据、readers 或真实运行日志。沿用仓库 `PANS_MODEL_PATH/PANS_STORE_ROOT/PANS_PCACHE_DIR/PANS_SELECTOR_INDEX_DIR` 等环境变量。

## 2. 先复核现有证据，不急于增加共享数

```bash
python scripts/audit_published_v1.py \
  --csv docs/layerkv/results/comparison.csv \
  --output results/v21_published_audit.json
```

P8 复用 IDs，不是 K/V 张量。固定 32 条对照：physical payload 减少 3.5598%；host 来源 payload 计数减少 17.5626%；TTFT 增加 17.6339%。两种字节口径不能混称 H2D。在线128条是另一组实验，不能和固定32条均值直接配对。

原路径到 identity 已改变后端、mask/cache 组织和推测预取。`selection_agreement_vs_original=0.0234375` 是整份28层计划 hash，仅3/128完全一致，不是说token overlap只有2.34%。

单层对 26→27 位于模型末端；这不能证明早层共享可行。原3对是26→27、3→2、18→19；7对还加入22→23、15→14、10→11、6→7。先分别测试每一对。

```bash
# 示例：从现有dense7中仅保留3→2；不重新拟合，不改变其矩阵。
python scripts/slice_layerkv_v21.py --input assets/layerkv_dense7/trec.pt \
  --targets 2 --output assets/v21_isolated2/trec.pt
# 相同来源与消费者的直接共享对照
python scripts/slice_layerkv_v21.py --input assets/layerkv_dense7/trec.pt \
  --targets 2 --direct --output assets/v21_direct2/trec.pt
```

## 3. original / identity / shared 三组先通过正确性检查

下面的 `data/sharedkv_splits/test` 是已被观察过的旧测试池；现在只能用于回归/开发，不能重新称最终未见测试集。最终结论要有新的未参与设计的数据。

```bash
python scripts/run_sharedkv_v21.py --mode original -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir data/sharedkv_splits/test --output results/v21_original_smoke

python scripts/run_sharedkv_v21.py --mode identity --backend vllm2 \
  --stats results/v21_identity_smoke.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir data/sharedkv_splits/test --output results/v21_identity_smoke

python scripts/run_sharedkv_v21.py --mode shared --backend vllm2 \
  --adapters assets/layerkv_dense1 --allow-candidate \
  --stats results/v21_shared_smoke.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir data/sharedkv_splits/test --output results/v21_shared_smoke
```

`--allow-candidate`仅允许在独立开发集评价未经认证的checkpoint，不证明它已通过局部或端到端门限。

比较 scored_records：

```bash
python scripts/compare_layerkv_v21.py \
  --baseline results/v21_identity_smoke --student results/v21_shared_smoke \
  --output results/v21_smoke_compare.json
```

## 4. 定位P8与速度：先改变payload预取，不改变P8选择

先固定 `--period 8`、相同bundle、相同样本、相同精度，做2×2：

| 选择周期 | payload调度 | identity | shared |
|---|---|---|---|
| P8 | `--schedule window` | 跑 | 跑 |
| P8 | `--schedule demand` | 跑 | 跑 |

`demand`仅使payload在消费时由原loader.resolve加载；不关闭选择器、不改变4个P8 leader，也不关闭下一leader的selector-key预取。用于判断现有payload预取是否隐藏了共享可省的成本。主结果仍必须对照原本启用预取的强系统。

固定计划可使用已保存的v1 logits目录：

```bash
python scripts/run_sharedkv_v21.py --mode shared --backend vllm2 \
  --adapters assets/layerkv_dense1 --allow-candidate --diagnostic \
  --selection-plan results/original_logits --schedule window \
  --stats results/v21_fixed.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir data/sharedkv_splits/test --output results/v21_fixed
```

这里仍完整执行INT4 index加载、打分和原选择函数，只替换最终IDs。`--replay-dir`接受v21 traces目录，也采用相同语义；和`--selection-plan`互斥。

### 冗余加载对照

给上面相同命令增加 `--shadow-load` 并更换输出路径。attention仍读完全相同的source和reader，但恢复consumer自身的KV、转换并保留，随后不用它参与attention。如此可在相同读侧数学下观察去掉一份payload恢复的效果。

必须检查两组 logits/选择一致。该对照会改变缓存驻留与队列，是诊断而不是严格的生产baseline；时间差不能直接当作硬件memcpy时间。所有shadow bytes/loads单独记录。

### 接着才测试P2/P4/P8

相同26→27在P2/P4/P8结构上都可表达；但bundle绑定period、Q分布与选择协议，必须各自重新校准/验证。不得把P8 bundle改一个JSON字段就称P2有效。

P1会把所有层当作selector leader，当前保护leader协议不允许共享consumer；代码明确拒绝。P1可以作为原系统选择开销基线，但不能冒充当前实现已经支持P1共享。

## 5. 速度优化必须逐项消融

顺序试：`--metadata-cache`；然后加 `--packed-ops`；最后比较 `--layout split` 与 `--layout hybrid`。

- metadata cache是请求内长度数组复用，未缓存Q/K/V或attention结果。miss发生在计时区内。
- packed ops预备A.T与FP32 key bias；导出矩阵仍是dense。新增副本计入`.v21.jsonl`的`packed_reader_extra_storage_bytes`。
- hybrid只对无reader的普通层拼接自己的prefix/tail；consumer仍读共享KV。普通层的GPU拼接分配属于计时的一部分，未隐藏。它可能比split慢。
- 统计cu缓存包括标签续写；selector计数仅首轮prefill。不要把whole-request计数当首token计数。
- 原v1的CUDA峰值统计是进程累计且位于标签评分之后，本补丁没有把它重命名为TTFT峰值。精确内存比较应另做隔离诊断。

```bash
python scripts/bench_sharedkv_v21.py --backend vllm2 \
  --adapter assets/layerkv_dense1/trec.pt --consumer 27 \
  --selected-tokens 1228 --query-tokens 64 \
  --iterations 40 --warmup 10 --output results/v21_micro.json
```

1228是公布数据中该consumer所选字节推导的token数；query=64只是示例，必须换成真实长度分布。微基准数据为合成resident tensor，不是Qwen质量、H2D或TTFT。`--compile`只试reader前后的小函数，不是完整attention融合。

诊断加 `--nvtx --diagnostic`，用外部Nsight采集。无profiler、无logits转储的正式组交替顺序至少重复若干轮。不要把128请求当成128轮硬件独立复测。

## 6. 质量改进：闭式非对称校准 → 可选reader训练

teacher使用同一新执行后端的无共享identity；它不是逐位等价于原SDPA路径，必须另过original/identity验收。teacher不应是上一轮student。

先建立独立fit/val目录（每个task文件格式沿用v1）；旧128条已用于设计，不作最终holdout。

```bash
python scripts/run_sharedkv_v21.py --mode identity --backend vllm2 --diagnostic \
  --trace-dir runtime_data/v21_teacher_fit --trace-layers 2,19,27 -- \
  --task trec --budget 25 --period 8 --samples 64 --warmup 0 \
  --bundle-dir /data/new_cal_fit --output results/v21_teacher_fit
# val用/data/new_cal_val、runtime_data/v21_teacher_val和新的results路径，独立再跑一次。
```

以已有26→27 reader作为初值，仅重校准B：

```bash
python scripts/refine_sharedkv_v21.py --store-root "$PANS_STORE_ROOT" --task trec \
  --teacher-fit runtime_data/v21_teacher_fit --teacher-val runtime_data/v21_teacher_val \
  --student-fit runtime_data/v21_teacher_fit --student-val runtime_data/v21_teacher_val \
  --source 26 --target 27 --initial-reader assets/layerkv_dense1/trec.pt \
  --steps 0 --output assets/v21_refit27/trec.pt
```

A固定时，设计矩阵为 `[beta_s * U_s, beta_s]`，目标为 `O_teacher - (1-beta_s)*U_private_student`。这是无梯度校准，不更新主干。

另开输出目录做 `--steps 200 --eval-every 20 --lr 0.0001 --device cuda --bf16-ste` 对照，只优化A/B/bk/bv；此时明确称reader蒸馏训练。损失含prefix各token+整个private-tail桶的KL、attention输出误差、o_proj之后pre-MLP残差误差以及trust正则。样本数/学习率/步数是起点，未GPU调优。

**多对顺序**：例如3→2、18→19、26→27，必须按target=2,19,27顺序推进。不是先接受最后一层再直接向前叠加。

每接受一个上游checkpoint，在相同fit/val请求上重跑student，用teacher traces回放相同IDs：

```bash
python scripts/run_sharedkv_v21.py --mode shared --backend vllm2 --diagnostic \
  --adapters assets/accepted_upstream --trace-dir runtime_data/v21_student_fit \
  --replay-dir runtime_data/v21_teacher_fit --trace-layers 19,27 -- \
  --task trec --budget 25 --period 8 --samples 64 --warmup 0 \
  --bundle-dir /data/new_cal_fit --output results/v21_student_fit
```

对应val也重跑。随后refine使用这些student目录和 `--carry assets/accepted_upstream/trec.pt`。代码核对上游reader fingerprint，不允许拿旧teacher-state当新的student。没有自动认证/人工篡改通过状态的脚本；每一步的独立端到端验收由实验记录承担。

局部门限保持0.03/0.08/0.03。新候选不通过也可显式诊断，但不能宣称已验收。即使局部通过，仍要检查完整模型与在线selector反馈。B/A矩阵没有能力凭空恢复source中缺失的特征；完整矩阵+直接蒸馏仍失败时，才考虑head匹配或xKV式共同子空间，而不是不停调rank。

## 7. 质量审计与更强的资源贡献

运行时加 `--logits results/new_logits` 会在计时之后导出完整首token logits、选择计划，以及调用方实际传入的标签续写得分。主TTFT不要开转储；转储可扰动下一请求的缓存/主机行为。

```bash
python scripts/compare_layerkv_v21.py --baseline results/base --student results/student \
  --baseline-logits results/base_logits --student-logits results/student_logits \
  --margin-pp 1 --output results/paired_quality.json
```

新报告分清首token完整词表KL、标签续写归一化分数KL、标签margin、逐层token Jaccard和整份计划hash。旧capture没有label得分时报告为空，不伪造。

128条全部没有额外错误时，paired bootstrap可能退化为[0,0]。代码增加保守的单侧95%有害翻转率上界；n=128且0次有害翻转时约2.313%，不能仅用退化bootstrap认证1pp非劣。该保守界不计有益翻转，需独立query假设，不替代集群/多重选择校正。

资源贡献需分别报告：(1) selected唯一payload，(2)含读取放大的实际恢复payload，(3)host/SSD来源计数，(4)Nsight硬件H2D，(5)加readers及packed/workspace后的总显存，(6)原系统TTFT/P95。SSD原文件仍未删除，不能声称磁盘容量压缩。

建议先固定质量门限改善帕累托前沿，再用相同总字节预算增加保留token或提高prefix命中率。token预算增加并不保证TREC准确率单调改善。更长自然prefix、多prefix竞争和新的多token任务比只有同一短TREC分类prefix更能说明通用价值。本包没有batch>1/并发实现，不能据单请求结果宣称服务吞吐提升。
