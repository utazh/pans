# SharedKV v2.6：冻结 v2.5 方法后的论文证据包

基线：utazh/pans 的 v2.5 提交 `5e523ec7f48b99b54cf449ace7b8f95e94c7e8db`。
本包不是下一版训练配方，不新增 G/R 搜索，不修改 reader 校准和冷状态定义。目标是核对机制、加强无共享对照、扩展有代表性的计时样本，并开始论文写作。

## 1. 实现范围

1. **固定 checkpoint 的原生块消融**：保持 source 图、reader、选择器和物理加载不变，将小原生保留区的内容替换为同位置的 `K_source @ A + bk`、`V_source @ B + bv`。只用于质量诊断，显式限制重建的小块 token 数，绝不重建整个 consumer prefix。
2. **固定结构的直接共享消融**：导出 A/B 为单位矩阵、偏置为零的诊断 checkpoint；保留 G16/R4 图、原生块和数据禁用名单。不是 KVSharer 原论文完整算法。
3. **可选更强恢复基线**：缓存晋升到当前 GPU 时，利用预取已经返回的完整 FP16 chunk，复制成独立缓存张量，避免从文件再恢复一次。分配、复制、注册开销全部在 TTFT 中。identity 和 shared 都用这一选项；不能将其通用优化收益算成共享的独立收益。
4. **冻结实验编排**：不同 block 使用不重叠 query，四臂位置平衡，每臂独立进程，保持 native/preserve 同策略配对。所有质量统计仍按独立 query，不把种子或计时重复展开。
5. **维持原非劣规则**：harm/benefit 的两项精确二项界，默认每路线 alpha=.025、margin=.01；不会事后更换规则把 v2.5 的失败改写为通过。

没有实现：新任务数据集/新前缀生成、xKV/MiniCache 完整基线、14B 多卡、R位置训练搜索、完整生成质量基准。已有 547 项服务器测试也没有在这里重跑。

## 2. 安装

```bash
git fetch origin v2.5
git worktree add ../pans-v26 5e523ec7f48b99b54cf449ace7b8f95e94c7e8db
python /path/to/pans_sharedkv_v26/scripts/install_sharedkv_v26.py --repo ../pans-v26
cd ../pans-v26
PYTHONPATH=src python -m pytest -q tests/test_sharedkv_v26.py
```

仅新增 `sharedkv_v26`、新脚本、文档和测试；原代码文件不覆盖。安装器检查六个直接相关接口，新入口仍执行原 v25 的依赖 guard。需要原环境中的 torch、numpy、scipy、pytest；GPU 路径沿用实际 vLLM FA2。

## 3. 先冻结主方法

建议主配置固定为 G16/R4，主计时 seed24。G14、R0 放在覆盖率/消融曲线，不为了更大数字继续搜索 G12/G8。

保存主 checkpoint SHA256、图、R规则、token预算、训练种子、软件环境和数据 manifest。final1536 已经观察过，此后机制探索只把它称为历史回归；新方法选择不得又把它称作新测试。

## 4. 机制对照（不训练，不测速度）

### 4.1 直接共享控制

```bash
python scripts/export_direct_v26.py \
  --input /path/to/frozen_G16_R4_seed24/trec.pt \
  --output assets/v26_direct_g16_r4/trec.pt
```

该操作不修改源 checkpoint。它回答相同来源/原生块下，读取变换是否必要；没有与 learned reader 等训练成本，故仅为固定结构机制对照。

### 4.2 原生块内容替换

```bash
python scripts/run_evidence_v26.py \
  --repair-values source-proxy \
  --audit-jsonl results/v26_proxy.extra.jsonl -- \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --store-root /path/to/prefix_store \
  --selector-index-dir /path/to/selector_index_k4_g32 -- \
  --pcache-root /path/to/private_pcache -- \
  --mode shared --backend vllm2 --diagnostic --allow-candidate \
  --adapters /path/to/frozen_G16_R4_seed24 \
  --metadata-cache --packed-ops --layout hybrid -- \
  --task trec --budget 25 --period 8 --samples 128 --warmup 8 \
  --bundle-dir /path/to/observed_development_bundle \
  --output results/v26_proxy
```

共四次 `--`：v26 / v25 / v23 / v21 / round4。对照运行改为 `--repair-values native` 并使用新输出路径，其余条件一致。

原生小块仍按原路径加载，source-proxy 只是 attention 时不用真实内容。理论实数计算等价于这个固定 reader 对全部 prefix 使用仿射共享近似，但 BF16 的显式小块重建有额外舍入点。因此需要记录 BF16与参考计算差异；不能把它等同于重新训练过的 R0，也不能从其 TTFT 推断加速。

这一实验能够检验“这些本层原生内容对固定 checkpoint 是否重要”，**不能证明最后4块比任意位置更好**。要声称位置优越，仍需公平的匹配大小、重新校准的位置对照；本包没有实现该搜索，建议论文先不作该主张。

## 5. 更强恢复基线：先过数值和生命周期检查

原系统可能对同一个 chunk 先做预取恢复，再在 CKLFU 晋升时重新读文件。可选 `reuse-prefetch` 只消除匹配到的 disk→当前GPU 的重复恢复，CPU 晋升仍走原路径。

- 只接受已落盘、分离K/V、完整FP16块；拒绝不完整块、重排、推测预取及跨GPU恢复。
- 完整块必须由原 `_resolve_entry` 返回，来源已同步就绪；不从目标答案构造数据。
- 缓存准入、评分、逐出决定仍由原 ScoreUnit 执行。
- 缓存拿到独立 clone，不保留整份 request buffer 的 view。
- 不更改文件、不改 DONTNEED、不改 mincore、不清全系统缓存。
- 注册表可能短时持有原 FP16 backing storage，必须实测总峰值显存；统计的view字节不是完整storage上界。
- 新代码主要经过CPU模拟检查，**没有真实GPU堆/缓存容量认证**。

先在已观察开发集运行 8–32 条 original-admission/reuse-prefetch，每个模型分别对比首token logits、全部候选token logprob、选择IDs、device_map、堆容量、冷认证、文件hash及private KV边界。预测相同不够。

示例冷恢复检查：

```bash
python scripts/run_evidence_v26.py \
  --admission reuse-prefetch \
  --audit-jsonl results/v26_reuse_probe.audit.jsonl -- \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --store-root /path/to/prefix_store \
  --selector-index-dir /path/to/selector_index_k4_g32 -- \
  --pcache-root /path/to/private_pcache \
  --cold ssd-cold --read-advice preserve \
  --owned-cache-root /path/to/private_pcache \
  --cold-log results/v26_reuse_probe.cold.jsonl -- \
  --mode identity --backend vllm2 --metadata-cache --layout hybrid \
  --stats results/v26_reuse_probe.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 16 --warmup 8 \
  --bundle-dir /path/to/observed_development_bundle \
  --output results/v26_reuse_probe
```

将 `admission original` 作为同模型对照。shared 再作相同检查。代码不自动认为检查通过。

如数值/容量/生命周期失败，先修复，不进入速度主表。若该优化没有加速，也如实记录；它是更强基线实验，不保证效果。普通的代码注册/clone成本可能抵消小请求收益。

## 6. 有限的论文计时研究

编辑 `configs/v26/evidence_study.example.json` 中所有路径、checkpoint SHA。`method_frozen` 只在真的冻结后设 true；`admission_gpu_parity_passed` 只在上节实际GPU审计通过后设 true。

```bash
python scripts/plan_study_v26.py \
  --spec configs/v26/evidence_study.example.json \
  --output studies/v26_g16_frozen

# 默认不执行。显式前台串行执行；失败保留，禁止覆盖。
python scripts/run_study_v26.py \
  --manifest studies/v26_g16_frozen/study.json --execute --max-jobs 1

# 验收第一臂后，去掉 --max-jobs 继续，不会重新覆盖已完成臂。
python scripts/run_study_v26.py \
  --manifest studies/v26_g16_frozen/study.json --execute

python scripts/analyze_study_v26.py \
  --manifest studies/v26_g16_frozen/study.json \
  --output studies/v26_g16_frozen/summary.json
```

默认设计12个block，每个32条不重复query。每block四臂：无共享/共享 × 原准入/预取复用准入，使用独立进程。默认 preserve、warm 两种条件，共96个臂。它可能需要较长时间；计划器不自动开后台作业。预算有限时预登记8块，而不是跑完再删除慢块。

native 策略可加入，但不能将不同策略混算。不同condition使用相同query块是配对设计，不是新质量样本。所有条件都保持同P8、25%预算、在线选择和正常模型预热；冷条件每请求重新通过原冷认证。

四个循环Latin位置次序用于平衡位置；它不覆盖全部二阶carryover，也不消除主机负载相关性。每臂独立进程，输出所有block方向、均值、P95等原日志。统计以run block而非每次重复请求为单位。

## 7. 质量规则不后改

```bash
python scripts/audit_quality_v26.py \
  --baseline results/identity/scored_records.jsonl \
  --student results/shared/scored_records.jsonl \
  --alpha-per-route 0.025 --margin 0.01 \
  --output results/v26_quality_audit.json
```

本脚本复现v2.5的保守二项界，三个种子仍分别计算。同一1536题不能合成4608独立题。新方法不能靠改 alpha/margin 或事后选种子升级旧失败；脚本从不修改 checkpoint 的诊断状态。

## 8. 接下来不是继续堆模块

- 同7B至少再做两个不同任务/两三个独立prefix，分别构建和校准；teacher/test禁用名单继续保留。
- 第二模型可后续做；14B未适配多卡，不把单24GiB不支持的路径写成已完成。
- 外部强基线至少落实一个近邻：KVSharer或xKV/MiniCache。按同质量/总字节预算对齐，并保持前缀命中恢复场景。
- 当前报告只证明候选标签分类，不等于自由生成或long-context QA泛化。先限制论文主张，或另补生成基准。
- 报告完整离线成本：teacher采集、父reader、搜索、额外种子、新训练，而不是只用主checkpoint训练时间。

具体论文结构和已有证据可写的主张见 `docs/MANUSCRIPT_OUTLINE_ZH.md`。
