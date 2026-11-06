# SharedKV v2.4：逐步减少全模型 source 数，直接校准实际候选评分

对应 `utazh/pans` 的 `v2.3` 提交 **3868c6b64332f89cc0b301c75b53c35630302f0e**。
本包是研究代码，未在真实 Qwen/RTX3090 上验证新效果，不是已经质量验收的部署版本。

## 这次实际改变什么

推理架构不变：共享真实 prefix K/V，A/bK 作用于 prefix 读侧，B/bV 作用于 prefix 输出；private query/generated KV 逐层独立。复用 v23 已有 SSD 冷恢复和 v21 优化运行时。

新增：
1. 从真实 scorer 采集全部候选标签的 token IDs、逐 token logprob、**平均**续写分数。无 gold label、无正确性信息；不从旧 group hidden traces 猜标签分数。
2. 一次重放完整实际 student（全部 decoder 层），联合优化指定 source-map 的 readers，直接约束最终候选评分。训练固定 teacher 的 token 选择；线上仍执行真实 P8 selector。
3. 从 calibration-fit 的真实 IDs 生成 26→24→…→16 来源的结构候选；默认保护前4层。显式 `--protect-first 0` 才生成尝试14来源的完整候选。实际ID等价类不足时，来源数未必正好每次减2。
4. 可选的预取/缓存晋升读取路径审计。它不修改缓存策略、不消除重复加载，不是速度优化。

这不是全参数微调。只训练 A/B/bK/bV。经典蒸馏/排序损失不是新发明，本实验的研究问题是：在固定模型和精确物理共享下，面向最终评分的联合校准能否支持更多消费者。

## 环境及安装

继续使用服务器原 PyTorch/Transformers/vLLM 环境；质量审核脚本还需要 SciPy。不要为了本包盲目升级依赖。

```bash
git fetch origin v2.3
git worktree add ../pans-v24 3868c6b64332f89cc0b301c75b53c35630302f0e
python /absolute/path/pans_sharedkv_v24/scripts/install_sharedkv_v24.py --repo ../pans-v24
cd ../pans-v24
PYTHONPATH=src python -m pytest -q tests/test_sharedkv_v24.py
```

安装只新增文件，9个依赖接口按实际Git blob hash核对。不要绕过 guard；本地版本不同应重新审核。
新 worktree 不含模型、TREC、prefix store、Pcache、K4索引、旧reader。沿用原项目路径配置；`--pcache-root` 应明确指向独立数据副本。不得把服务器的绝对路径当成本包已经存在的资源。

## 数据先行

官方500条已经观察过，是固定历史回归，不是本轮的新holdout。至少区分：
- fit：校准和结构生成；
- validation：checkpoint选择；
- development：完整模型阶段准入，不能参与梯度；
- final：拓扑和全部超参数冻结后的独立验证。

可从官方训练池预先划分并排除prefix示例、此前用作开发/回归的query。按**实际tokenized query hash**查重；不要只改UID。不同训练seed不增加独立质量样本数。

第一轮可沿用64/64的规模作同数据消融，随后在事先划分的较大fit/val/dev上重跑；新增样本量本身也是变量，要给旧v23控制同样数据。不要看到500结果后再选checkpoint。

旧v23 bundle通常只有fit/val合并的禁用hash列表，没有角色信息。使用 `--initial` 时，若无法确认旧split角色，代码会拒绝与新fit/val重叠；可以不指定initial从相同仿射初始化做对照，或改用与旧calibration集合不交的新fit/val。不能伪造角色字段来通过检查。新的v24连续阶段记录并保留split角色。

## 1. 采集紧凑teacher评分

以下 wrapper 有**三次 `--`**：v24选项、v23选项、v21选项、round4选项。capture不含大hidden-state traces。

```bash
python scripts/run_sharedkv_v24.py \
  --teacher-records runtime_data/v24_teacher_fit --split fit -- \
  --pcache-root /absolute/path/to/private_pcache -- \
  --mode identity --backend vllm2 --diagnostic \
  --metadata-cache --layout hybrid -- \
  --task trec --budget 25 --period 8 --samples 64 --warmup 0 --max-tokens 1 \
  --bundle-dir /absolute/path/to/fit_bundle \
  --output results/v24_teacher_fit
```

验证集使用独立路径、`--split validation` 和独立 bundle，其他协议相同。校准 warmup=0 只避免重复写同query，不是冷缓存定义。capture跑出的TTFT不可用于性能结论。

记录包含原生teacher实际候选token列表及全部逐token得分，包括候选最后一个token。保存平均分数，验证 `sum(logprobs)/len(tokens)`，拒绝偷换成总log-likelihood。record不保存gold。

## 2. 生成来源数阶梯（结构候选，不是质量已通过）

```bash
python scripts/plan_sources_v24.py \
  --teacher-fit runtime_data/v24_teacher_fit --task trec \
  --protect-first 4 --block-size 4 --output-dir configs/v24_ladder
```

每个4层块保留第一层，另外三层中寻找跨所有fit请求都具有相同 IDs 的最大等价类；选该类中间层为source。该规则不使用标签、精度、test或延迟。候选按后段→前段排列，第一组在匹配实际IDs时正是26供25/27使用。

预计可生成 `g26.json, g24.json, ... g16.json`，但必须看实际输出，不要把文件名数字当成保证。每阶段训练还会验证val的IDs；任何不一致都会拒绝，绝不隐式union、截断或改变25%预算。

先验证g26，再评估g24等。`--protect-first 0` 是未来研究全部7组、尝试g14的显式选项，不能一上来声称14来源质量合格。不同来源数不会自动换算成比例相同的TTFT改善。

## 3. 直接校准最终评分

```bash
python scripts/train_budget_sharedkv_v24.py \
  --model-path /absolute/path/to/Qwen2.5-7B-Instruct \
  --store-root /absolute/path/to/prefix_store --task trec \
  --teacher-fit runtime_data/v24_teacher_fit \
  --teacher-val runtime_data/v24_teacher_val \
  --plan configs/v24_ladder/g26.json \
  --steps 200 --eval-every 25 --seed 24 --device cuda \
  --output assets/v24_g26_s24/trec.pt
```

可以添加 `--initial /absolute/path/to/v23_or_v24/trec.pt`，但必须遵守上述split角色和来源所有权检查。使用相同query样本对比v23 group-only和v24 score目标时，控制初始化、数据、seed，并记录计算量。200次v24更新会评估多条候选，不等于v23每步抽一条分支的训练成本。

### 训练的精确含义

`S(y)=mean_j log p(y_j | query, y_<j, selected-prefix)`。
目标含：候选分布KL（softmax(S)，不是声称LM原生定义了此类别概率）、中心化S的MSE、逐token得分MSE、以teacher优胜候选为参照的截断margin保护。无gold分类CE，也不挑选真实标签分支。

从当前student的embedding开始真实执行所有28层；当前query/候选prefix K/V每层重新计算。后面的冻结层在梯度图中。与旧代码不同，不使用预先采集的teacher/private tail输出来替代改变后的student状态。

训练固定teacher选择集合，选择器不在反向图中。正式质量评估必须用真实在线P8选择；如果组合后selector发生变化，不能用固定plan结果认证线上质量。

每次开始先把**无reader原生全网络重放**与实际teacher分数比较。默认最大平均分数差0.02、token logprob差0.10，都是预设工程数值门限，不是任务非劣界限。失败会输出 `.rejected_replay.json` 并中止；先修数值/后端，勿为了跑通临时加大阈值。

为减小显存，默认启用非重入activation checkpoint；并以两遍精确链式法则逐候选反传，只保留一个候选的模型图。CPU测试验证其梯度与一次性完整图一致。代价是更多训练计算。报告包含显式候选前向/反向数、墙钟和CUDA峰值；没有在RTX3090上实际验收峰值。OOM时先缩短明确的max-sequence或减少训练分支需求，不可静默截断任务。

默认只支持batch=1、静态RoPE、同编号head、相同选择集合、一跳source。不是通用开放生成蒸馏。任务条件化reader需要额外跨任务/自然长生成验证。

### 分阶段扩展

g26在独立dev达到预设准入条件后：

```bash
python scripts/train_budget_sharedkv_v24.py \
  --model-path /absolute/path/to/Qwen2.5-7B-Instruct \
  --store-root /absolute/path/to/prefix_store --task trec \
  --teacher-fit runtime_data/v24_teacher_fit --teacher-val runtime_data/v24_teacher_val \
  --plan configs/v24_ladder/g24.json --initial assets/v24_g26_s24/trec.pt \
  --steps 200 --eval-every 25 --seed 24 --device cuda \
  --output assets/v24_g24_s24/trec.pt
```

此时此前和新加入的reader一起优化；所有student状态每次实时重算，所以不需要旧式ascending carry traces。不能直接拼接分别拟合的多组reader当成联合结果。输出始终是diagnostic候选；局部旧门限没有在此trainer重新计算，明确记作未重新验收，不会伪造通过。

## 4. 推理与冷测试不需要换执行内核

导出的是原有标准reader格式，使用已有v23入口即可，避免把新推理工程优化与方法收益混起来。

```bash
python scripts/run_sharedkv_v23.py \
  --pcache-root /absolute/path/to/private_pcache -- \
  --mode shared --backend vllm2 --adapters assets/v24_g26_s24 --allow-candidate \
  --metadata-cache --packed-ops --layout hybrid \
  --stats results/v24_g26_dev.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 128 --warmup 8 \
  --bundle-dir /absolute/path/to/new_dev_bundle --output results/v24_g26_dev
```

正式冷测试沿用验证过的v23协议：

```bash
python scripts/run_sharedkv_v23.py \
  --cold ssd-cold --read-advice native \
  --pcache-root /absolute/path/to/private_pcache \
  --owned-cache-root /absolute/path/to/private_pcache \
  --cold-log results/v24_g26_cold_native.r0.jsonl -- \
  --mode shared --backend vllm2 --adapters assets/v24_g26_s24 --allow-candidate \
  --metadata-cache --packed-ops --layout hybrid \
  --stats results/v24_g26_cold_native.r0.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir /absolute/path/to/timing_bundle --output results/v24_g26_cold_native_r0
```

同时测相同identity；再成对测 `ssd-cold/preserve`。不能拿native基线对比preserve共享。每次请求仍须通过原来的应用驱逐+mincore零页认证；不再新增全局drop_caches、empty_cache或SSD写操作。原冷准备在TTFT外模拟已发生的驱逐，约4.9s不能当作生产请求中的“免费优化”，也不应加到自然冷恢复延迟上。

## 5. 针对重复恢复的诊断（不是性能测量）

```bash
python scripts/run_sharedkv_v24.py \
  --audit-restore results/v24_reads_identity.jsonl -- \
  --cold ssd-cold --read-advice native \
  --pcache-root /absolute/path/to/private_pcache \
  --owned-cache-root /absolute/path/to/private_pcache \
  --cold-log results/v24_reads_identity.cold.jsonl -- \
  --mode identity --backend vllm2 --diagnostic --metadata-cache --layout hybrid -- \
  --task trec --budget 25 --period 8 --samples 8 --warmup 8 \
  --bundle-dir /absolute/path/to/timing_bundle --output results/v24_reads_identity
```

对shared/native、identity/preserve、shared/preserve各跑独立诊断。新增审计记录“prefetch磁盘读取尝试”“晋升时磁盘读取尝试”与两路径重叠文件；它**不表示每次尝试都命中物理盘**。与/proc/read_bytes、Nsight和读取建议结合解释。诊断有Python封装/队列屏障，不用于正式TTFT。

本包没有实现prefetch结果直接adopt到CKLFU。这样可以避免在未证明重复恢复程度前贸然改变缓存策略或生命周期。确需后续admission优化，应同样提供给identity/shared，重新测共享增量。

## 6. 质量准入

```bash
python scripts/audit_quality_v24.py \
 --baseline results/identity_dev/scored_records.jsonl \
 --student results/v24_g26_dev/scored_records.jsonl \
 --margin 0.01 --output results/v24_g26_dev_quality.json
```

脚本输出净差、有害/有益翻转和保守单侧界，不自动修改checkpoint状态。统计假设为固定比较下的独立query；对多次拓扑选择/同prefix聚类没有自动校正。McNemar不显著不等于非劣。新增独立训练seed也不能扩充query样本量。

建议先修复g26开发集退化，预登记seed 24/25/26；然后再走g24/g22/g20。可以同时筛查各层段作为探索，但不能越过已失败的开发验收宣称大范围共享可部署。最终跨新prefix/任务验证后，再谈一般性贡献。

## 文件/验证范围

本包未修改v23 SSD-cold/native实验结果，未改原P8、保留率、缓存容量或代码后端。未做真实Qwen/GPU性能测试；数学与模拟接口测试见 TEST_RESULTS.md。
