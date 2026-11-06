# v2.8：冻结 R4/R0 完整矩阵与跨层基线质量复核

针对 `utazh/pans` v2.7 commit `681fe495028598e9d51f2f57220ecb96ed0e5f35`。

本包不改主干、reader 张量、训练目标、selector、attention 内核、缓存晋升或 SSD 冷认证。所有执行均为前台串行、每臂新进程；不启动后台监控，不杀其他任务。

## 1. 本轮固定的研究问题

四任务 SST-2/SUBJ/TREC/RTE × 5/10/25/50%：每格复用该格已有、已纠正数据的 R4 checkpoint，导出相同权重、相同 source 图的 frozen-R0。

`R0` 始终指 **R4 训练得到的 reader，推理时 blocks=0**。不是历史从头训练的 R0，更不表示整个方法无需训练。

每格主比较 I/R4/frozen-R0。完整评估人数分别 872、1003、500、277；SUBJ 是修正后的固定自定义划分，不是官方 1000 条。历史完整划分继续是回归/泛化覆盖证据，不重新称为未观察最终留出。

KVSharer-Prefix G24/G20/G16 和 xKV-CLF/P8 的两档秩 256/384、384/576，分别 FP32/BF16，仍为 **质量接入**。不读取它们原 native-loader 路径的 TTFT 来排名。主方法计时另表。

## 2. 安装

```bash
git fetch origin v2.7
git worktree add ../pans-v28 681fe495028598e9d51f2f57220ecb96ed0e5f35
python /absolute/path/to/pans_sharedkv_v28/scripts/install_sharedkv_v28.py --repo ../pans-v28
cd ../pans-v28
PYTHONPATH=src python -m pytest -q tests/test_sharedkv_v28.py
```

安装器检查七个已审接口，拒绝覆盖旧代码；不替换 v27 的数值修复。代码引用已有的 v24/v25/v26/v27 模块，不需要再安装或复制旧 zip。原服务器完整 GPU 回归仍需再次运行，本包 CPU 测试不等于该回归。

## 3. 路径与冻结

模型、缓存、reader、teacher 和数据都不在包里。新 worktree 通常不包含被 `.gitignore` 忽略的实验产物，必须填入其真实绝对路径。缓存必须是自己独立 inode 的副本；不使用与其他作业共享 inode 的硬链接。

```bash
PY=/absolute/path/to/vllm-stable/bin/python
$PY scripts/matrix_v28.py template \
  --model-path /absolute/path/to/Qwen2.5-7B-Instruct \
  --pcache-root /absolute/path/to/private_pcache \
  --python "$PY" --output configs/v28_spec.json
```

**生成的是模板，不能原样默认有效。**逐格核对 `r4, teacher_fit, teacher_val, eval_file, validation_file`。原脚本里 SUBJ 名称应含 `_canonical`；数据是 `canonical_full_eval/subj.jsonl`，不能用只含部分题目的 `canonical_eval` 或未纠正文件。

TREC/25% 对应历史 `assets/v25_g16_r4_s24/trec.pt` 和 `runtime_data/v24_teacher_fit/val`。这个 teacher 的验证原文文件未必位于新的 v26/val，需填入与它 token hash 对应的原验证文件，不能为了过检查替换 teacher。

同一任务四预算必须指向同一个完整评估文件和相同 prefix。每个预算使用自己的 reader、teacher 和身份元数据；不把 TREC25 的矩阵用于其他预算/任务。

设置空闲 `gpu`（物理 GPU 编号），检查 `python`。核对后明确将 `paths_are_templates_verify_before_prepare` 改为 `false`。

```bash
$PY scripts/matrix_v28.py prepare --spec configs/v28_spec.json --output studies/v28_frozen
```

prepare 验证完整人数、原始字段、UID/规范文本重复、teacher fit/validation 角色、评估与校准 token hash 隔离、来源图/同 ID 合约，并逐张量检查 frozen-R0 未改变 A/B/bias。保存 SHA256 后不再改 spec/manifest/输入文件；更改协议要新建研究目录。

它不补造缺失 teacher/checkpoint，也不放宽历史非劣判定。缺文件时停止并给出实际路径。目录构建不运行模型前向。

## 4. 第一阶段：主方法完整质量矩阵，不重训

```bash
$PY scripts/matrix_v28.py run --manifest studies/v28_frozen/manifest.json --stage quality --cells trec_b25
# 上面只打印命令。实际执行：
$PY scripts/matrix_v28.py run --manifest studies/v28_frozen/manifest.json --stage quality --cells trec_b25 --execute
# 首格通过接口检查后运行全部 16 格：
$PY scripts/matrix_v28.py run --manifest studies/v28_frozen/manifest.json --stage quality --execute
```

总计 16×3=48 个完整质量臂；校准不新增。每格 I/R4/R0 都执行全部冻结评估题，质量运行的时间不加入正式速度表。成功作业再次运行会校验哈希后跳过；失败/半写目录不覆盖，需归档失败后建立新 study 或明确的新尝试目录，不伪造 success marker。

本轮不根据测试集分数选择每格 R0 或 R4，然后只报告最大值。两条路线全部保留。多次观察过的完整划分不用于新的最终认证。

## 5. 外部基线：用已有适配逻辑，只补泛化参数和统一分解

### 5.1 xKV

```bash
$PY scripts/matrix_v28.py compile-baselines --manifest studies/v28_frozen/manifest.json --part factors --cells trec_b25
# 去掉 --cells 依次处理全部单元
```

每个固定 prefix 只做一次完整 FP64 精确 SVD master，预算间共享 master。K 在缓存空间逆固定 RoPE，再与 V 分别跨四层拼接分解。主因子格式统一为 U、S×Vh。

从**同一个 master**裁剪两档秩，并各自导出 FP32/BF16。不存在一条随机 SVD、另一条精确 SVD的混杂；也不通过除以零奇异值得到 U。CPU 使用精确 LAPACK，CUDA 明确 `gesvd`。同一 master 的 FP32/BF16 改变的是因子及其重建计算精度，不仅是磁盘存储位数。

另导出 FP64 全秩参考。精确主分解有较高离线时间和磁盘成本，master可复用，不把 full64 参考算成部署配置。默认CPU，服务器可在冻结 spec 前选择CUDA；不得在时间测量同时分解。

每格编译输出独占；同格失败后不覆盖旧输出。编译完一格可用 `--cells` 跑下一格，避免对已存在输出再次执行编译。

### 5.2 KVSharer

```bash
$PY scripts/matrix_v28.py compile-baselines --manifest studies/v28_frozen/manifest.json --part kvsharer --cells trec_b25
```

复用 v27 实际服务器搜索定义：30条按hash排序 FIT，K/V分别做token/head向量L2后求均值，距离降序、原remap、最终normalized-hidden cosine均值 >0.5。每个任务/预算重新做条件化搜索，不沿用TREC图，不借用我们G16/单位reader。源可跨P8、消费者读取自己的选中位置，private KV保持原生。

搜索本身不读取gold label，但会和teacher原首token得分做原生数值回放检查。目标G未达到就保留 `unreached`，不会放宽0.5或随意凑来源。

### 5.3 必须先过每格的实际模型接入检查

```bash
$PY scripts/matrix_v28.py run --manifest studies/v28_frozen/manifest.json --stage baseline-quality --cells trec_b25 --execute
```

每格16条明确的teacher validation请求，先记录identity的实际IDs；direct identity和FP64全秩CLF在同IDs下执行完整候选评分。保持原最大标签平均分差 `<0.05` 门限；失败就停止该格，不把张量RMSE当成模型通过。

随后才对该格所有低秩/搜索图跑完整质量评估。一般每格3个KVSharer+4个CLF=7臂，共112个外部完整质量臂（未达到的G记录缺失，不造结果）。冷运行、profiler和外部TTFT自动排除。

只有生成 `quality_records_v27.jsonl` 的外部质量字段才进入主质量分析；用于数值验证的原始 scorer 输出另存、SHA固定，不掩盖其中没有意义的基线时间。

## 6. 正式计时：只测真实恢复路径 I/R4/R0

```bash
$PY scripts/matrix_v28.py run --manifest studies/v28_frozen/manifest.json --stage timing --cells trec_b25
$PY scripts/matrix_v28.py run --manifest studies/v28_frozen/manifest.json --stage timing --execute
```

默认每格6个不同query块×16条，每块I/R4/R0新进程，六种顺序完整平衡。每任务用96条不同请求；同任务四预算复用相同query块便于配对。每臂4次运行时warm-up，正式请求每次仍执行v23应用缓存/文件页冷认证。

主策略 `SSD-cold/preserve + reuse-prefetch`，另普通缓存。16格冷侧正式请求量=16×6×16×3=4608，普通侧同样4608；warm-up不加入该计数。可以在第一次prepare前预登记其它3的倍数，但不要看结果后加轮直到显著。

本地 foreground lock 防止同一研究多进程并发执行；它不能锁住其他研究目录或其他用户GPU/CPU/SSD。使用者须确保自己的训练/质量任务不与正式计时重叠，记录整机负载。不结束他人作业，不允许任何外部baseline quality桥进入冷速度表。

## 7. 汇总与资源口径

```bash
$PY scripts/matrix_v28.py analyze --manifest studies/v28_frozen/manifest.json --output results/v28_summary
```

输出 `quality.csv`、`timing.csv`、`external_theory_only.csv`、完整 JSON。缺臂、失败或未达目标G明确列出，不能静默忽略。

主质量同时报告I→R4、I→R0、R4→R0，逐UID有害/有益翻转和旧式保守下界，全部 `certified=false`。计时按块均值重采样，不把同题不同预算/种子当成独立样本，不将描述性区间叫成正式稳定加速认证。

external_theory_only仅是FIT选择支持下的因子/source并集算术，分开计常驻right因子和选中left行；不是实际恢复字节或模型总显存。外部质量桥仍读原生payload，故本包没有任何可用于外部SSD速度排名的实现。主方法真实资源仍看旧的 `*.extra.jsonl.v25.jsonl` 以及Nsight；不要与理论外部字节混表。

## 8. 本轮不做什么

不新增训练，不进一步减G，不改R4位置，不再运行proxy以追求更高已观察测试成绩，不回退旧错误SUBJ，不换注意力/softmax精度来给某方法单独调优；不称KVSharer-Prefix为另一篇KVShare，也不称CLF/P8为完整xKV-SR。

若后续要声称“比KVSharer/xKV更快”，必须另做它们真实source/因子恢复与重建执行，并与质量桥数值核对。当前最小改动版本只支持**完整质量对比＋主方法真实恢复对比**。
