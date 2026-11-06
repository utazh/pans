# v2.7：质量边界、机制收敛与论文基线准备

基准代码：utazh/pans v2.6 @ `11fde71c36ceecded60d560620900dec5c3a5d76`。

## 决策

现在可以写方法、系统与结果章节；不能把 v2.6 包装成普遍无损或已超过 ContiguousKV。
报告的 I 是同后端、同 P8 的无共享 identity，并不是自动等同完整 ContiguousKV。
四任务×四预算的 15/16 个质量下降单元必须保留。该版本不继续堆叠在线算法；只提供有限诊断、条件质量曲线与外部表示基线的质量接入。

## 已实现 / 未实现

- **已实现**：固定 R4 checkpoint 无训练导出 R0；source-proxy 与 R0 的精确算术等价参考；受层距约束的共享图；fit-only 计划设计；成对质量和同协议 Pareto 分析。
- **已实现**：xKV-CLF 数学参考的离线因子构建、选中行重建和 Qwen 在线质量接入；外部直接 source-map 的相同质量接入；KVSharer-style 搜索函数（需要调用方提供 final-hidden cosine evaluator）。
- **未实现**：完整作者版 KVSharer notebook 的 Qwen 移植与自动搜索执行；xKV-SR landmark selector 和 SSD-native 因子 I/O/预取；外部基线的有效端到端速度结果；14B/多卡验证；自由生成质量认证。
- 数学参考质量接入故意仍执行原生 payload 恢复，随后用因子/外部图替换 attention 输入。**它的 TTFT/物理流量无效，不能做速度基线。**输出有 quality-only 标记，并另存不含时间字段的质量记录。

## 安装

```bash
git fetch origin v2.6
git worktree add ../pans-v27 11fde71c36ceecded60d560620900dec5c3a5d76
python /path/to/pans_sharedkv_v27/scripts/install_sharedkv_v27.py --repo ../pans-v27
cd ../pans-v27
PYTHONPATH=src python -m pytest -q tests/test_sharedkv_v27.py
```

安装只新增，检查7个实读接口；不修改旧训练、晋升、共享或冷状态判断。重复安装会拒绝覆盖。
本次 54 项 CPU 单测通过、1 项 CUDA 单测跳过；7 个命令行帮助入口通过。新增工具依赖现有 PyTorch/NumPy，以及 SciPy 与 pytest；服务器必须先跑旧完整回归与真实 Qwen 小样本桥接。

## A. 先做不需训练的固定-reader控制

```bash
python scripts/export_frozen_r0_v27.py \
  --input /path/to/frozen_g16_r4_seed24/trec.pt \
  --output assets/v27_frozen_r0/trec.pt
```

只把 `native_policy.blocks` 改为 0，保存父文件 SHA、所有数据禁用名单和 reader 张量，不重训、不改来源图、不把父报告当成新结果。
这不是历史从 R0 训练出来的模型。旧 R4、v26 source-proxy、frozen-R0 三臂均保留，不根据已观察 TREC500 的 76.8% 改选主方法。

运行示例：沿用现有入口，避免新增推理kernel。

```bash
python scripts/run_evidence_v26.py --admission reuse-prefetch \
  --audit-jsonl results/v27_frozen_r0.v26.jsonl -- \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --store-root /path/to/prefix_store \
  --selector-index-dir /path/to/selector_index -- \
  --pcache-root /path/to/private_pcache -- \
  --mode shared --backend vllm2 --layout hybrid --metadata-cache \
  --packed-ops --adapters assets/v27_frozen_r0 --allow-candidate \
  --stats results/v27_frozen_r0.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 128 --warmup 8 \
  --bundle-dir /path/to/development_bundle --output results/v27_frozen_r0
```

先在**相同选中计划**下检查数值，再恢复在线 selector。
生成评估计划要用 clean identity 的真实 `--logits` 捕获；固定计划在已有 v21 参数段用 `--selection-plan ... --diagnostic`，selector 计算仍保留，不能作为正式线上速度。
官方校准 records 的计划导出脚本只面向 fit/val诊断；不得绕过原模型的 query-hash 禁用名单在拟合集评估准确率。

对于 frozen-R0：无原生加载、无 proxy 构造，因此它可以在质量验证后使用原冷认证做独立计时。
对于 source-proxy：仍真实加载原生块且有额外小矩阵构造，只允许质量诊断。

## B. 停止为凑 G16 而放大层距

```bash
python scripts/plan_local_frontier_v27.py \
  --teacher-fit /path/to/teacher_fit --task subj \
  --max-distance 1 --max-consumers 2 --protect-first 4 \
  --budgets 16,20,24,28 --output-dir configs/v27_subj_b25_local
```

所有 fit 请求均核验 IDs；只允许同 P8、一跳、leader原生；不因 G16 不可行而扩大距离。
`feasibility.json` 会给出不可行目标。28来源是身份端点，不需要训练reader。
最大距离2作为**单独预登记的**对照，而非自动兜底。

新图要从相同初始化策略重新拟合，不能把旧远距 reader 直接改指向相邻层；旧 `train.initialize` 会拒绝不一致所有权。可用原 `train_native_prefix_v25.py`，省略 `--initial` 从仿射初始化，R=0/4、200步、64fit/64val（需要时另开较大fit的匹配计算量对照）。
主质量前沿：25%和50%，G16/20/24/28。5/10%保留负结果和资源账，不宣称普遍适用。

## C. 成对质量、Pareto与模型选择

```bash
python scripts/analyze_pair_v27.py \
  --baseline results/I/scored_records.jsonl \
  --student results/S/scored_records.jsonl \
  --output results/v27_pair.json
```

不合并重复UID，不跨任务/前缀/运行温度做Pareto，不把低于基线的点删掉。代码保留有害/有益翻转的保守二项界；新输出始终 `certified=false`。旧非劣失败不重算为通过。
可以在独立development选择给定质量容差下的最低成本点，然后冻结并在新的未见数据评估；该数据驱动操作点不是自动风险保证或新核心算法。

## D. xKV跨层因子的质量参考（不是完整xKV-SR系统）

离线采用论文的跨层拼接与SVD，K采用从已缓存固定RoPE结果逆旋转得到的近似pre-RoPE，V不旋转；重建仅选中行再按原绝对位置加回RoPE。先用小数据的全秩roundtrip核验，再做低秩扫描。默认float32因子空间不能当成BF16物理压缩效果。

```bash
python scripts/compile_prefix_baseline_v27.py --method clf \
  --teacher-fit /path/to/teacher_fit --store-root /path/to/prefix_store \
  --task trec --model-config /path/to/Qwen2.5-7B-Instruct/config.json \
  --window 4 --rank-k 256 --rank-v 384 --device cpu \
  --output assets/v27_clf_quality_trec.pt
```

rank仅是起点，不是等预算配平的结论；尾组rank受矩阵几何截断，实际rank和全部因子字节写入文件。

```bash
python scripts/run_baseline_quality_v27.py \
  --representation assets/v27_clf_quality_trec.pt -- \
  --admission reuse-prefetch --audit-jsonl results/v27_clf.v26.jsonl -- \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --store-root /path/to/prefix_store --selector-index-dir /path/to/selector_index -- \
  --pcache-root /path/to/private_pcache -- \
  --mode identity --backend vllm2 --layout hybrid --metadata-cache --diagnostic \
  --stats results/v27_clf.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 128 --warmup 0 \
  --bundle-dir /path/to/development_bundle --output results/v27_clf
```

只读取 `quality_records_v27.jsonl` 做质量比较。其他原runner时间/字节只是其不可抑制的原生日志，不是该因子baseline的有效性能。

## E. KVSharer

作者代码入口和固定SHA见 `docs/BASELINES_ZH.md`。`baseline_math.kvsharer_style_search` 提供论文搜索逻辑的参考：距离降序、逐步尝试、final-hidden cosine门限；只做一跳source，固定prefix与原论文校准不同，明确标注。

`compile_prefix_baseline_v27.py --method direct-map --direct-plan map.json ...` 接入外部图，允许不同层读取source的**自身所选位置**，保持private cache原生；不冒充作者完整模型KV替换版本，也不宣称完成其SSD计时。

## 论文材料

- `paper/draft.tex`：方法与当前证据的英文初稿，包含负结果和未完成项，不是投稿模板。
- `paper/references.bib`：核对过的论文条目。
- `docs/PAPER_PLAN_ZH.md`：论文主张、实验矩阵、停止条件。
- `docs/AUDIT_V26_ZH.md`：事实、推断、实验三者分离。

正式投稿前必须完成：明确的ContiguousKV基线来源；作者版或清楚标注适配的KVSharer与xKV对照；至少一个质量匹配的实际操作点；固定proxy机制解释的边界。不以“测试多”或“CCF B”替代这些证据。
