# SharedKV v2.2：按实际读取误差对齐 GQA 组

本补丁只增加一个方法变量：目标层的第 h 个 KV 组不必读取 source 层同编号的 h 组，可以读取离线选择的 π(h) 组。A/B/bK/bV、P8、主干、source bank 和 private tail 的语义保持不变。K 和 V 必须使用相同的一对一映射。

**没有真实 Qwen 新结果；不承诺质量恢复或加速。** CPU 测试覆盖精确代数、合成拟合、数据边界和模拟接口，CUDA/Qwen 仍需服务器验收。此方法的第一关是：4×4 读取误差矩阵是否真的存在有价值的非对角匹配。没有就停止这一分支。

## 审核基线

GitHub `utazh/pans` 的 `v2`，提交 `b48c45d779ecde8ae960a6e4837d8b6673c57600`。不是默认 main，也不是只含 v1 的分支。安装器只增加文件，不改原 runner、Pcache、v21 代码或模型权重。

运行依赖现有 v2 的 PyTorch、Transformers、vLLM FA2 环境，不增加 SciPy/FlashInfer 依赖。4! 种一对一映射直接穷举。

```bash
git fetch origin v2
git worktree add ../pans-headalign-v22 b48c45d779ecde8ae960a6e4837d8b6673c57600
python /path/to/pans_sharedkv_v22/scripts/install_sharedkv_v22.py --repo ../pans-headalign-v22
cd ../pans-headalign-v22
python -m pytest -q tests/test_headalign_v22.py
```

新 worktree 不包含模型、Pcache、selector index、prefix store、traces、readers 或数据。继续设置已有 `PANS_MODEL_PATH/PANS_STORE_ROOT/PANS_PCACHE_DIR/PANS_SELECTOR_INDEX_DIR`，并用绝对路径指向旧实验数据，不要把已有文件移走。安装器和新入口会检查旧接口哈希；变化时应审核，不应删 guard 强行运行。

## 1. 先做无梯度筛查，不先跑更多层

优先诊断 18→19，因为现有报告说明其单独误差偏大；这只是开发阶段的故障定位，不是依据旧 holdout 挑选新测试赢家。

复用 v2 已有的 clean identity teacher fit/val traces。单对隔离时，student=teacher。原 fit=64 会按 query 再分成 48 条 edge-fit / 16 条 assignment-selection，外部 val 不参与头对应关系选择，只用于最终局部报告和可选训练的 checkpoint 选择。

```bash
export TEACHER_FIT=/absolute/path/to/v21_clean_teacher_fit
export TEACHER_VAL=/absolute/path/to/v21_clean_teacher_val
export PREFIX_STORE=/absolute/path/to/prefix_store

python scripts/compile_headalign_v22.py \
  --store-root "$PREFIX_STORE" --task trec \
  --teacher-fit "$TEACHER_FIT" --teacher-val "$TEACHER_VAL" \
  --student-fit "$TEACHER_FIT" --student-val "$TEACHER_VAL" \
  --source 18 --target 19 --mapping search --steps 0 \
  --device cuda --output assets/v22_map19_s0/trec.pt
```

输出 `trec.report.json` 包含 4×4 cost_matrix、每条边的输出误差/KL/mass、identity 与最佳匹配的代价、内部 query 划分及外部 val 指标。`cost[h][s]` 是目标组 h 读取 source 组 s 的误差；不是余弦相似度。最终只保留四条边的读取矩阵，不保留 16 套在线矩阵。

身份映射控制使用**完全相同的拟合流程**，只固定 π=[0,1,2,3]：

```bash
python scripts/compile_headalign_v22.py \
  --store-root "$PREFIX_STORE" --task trec \
  --teacher-fit "$TEACHER_FIT" --teacher-val "$TEACHER_VAL" \
  --student-fit "$TEACHER_FIT" --student-val "$TEACHER_VAL" \
  --source 18 --target 19 --mapping identity --steps 0 \
  --device cuda --output assets/v22_diag19_s0/trec.pt
```

初始原矩阵可通过 `--initial-reader /path/to/v21_refit19_s200/trec.pt` 加入对角边。使用它时 search/identity 两组必须同时使用；其所有训练/验证 query 会进入排除列表。不要把“更多离线计算”的收益误归因于映射：还应保留原 v21 reader 和等步数的身份映射重校准。

## 2. 只有映射筛查有信号时，再蒸馏读取器

在上面命令分别加：

```text
--steps 200 --eval-every 20 --lr 0.0001 --bf16-ste
```

A/B/bK/bV 被优化；主干完全冻结。π 是离线固定映射，不逐请求自适应。两组使用相同步数/损失/切分。赋值阶段最小化的是逐组 surrogate 的和，不代表 o_proj 后误差或真实分类必然改善。

默认局部门限保持 0.03 / 0.08 / 0.03，不提供放宽参数。所有输出仍为 diagnostic_only，不自动生成部署认证。矩阵 A/B 并未约束为可逆，也没有迁移 RoPE 位置。

## 3. 在线验证：KV 不重排，只重排短 Q 和 prefix 输出

新 bundle 有不同的 format；旧 v21 入口应拒绝，而不是静默忽略 head_maps。新入口复用 v21 的全部 CLI：

```bash
python scripts/run_sharedkv_v22.py --mode shared --backend vllm2 \
  --adapters assets/v22_map19_s200 --allow-candidate \
  --metadata-cache --packed-ops --layout hybrid \
  --stats results/v22_map19.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir /absolute/path/to/independent_development_bundle \
  --output results/v22_map19
```

上述命令需要你确实先生成 `v22_map19_s200/trec.pt`。目录名只表示输出组织，不表示已经通过验证。

同执行路径 identity：`--mode identity`，去掉 adapters/allow-candidate/packed-ops，保持 metadata-cache/hybrid。旧 v21 readers 也可以直接通过新入口运行（没有 head_maps 时等同恒等映射），用来检查桥接本身是否改变输出。

正式计时不用 `--logits/--trace-dir/--diagnostic/--nvtx/--selection-plan`；质量和固定计划诊断分开跑。保留原 original 和最强 hybrid identity，不能只对照更慢的旧 split。

在线 source bank、共享 alias、P8 配置、loader 调度以及 private DynamicCache 未改。只有 prefix Q 被按完整 GQA 组送至正确的 source head；prefix 输出/LSE 再返回目标顺序，然后应用本层 B/bias 与原生 tail 合并。不会重排长 K/V，也不会把 K/V heads 从 4 展开成 28。

每个 consumer 新增两份 int64 的 28-head 索引，共 448 bytes（身份映射也会准备索引，但执行时直接走旧 fast path）。另外有 Q、输出和 LSE 的临时 index_select；这些开销在 TTFT 内，**不能声称映射零成本**。没有实现这些 gather 的 Triton 融合。

## 4. 顺序组合：含头映射的 fingerprint 必须匹配

先验收单对，再按 consumer 递增加入。新上游模型要用 `run_sharedkv_v22.py` 重采实际 student：

```bash
python scripts/run_sharedkv_v22.py --mode shared --backend vllm2 \
  --adapters assets/previous_stage --allow-candidate --diagnostic \
  --trace-dir runtime_data/v22_student_fit --replay-dir "$TEACHER_FIT" \
  --metadata-cache --packed-ops --layout split -- \
  --task trec --budget 25 --period 8 --samples 64 --warmup 0 \
  --bundle-dir /absolute/path/to/fit_bundle --output results/v22_capture_fit
```

val 用对应独立目录再采一次。然后编译下一层时传入 `--carry assets/previous_stage/trec.pt` 及新 student-fit/student-val。每个对应 trace 必须含所需 target 的层数据。

新 fingerprint 包含所有非恒等 head maps。即使 A/B 完全相同，改变了上游 π 后也不能继续使用旧 student traces。普通旧 v21 refine 工具不理解头映射；应使用新 compile 工具。

## 5. 速度与字节：不要承诺本轮会降低 TTFT

本轮不再重写 selector 或缓存分数系统。主要目标是以相同 prefix KV / 矩阵参数量降低共享误差，让更多真正合格的层对有机会进入共享集合。即使映射成功，同一层对的 KV 字节仍与旧共享完全一样。

新增 resident 微基准只隔离映射开销：

```bash
python scripts/bench_headalign_v22.py --adapter assets/v22_map19_s200/trec.pt \
  --target 19 --backend vllm2 --prefix-tokens 1228 --query-tokens 41 \
  --iterations 100 --warmup 20 --output results/v22_map_overhead.json
```

原生路径与映射路径在非恒等 π 下计算的函数不同，这里只比较算子成本，不比较质量。真实质量要用 model runner。CUDA 不可用时脚本报错，绝不拿 CPU reference 时间冒充 GPU。

`budget_frontier_v22.py` 是物理预算**计算器**，不是执行器：

```bash
python scripts/budget_frontier_v22.py --consumers 1 2 3 6
```

它使用统一完整 16-token block 的保守模型，计算器的 77-block 单源与报告中含 12-token 尾块的 1228-token 单源不同。不能用其值替代实际逐请求 source 并集、加载计数、H2D 和峰值显存。

包内没有自动修改 keep_ratio 或 exact-layer budget，也没有实现 iso-byte 增加 token 的在线路径。后续做等字节质量曲线，需要显式为各预算重做校准/验证和原 loader 预算设置，不得改 bundle 的 keep_ratio 欺骗 guard。

## 6. 建议的停止条件

- 4×4 表基本对角最好：头错配假设不受支持，停止这一分支，不继续调 20 组参数。
- 匹配集下降而独立 val/完整模型不改善：视为过拟合或目标不匹配，不能只报告局部最低值。
- 质量改善但额外 gather 太贵：先报告同预算质量与真实数据量，不宣称 TTFT；下一步才考虑小型融合 kernel。
- 多层仍通不过：保留原生层，不强行加入 18→19。现在看到过的 175 条只能做回归/开发，新的最终结论应使用未参与设计的 prefix/query/task。

## 7. 文件

`matching.py`：映射/穷举/fingerprint；`calibrate.py`：16 条边、闭式校准与可选蒸馏；`ops.py`：读侧排列；`runtime.py`：v2 桥接；`budget.py`：预算算术；`tests/test_headalign_v22.py`：数学、合成闭环、错误输入与 CUDA 条件测试。

服务器原始 v2 数据不在本包内。本地测试不会被写成真实模型的质量/TTFT 结果，详见 TEST_RESULTS.md。
