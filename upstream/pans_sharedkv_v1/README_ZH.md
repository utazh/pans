# pans 读侧适配跨层 Prefix KV 共享：实验附加包 v1

**目标**：同一请求的多层 attention 读取同一份物理 prefix K/V；通过短 query 侧和聚合输出侧的离线校准矩阵补偿层间差异。降低的是重复恢复、gather、传输和工作集，不是跳过 attention/MLP。

**状态**：已做 CPU 数学与接口模拟测试；未在真实 Qwen2.5-7B、你的服务器、真实 Pcache/CUDA 环境执行。不是已经验证精度或 TTFT 的发布版本。测试通过不能替代模型验收。

目标原仓库：`utazh/pans`，commit `e32c4a5605ca55d2904b33065d67a1d71667d47c`。本包仅新增文件，不修改原选择器、Pcache 或 run_round4.py；启动时校验三个关键源文件的 Git blob SHA，避免对不匹配版本静默打补丁。请在独立实验分支使用。

## 1. 已实现内容

|文件|作用|
|---|---|
|`src/contiguous_fuxian/sharedkv/core.py`|只读共享 prefix + 私有 tail；共同 softmax；仿射偏移；GQA；流式参考 attention 和 FlashAttention-2 路径|
|`fit.py`|RoPE 同频率块结构拟合、query 度量的 reduced-rank ridge、attention-output 仿射回归；不用反向传播|
|`collect.py`|在原 runner 上捕获独立校准请求的所有 GQA 组 Q、私有分支统计和实际选择 IDs|
|`compile.py`|读已有 BF16 prefix store，拟合层对，独立局部验证，不合格层对保留原始 KV|
|`runtime.py`|只按物理 source 调度/resolve；共享缓冲区；保留原 greedy/标签续写评分；DynamicCache 只存私有 tail|
|`scripts/run_sharedkv.py`|复用原 `run_round4.py` 的入口；original/collect/identity/shared 四模式|
|`scripts/compile_sharedkv.py`|编译矩阵入口|
|`scripts/bench_sharedkv.py`|合成 GPU 传输+attention 微基准，**不等于 TTFT 或模型质量测试**|
|`tests/test_sharedkv.py`|数学、RoPE 配对、GQA、所有权、缓存 fork、编译链路等 CPU 测试；另有 GPU 测试|

`reference` 只用于正确性和离线校准，不得用来宣传速度。`flash2` 路径没有静默回退，要求 CUDA、FP16/BF16 和支持返回 LSE 的 FlashAttention-2。独立核对你的驱动/PyTorch/CUDA/flash-attn 编译组合，不要为了本包盲目更换原环境。

## 2. 安装附加文件

解压后，将以下新增文件复制到原仓库对应位置。不要把本 README 覆盖原 README。

```bash
cd /path/to/pans
# ADDON 指向解压后的 pans_sharedkv_v1 目录。
export ADDON=/path/to/pans_sharedkv_v1
cp -r "$ADDON/src/contiguous_fuxian/sharedkv" src/contiguous_fuxian/
cp "$ADDON/scripts/run_sharedkv.py" scripts/
cp "$ADDON/scripts/compile_sharedkv.py" scripts/
cp "$ADDON/scripts/bench_sharedkv.py" scripts/
cp "$ADDON/tests/test_sharedkv.py" tests/
PYTHONPATH=src python -m pytest tests/test_sharedkv.py -q
```

已有依赖仍由原仓库环境提供。本包 CPU 测试只需 torch、numpy、pytest；模型执行还需原环境的 transformers/FlexGen 依赖；性能路径另外需要 FlashAttention-2。

先运行原仓库的完整测试，再运行新测试。交付环境没有原仓库完整执行环境，因此**没有声称重新运行过原来的 307 项测试**。

## 3. 先验收 identity，不要先看共享速度

沿用原来模型、prefix store、Pcache、K4 index 的环境变量设置。对同一小规模测试集分别运行：

```bash
python scripts/run_sharedkv.py --mode original -- \
  --task trec --budget 25 --period 8 --samples 16 --warmup 0 \
  --output results/sharedkv_original_smoke

python scripts/run_sharedkv.py --mode identity --backend flash2 \
  --stats results/sharedkv_identity_smoke.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 16 --warmup 0 \
  --output results/sharedkv_identity_smoke
```

identity **不共享跨层 KV、不做矩阵变换**，但使用新增的两分支 attention、私有 tail 缓存与 source 调度器。因此它能暴露 mask、位置、精度、label continuation 等集成错误，也能测新增执行路径的成本。模型 logits 可能因 FP 舍入和 attention 后端不同而不逐位一致，必须先量化差异，确认没有显著劣化或明显位置错误，再运行 shared。

新测试中的 CUDA 两项必须实际通过，不能将 skipped 当通过。建议在 identity 下另外增加多 token 标签/生成检查。原 `run_round4.py` 固定 `max-tokens=1`，并做标签续写评分；长生成评测需通过原底层 runner 暴露的参数另建入口，本固定入口不会悄悄改变它。

原始基线和 identity 都要保留。不能只比 shared 与较慢的 identity，然后宣称比原框架快。

## 4. 校准数据必须与最终测试 query 隔离

准备两个与原 bundle 格式相同的开发数据目录，例如：

- `/data/cal_fit/trec.jsonl`：建议从 32–64 个独立 query 起步。
- `/data/cal_val/trec.jsonl`：建议另取 32–64 个独立 query。

以上是起步配置，不是已经验证的充分样本数。两组的 `prefix_text` 必须与已有 prefix store 完全一致，但 query 不能来自最终测试集。不要直接拿原 strict 测试集前几十条拟合，再用同一批条目宣称质量保持。

```bash
python scripts/run_sharedkv.py --mode collect \
  --trace-dir runtime_data/sharedkv_cal_fit --max-query-samples 8 -- \
  --task trec --budget 25 --period 8 --samples 64 --warmup 0 \
  --bundle-dir /data/cal_fit --output results/sharedkv_collect_fit

python scripts/run_sharedkv.py --mode collect \
  --trace-dir runtime_data/sharedkv_cal_val --max-query-samples 8 -- \
  --task trec --budget 25 --period 8 --samples 64 --warmup 0 \
  --bundle-dir /data/cal_val --output results/sharedkv_collect_val
```

collect 会额外投影、拷贝、计算参考 attention 并保存文件。**它的时间不能作为正式结果。**捕获所有 28 个 Q heads，而非只用 4 个 selector heads。每个请求最多采样 8 个 query 位置，覆盖首尾；这是离线校准节省成本的近似，也需要做样本数消融。

每个 trace 存 q、private conditional output、private log Z、选择 token IDs、模型/前缀身份。长 prefix KV 不重复转储，编译时读取既有 `layer_XX_key.bf16` / `layer_XX_value.bf16`。读入时模拟 BF16 store → FP16 payload → BF16 compute 的数值路径。

不同预算分别校准；本版要求 fit/val 的预算、period、前缀及模型身份一致。

## 5. 先验证两个完整矩阵的能力，再看结构化限制

先只试层对 2→3（0-based），确认读侧仿射表达能力。第一轮不要共享八层。

```bash
python scripts/compile_sharedkv.py \
  --store-root "$PANS_STORE_ROOT" --task trec \
  --fit-traces runtime_data/sharedkv_cal_fit \
  --validation-traces runtime_data/sharedkv_cal_val \
  --pairs 2:3 --key-mode dense --ridge 0.001 \
  --output assets/sharedkv_dense/trec.pt
```

再尝试结构化 K reader：

```bash
python scripts/compile_sharedkv.py \
  --store-root "$PANS_STORE_ROOT" --task trec \
  --fit-traces runtime_data/sharedkv_cal_fit \
  --validation-traces runtime_data/sharedkv_cal_val \
  --pairs 2:3,4:5,6:7,10:11,12:13,14:15,18:19,20:21,22:23 \
  --key-mode rope_rr --rank 16 --ridge 0.001 \
  --output assets/sharedkv_r16/trec.pt
```

支持 `direct`（直接共享）、`dense`（完整 A）、`rope`（仅同频率块）、`rope_rr`（同频率块 + reduced-rank residual）。同一命令的多个 pairs 必须互不重叠；v1 不做自动组搜索。反方向如 `3:2` 可作为独立实验，但不能同时与 `2:3` 构成环。

推荐 rank 对照：0/8/16/32/128，而不是只报最佳 rank。这里的 rank 约束作用于 K 的补偿残差；为避免多个小 kernel，本版在线导出/使用完整稠密 A 和 B。因此**较低 rank 在本版是拟合正则化，不等于更低的在线参数量或计算量**。

局部门限默认：整体 attention 输出相对 RMSE ≤ 3%、逐请求最大相对 RMSE ≤ 8%、prefix mass 绝对误差 P95 ≤ 0.03。它们只是排除明显失败的初始门限，不是任务精度保证。阈值必须在开发集确定，不能在最终测试结果上反复选择。

未通过局部门限的 consumer 不进入共享表；它仍加载自身 KV。输出会明确显示接受了几对。全部拒绝意味着当前配置**没有跨层共享收益**，不要隐藏这个结果。

## 6. 正式共享路径

```bash
python scripts/run_sharedkv.py --mode shared --backend flash2 \
  --adapters assets/sharedkv_r16 \
  --stats results/sharedkv_trec25.extra.jsonl -- \
  --task trec --budget 25 --period 8 --warmup 32 \
  --output results/sharedkv_trec25
```

运行前，reader 按 task 装入 GPU（单独记录准备时间；不算进与原仓库一致的 logits-ready TTFT）。模型/前缀 hash、RoPE、精度、period、预算不符则报错。测试 query 的 token hash 若出现在 fit/val 中，拒绝执行。

每个 P8 leader 使用原 selector 生成各层选择，随后只 schedule 唯一 source。不对消费者调用原 next-layer/period speculative prefetch。第一次读取 source 时做一次 FP16→BF16 和布局整理，后续层引用同一 tensor；当前 query 和后续 token 的 KV 逐层独立计算。

`P_index=8` 保持不变，`P_KV` 是独立的共享层对。leader 0/8/16/24 不能是 consumer，第一版保持其自身 KV 和原 K4 selector 语义。受上游 hidden-state 近似影响，shared 与 original 的 leader 选择仍可能不同；这里保证的是**同一方法内共享层对的 IDs 相同**，并没有宣称跨方法选择逐项一致。固定旧选择计划可用于诊断，但不能把离线选好计划的时间排除后当在线 TTFT 结果。

## 7. 统计与公平性

原 runner 的正式输出、TTFT 边界和标签评分被保留。新增 JSONL 提供 source 表、实际 source resolve 次数、逻辑/唯一所选 payload 字节、reader 和 request prefix buffer 字节。

**逻辑所选字节不是实际硬件 H2D 计数。**实际传输请结合原 loader 的 source/physical 计数及单独 Nsight 运行；计入 chunk 读放大、selector、metadata、reader 冷加载。CPU-hit 与 GPU-hit 分开报告，GPU 已有副本时本来就可能没有 H2D 可省。

矩阵不会自动从原 55 MiB payload cache 配额中扣除。公平资源约束需要报告总 CUDA peak/RSS，或另调 payload 容量使总预算相同；不能只固定旧 payload 配额而忽略新增常驻矩阵、转换和私有 tail。多个 task 的矩阵也要累计。

本版不删除原 SSD 的逐层 KV 文件，不重写持久化 Pcache 格式；目标是减少在线恢复与传输。不能把它报告成已经实现了 SSD 存储压缩。

正式实验使用 original、identity、shared 三组，相同 UID、暖机、GPU/CPU 环境和缓存条件，交替顺序至少重复 5 次；Profiler 与正式计时分开。若 shared 仅好于 identity 而不优于 original，则尚未实现你的 TTFT 目标。

质量验收至少包括相对当前 P8 基线和 Full-KV 参考的任务质量、逐请求 logits/标签排序、长生成检查；平均变化不显著不等于“无损”。预先定义非劣界限（如 1 个百分点，仅为候选）并看配对置信区间，样本不足则如实报告。

## 8. 合成性能预检（非必要）

```bash
python scripts/bench_sharedkv.py --prefix-tokens 8192 --query-tokens 64 \
  --iterations 30 --output results/sharedkv_synthetic.json
```

该测试只含已知大小 pinned CPU payload→GPU + 两分支 attention。它不含 SSD、host gather、selector、MLP 和真实层间差异，输出不能称作模型 TTFT。

## 9. 明确未实现/未验证的内容

真实 GPU/model 质量和性能验证尚未执行；跨请求统一 alias-aware 缓存策略、物理 source 的跨请求收益评分、独立 fused 三分支/多 source Triton kernel、在线自适应共享、不同选择集合的并集加载、跨 P8 窗口共享、动态 RoPE/YaRN、QK-norm/其他模型、批处理、端到端蒸馏和共享后回灌校准均不在 v1 中。

重点先验证：完整 A/B 是否能让至少一对真实层在独立 query 上保持质量。如果做不到，不能靠给矩阵换个结构名称解决。局部成功后再做全模型质量—传输—TTFT 的 Pareto 曲线。

理论推导与参考来源见 `docs/DESIGN_ZH.md`、`docs/SOURCES.md`。测试状态见 `TEST_RESULTS.md`。
