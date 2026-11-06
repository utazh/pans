# v30 任务状态（3090，Qwen2.5-7B）

范围：只补实验和服务器代码，不改论文。按作者要求，从 GPT 两份方案里挑与本轮直接相关、且有必要的部分执行。

## 一、本轮实验

| 编号 | 内容 | 状态 | 结果位置 |
|---|---|---|---|
| E-P4a | NS 在 P4 与 P8 下的对照：4 任务 × 预算 5/10/25/50，dev500 + 冻结测试集 | 运行中 | `studies/v30/quality/runs/ns_p4_{dev,test}_*`，汇总 `results/v30/p4_vs_p8.csv` |
| E-P4b | P4 下的 RAPS（25%）：P4 teacher → G16 几何图 → R4 训练（200 步）→ R0 → G20/G24；dev 选择用冻结规则 rev3，另跑测试集 | 运行中 | `studies/v30/p4_readers/*`，`results/v30/p4_selection.json` |
| E-EQ | NS 降预算到与共享读取量对齐的 14.2857/17.8571/21.4286%（= 25% × 16/20/24 ÷ 28），P8 与 P4 都跑，测试集 | 运行中 | `ns_eq_*`，`results/v30/quality_points.csv`、`frontier.csv` |
| E-CAKE / E-LAVa | 两个层级预算分配 + 淘汰基线，接入同一 SSD 恢复引擎；预算 10/14.29/17.86/21.43/25/50% | 排队 | `{cake,lava}_static_*`、`*_budget_*`、`*_oracle_*` |
| E-T | SSD 冷启动计时：每任务 25% 一格，11 个臂 × 6 块 × 16 请求，按块配对 | 排队（质量跑完后自动开始） | `studies/v30/timing`，`results/v30/timing.csv` |

**SST-2 的 P4 适配（如实记录）：**

- 现象：按冻结的 P4 几何规则，SST-2 最多只能有 7 个消费层，因此候选集从 G16/G20/G24 变为 **G21/G24**。
- 原因：SST-2 前缀共 239 个块，25% 预算下各层交替分到 60 块和 59 块；P4 的 4 层周期内，只有奇偶相同的层选中的块完全一致。
- 做法：只对 SST-2 启用"可行的最大共享"回退（`--max-feasible`），其余任务不受影响。回退的原因和最终候选集都记在 `readers_done.json` 中。

## 二、CAKE / LAVa 的接入方式（避免不公平比较）

- **依据：** CAKE 按官方代码 `antgroup/cakekv`（提交 c4675ce）实现，LAVa 按 arXiv 2509.09754 的式 5–7 实现。
  - 超参数：
    - CAKE：Qwen2.5-7B 用官方表中的 τ。每层平均保留不少于 362 个 token 时用 1024 档（1.5, 0.6），否则用 128 档（0.4, 0.6）；γ=200，观察窗口 32，平均池化核大小 5。
    - LAVa：最大池化核大小 7；GQA 下组内取最大值；层预算按归一化熵分配。
- **prefix 版（可部署，报告质量、字节和 TTFT）：** 观察窗口取 prefix 末尾 32 个 token，在注册 prefix 时就算好，与具体请求无关；每个 prefix、每个预算只有一份固定选块。
- **oracle 版（只作质量参照）：** 观察窗口取请求自己的后缀，并且要用到完整的 prefix key。这等于先看过全部缓存再决定读哪些，所以不报告 TTFT。
- **budget 版（消融）：** 只采用 CAKE/LAVa 的层预算，具体选哪些块仍由我们的在线 selector 决定。
- **粒度适配：** 引擎以"一层、全部 KV 头的 16-token 块"为读取单位，所以 token 分数先在各 KV 组上求和，再在块内求和。LAVa 原本的逐 head 动态预算因此变成逐层预算；prefix 版中观察窗口所在的块始终保留。
- **总预算：** 与 NS 相同，即 T = floor(ratio × 块数 × 28 + 0.5)；实际读取字节由引擎记录（`mean_physical_prefetch_kv_bytes`）。
- **TTFT 口径：** 固定选块复用引擎已有的 v24 replay 钩子。在线 selector 仍会运行并计入时间，只是被替换掉选出的 ID，所以这两个基线在 TTFT 上略吃亏。这一点在报告中注明。

## 三、代码改动（只新增文件，受 guard 保护的文件一个未改）

| 文件 | 作用 | 验证 |
|---|---|---|
| `scripts/v30_round4_ext.py` + `scripts/v30_entry.py` | 小数预算、按层预算配置、固定选块（按 prefix 或按请求）；只替换原链路最内层的 runpy 调用 | 16 条 TREC dev：预算 25% 时预测和标签对数概率与原链路逐条相同，字节相同；14.29% 时字节是 25% 的 0.570 倍 |
| `scripts/v30_common.py` | **严格配对**：UID 不唯一、条数不符、标签或查询不一致时直接报错；输出 harmed / rescued / disagree；任务可续跑，失败的产物改名保留，从不覆盖 | 冒烟测试 |
| `scripts/v30_kvscore.py` / `v30_select_build.py` | CAKE/LAVa 统计量和选块生成；固定选块文件附带源统计文件的 sha256 | 重算的第 0 层 key 与 prefix_store 完全一致（相对误差 0）；固定选块注入后，标签对数概率确实改变，字节与 NS 25% 相同 |
| `scripts/v30_p4_readers.py` | P4 reader（v29_subj_p4 流程参数化）；写入**输入签名**：plan、checkpoint、fit/val 数据和各 reader 的 sha256 | — |
| `scripts/v30_quality.py` / `v30_timing.py` / `v30_report.py` | 质量批量评估、冷启动计时（沿用 v29 计时库 rev3）、汇总报告 | — |

## 四、GPT 清单中本轮没有做的（及原因）

| 项目 | 处理 | 原因 |
|---|---|---|
| C01 Llama RoPE / 配置校验 | 不适用 | 只影响 Llama overlay，本轮只用 Qwen7B |
| C02 pipeline 签名 | 部分完成 | 本轮新产物都写入了签名；历史的 v29m_pipeline 在 A100 备份里，本轮未动 |
| C03 逐请求配对检查 | 完成（新脚本内） | — |
| C06 完整资源账本 | 未做 | 现有字节字段已足够支撑本轮 Pareto 对比；完整的 H2D/显存账本工程量大，收益有限 |
| C08 dev 提前停止 | 未启用 | 可选项，不影响结论 |
| E03 冻结后的新任务评估、E04 QuALITY 扩充、E08 多 seed、E09 自由生成 | 未做 | 作者本轮未要求，可后续再定 |
| M01 / M02 / M03 方法增强 | 未启用 | 方案本身规定默认关闭 |
