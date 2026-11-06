# SharedKV v2.5：扩大主KV共享范围 + 有界的原生prefix补丁

基线：`utazh/pans` 的 `v2.4` 提交 `a6be130c921024a4cd358ccf5841e4a46f547ae2`。
本包是研究代码，没有真实Qwen/GPU收益结果，不自动认证任何checkpoint。

## 1. 本轮要验证什么

不再把G26的统计非劣失败作为禁止研究G24/G22的前提。原v2.4验收结果不变；另建有限的预算探索，所有未认证结果保持candidate。

同时比较两个结构：

- `R=0`：原来每个consumer的整个已选prefix均从source读取。
- `R=4`：每个consumer保留**原选择集合最后4个物理块**的本层原生K/V，剩余部分从同一个source读取并适配。

不是把任意未选中的最近64个token额外加入attention。仍然是同一个25%已选支持集，不增删token，不改精度、P8或gold标签。
这不是DeepSeek的SWA或Bounded Replay实现：这里是从已有SSD KV中读取少量本层数据，不重新计算prefix。

目的：检验付出少量原生prefix字节，是否能让更多层共享其余大部分KV。不能预先假定局部窗口一定解决Qwen的跨层误差。

## 2. 执行顺序

先用v24的原始teacher记录生成G26/G24/G22/G20/G16/G14研究计划。
先运行G24、G22的R=0与R=4，同seed24、同初值、同训练设置；可并行作为独立研究候选，不需要父模型已经获得非劣认证。
只在fit/val/探索dev上筛查。不可用已观察的final768挑选并再宣称其为新最终集。
优先给少数有希望的组合增加24/25/26三个种子和独立最终质量验证；预注册最终非劣标准，不改旧门限，不把探索通过当认证。

预算说明：G只数“保存完整已选prefix的主来源”，不是所有本层KV对象数。R>0时consumer仍有少量原生补丁。实际加载次数可能回到28；加载字节应显著小于28份完整prefix。不得只报告旧统计中的G或unique_selected_bytes。

## 3. 安装

```bash
git fetch origin v2.4
git worktree add ../pans-v25 a6be130c921024a4cd358ccf5841e4a46f547ae2
python /path/to/pans_sharedkv_v25/scripts/install_sharedkv_v25.py --repo ../pans-v25
cd ../pans-v25
PYTHONPATH=src python -m pytest -q tests/test_sharedkv_v25.py
```

模型、SSD Pcache、prefix store、K4 index和数据必须沿用已有可复现环境。安装器仅新增文件，拒绝覆盖，核对关键接口hash。整包不包含模型、数据、teacher记录、reader二进制或原始GitHub仓库。

## 4. 计划与训练

```bash
python scripts/plan_exploration_v25.py \
  --teacher-fit /path/to/v24_teacher_fit --task trec \
  --protect-first 0 --budgets 26,24,22,20,16,14 \
  --native-blocks 0,4 --seeds 24 \
  --output-dir configs/v25_explore
```

输出study.json和标准v24拓扑JSON。拓扑只使用fit的几何与IDs，不读取开发/测试准确率。未合法产生的预算会标注unavailable，不补造。

```bash
python scripts/train_native_prefix_v25.py --native-blocks 0 -- \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --store-root /path/to/prefix_store --task trec \
  --teacher-fit /path/to/v24_teacher_fit \
  --teacher-val /path/to/v24_teacher_val \
  --plan configs/v25_explore/g24.json \
  --initial /path/to/frozen_v24_seed24/trec.pt \
  --steps 200 --eval-every 25 --seed 24 --device cuda \
  --output assets/v25_g24_r0_s24/trec.pt

python scripts/train_native_prefix_v25.py --native-blocks 4 -- \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --store-root /path/to/prefix_store --task trec \
  --teacher-fit /path/to/v24_teacher_fit \
  --teacher-val /path/to/v24_teacher_val \
  --plan configs/v25_explore/g24.json \
  --initial /path/to/frozen_v24_seed24/trec.pt \
  --steps 200 --eval-every 25 --seed 24 --device cuda \
  --output assets/v25_g24_r4_s24/trec.pt
```

两个R使用同一个initializer；它必须与当前fit/val角色及模型/prefix/budget相容。初值不是直接拼接上线：所有启用的reader在完整student上继续共同优化。
训练仍用v24的候选得分损失、分支流式反向和identity回放门限，没有新增loss项。本包保留服务器修复后的FA2前向及LSE反向，不能退回早期参考重放。

输出新格式 `pans-shared-prefix-native-suffix-v25`。旧运行器会拒绝而非静默忽略补丁。所有输出诊断标志保持True。

## 5. 普通运行和冷恢复

使用v25入口，即便R=0也便于同运行框架比较。**有三次`--`**：v25选项 / v23选项 / v21选项 / round4选项。

```bash
python scripts/run_native_prefix_v25.py -- \
  --pcache-root /path/to/private_pcache -- \
  --mode shared --backend vllm2 \
  --adapters assets/v25_g24_r4_s24 --allow-candidate \
  --metadata-cache --packed-ops --layout hybrid \
  --stats results/v25_g24_r4.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir /path/to/exploration_dev \
  --output results/v25_g24_r4
```

无共享对照使用`--mode identity`，移除adapters/allow-candidate/packed-ops，其他相同。R=0是同一新wrapper中的純共享对照。另保留旧v24 identity作为桥接检查。

SSD-cold/preserve：

```bash
python scripts/run_native_prefix_v25.py -- \
  --cold ssd-cold --read-advice preserve \
  --owned-cache-root /path/to/private_pcache \
  --pcache-root /path/to/private_pcache \
  --cold-log results/v25_cold_g24_r4.r0.jsonl -- \
  --mode shared --backend vllm2 \
  --adapters assets/v25_g24_r4_s24 --allow-candidate \
  --metadata-cache --packed-ops --layout hybrid \
  --stats results/v25_cold_g24_r4.r0.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 32 --warmup 8 \
  --bundle-dir /path/to/exploration_dev \
  --output results/v25_cold_g24_r4_r0
```

`native`另做一组同策略配对。冷认证、预热和计时边界直接委托v23，未更改零页条件。每个arm使用独立进程，避免已知mmap残留。正式计时不加trace/logits/NVTX；profiler另跑。R=4补丁文件同属目标prefix，包含在冷准备范围中，不提前偷偷加载补丁。

## 6. 新的资源账本

原v1/v21 `.extra.jsonl`只知道完整source bank，会漏算补丁，因此必须合并读取：

- `*.extra.jsonl`：旧source/reader信息；
- `*.extra.jsonl.v21.jsonl`：packed、selector和元数据缓存；
- `*.extra.jsonl.v25.jsonl`：native补丁字节、实际consumer repair次数、cut位置、完整/补丁存储地址、全部所选payload。

`total_resolve_calls = full_source_resolve_calls + native_repair_resolve_calls`。
`total_selected_payload_bytes = shared_base_payload_bytes + native_repair_payload_bytes`。
矩阵、packed额外副本、工作区、索引和整个模型内存仍需另计。不把selected bytes当硬件H2D。没有删除SSD副本，不声称磁盘压缩已实现。

每个consumer在cache admission中只晋升它的补丁IDs，禁止把完整逻辑选择重新晋升；全局精确块预算和selector原始IDs保持不变。source仍由原路径评分，不新增“按消费者数加权”的缓存策略。

## 7. 扩到Qwen2.5-14B

```bash
python scripts/inspect_model_v25.py --model-path /path/to/Qwen2.5-14B-Instruct
```

检查输出应是48层/40Q/8KV/d128；代表Q heads为0,5,10,15,20,25,30,35。
14B仍需重建prefix store、物理Pcache、8-head K4 index，重新采集teacher、设计拓扑和训练reader。7B矩阵和缓存不允许复用。

新入口可显式覆盖原runner里的资源参数：

```bash
python scripts/run_native_prefix_v25.py \
  --model-path /path/to/Qwen2.5-14B-Instruct \
  --store-root /path/to/prefix14 \
  --selector-index-dir /path/to/k4_14 \
  --auto-probe-heads -- \
  --pcache-root /path/to/pcache14 -- \
  --mode identity --backend vllm2 --metadata-cache --layout hybrid -- \
  --task trec --budget 25 --period 8 --samples 8 --warmup 8 \
  --bundle-dir /path/to/bundle14 --output results/identity14_smoke
```

这不是完整的14B验收。现有训练/运行是单GPU `.to(device)`，不支持TP或device_map分片。14.7B BF16权重约29.4GB（27.4GiB），单24GiB显卡不够；激活、KV、reader另计。用容量足够的单GPU先验收，不能只加`device_map=auto`便宣称多卡已适配。
模型类型/QK norm/RoPE缩放/滑窗或kernel API变化需要单独适配和回放测试。

## 8. 必做服务器验收

R=0对旧v24使用相同reader与IDs，先测logits/完整候选分数一致；然后R=4的线上结果与新训练重放一致。新增CPU桥接测试不能取代真实FA2模型校验。
验证prefix远端是source视图、补丁是consumer自身数据、二者位置无交集且覆盖原IDs、私有cache长度不含补丁、标签fork不交叉污染。
在7B上完成后，再运行原有全部CPU/CUDA测试以及14B原生回放。无法通过时停止并保留日志，不绕过hash/数值/冷状态检查。

## 9. 研究结论边界

已知证据只来自v2.4报告；本包无真实Qwen新结果。默认4块是小范围起点，不是验证过的最优值。它不是绝对位置SWA、不是无训练、不是精确复原所有原层KV。
若R=4未改善同预算的质量，保留负结果，回到R=0的预登记多预算曲线，不继续无休止调局部补丁。
旧768非劣失败保留。新研究扫描不会自动把它改成通过；独立最终评估与多次选择问题需重新制定协议。
