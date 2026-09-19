# Prism LayerKV：服务器独立实验目录

位置：`/home/panzihang/src/prism_layerkv`。从服务器当前 `prism_pans` 工作树独立复制，叠加用户提供的 `pans_sharedkv_v1.zip`。没有更新 Qwen 主干参数。

## 运行语义

- 每层正常计算自己的 Q、K、V，并按原位置应用 RoPE。
- 历史 prefix 使用请求级、只读的物理 source K/V；consumer 不加载或重建自己的 prefix K/V。
- consumer 用 `Q @ A.T` 读取 source；将 prefix 条件输出做 `U @ B + bv`。
- key 截距 `Q @ bk / sqrt(d)` 加回 prefix logsumexp。prefix 与私有 tail 通过 logsumexp 共同归一化。
- 当前 query 与后续生成 token 的 K/V 保存在每层独立 DynamicCache，prefix 不进入该 cache。
- P8 selector leaders 保持原样。共享限于同一 P8 窗口、相同选择集合，物理源可早于或晚于 consumer。
- 没有压缩或删除原始磁盘 KV 文件；目前验证在线加载、驻留和读取行为。

## 与原型相比的服务器适配

服务器没有独立 `flash_attn` 包，默认使用已安装 vLLM 的 FlashAttention-2 varlen 接口（`--backend vllm2`），返回输出和 LSE，保持 GQA 的物理 KV head 数，不 repeat prefix KV。新增真实 Qwen 小模型 prefill/decode、GPU GQA/LSE 和拒绝未通过门限 bundle 的测试。

入口增加独立 bundle、max-tokens、全词表 logits 保存、固定选择计划和诊断导出。修复原入口 `--warmup 0` 与正数 warmup-sample 参数的冲突。原核心 runner、Pcache、selector 逻辑保留复制时内容。

`sharedkv/io.py` 中继承的 `BASE_COMMIT=e32c4a...` 是原型兼容标识，不是此次服务器工作树提交。服务器源仓库 HEAD 为 `61318220e7df80488292b54b00a87134a031f169`，且含用户现有未提交修改。实际复制内容见 `docs/layerkv/audit/source_before.json` 和接口 guard `docs/layerkv/audit/integration_hashes.json`。

## 快速复现

所有命令在本目录运行，输出目录必须尚不存在。

```bash
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/home/panzihang/venvs/vllm-stable/bin/python

# 原始执行路径
$PY scripts/run_sharedkv.py --mode original -- \
  --task trec --budget 25 --period 8 --samples 128 --warmup 8 \
  --bundle-dir data/sharedkv_splits/test --output results/reproduce_original

# 新的两分支路径，不跨层共享
$PY scripts/run_sharedkv.py --mode identity --backend vllm2 \
  --stats results/reproduce_identity.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 128 --warmup 8 \
  --bundle-dir data/sharedkv_splits/test --output results/reproduce_identity

# 第 26 层作为第 27 层的物理源（0-based）
# 该 bundle 未通过严格局部门限，只能显式用于诊断；不要当成已验收的部署配置。
$PY scripts/run_sharedkv.py --mode shared --backend vllm2 \
  --adapters assets/layerkv_dense1 --allow-diagnostic \
  --stats results/reproduce_dense1.extra.jsonl -- \
  --task trec --budget 25 --period 8 --samples 128 --warmup 8 \
  --bundle-dir data/sharedkv_splits/test --output results/reproduce_dense1
```

将 `--max-tokens 8` 放在 `--` 后可检查连续生成。将 `--selection-plan results/original_logits` 放在 `--` 前可使用已记录的固定选择集合。

## 校准与门限

拟合、验证、测试以请求为单位随机划分，seed=20260918：32/32/128。UID 清单在 `data/sharedkv_splits/manifest.json`，实际 query hash 交集检查在 `docs/layerkv/results/split_validation.json`。这些请求来自现有 strict 数据池，并非新外部测试集。

```bash
$PY scripts/compile_sharedkv.py \
  --store-root runtime_data/prefix_store --task trec \
  --fit-traces runtime_data/cal_fit --validation-traces runtime_data/cal_val \
  --pairs 26:27 --key-mode dense \
  --output assets/reproduce_strict/trec.pt \
  --diagnostic-output assets/reproduce_diagnostic/trec.pt
```

默认局部门限为整体输出相对 RMSE <= 0.03、单请求最大 <= 0.08、prefix mass 绝对误差 P95 <= 0.03。不合格层对从严格 bundle 中排除。诊断 bundle 保留不合格候选，运行时必须显式 `--allow-diagnostic`。

支持 `dense / rope / rope_rr / direct`。rank 约束用于离线拟合；在线导出仍为稠密 A/B，不能声称已有低秩算子加速。

## 文件

- `src/contiguous_fuxian/sharedkv/`：数学内核、采集、拟合、编译、运行时。
- `scripts/run_sharedkv.py`：单次实验入口。
- `scripts/run_layerkv_suite.py`：多组顺序运行，复用冻结模型及已验证指纹，每组重新创建并关闭 Pcache store；模型文件 stat 改变时重新加载/验证。
- `configs/layerkv_baselines.json`、`configs/layerkv_experiments.json`：本次完整参数。
- `scripts/analyze_layerkv.py`：生成对比 CSV/JSON、泄漏检查、源文件完整性和补丁。
- `EXPERIMENT_REPORT_ZH.md`：完成后的效果报告。
- `upstream/pans_sharedkv_v1/`：上传原型的原始解压副本，便于检查改动。

## 结果口径

TTFT 是原 runner 的 logits-ready 边界，不含模型加载、离线校准和标签续写评分。额外 stats 位于每次请求计时之后。物理 selected bytes、loader physical prefetch bytes、host 来源字节和硬件 PCIe/H2D 字节不是同一个量；本次未使用 Nsight 对 PCIe 总流量作认证。

保留 original/identity/shared 对照。新路径关闭了原路径的推测 next-layer 预取，identity 的调度/两分支计算差异可能单独影响时间。共享收益应同时对照 identity 和 original。服务器并非独占，延迟仅描述本次共享负载下的运行，不能推断独占硬件稳定收益。

最终固定集合对照使用 configs/layerkv_fixed_verification.json，结果目录为 identity_fixed_v2 / dense1_fixed_v2；旧 fixed 结果仅作引用循环修复前的调试记录。完整效果见 EXPERIMENT_REPORT_ZH.md。

## GitHub v1 发布范围

该分支基于 GitHub main 的 e32c4a5605ca55d2904b33065d67a1d71667d47c，合入服务器最新 LayerKV 实现；保留 main 已有的可配置路径及隔离测试。run_round4.py 合入 max-tokens、100% 预算和零预热修复，不退回服务器硬编码路径。

本仓库附带代码、配置、测试、上传原型快照和小型结果摘要。模型权重、数据集、prefix store、Pcache、selector index、校准 traces、adapter 二进制和原始运行日志不随代码发布，需要按主 README 准备，并在目标环境重新校准。上文命令描述原服务器环境；其他环境请替换 Python 路径，并通过 PANS_MODEL_PATH/PANS_BUNDLE_DIR/PANS_STORE_ROOT/PANS_PCACHE_DIR/PANS_SELECTOR_INDEX_DIR 或 run_round4 参数指定资源。套件 JSON 保留原实验参数。

共享模块、新增 Python 入口、配置和测试均复制自服务器最新代码。原服务器的一次性 stage/finalize shell 脚本依赖绝对路径或历史 PID，不作为通用运行入口发布。upstream 是用户提供原型的原始快照，不代表最终服务器实现或当前验证结论。docs/layerkv/audit 中哈希记录属于原服务器实验快照；该发布分支的文件以 Git 提交为准。

历史 GPU 测试记录见 docs/layerkv/validation/server_gpu_tests.txt；发布副本的 CPU 检查另存于同目录。全部发布整理与检查在服务器独立 publication 目录完成，原实验目录代码未改写。
