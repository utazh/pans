# 一手参考文献、代码及其与本方案的关系

查阅日期：2026-09-18。以下均是作者论文、作者仓库或官方实现。代码路径随主分支更新可能变化；安装请隔离环境，不要盲目覆盖 pans 的依赖版本。

## 当前用户仓库

- 仓库：https://github.com/utazh/pans
- 基准提交：`e32c4a5605ca55d2904b33065d67a1d71667d47c`
- 原入口：https://github.com/utazh/pans/blob/e32c4a5605ca55d2904b33065d67a1d71667d47c/scripts/run_round4.py
- attention/selector 路径：`src/contiguous_fuxian/flexgen_qwen_reprefill.py`
- 物理恢复/缓存：`src/contiguous_fuxian/flexgen_pcache.py`
- prefix BF16 原始文件：`src/contiguous_fuxian/sparse_qwen_reprefill.py`
- INT4 selector：`src/contiguous_fuxian/quantized_key_index.py`

## Qwen2.5-7B 与官方 attention 实现

- 模型 config：https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json
- 已核对实现：https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py
- 关注：Q/K/V bias、GQA 分组、split-half RoPE、o_proj、cache.update、decoder 残差顺序。
- 这不等于本包已在该 transformers 版本上实际运行；真实依赖由服务器验收。

## DeepSeek-V2 / MLA

- 论文：https://arxiv.org/html/2405.04434v5
- 关注：K/V 投影分别向 Q 和 attention 输出侧吸收；普通 RoPE 与投影吸收之间的不交换问题。
- 借鉴：读侧计算与位置约束。
- 区别：MLA 是模型架构；本方案是冻结已有模型、固定 prefix 的跨层近似共享。
- 矩阵移项本身不是本方案的原创贡献。

## ReCalKV

- 论文：https://arxiv.org/abs/2505.24357
- 详细版本：https://arxiv.org/html/2505.24357v3
- 作者代码：https://github.com/XIANGLONGYAN/ReCalKV
- 重点路径：`palu/decomposition.py` 的 `compress_model_ours`；`palu/model/modules/svd_linear.py` 的 `from_linear_adasvd`、`from_linear_whiten_reorder`；`kernel/`。
- 借鉴：K/V 非对称处理、离线 Value 校准、输出侧矩阵融合。
- 区别：不能直接把其低秩压缩/输出融合代码复制到“共享 prefix + 原生 private tail”的双分支中；只有 prefix 分支可以应用本方案的 B。

## xKV / xKV-SR

- 论文：https://arxiv.org/abs/2503.18893
- 详细版本：https://arxiv.org/html/2503.18893v2
- 作者代码：https://github.com/abdelfattah-lab/xKV
- 重点入口：`evaluate/eval_acc.py`；`examples/xKV/`；`examples/xKV_SR/`。
- 借鉴：跨层共享子空间、Selective Reconstruction、质量与实际效率共同评价。
- 必须作为强相关工作/对照，而非只对比直接共享。
- 区别：本方案直接读取源层已存在的物理 KV，在线不重建消费者 prefix KV。但 xKV 使用的跨层共享子空间与矩阵分解思路存在实质关联，不能仅凭是否重建就宣称完全不同。

## SVD-LLM

- 论文：https://arxiv.org/abs/2403.07378
- 作者代码：https://github.com/AIoT-MLSys-Lab/SVD-LLM
- 借鉴：激活分布/白化感知的截断，而不是直接对权重矩阵做无度量 SVD。
- 区别：这里的 X 是 prefix source key，右度量来自本层 Q；具体 reduced-rank 回归和前缀归一化是本包所明确写出的目标，不能声称原论文已经验证该目标的效果。

## KVSharer

- 论文：https://arxiv.org/abs/2410.18517
- 作者代码：https://github.com/yangyifei729/KVSharer
- 重点：`llama_real_share/cache_utils.py`、`internlm2_real_share/cache_utils.py` 的共享缓存实现和示例 notebook。
- 借鉴：真实物理 KV 跨层共享的 cache 组织；不要把复用 ID 当作复用内容。
- 区别：本方案加入校准读取器，且针对已经缓存的固定 prefix；是否值得增加这些组件必须以同预算对照证明。

## Eigen Attention

- 论文：https://aclanthology.org/2024.findings-emnlp.899/
- arXiv：https://arxiv.org/abs/2408.05646
- 作者仓库：https://github.com/UtkarshSaxena1/EigenAttn
- 借鉴：在 attention 低秩空间计算而非无条件恢复所有高维 KV。
- 本次查阅该仓库 README 仍标注代码待添加，不能当作本包可直接依赖的完整实现。

## FlashAttention-2

- 官方代码：https://github.com/Dao-AILab/flash-attention
- 已核对文件：https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py
- 已读文件 blob：`1edab572c1f81ae6fd13bf1c43cf9e022c531adc`
- 本包使用公共 `flash_attn_func` 返回 out/LSE；dropout=0 时，上游 forward 将实际 attention mask 输出开关关闭。
- 本包显式检查返回 mask 为空与 LSE 形状；接口不符合即报错，不在计时中静默切换到参考实现。
- GPU kernel 数值与性能尚需服务器实测。

## 引用与发布说明

本包新增实现为本次任务编写，未复制上述仓库源文件到交付包。现有 pans 与其 vendor 的许可和出处不变；本包不为整个用户仓库重新授予许可证。论文采用上述方法思想时应引用原作者，而不是把所有组成部分都称为本方案首创。
