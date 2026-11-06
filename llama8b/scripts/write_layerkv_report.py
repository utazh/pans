#!/usr/bin/env python3
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
rows=json.loads((R/'results/comparison.json').read_text())
by={x['variant']:x for x in rows}
table=['|方案|请求数|准确率 %|TTFT ms|原路径 logits KL|实际预取 payload MB|物理 resolve 次数|',
       '|---|---:|---:|---:|---:|---:|---:|']
for name in ['fullkv','original','identity','original_context','direct1','dense1','dense3','dense7','dense1_repeat','original_repeat','identity_fixed_v2','dense1_fixed_v2']:
 if name not in by:continue
 x=by[name];kl=x.get('mean_logits_kl')
 table.append(f"|{name}|{x['n']}|{100*x['accuracy']:.2f}|{x['mean_ttft_ms']:.2f}|{kl:.6f}" if kl is not None else f"|{name}|{x['n']}|{100*x['accuracy']:.2f}|{x['mean_ttft_ms']:.2f}|—")
 table[-1]+=f"|{x['mean_physical_prefetch_bytes']/1e6:.2f}|{x.get('physical_resolve_calls','—')}|"
local=['|K reader|局部输出相对 RMSE|prefix mass P95 误差|','|---|---:|---:|']
for title,path in [('直接共享','layerkv_direct_strict'),('稠密 A/B','layerkv_strict_forward'),('RoPE 块','layerkv_rope_strict'),('RoPE + rank16','layerkv_rope_rr_strict')]:
 b=json.loads((R/f'assets/{path}/trec.report.json').read_text());v=b['reports']['27']
 local.append(f"|{title}|{v['output_relative_rmse']:.6f}|{v['prefix_mass_p95_absolute_error']:.6f}|")
split=json.loads((R/'results/split_validation.json').read_text())
integrity=json.loads((R/'audit/source_integrity.json').read_text())
dense=by['dense1'];direct=by['direct1'];original=by['original'];identity=by['identity']
payload_saved=1-dense['mean_unique_selected_bytes']/identity['mean_unique_selected_bytes']
kl_reduction=1-dense['mean_logits_kl']/direct['mean_logits_kl']
ci=dense['paired_bootstrap95_delta']
text=f"""# 跨层共享 prefix KV 实现与服务器实测

功能已实现，单层对可以显著降低直接共享造成的输出分布偏移；扩大到 3 或 7 个层对后，本次分类质量明显下降。所有 14 个稠密候选方向均未通过原型默认的严格局部门限。固定选择集合的对照中，共享版慢于 identity，尚未证明共享本身带来稳定延迟收益。当前交付为可运行、可验证的研究实现，不能把这些诊断配置视为已验收的部署配置。

## 实验设置

- 路径：/home/panzihang/src/prism_layerkv；原目录 /home/panzihang/src/prism_pans 保留。
- Qwen2.5-7B-Instruct，28 层、28 Q heads、4 KV heads、head_dim=128；FP16 payload、BF16 模型与在线 readers。
- PyTorch 2.10.0+cu130、Transformers 4.57.6，服务器现有 vLLM FlashAttention-2，GPU 2 / RTX 3090。
- TREC 固定 prefix 长 4940 tokens，P8 + K4 selector，25% 精确全局块预算；Full-KV 对照用 100%。
- 拟合 32 条、验证 32 条、测试 128 条；seed=20260918。来自现有 strict 数据池，不是全新外部 holdout。实际 query hash 两两无交集：{split['pairwise_disjoint']}。
- 测试请求未用于拟合或层对筛选。拟合使用无标签中间态、闭式回归和 SVD，主干参数不更新。
- 主要质量组预热 8 条、评测相同 128 条；固定集合组 32 条，连续生成组 8 条，人工长 query 组 4 条且不预热。使用完整标签续写对数似然打分；TTFT 只计 logits-ready。
- 批量实验共享已加载的冻结模型，模型文件 stat 未变时复用已计算的内容指纹；每组新建、关闭 Pcache store。模型加载、矩阵加载/预热及校准不计入 TTFT。
- GPU 与服务器有其他负载。本次仅两次单层对与原路径的重复运行，不能据此宣称独占环境稳定加速或统计非劣。

## 核心实现

当前层正常生成自己的 Q/K/V。历史 prefix 读取时，在 post-RoPE Q 上应用 A 的转置；源 K/V 在请求级 bank 中仅加载和转换一次。prefix 条件输出经 B 和 bv 变换，bk 对 prefix logsumexp 的贡献显式保留，再与原生私有 tail attention 共同归一化。DynamicCache 中只保存 query/生成 token 的本层 K/V，标签续写 fork 不复制 prefix。

新增 vllm2 后端不展开 GQA 的 KV heads，不重建 consumer prefix KV。每个 consumer 的 BF16 A/B/bias 占 264192 bytes。在线仍为稠密矩阵，rank 约束只用于拟合。

原 runner 和 Pcache 的核心源文件没有改写。新增入口支持独立数据集、连续生成、固定选择计划、全词表 logits 保存、诊断导出。原型的 warmup=0 参数冲突已修复；接口 hash guard 针对实际复制的工作树重新记录。

## 独立测试结果

{chr(10).join(table)}

original_context / original_repeat 是同一评测批次前后的原路径对照；dense1_repeat 为单层对重复。identity 使用新 attention/调度路径但不跨层共享。直接共享使用同一层对及单位 A/B，以控制读取路径。

identity_fixed_v2 与 dense1_fixed_v2 使用 original 保存的相同选择集合，32 条请求，不能直接与 128 条均值作成对比较。这两组为最终有效结果；旧 fixed 目录仅保留为引用循环修复前的调试记录。共享版 288.61 ms 慢于 identity 的 245.35 ms，因此本次没有证实稳定的独立共享加速。详细逐请求记录、P95、配对 bootstrap 区间和来源字节在 results/comparison.json、comparison.csv。

单层对选择为 **0-based 26 → 27**：
- 平均 logits KL：直接共享 {direct['mean_logits_kl']:.6f}，补偿共享 {dense['mean_logits_kl']:.6f}，降低 {100*kl_reduction:.2f}%。
- 准确率：原路径 {100*original['accuracy']:.2f}%，直接共享 {100*direct['accuracy']:.2f}%，补偿共享 {100*dense['accuracy']:.2f}%。补偿并未在这批分类准确率上超过直接共享；它的明确作用是更接近原模型分布。
- 补偿共享相对 original 准确率差的配对 bootstrap 95% 区间为 [{100*ci[0]:.2f}, {100*ci[1]:.2f}] 个百分点。样本量不足以认证 0.5 或 1 个百分点的非劣界限。
- 唯一加载次数 28 → 27，所选物理 payload 降低 {100*payload_saved:.2f}%。来源与消费者 data_ptr 一致，未重建长 prefix。
- 单层对 TTFT 相比 original 有下降，但 identity 自身也更快，因此不能将整个下降归因于 KV 共享或 reader。固定集合对照与重复运行见表。
- 3 个层对为 26→27、3→2、18→19；7 个层对再增加 22→23、15→14、10→11、6→7。层对依据验证集局部误差选择，不依据测试准确率。

Full-KV 在本批数据上的分类准确率低于当前稀疏原路径，这是实测结果，不能预设 Full-KV 为这项分类协议的准确率上界。

## 校准与结构消融

对同一 26→27 层对，32 条独立验证请求的 teacher-state 局部指标：

{chr(10).join(local)}

默认接受门限是整体输出相对 RMSE <= 0.03、单请求最大 <= 0.08、prefix mass P95 误差 <= 0.03。14 个稠密方向通过数为 0，严格 bundle 自动保留原层 KV。诊断 bundle 必须显式 --allow-diagnostic，失败门限没有为了跑通而放宽。

局部改善不能代替共享后的端到端评测。纯 RoPE 块表达能力不足；rank16 残差接近稠密结果，但并未达到严格门限。

## 物理与计时口径

额外 stats 记录每层 prefix 存储地址、实际 source resolve 次数、唯一所选 payload、reader bytes、请求级 prefix buffer 和私有 tail 长度。结果中的 physical_alias_verified / private_cache_bounded 应全部为 true。

物理预取 payload 包括 GPU/CPU/SSD 来源，**不是 PCIe 总传输字节**。comparison.json 另给出 loader 来源计数推导的 host payload bytes；它也不是 Nsight 的硬件 H2D 认证。没有删除或压缩逐层 SSD 文件，不能宣称磁盘存储减少。

新路径禁用了原路径的推测 next-layer 预取。原路径相对 identity 的差异还包含后端、mask、cache 组织和调度成本。跨请求的 cache score 没有扩展为按逻辑消费者加权的 alias-aware 策略。不同选择集合的 union 加载、batch>1、动态 RoPE 和融合 reader kernel 均未实现。

## 正确性、连续生成与长 query

最终完整测试为 342 passed、2 skipped、87 subtests passed；2 项跳过的是需要独立 flash_attn 包的测试，vLLM FA2 GPU 测试已通过。记录见 logs/final_tests.log，包括数学恒等式、共同归一化、RoPE、GQA、source 去重、真实 Qwen prefill/decode、私有 cache 及诊断 gate。

8 条最多 8-token 连续生成的原路径/共享输出文本全部一致，私有 cache 长度检查通过；逐条文本见 results/generation_examples.json。长 query 压力检查使用 4 条测试请求加中性 padding，检查位置与 cache 生命周期，不作为自然任务质量基准；query 长度 687–696 tokens，4 条中 3 条生成文本完全相同，平均 logits KL 0.01666，私有 cache 长度检查通过。结果见 results/long_query_stress.json。

## 文件完整性和复现

源代码/配置/脚本/vendor 检查 {integrity['checked_files']} 个文件，原目录内容哈希均未变化：{integrity['unchanged']}。实现补丁见 audit/implementation.patch；源仓库 HEAD=61318220e7df80488292b54b00a87134a031f169，同时保留其未提交修改。

全部实验与结果在服务器。README_LAYERKV.md 提供单次命令，configs/layerkv_baselines.json 与 configs/layerkv_experiments.json 记录本次全部参数，scripts/analyze_layerkv.py 可重建对比与完整性检查。
"""
(R/'EXPERIMENT_REPORT_ZH.md').write_text(text)
print('Wrote',R/'EXPERIMENT_REPORT_ZH.md')
