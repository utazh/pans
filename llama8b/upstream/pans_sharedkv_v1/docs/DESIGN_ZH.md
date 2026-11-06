# 结构化读侧适配：设计、数学与研究边界

本文是待实测的研究方案，不是已经得到质量和性能结论的论文方法。新增代码对 `utazh/pans` 的固定提交提供实验入口；真实 GPU 和模型验收步骤见 README。

## 1. 研究对象与目标

对象是已经完成全层计算、位置和内容固定、会重复被查询的 prefix。当前请求仅执行 query 的 prefill。拟减少的是逐层恢复 prefix KV 的重复物理数据，而不是重新计算 prefix 的 FLOPs，更不是跳过 query 的模型层。

将当前层 l 的 prefix KV 表示成共享源 a 的近似仿射读出：

```
K_hat_l = K_a A_l + 1 bK_l
V_hat_l = V_a B_l + 1 bV_l
```

代码不在线构造 K_hat_l/V_hat_l。每个物理 KV head 有一组 A、B、bK、bV；Qwen2.5-7B 的同一 GQA 组中 7 个 Q heads 使用同一 reader。不同层的 reader 不同，attention 权重/输出也不共享。模型主干、Q/K/V 投影、o_proj、MLP 和规范化参数均不更新。

本版 reader 针对 prefix、模型、预算和位置配置校准。不能把它称为适用于任意新前缀/任意位置的通用模型转换。

## 2. 为什么采用非对称的两个 reader

K 决定打分及 prefix 与当前输入之间的质量分配；V 决定加权读出的内容。Qwen 对 Q/K 使用 RoPE，对 V 不使用；Q/K/V 投影还含偏置。因而让 A=B、B=A 的逆、或统一对两个张量做相同低秩压缩，没有本方案中的数学依据。

K 侧重点：位置结构、当前层 Q 可观测的误差、prefix 归一化量。

V 侧重点：经过实际 attention 之后的输出误差，而非只拟合所有原始 V 行的均方误差。

这里“有结构”不是“结构越强越好”。原模型不同层可采用不同表示基底；纯 RoPE 块结构可能严重欠拟合。先做完整 A/B 的能力检验，再讨论结构化正则是否提升泛化或减少校准数据。

## 3. K reader：RoPE 同频率块先验 + query 度量残差

设 head_dim=d，Qwen split-half RoPE 的频率对为 (j,j+d/2)，不是 (2j,2j+1)。对每个频率对拟合一个块：

```
C_j = [[a_j, -b_j],
       [b_j,  a_j]]
```

把这些块放回 split-half 维度顺序得到 C。它是该频率二维平面上的缩放和旋转，与同一频率的 RoPE 旋转交换。它是先验，不是对原模型跨层关系的断言。

使用 A=C+D，并限制 rank(D)<=r。D 允许修复跨通道、跨频率的残余差异。**只有 C 具有上述交换性；加入一般 D 和 post-RoPE 仿射偏置后，整个 reader 不保证位置等变。**因此本版在实际 post-RoPE 缓存空间拟合，绑定原 prefix 位置和固定 RoPE 配置；不把 A 吸收进 q_proj，也不承诺任意位置迁移或长度外推。

原始 K 误差 e 对某个 Q 的 logit 误差为 q e^T / sqrt(d)。忽略常数后，可用二阶矩：

```
M_Q = E[q^T q] + epsilon I
E[(q e^T)^2] = e M_Q e^T
```

所有校准请求中该 GQA 组的全部 Q heads 都进入 M_Q。四个代表性 selector heads 是在线选择的成本优化，不足以代表所有 attention heads 的读取需求。

将 source/target K 中心化为 X/Y，拟合以下正则化 reduced-rank 回归：

```
min_D || (X D - (Y-X C)) L_Q ||_F^2 + lambda ||D L_Q||_F^2
s.t. rank(D)<=r, M_Q=L_Q L_Q^T
```

实现先做输入 Gram 的 Cholesky 白化，再在 query 度量下截断 SVD，最后通过三角方程恢复 D。没有显式求大矩阵逆，没有反向传播。bk=mean(K_target)-mean(K_source)A。

限制说明：M_Q 是请求分布的二阶近似，不是每个 query-token 联合分布的完整拟合，也没有 token attention 权重。完整无秩约束回归中，右侧正定度量在一些目标形式下会消去；它的主要作用应通过有秩约束的对照证实，不能预先宣称全面提升。

代码模式：

- `dense`：完整 A；最先运行，排除表示容量不足。
- `rope`：仅同频率 C；检验强结构先验是否过度限制。
- `rope_rr`：C 加秩 r 残差；r=0/8/16/32/128 可作对照。
- `direct`：A/B 单位矩阵、偏置零；直接共享的对照。

本版将 A 导出为稠密矩阵，B 也为稠密矩阵。因此 r 是离线拟合限制，不减少本版的在线矩阵存储和乘法数量。分解形式和融合 kernel 属后续性能工作，不能提前计入收益。

## 4. 偏置不能在 prefix/query 分支竞争中丢弃

令 prefix 条件打分为 S=Q A^T K_a^T/sqrt(d)。仿射 K 给每个 prefix token 增加同一偏移：

```
delta = Q bK^T / sqrt(d)
```

它不改变 prefix **内部**的 softmax 权重，但改变 prefix 的分区函数：

```
log Z_prefix = logsumexp(S) + delta
```

当前 query/private tail 仍使用原 Q 和本层原生新 KV，其条件输出与 log Z 为 u_tail,z_tail。共享 prefix 条件输出为 u_prefix，经过仿射 V reader 后为 u_prefix B+bV。

共同归一化的准确合并是：

```
z = logaddexp(z_prefix, z_tail)
beta = exp(z_prefix-z)
out = beta*(u_prefix B+bV) + (1-beta)*u_tail
```

因此 bV 的最终贡献为 beta*bV，而不是直接给整个输出加 bV。独立 softmax 后直接相加、把 B 应用于完整 attention 输出、中心化 K 后忽略偏置，都会改变模型语义。

这些公式与显式构造仿射近似 KV 的 attention 数学等价；不代表仿射近似 KV 等于原模型 KV。实际低精度执行还会引入舍入差异。

## 5. V reader：用实际聚合输出拟合

固定 A/bK 后，对独立校准 query 收集：

- u_s：共享源 V 经新 prefix 权重聚合后的条件输出；
- u_t：原目标层 V 经原 prefix 权重聚合后的条件输出；
- u_private：原模型 private tail 条件输出；
- beta_s/beta_t：新/旧 prefix 在整个 attention 中的质量。

目标不是简单 u_s B≈u_t，而是最终混合输出相近：

```
beta_s*(u_s B+bV)+(1-beta_s)*u_private
  ≈ beta_t*u_t+(1-beta_t)*u_private
```

这是 B/bV 的线性回归：

```
设计矩阵：[beta_s*u_s, beta_s]
回归目标：beta_t*u_t+(beta_s-beta_t)*u_private
```

按物理 KV head 汇集对应的全部 Q heads，使用 ridge 正则。原始 V 的仿射拟合只是先验，避免小样本输出校准任意放大不受约束的方向。这样 V reader 补偿的还包括 A 改变 attention 权重后产生的可线性补偿部分。

局部校准沿用原 P8 sparse teacher 的中间状态。多个层对同时替换后，后续 Q、private KV 和 selector 结果可能改变，局部误差会累积。必须做真实修改模型的端到端验证；本版没有实现共享模型状态的再采集/回灌校准，也没有实现梯度蒸馏。

## 6. 训练、数据与验证的准确说法

默认方案需要离线推理收集数据、解线性方程和 SVD，**不需要主干训练或反向传播**。应称为“冻结主干的无梯度离线校准”，而不是“完全无需校准/数据”。标签不参与 reader 拟合，但最终任务质量验收仍需要任务评测。

fit、局部 validation、最终 end-to-end evaluation 按整个 query 隔离。代码以 query token hash 拒绝重复使用拟合/验证请求进行在线评测。前缀可相同，query 不可重合。验证集筛选层对会产生选择偏差，因此还需保留完全独立的最终测试集。

未来可以尝试只更新 reader 的蒸馏，或共享后重新采集 Q 再拟合；若采用，则不再把该版本宣称为无需反向传播，且必须报告校准/训练成本。当前交付未实现这一阶段。

## 7. 为什么先限定物理共享形式

本版按整个层的四个 KV heads 共享；不做跨层 head permutation。即使目标与源的语义 head 编号不对应也不自动重排，这是实用限制。完整 A 仍失败时，不能断言一定是优化不够：源 head 可能没有目标 head 所需的信息。扩大源子空间、多个源、head 匹配都是后续方向，也会改变传输和 kernel 成本。

P_index=8 不代表 P_KV=8。先做互不重叠的两层共享，且不跨 index 窗口；leader 0/8/16/24 保持 exact，原 K4 selector 不变。每个共享组的选择 IDs 必须相同。低层可读已离线完成的高层 prefix，但 source 所有权必须一跳、无环。

loader 仅 schedule/resolve 唯一物理 source；同一 source 的 FP16→BF16 转换和布局整理只做一次。共享 prefix 不写入每层 DynamicCache，避免尾部追加时产生整段复制；cache 仅保存本层私有新 KV。标签候选缓存 fork 也只复制私有 tail，使用请求级同一只读 prefix。

这一实现仍保留原 SSD 的逐层文件，不等于实现了磁盘压缩。原缓存评分只更新被物理加载的 source；更完整的跨请求 alias-aware 评分不在本版中，应单独研究。

## 8. 评价与否证条件

至少保留 original、identity（新执行路径但无共享）、shared 三组。identity 确认掩码/位置/标签续写/数值正确；shared 不能只打败更慢的 identity，必须与原实现比较。

直接共享和适配共享的机制对比必须使用同一层对/共享预算，否则可能只是后者接受了不同层数。`direct` 默认也经过局部门控；用于固定映射消融时，应记录并核对实际 sources，必要时仅在开发诊断中显式放宽局部门限以保留同一映射，不能混入最终质量安全模式结果。

在相同精度与选择下，某共享组原始所选 payload 为 sum_l |I_l|*2*Hkv*d*b；共享后至少需要并集 |union_l I_l|*2*Hkv*d*b。本版要求相同 I，且每个组只恢复一次。示例九对/28 层全部接受时，等长逻辑 payload 理想减少 9/28≈32.1%；它不是硬件传输或 TTFT 保证。GPU 已命中的块本来就不需要 H2D。

每个 BF16 dense reader 的体积为 2*Hkv*d*d*2+2*Hkv*d*2 bytes；Hkv=4,d=128 时为264192 bytes。请计入全部 task 的 reader、prefix BF16 工作集、private tail、staging、index 和实际缓存副本。

若读取适配与第二分支 attention 的额外关键路径成本超过恢复数据的节省，TTFT 会回退。参考 attention 只用于数学正确性，不是可比较的性能后端。FlashAttention 路径也不是已经融合到一次 kernel 的最终实现。

建议先看以下否证条件：完整 A/B 是否在独立请求上仍严重损失；结构先验是否比完整 A 更差；多层累计替换是否放大误差；物理传输是否真的下降；shared 是否仍慢于 original。报告不满足的结果，而不是隐藏拒绝层对或仅改变相对基线。

主张边界：矩阵移项、低秩/白化、离线 value 校准、物理 KV 跨层共享分别已有先例。潜在贡献只能通过系统实验建立：固定前缀条件下，结构化、归一化保持的读侧适配是否能以可接受的校准成本，把已有冻结模型转成真正减少 prefill 数据恢复的共享缓存形式。不预先断言“首次”或“必然可发表”。
